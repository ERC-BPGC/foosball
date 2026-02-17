#!/usr/bin/env python3
"""
infer_with_filter.py
Fully patched version that integrates YOLOv8 ONNX head detector as a filter
for DFINE ball detections.

Key behavior:
 - Run primary detector (Torch_model) on each frame.
 - For each ball detection, crop an ROI (with configurable padding).
 - Run head detector on ROI; convert ROI-local head boxes back to full-frame coords.
 - Compute IoU between ball box and head boxes (global coords).
 - If IoU > cfg.filter.iou_threshold OR head center inside ball AND head score >=
   cfg.filter.head_conf_threshold, mark as distractor.
 - Save ROI crops, annotated images, YOLO-format label text files, debug visuals.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Sequence

import cv2
import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.dl.utils import (
    abs_xyxy_to_norm_xywh,
    get_latest_experiment_name,
    vis_one_box,
)
from src.infer.torch_model import Torch_model
from src.filters.yolov8_head_detector import YOLOv8HeadDetector


def figure_input_type(folder_path: Path) -> str:
    video_types = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']
    img_types = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
    data_type = 'images'
    for file in os.listdir(folder_path):
        if Path(file).suffix in video_types:
            data_type = 'videos'
            break
        elif Path(file).suffix in img_types:
            data_type = 'images'
    logger.info(f'Data type in {folder_path} is {data_type}')
    return data_type


def save_yolo_annotations(output_path: Path, img_name: str,
                          results: Dict[str, np.ndarray],
                          or_img_shape: Tuple[int, int, int]) -> None:
    """
    Save YOLO-format annotations (class + normalized xywh) for an image.
    Expects results['boxes'] to be absolute xyxy arrays.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    height, width = or_img_shape[:2]
    if results['boxes'].shape[0] > 0:
        with open(output_path / f'{img_name}.txt', 'a') as f:
            for class_id, box in zip(results['labels'], results['boxes']):
                box_2d = box[None]  # (4,) -> (1,4)
                norm_box = abs_xyxy_to_norm_xywh(box_2d, height, width)[0]
                f.write(f'{int(class_id)} {" ".join(map(str, norm_box))}\n')


def visualize(output_path: Path, img_name: str, or_img: np.ndarray,
              results: Dict[str, np.ndarray], label_to_name: Dict[int, str],
              mode: str) -> None:
    """
    Draw boxes (using vis_one_box) and save annotated image.
    mode receives 'ball' or 'distractor' -> mapped to 'pred' for vis_one_box.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    vis_mode = "pred" if mode in ["ball", "distractor"] else mode

    if results['boxes'].shape[0] > 0:
        for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
            box = np.array(box).flatten()
            vis_one_box(
                img=or_img,
                box=box,
                label=int(label),
                mode=vis_mode,
                label_to_name=label_to_name,
                score=float(score),
            )
        cv2.imwrite(str(output_path / f'{img_name}.jpg'), or_img)


# Robust IoU calculation (works for lists, tuples, ndarrays)
def local_iou(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """
    Compute IoU between two boxes specified as [x1, y1, x2, y2].
    Returns IoU in [0,1]. Safe for degenerate boxes.
    """
    a = np.array(boxA, dtype=float)
    b = np.array(boxB, dtype=float)

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    areaA = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    areaB = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

    denom = areaA + areaB - interArea
    if denom <= 0.0:
        return 0.0
    return float(interArea / denom)


def head_center_inside(ball_box: Sequence[float], head_box: Sequence[float]) -> bool:
    """
    Return True if the head box center lies inside the ball box.
    Boxes are [x1,y1,x2,y2].
    """
    hx1, hy1, hx2, hy2 = head_box
    cx = (hx1 + hx2) / 2.0
    cy = (hy1 + hy2) / 2.0
    return (ball_box[0] <= cx <= ball_box[2]) and (ball_box[1] <= cy <= ball_box[3])


def draw_heads_on_image(img: np.ndarray, head_boxes_global: List[Sequence[int]]) -> np.ndarray:
    """
    Draw red rectangles for all head boxes on a full-frame copy and return it.
    """
    out = img.copy()
    for hb in head_boxes_global:
        x1, y1, x2, y2 = [int(round(v)) for v in hb]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out


def run_inference_on_frames(cfg: DictConfig, torch_model: Torch_model,
                            head_detector: Optional[YOLOv8HeadDetector],
                            folder_path: Path, output_path: Path):
    """
    Main processing loop. Saves same outputs as original infer.py plus debug visuals:
      - output/infer/.../ball/images and labels
      - output/infer/.../distractors/images and labels
      - output/infer/.../roi_crops/ball and /distractors
      - output/infer/.../debug_visuals (roi-local annotated)
      - output/infer/.../debug_heads (full-frame with head boxes)
    """
    distractors_img_path = output_path / 'distractors' / 'images'
    distractors_label_path = output_path / 'distractors' / 'labels'
    ball_img_path = output_path / 'ball' / 'images'
    ball_label_path = output_path / 'ball' / 'labels'
    debug_vis_path = output_path / 'debug_visuals'
    roi_ball_path = output_path / 'roi_crops' / 'ball'
    roi_distractor_path = output_path / 'roi_crops' / 'distractors'
    debug_heads_path = output_path / 'debug_heads'

    for p in [distractors_img_path, distractors_label_path, ball_img_path, ball_label_path,
              roi_ball_path, roi_distractor_path, debug_vis_path, debug_heads_path]:
        p.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([folder_path / f for f in os.listdir(folder_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    total_images = len(image_paths)
    logger.info(f"Found {total_images} images in {folder_path}")

    total_ball = 0
    total_distractors = 0

    for img_idx, img_path in enumerate(tqdm(image_paths, desc='Processing frames'), start=1):
        img_name = img_path.stem
        or_img = cv2.imread(str(img_path))
        if or_img is None:
            logger.warning(f'Could not read image: {img_path}, skipping.')
            continue
        h_img, w_img = or_img.shape[:2]

        # -------------- run DFINE / Torch_model ----------------
        try:
            detections = torch_model(or_img)
            if not detections or not isinstance(detections, list) or not isinstance(detections[0], dict):
                logger.error(f"Invalid model output format on {img_name}: {detections}")
                continue
            boxes = detections[0].get('boxes', np.empty((0, 4)))
            scores = detections[0].get('scores', np.empty((0,)))
            labels = detections[0].get('labels', np.empty((0,)))
        except Exception as e:
            logger.error(f'Inference failed on {img_name}: {e}')
            continue

        initial_detections = {'boxes': boxes, 'scores': scores, 'labels': labels}
        final_ball_detections = {'boxes': [], 'scores': [], 'labels': []}
        distractor_detections = {'boxes': [], 'scores': [], 'labels': []}
        frame_head_boxes_global: List[List[int]] = []  # accumulate for global debug overlay

        # If there are any detections
        if initial_detections['scores'].size > 0 and initial_detections['scores'].any():
            for i, (box, score, label) in enumerate(zip(initial_detections['boxes'],
                                                        initial_detections['scores'],
                                                        initial_detections['labels'])):
                # Expect box = [x1, y1, x2, y2]
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)

                # Expand ROI by padding_scale (configurable)
                pad_w = int(box_w * cfg.filter.padding_scale)
                pad_h = int(box_h * cfg.filter.padding_scale)

                roi_x1 = max(0, x1 - pad_w)
                roi_y1 = max(0, y1 - pad_h)
                roi_x2 = min(w_img, x2 + pad_w)
                roi_y2 = min(h_img, y2 + pad_h)

                # Correct ROI cropping
                roi_crop = or_img[roi_y1:roi_y2, roi_x1:roi_x2].copy()

                # If ROI is too small, skip head check and keep detection
                if roi_crop.size == 0 or roi_crop.shape[0] < cfg.filter.min_roi_size or roi_crop.shape[1] < cfg.filter.min_roi_size:
                    final_ball_detections['boxes'].append(box)
                    final_ball_detections['scores'].append(score)
                    final_ball_detections['labels'].append(label)
                    try:
                        cv2.imwrite(str(roi_ball_path / f"{img_name}_det_{i}.jpg"), roi_crop)
                    except Exception:
                        pass
                    continue

                # Run head detector on ROI (returns list of (x, y, w, h, score) relative to ROI)
                try:
                    head_results = head_detector.detect(roi_crop.copy()) if head_detector is not None else []
                except Exception as e:
                    logger.warning(f"[{img_name}] Head detector failed on ROI: {e}")
                    head_results = []
                logger.debug(f"[{img_name}] det#{i} head_results: {head_results}")

                max_iou = 0.0
                best_head_roi = None
                head_boxes_global_for_roi: List[List[int]] = []

                # Parse head_results robustly
                if head_results:
                    for hr in head_results:
                        # hr expected to be (x, y, w, h, score) using ROI-local coords OR sometimes (x1,y1,x2,y2,score)
                        try:
                            if len(hr) >= 5:
                                # candidate common layout: (x, y, w, h, score)
                                hx, hy, hw, hh, hscore = hr[:5]
                                # If hw/hh look like widths/heights (positive) use that, else interpret as x1,y1,x2,y2
                                if hw > 0 and hh > 0:
                                    hx1_local = int(round(hx))
                                    hy1_local = int(round(hy))
                                    hx2_local = int(round(hx + hw))
                                    hy2_local = int(round(hy + hh))
                                else:
                                    # fallback treat hr as xyxy maybe
                                    hx1_local = int(round(hr[0]))
                                    hy1_local = int(round(hr[1]))
                                    hx2_local = int(round(hr[2]))
                                    hy2_local = int(round(hr[3]))
                                    hscore = float(hr[4])
                            else:
                                # not the expected length; skip safely
                                continue
                        except Exception as e:
                            logger.debug(f"[{img_name}] Skipping malformed head result {hr}: {e}")
                            continue

                        # convert ROI-local to global coords
                        hx1_global = roi_x1 + hx1_local
                        hy1_global = roi_y1 + hy1_local
                        hx2_global = roi_x1 + hx2_local
                        hy2_global = roi_y1 + hy2_local

                        head_boxes_global_for_roi.append([hx1_global, hy1_global, hx2_global, hy2_global])

                        # compute IoU against the ball box (global)
                        ball_box_global = [float(x1), float(y1), float(x2), float(y2)]
                        head_box_global = [float(hx1_global), float(hy1_global), float(hx2_global), float(hy2_global)]

                        try:
                            iou_val = local_iou(ball_box_global, head_box_global)
                        except Exception as e:
                            logger.error(f"[{img_name}] IoU calc failed: {e}")
                            iou_val = 0.0

                        # debug log for every head vs ball
                        logger.debug(f"[{img_name}] det#{i} head_iou={iou_val:.3f} head_score={float(hscore):.6f}")

                        if iou_val > max_iou:
                            max_iou = iou_val
                            best_head_roi = (hx1_local, hy1_local, hx2_local, hy2_local, float(hscore))

                # add heads found in this ROI to the frame-wide list for global debug overlay
                for hb in head_boxes_global_for_roi:
                    if hb not in frame_head_boxes_global:
                        frame_head_boxes_global.append(hb)

                # Decide distractor based on IoU threshold OR center-inside + head score
                is_distractor = False
                reason = None
                if max_iou > cfg.filter.iou_threshold:
                    is_distractor = True
                    reason = f"iou>{cfg.filter.iou_threshold:.2f}"
                else:
                    # center-inside fallback using best_head_roi
                    if best_head_roi:
                        bx1_l, by1_l, bx2_l, by2_l, best_score = best_head_roi
                        bx1_g = roi_x1 + bx1_l
                        by1_g = roi_y1 + by1_l
                        bx2_g = roi_x1 + bx2_l
                        by2_g = roi_y1 + by2_l
                        if head_center_inside([x1, y1, x2, y2], [bx1_g, by1_g, bx2_g, by2_g]):
                            if float(best_score) >= cfg.filter.head_conf_threshold:
                                is_distractor = True
                                reason = "center-inside+score"

                if is_distractor:
                    distractor_detections['boxes'].append(box)
                    distractor_detections['scores'].append(score)
                    distractor_detections['labels'].append(label)
                    total_distractors += 1
                    # save ROI distractor crop
                    try:
                        cv2.imwrite(str(roi_distractor_path / f"{img_name}_det_{i}.jpg"), roi_crop)
                    except Exception:
                        pass

                    # save debug visual of ROI with ball and best head
                    if cfg.filter.save_debug_visuals:
                        debug_img = roi_crop.copy()
                        ball_local = (x1 - roi_x1, y1 - roi_y1, x2 - roi_x1, y2 - roi_y1)
                        cv2.rectangle(debug_img, (int(ball_local[0]), int(ball_local[1])),
                                      (int(ball_local[2]), int(ball_local[3])), (0, 255, 255), 2)
                        if best_head_roi:
                            bx1, by1, bx2, by2, _ = best_head_roi
                            cv2.rectangle(debug_img, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 2)
                        cv2.putText(debug_img, f"maxIoU:{max_iou:.3f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)
                        try:
                            cv2.imwrite(str(debug_vis_path / f'{img_name}_det_{i}_distractor.jpg'), debug_img)
                        except Exception:
                            pass
                else:
                    final_ball_detections['boxes'].append(box)
                    final_ball_detections['scores'].append(score)
                    final_ball_detections['labels'].append(label)
                    total_ball += 1
                    try:
                        cv2.imwrite(str(roi_ball_path / f"{img_name}_det_{i}.jpg"), roi_crop)
                    except Exception:
                        pass
        else:
            logger.debug(f"[{img_name}] No detections above confidence threshold.")

        # Convert lists to numpy arrays for saving/visualization
        for d in [final_ball_detections, distractor_detections]:
            d['boxes'] = np.array(d['boxes']) if d['boxes'] else np.empty((0, 4))
            d['scores'] = np.array(d['scores']) if d['scores'] else np.empty((0,))
            d['labels'] = np.array(d['labels']) if d['labels'] else np.empty((0,))

        # Save full-frame head debug overlay (all heads detected in this frame)
        if frame_head_boxes_global:
            try:
                global_debug_img = draw_heads_on_image(or_img, frame_head_boxes_global)
                cv2.imwrite(str(debug_heads_path / f"{img_name}_heads.jpg"), global_debug_img)
            except Exception:
                pass

        # Save annotated images and YOLO labels as in original infer.py
        vis_img = or_img.copy()
        if distractor_detections['boxes'].shape[0] > 0:
            visualize(distractors_img_path, img_name, vis_img, distractor_detections, cfg.train.label_to_name, mode="distractor")
            save_yolo_annotations(distractors_label_path, img_name, distractor_detections, or_img.shape)

        if final_ball_detections['boxes'].shape[0] > 0:
            visualize(ball_img_path, img_name, vis_img, final_ball_detections, cfg.train.label_to_name, mode="ball")
            save_yolo_annotations(ball_label_path, img_name, final_ball_detections, or_img.shape)

        # Logging per-image summary
        logger.debug(f"[{img_name}] final_ball: {len(final_ball_detections['boxes'])}, distractors: {len(distractor_detections['boxes'])}")
        logger.debug(f"Processed {img_idx}/{total_images}")

    # Summary
    logger.info(f"Processed {total_images} images")
    logger.info(f"Ball detections kept: {total_ball}")
    logger.info(f"Distractor detections filtered: {total_distractors}")
    logger.info(f"ROI ball crops saved: {len(list(roi_ball_path.glob('*.jpg')))}")
    logger.info(f"ROI distractor crops saved: {len(list(roi_distractor_path.glob('*.jpg')))}")
    if cfg.filter.save_debug_visuals:
        logger.info(f"Debug ROI visuals saved: {len(list(debug_vis_path.glob('*.jpg')))}")
    logger.info(f"Global head debug images saved: {len(list(debug_heads_path.glob('*.jpg')))}")


@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    # Set experiment name
    cfg.exp = get_latest_experiment_name(cfg.exp_name, cfg.train.path_to_save)
    logger.info(f'Current experiment: {cfg.exp}')

    # Instantiate Torch model (DFINE)
    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=cfg.train.pretrained_model_path,
        n_outputs=1,
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=cfg.train.conf_thresh,
        keep_ratio=cfg.train.keep_ratio,
        half=cfg.train.amp_enabled,
        use_nms=True,
        device=cfg.train.device,
    )

    # Instantiate YOLOv8 head detector if enabled
    head_detector = None
    if cfg.filter.enabled:
        logger.info('Initializing YOLOv8 head detector for filtering...')
        head_detector = YOLOv8HeadDetector(
            model_path=cfg.filter.head_model_path,
            conf_threshold=cfg.filter.head_conf_threshold,
            iou_threshold=getattr(cfg.filter, "head_nms_threshold", 0.45),
            input_size=getattr(cfg.filter, "head_input_size", 640),
        )
        logger.info('Head detector initialized successfully.')

    folder_path = Path(cfg.train.path_to_test_data)
    output_path = Path(cfg.train.infer_path) / cfg.exp

    if output_path.exists():
        logger.warning(f'Output path {output_path} already exists. Removing it.')
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # Confirm input type and run inference loop
    figure_input_type(folder_path)
    run_inference_on_frames(cfg, torch_model, head_detector, folder_path, output_path)

    logger.info('Inference with filtering completed.')
    logger.info(f'Outputs saved to: {output_path}')


if __name__ == '__main__':
    main()
