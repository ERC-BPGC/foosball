# code/custom_d_fine/src/dl/infer.py
from pathlib import Path
from shutil import rmtree
import csv

import cv2
import hydra
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

from src.dl.utils import abs_xyxy_to_norm_xywh, get_latest_experiment_name, vis_one_box
from src.infer.torch_model import Torch_model
from src.filters.head_filter import (
    HeadFilter,
    expand_around_box,
    is_head_distractor,
)

def figure_input_type(folder_path: Path):
    video_types = ["mp4", "avi", "mov", "mkv"]
    img_types = ["jpg", "png", "jpeg"]
    data_type = "image"
    for f in folder_path.iterdir():
        if f.is_file() and f.suffix[1:].lower() in video_types:
            data_type = "video"
            break
        elif f.is_file() and f.suffix[1:].lower() in img_types:
            data_type = "image"
            break
    logger.info(f"Inferencing on data type: {data_type}, path: {folder_path}")
    return data_type

def draw_box(img, box, color, thickness=2, label_text=None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
    if label_text:
        cv2.putText(img, label_text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def save_yolo_annotations_with_conf(res, output_path, img_stem, img_shape):
    """
    Write 'class x y w h conf'
    """
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{img_stem}.txt", "a") as f:
        for class_id, box, score in zip(res["labels"], res["boxes"], res["scores"]):
            norm_box = abs_xyxy_to_norm_xywh(np.asarray([box]), img_shape[0], img_shape[1])[0]
            f.write(f"{int(class_id)} {norm_box[0]} {norm_box[1]} {norm_box[2]} {norm_box[3]} {float(score):.6f}\n")

def ensure_clean_dir(p: Path):
    if p.exists():
        rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def run_images(torch_model, head_filter, cfg, folder_path, out_root, label_to_name):
    """
    Full-frame inference with D-FINE -> head filter per ball box.
    """
    # Outputs
    detections_dir = out_root / "detections"
    distractors_dir = out_root / "distractors"
    crops_dir = out_root / "dfine_crops"

    # create subfolders for images + labels
    (detections_dir / "images").mkdir(parents=True, exist_ok=True)
    (detections_dir / "labels").mkdir(parents=True, exist_ok=True)
    (distractors_dir / "images").mkdir(parents=True, exist_ok=True)
    (distractors_dir / "labels").mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    img_paths = sorted([p for p in folder_path.iterdir() if p.is_file() and not p.name.startswith(".")])

    crop_scale = float(cfg.filter.crop_scale)
    prox_factor = float(cfg.filter.proximity_factor)
    min_overlap = float(cfg.filter.min_overlap)
    save_dfine_crops = cfg.outputs.save_dfine_crops
    save_labels = cfg.outputs.save_labels
    save_detections = cfg.outputs.save_detections
    save_distractors = cfg.outputs.save_distractors  # for labels only

    for img_path in tqdm(img_paths, desc="Frames"):
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to read {img_path}")
            continue
        or_img = img.copy()
        res = torch_model(img)
        res = {"boxes": res[0]["boxes"], "labels": res[0]["labels"], "scores": res[0]["scores"]}

        kept_boxes, kept_scores, kept_labels = [], [], []
        dropped_boxes, dropped_scores, dropped_labels = [], [], []
        heads_this_frame = 0
        ball_candidates = [(b, s, l) for b, s, l in zip(res["boxes"], res["scores"], res["labels"])]

        for box, score, label in ball_candidates:
            x1, y1, x2, y2 = map(int, box.tolist())
            cx1, cy1, cx2, cy2 = expand_around_box((x1, y1, x2, y2), img.shape[0], img.shape[1], scale=crop_scale)
            crop = or_img[cy1:cy2, cx1:cx2]

            if save_dfine_crops:
                cv2.imwrite(str(crops_dir / f"{stem}_{x1}_{y1}_{x2}_{y2}.jpg"), crop)

            head_dets = head_filter.detect(crop) if head_filter is not None else []
            head_dets_global = []
            for hd in head_dets:
                hx1, hy1, hx2, hy2 = hd.box
                head_dets_global.append(type(hd)(box=(hx1 + cx1, hy1 + cy1, hx2 + cx1, hy2 + cy1), conf=hd.conf))

            is_dist, culprit = is_head_distractor(
                (x1, y1, x2, y2),
                (cx1, cy1, cx2, cy2),
                head_dets_global,
                proximity_factor=prox_factor,
                min_overlap=min_overlap,
            )
            if is_dist and (culprit is not None) and culprit.conf >= float(cfg.filter.head_conf_threshold):
                dropped_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                dropped_scores.append(score)
                dropped_labels.append(label)
                heads_this_frame += 1

                # Always save annotated distractor image with ALL heads
                dbg = or_img.copy()
                draw_box(dbg, (x1, y1, x2, y2), (0, 0, 255), 2, label_text="BALL_DROP")
                for hd in head_dets_global:
                    color = (0, 255, 255) if hd is culprit else (255, 0, 0)
                    tag = f"HEAD {hd.conf:.2f}"
                    draw_box(dbg, hd.box, color, 2, label_text=tag)
                cv2.imwrite(str(distractors_dir / "images" / f"{stem}.jpg"), dbg)

            else:
                kept_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                kept_scores.append(score)
                kept_labels.append(label)

        kept = {
            "boxes": np.array(kept_boxes, dtype=np.float32) if kept_boxes else np.zeros((0,4), dtype=np.float32),
            "labels": np.array(kept_labels, dtype=np.int64) if kept_labels else np.zeros((0,), dtype=np.int64),
            "scores": np.array(kept_scores, dtype=np.float32) if kept_scores else np.zeros((0,), dtype=np.float32),
        }
        dropped = {
            "boxes": np.array(dropped_boxes, dtype=np.float32) if dropped_boxes else np.zeros((0,4), dtype=np.float32),
            "labels": np.array(dropped_labels, dtype=np.int64) if dropped_labels else np.zeros((0,), dtype=np.int64),
            "scores": np.array(dropped_scores, dtype=np.float32) if dropped_scores else np.zeros((0,), dtype=np.float32),
        }

        if cfg.outputs.save_images and save_detections and len(kept["boxes"]):
            img_keep = or_img.copy()
            for b, l, s in zip(kept["boxes"], kept["labels"], kept["scores"]):
                vis_one_box(img_keep, b, l, mode="pred", label_to_name=label_to_name, score=float(s))
            cv2.imwrite(str((detections_dir / "images" / f"{stem}.jpg")), img_keep)

        if save_labels:
            if save_detections and len(kept["boxes"]):
                save_yolo_annotations_with_conf(kept, detections_dir / "labels", stem, img_shape=img.shape)
            if save_distractors and len(dropped["boxes"]):
                save_yolo_annotations_with_conf(dropped, distractors_dir / "labels", stem, img_shape=img.shape)

        stats_rows.append({
            "frame": stem,
            "dfine_candidates": len(ball_candidates),
            "kept_after_head_filter": int(len(kept["boxes"])),
            "distractors": int(len(dropped["boxes"])),
            "heads_detected_in_crop": int(heads_this_frame),
        })

    if cfg.outputs.csv_stats and stats_rows:
        csv_path = out_root / "all_frame_stats.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            w.writeheader()
            w.writerows(stats_rows)

        total_frames = len(stats_rows)
        total_candidates = sum(r["dfine_candidates"] for r in stats_rows)
        total_kept = sum(r["kept_after_head_filter"] for r in stats_rows)
        total_dropped = sum(r["distractors"] for r in stats_rows)
        total_heads = sum(r["heads_detected_in_crop"] for r in stats_rows)

        logger.info(f"Processed {total_frames} frames")
        logger.info(f"Total ball candidates: {total_candidates}")
        logger.info(f"Kept after head filter: {total_kept}")
        logger.info(f"Dropped as distractors: {total_dropped}")
        logger.info(f"Heads detected in crops: {total_heads}")
        logger.info(f"CSV stats saved to {csv_path}")

def run_videos(*args, **kwargs):
    raise NotImplementedError("Video mode not implemented for head-filter yet. Convert to frames first.")

@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=cfg.train.pretrained_model_path,
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=cfg.train.conf_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
    )

    head_filter = None
    if bool(cfg.filter.enabled):
        logger.info(f"Loading head filter: {cfg.filter.head_model_path}")
        head_filter = HeadFilter(
            model_path=str(cfg.filter.head_model_path),
            conf_threshold=float(cfg.filter.head_conf_threshold),
        )

    folder_path = Path(str(cfg.train.path_to_test_data))
    data_type = figure_input_type(folder_path)

    out_root = Path(cfg.train.infer_path)
    ensure_clean_dir(out_root)

    if data_type == "image":
        run_images(
            torch_model=torch_model,
            head_filter=head_filter,
            cfg=cfg,
            folder_path=folder_path,
            out_root=out_root,
            label_to_name=cfg.train.label_to_name,
        )
    else:
        run_videos()

if __name__ == "__main__":
    main()
