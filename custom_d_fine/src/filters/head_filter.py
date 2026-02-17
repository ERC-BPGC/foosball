# code/custom_d_fine/src/filters/head_filter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO

@dataclass
class HeadDet:
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2
    conf: float

class HeadFilter:
    """
    Thin wrapper around a YOLOv8 head detector.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.15,
        device: Optional[str] = None,
    ):
        self.model = YOLO(model_path)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.conf_threshold = conf_threshold

    @torch.inference_mode()
    def detect(self, img_bgr: np.ndarray) -> List[HeadDet]:
        """
        Runs YOLOv8 on a BGR crop and returns head detections in xyxy (abs) + conf.
        """
        # Ultralytics YOLO expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.model.predict(
            img_rgb, imgsz=max(img_rgb.shape[:2]), conf=self.conf_threshold, verbose=False, device=self.device
        )
        out: List[HeadDet] = []
        if not res:
            return out
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return out

        boxes = r0.boxes.xyxy.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy().astype(float)
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            out.append(HeadDet((int(x1), int(y1), int(x2), int(y2)), float(c)))
        return out


def box_center_xyxy(xyxy: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_diag(xyxy: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = xyxy
    return math.hypot(x2 - x1, y2 - y1)


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def expand_around_box(
    box_xyxy: Tuple[int, int, int, int],
    img_h: int,
    img_w: int,
    scale: float,
) -> Tuple[int, int, int, int]:
    """
    Expand square region around a box center by (scale * max(w,h)).
    """
    x1, y1, x2, y2 = box_xyxy
    bw, bh = (x2 - x1), (y2 - y1)
    size = int(scale * max(bw, bh))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = size // 2
    nx1 = max(0, cx - half)
    ny1 = max(0, cy - half)
    nx2 = min(img_w - 1, cx + half)
    ny2 = min(img_h - 1, cy + half)
    # ensure >= 1px
    if nx2 <= nx1: nx2 = min(img_w - 1, nx1 + 1)
    if ny2 <= ny1: ny2 = min(img_h - 1, ny1 + 1)
    return int(nx1), int(ny1), int(nx2), int(ny2)


def is_head_distractor(
    ball_box: Tuple[int, int, int, int],
    crop_rect: Tuple[int, int, int, int],
    head_dets: List[HeadDet],
    proximity_factor: float,
    min_overlap: float,
) -> Tuple[bool, Optional[HeadDet]]:
    """
    Decide if at least one head is plausibly the 'distractor' for this ball detection.
    Guards against far audience by using:
      - proximity of head center to ball center, measured in ball diagonals
      - small but nonzero overlap between head and crop rect
    """
    if not head_dets:
        return False, None

    b_diag = max(1e-6, box_diag(ball_box))
    b_cx, b_cy = box_center_xyxy(ball_box)

    chosen: Optional[HeadDet] = None
    chosen_dist = float("inf")

    for hd in head_dets:
        hx, hy = box_center_xyxy(hd.box)
        dist = math.hypot(hx - b_cx, hy - b_cy) / b_diag
        ov = iou_xyxy(hd.box, crop_rect)  # quick overlap proxy
        if dist <= proximity_factor and ov >= min_overlap:
            if dist < chosen_dist:
                chosen = hd
                chosen_dist = dist

    return (chosen is not None), chosen
