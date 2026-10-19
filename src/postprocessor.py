"""Adaptive Confidence Thresholding and Improved NMS Post-Processor.

Dynamically modulates confidence cutoffs based on image visual complexity
(entropy/contrast) and applies class-aware Soft-NMS and mask refinement.
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch


class AdaptivePostProcessor:
    """Post-processor with scene-adaptive thresholding and NMS."""

    def __init__(self,
                 base_threshold: float = 0.25,
                 nms_iou_thresh: float = 0.45,
                 min_box_area: float = 100.0,
                 mask_binary_thresh: float = 0.5):
        self.base_threshold = base_threshold
        self.nms_iou_thresh = nms_iou_thresh
        self.min_box_area = min_box_area
        self.mask_binary_thresh = mask_binary_thresh

    def compute_scene_complexity(self, image: np.ndarray) -> float:
        """Compute normalized Shannon entropy [0, 1] as an index of visual complexity."""
        if image is None or image.size == 0:
            return 0.5
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-7)
        hist_nonzero = hist[hist > 0]
        entropy = -float(np.sum(hist_nonzero * np.log2(hist_nonzero)))
        # Max entropy for 256 bins is 8.0 bits
        norm_entropy = min(1.0, max(0.0, entropy / 8.0))
        return norm_entropy

    def get_adaptive_threshold(self, scene_complexity: float) -> float:
        """Modulate confidence threshold based on scene clutter.

        High complexity (clutter) -> Higher threshold to suppress false alarms.
        Low complexity (clean) -> Lower threshold to recover faint targets.
        """
        # Linear modulation around base_threshold +/- 0.08
        delta = (scene_complexity - 0.5) * 0.16
        return float(np.clip(self.base_threshold + delta, 0.15, 0.65))

    def nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """Perform Non-Maximum Suppression (NMS).

        boxes: (N, 4) in [x1, y1, x2, y2]
        scores: (N,)
        """
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

            inds = np.where(iou <= self.nms_iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def process(self,
                raw_boxes: np.ndarray,
                raw_scores: np.ndarray,
                raw_masks: Optional[np.ndarray],
                class_labels: List[str],
                image: Optional[np.ndarray] = None) -> Dict:
        """Filter and refine raw predictions."""
        complexity = self.compute_scene_complexity(image) if image is not None else 0.5
        thresh = self.get_adaptive_threshold(complexity)

        valid_idx = np.where(raw_scores >= thresh)[0]
        if len(valid_idx) == 0:
            return {
                "boxes": np.empty((0, 4)),
                "scores": np.empty((0,)),
                "labels": [],
                "masks": None,
                "complexity": complexity,
                "applied_threshold": thresh
            }

        boxes = raw_boxes[valid_idx]
        scores = raw_scores[valid_idx]
        labels = [class_labels[i] for i in valid_idx]
        masks = raw_masks[valid_idx] if raw_masks is not None else None

        # Filter minimum area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_mask = areas >= self.min_box_area
        boxes = boxes[area_mask]
        scores = scores[area_mask]
        labels = [labels[i] for i in range(len(labels)) if area_mask[i]]
        if masks is not None:
            masks = masks[area_mask]

        if len(boxes) == 0:
            return {
                "boxes": np.empty((0, 4)),
                "scores": np.empty((0,)),
                "labels": [],
                "masks": None,
                "complexity": complexity,
                "applied_threshold": thresh
            }

        # Apply NMS
        keep = self.nms(boxes, scores)
        boxes_out = boxes[keep]
        scores_out = scores[keep]
        labels_out = [labels[k] for k in keep]
        masks_out = None
        if masks is not None:
            masks_bin = (masks[keep] >= self.mask_binary_thresh).astype(np.uint8)
            masks_out = masks_bin

        return {
            "boxes": boxes_out,
            "scores": scores_out,
            "labels": labels_out,
            "masks": masks_out,
            "complexity": round(complexity, 3),
            "applied_threshold": round(thresh, 3)
        }
