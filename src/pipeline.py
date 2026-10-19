"""Unified End-to-End LLM Semantic Detection & Segmentation Pipeline.

Ties together:
1. LLM Semantic Enhancer (attribute augmentation & persistent caching)
2. Open-Vocabulary Dense Detector with Asymmetric LoRA Cross-Modal Alignment
3. Scene-Entropy Adaptive Post-Processor (Dynamic thresholding & NMS)
4. Comprehensive Visualizer (colored masks, bounding boxes, labels)
"""

import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from src.detector import OpenVocabularySemanticDetector
from src.llm_enhancer import LLMEnhancer
from src.postprocessor import AdaptivePostProcessor

logger = logging.getLogger("LLMSemanticPipeline")


class LLMSemanticPipeline:
    """Production open-vocabulary detection and segmentation pipeline."""

    def __init__(self,
                 llm_model: Optional[str] = "rule_based",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cache_file: str = "cache/semantic_cache.json",
                 base_threshold: float = 0.25):
        self.device = device
        logger.info("Initializing LLMSemanticPipeline on device: %s", device)

        self.enhancer = LLMEnhancer(model_name_or_path=llm_model, cache_file=cache_file, device=device)
        self.detector = OpenVocabularySemanticDetector(device=device)
        self.postprocessor = AdaptivePostProcessor(base_threshold=base_threshold)

    def predict(self,
                image: Union[str, np.ndarray],
                queries: List[str],
                enable_llm_enhancement: bool = True) -> Dict:
        """Run end-to-end open-vocabulary detection and segmentation.

        Args:
            image: File path string or BGR numpy array (HxWx3)
            queries: List of arbitrary object names or descriptions (e.g. ['car', 'solar panel'])
            enable_llm_enhancement: If True, uses LLM/ontology semantic expansion; if False, bare queries.

        Returns:
            Dictionary containing boxes, scores, labels, masks, timings, and metadata.
        """
        # Load image if path
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise FileNotFoundError(f"Image file not found: {image}")
        else:
            img_bgr = image.copy()

        t0 = time.time()

        # Step 1: LLM Semantic Enhancement
        t_llm_start = time.time()
        enriched_prompts = []
        for q in queries:
            if enable_llm_enhancement:
                data = self.enhancer.enhance(q)
                enriched_prompts.append(data.get("combined_prompt", q))
            else:
                enriched_prompts.append(q)
        llm_time_ms = (time.time() - t_llm_start) * 1000.0

        # Step 2: Open-Vocabulary Cross-Modal Dense Inference
        t_det_start = time.time()
        raw_res = self.detector.forward_prediction(
            image_bgr=img_bgr,
            query_texts=enriched_prompts,
            query_labels=queries
        )
        detector_time_ms = (time.time() - t_det_start) * 1000.0

        # Step 3: Adaptive Post-Processing & Mask Generation
        t_post_start = time.time()
        final_res = self.postprocessor.process(
            raw_boxes=raw_res["boxes"],
            raw_scores=raw_res["scores"],
            raw_masks=raw_res["masks"],
            class_labels=raw_res["labels"],
            image=img_bgr
        )
        post_time_ms = (time.time() - t_post_start) * 1000.0
        total_time_ms = (time.time() - t0) * 1000.0

        final_res["timings_ms"] = {
            "llm_enhancement": round(llm_time_ms, 2),
            "detector_inference": round(detector_time_ms, 2),
            "post_processing": round(post_time_ms, 2),
            "total_latency": round(total_time_ms, 2)
        }
        final_res["fps"] = round(1000.0 / max(1.0, total_time_ms), 1)
        final_res["enriched_prompts"] = enriched_prompts
        return final_res

    def visualize(self,
                  image: Union[str, np.ndarray],
                  result: Dict,
                  output_path: Optional[str] = None) -> np.ndarray:
        """Render colored instance masks, bounding boxes, and labels onto image."""
        if isinstance(image, str):
            canvas = cv2.imread(image)
        else:
            canvas = image.copy()

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        masks = result.get("masks", None)

        colors = [
            (255, 60, 60), (60, 220, 60), (60, 60, 255),
            (240, 200, 20), (220, 50, 220), (50, 220, 220)
        ]

        # Draw semi-transparent masks
        if masks is not None and len(masks) > 0:
            mask_overlay = canvas.copy()
            for i, mask in enumerate(masks):
                color = colors[i % len(colors)]
                mask_bool = mask.astype(bool)
                mask_overlay[mask_bool] = color
            cv2.addWeighted(mask_overlay, 0.45, canvas, 0.55, 0, canvas)

        # Draw bounding boxes and text
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            score = scores[i]
            label = labels[i]
            color = colors[i % len(colors)]

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            caption = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(canvas, caption, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), canvas)
            logger.info("Saved visualization to %s", output_path)

        return canvas
