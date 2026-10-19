"""CLI Demonstration for Lightweight LLM Semantic Detection & Segmentation."""

import argparse
from pathlib import Path
import sys
import numpy as np
import cv2

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import LLMSemanticPipeline


def create_synthetic_test_image(output_path: str = "demo/sample_scene.jpg") -> str:
    """Generate a synthetic urban scene with distinct objects if no input image is provided."""
    h, w = 640, 640
    img = np.full((h, w, 3), (230, 230, 230), dtype=np.uint8)

    # Road surface
    cv2.rectangle(img, (0, 380), (w, h), (70, 70, 70), -1)
    # Road dashed line
    for x in range(0, w, 60):
        cv2.rectangle(img, (x, 500), (x + 35, 510), (255, 255, 255), -1)

    # Car (Blue metallic box with roof and wheels)
    cv2.rectangle(img, (120, 360), (320, 470), (180, 50, 40), -1)  # Blue body (BGR)
    cv2.rectangle(img, (160, 310), (280, 360), (220, 180, 150), -1)  # Cabin/windows
    cv2.circle(img, (170, 475), 28, (30, 30, 30), -1)  # Front wheel
    cv2.circle(img, (270, 475), 28, (30, 30, 30), -1)  # Rear wheel

    # Traffic cone (Bright orange cone with white stripe)
    cone_pts = np.array([[460, 480], [420, 560], [500, 560]], dtype=np.int32)
    cv2.fillPoly(img, [cone_pts], (20, 140, 240))  # Orange (BGR)
    cv2.rectangle(img, (400, 560), (520, 575), (20, 140, 240), -1)  # Base
    # White retro-reflective stripe
    stripe_pts = np.array([[445, 510], [435, 530], [485, 530], [475, 510]], dtype=np.int32)
    cv2.fillPoly(img, [stripe_pts], (240, 240, 240))

    # Sky and sun
    cv2.circle(img, (540, 90), 45, (60, 220, 255), -1)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), img)
    return str(out_p)


def main():
    parser = argparse.ArgumentParser(description="LLM Semantic Detection CLI Demo")
    parser.add_argument("--image", type=str, default=None, help="Input image path (creates synthetic if omitted)")
    parser.add_argument("--queries", nargs="+", default=["car", "traffic cone"], help="Target object query prompts")
    parser.add_argument("--output", type=str, default="demo/detection_result.jpg", help="Output annotated image path")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device ('cuda' or 'cpu')")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM semantic expansion (raw query baseline)")
    args = parser.parse_args()

    image_path = args.image
    if image_path is None or not Path(image_path).exists():
        print("No input image specified. Generating synthetic multi-object scene...")
        image_path = create_synthetic_test_image()

    print("=" * 65)
    print(" Lightweight LLM-Augmented Semantic Detection Pipeline")
    print(f" Target Queries: {args.queries}")
    print(f" LLM Augmentation: {'Disabled' if args.disable_llm else 'Enabled'}")
    print(f" Device: {args.device}")
    print("=" * 65)

    pipeline = LLMSemanticPipeline(device=args.device)
    result = pipeline.predict(
        image=image_path,
        queries=args.queries,
        enable_llm_enhancement=not args.disable_llm
    )

    print("\n--- Detection Results ---")
    num_detected = len(result["boxes"])
    print(f"Detected {num_detected} object instance(s):")
    for i in range(num_detected):
        box = result["boxes"][i]
        score = result["scores"][i]
        label = result["labels"][i]
        print(f"  [{i+1}] Class: '{label}' | Confidence: {score:.3f} | BBox: {list(map(int, box))}")

    print("\n--- Pipeline Profiling & Latency ---")
    timings = result["timings_ms"]
    print(f"  LLM Semantic Enhancement: {timings['llm_enhancement']} ms")
    print(f"  Detector Feature Inference: {timings['detector_inference']} ms")
    print(f"  Adaptive NMS & Post-Proc: {timings['post_processing']} ms")
    print(f"  Total Latency: {timings['total_latency']} ms ({result['fps']} FPS)")

    out_canvas = pipeline.visualize(image_path, result, output_path=args.output)
    print(f"\nVisual output saved to: {args.output}")


if __name__ == "__main__":
    main()
