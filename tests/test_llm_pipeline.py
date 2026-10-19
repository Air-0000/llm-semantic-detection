import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

from src.prompt_template import build_semantic_query, parse_structured_response
from src.semantic_cache import SemanticCache
from src.lora_model import HierarchicalAsymmetricLoRA
from src.postprocessor import AdaptivePostProcessor
from src.pipeline import LLMSemanticPipeline


def test_prompt_template_and_parser():
    query = build_semantic_query("fire truck")
    assert "fire truck" in query
    assert "Morphology:" in query

    mock_llm_response = (
        "Morphology: large boxy vehicle with high ground clearance\n"
        "Material: red painted steel, reflective decals\n"
        "Salient Parts: emergency lights, ladder, water cannon\n"
        "Context: road, urban fire station"
    )
    parsed = parse_structured_response(mock_llm_response, default_name="fire truck")
    assert parsed["name"] == "fire truck"
    assert "ladder" in parsed["salient_parts"]
    assert "fire truck, characterized by" in parsed["combined_prompt"]


def test_semantic_cache_persistence():
    temp_dir = tempfile.mkdtemp()
    cache_path = os.path.join(temp_dir, "test_cache.json")
    try:
        cache = SemanticCache(cache_file=cache_path, max_entries=5)
        assert len(cache) == 0

        data = {"name": "car", "combined_prompt": "metallic vehicle with wheels"}
        cache.put("car", data)
        assert len(cache) == 1

        retrieved = cache.get("car")
        assert retrieved is not None
        assert retrieved["name"] == "car"

        # Test persistence
        cache2 = SemanticCache(cache_file=cache_path)
        assert len(cache2) == 1
        assert cache2.get("car")["name"] == "car"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_asymmetric_lora():
    lora = HierarchicalAsymmetricLoRA(
        in_dim=64, hidden_dim=64, out_dim=64,
        shallow_rank=4, deep_rank=16
    )
    assert lora.shallow_rank == 4
    assert lora.deep_rank == 16

    x = torch.randn(2, 64)
    out = lora(x)
    assert out.shape == (2, 64)

    audit = lora.get_trainable_param_count()
    assert audit["trainable_lora_params"] > 0
    assert audit["ratio_pct"] < 100.0


def test_adaptive_postprocessor():
    post = AdaptivePostProcessor(base_threshold=0.25)
    # Test blank image complexity vs textured image complexity
    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
    noisy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    c_blank = post.compute_scene_complexity(blank_img)
    c_noisy = post.compute_scene_complexity(noisy_img)
    assert c_noisy > c_blank

    thresh_blank = post.get_adaptive_threshold(c_blank)
    thresh_noisy = post.get_adaptive_threshold(c_noisy)
    # Noisy/cluttered scene should elevate threshold
    assert thresh_noisy > thresh_blank


def test_pipeline_end_to_end():
    # Use CPU for fast unit testing
    pipeline = LLMSemanticPipeline(device="cpu")
    dummy_img = np.full((320, 320, 3), 128, dtype=np.uint8)
    queries = ["car", "traffic cone"]

    result = pipeline.predict(dummy_img, queries=queries, enable_llm_enhancement=True)
    assert "boxes" in result
    assert "scores" in result
    assert "timings_ms" in result
    assert "fps" in result

    # Test visualization
    canvas = pipeline.visualize(dummy_img, result)
    assert canvas.shape == dummy_img.shape


if __name__ == "__main__":
    print("Running tests for LLM Semantic Detection Pipeline...")
    test_prompt_template_and_parser()
    print("  [PASS] test_prompt_template_and_parser")
    test_semantic_cache_persistence()
    print("  [PASS] test_semantic_cache_persistence")
    test_asymmetric_lora()
    print("  [PASS] test_asymmetric_lora")
    test_adaptive_postprocessor()
    print("  [PASS] test_adaptive_postprocessor")
    test_pipeline_end_to_end()
    print("  [PASS] test_pipeline_end_to_end")
    print("\nALL 5 TESTS PASSED SUCCESSFULLY!")

