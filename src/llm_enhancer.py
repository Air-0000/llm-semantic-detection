"""Lightweight LLM Semantic Enhancer.

Augments open-vocabulary category names into fine-grained visual-semantic descriptors.
Supports:
1. Qwen2.5 / Transformers text-only LLMs
2. High-speed compositional semantic ontology fallback
3. Integrated thread-safe LRU disk caching
"""

import logging
import time
from typing import Dict, List, Optional
import torch

from src.prompt_template import (
    build_semantic_query,
    parse_structured_response,
    STRUCTURED_SYSTEM_PROMPT
)
from src.semantic_cache import SemanticCache

logger = logging.getLogger("LLMEnhancer")


# Built-in visual ontology knowledge base for deterministic instantaneous fallback
ONTOLOGY_PRIORS = {
    "car": {
        "morphology": "aerodynamic metallic body with four wheels, windshield, and roofline",
        "material": "reflective painted sheet metal, glass windows, rubber tires",
        "salient_parts": "headlights, front grille, license plate, side mirrors",
        "context": "asphalt roads, parking spaces, street traffic"
    },
    "person": {
        "morphology": "upright bipedal silhouette with head, torso, and limbs",
        "material": "fabric clothing, skin texture, hair",
        "salient_parts": "face, hands, feet, shoes, clothing contours",
        "context": "sidewalks, indoor rooms, public plazas, crosswalks"
    },
    "dog": {
        "morphology": "quadrupedal mammal with elongated snout, tail, and fur-covered body",
        "material": "fur texture, wet nose, claws",
        "salient_parts": "ears, snout, paws, wagging tail, collar",
        "context": "parks, grass lawns, domestic interiors, sidewalks"
    },
    "traffic cone": {
        "morphology": "conical bright geometric marker with a flat square base",
        "material": "fluorescent orange flexible plastic or vinyl with reflective retro-bands",
        "salient_parts": "white or silver reflective horizontal stripes, pointed top",
        "context": "road construction zones, lane dividers, hazard areas"
    },
    "bicycle": {
        "morphology": "lightweight tubular triangular frame with two spoked wheels",
        "material": "metallic or carbon frame, rubber spoked tires, chain",
        "salient_parts": "handlebars, saddle seat, pedals, chain sprocket",
        "context": "bike lanes, sidewalks, racks, roadside"
    },
    "solar panel": {
        "morphology": "large planar rectangular grid matrix angled towards sky",
        "material": "photovoltaic dark blue/black monocrystalline silicon with silver busbars",
        "salient_parts": "metallic framing, solar cell grid lines, support mounting brackets",
        "context": "building rooftops, solar farms, open sunny fields"
    },
    "cell": {
        "morphology": "circular or biconcave discoid microscopic biological corpuscle",
        "material": "smooth lipid bilayer membrane, cytoplasmic interior",
        "salient_parts": "central pallor (RBC), lobed nucleus (WBC), granules",
        "context": "blood smear background, microscopic optical field"
    }
}


class LLMEnhancer:
    """Zero-shot visual semantic expander using lightweight LLM or ontological engine."""

    def __init__(self,
                 model_name_or_path: Optional[str] = "Qwen/Qwen2.5-0.5B-Instruct",
                 cache_file: str = "cache/semantic_cache.json",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_name = model_name_or_path
        self.cache = SemanticCache(cache_file=cache_file)

        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        """Attempt loading local or cached transformer LLM if available."""
        if not self.model_name or self.model_name == "rule_based":
            logger.info("Using high-speed ontological rule-based semantic enhancer.")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Attempting to initialize LLM: %s on %s", self.model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model.eval()
            logger.info("Successfully loaded LLM %s", self.model_name)
        except Exception as e:
            logger.info("Local transformer weights for %s not found locally (%s). Using ontological engine.",
                        self.model_name, e)
            self.model = None
            self.tokenizer = None

    def enhance(self, object_name: str) -> Dict[str, str]:
        """Enhance object name into structured visual semantic dictionary.

        Checks semantic cache first. On miss, queries LLM or ontological engine.
        """
        clean_name = object_name.strip()
        # 1. Check Cache
        cached = self.cache.get(clean_name)
        if cached is not None:
            return cached

        # 2. LLM or Ontological Generation
        start_t = time.time()
        if self.model is not None and self.tokenizer is not None:
            semantic_data = self._generate_with_llm(clean_name)
        else:
            semantic_data = self._generate_with_ontology(clean_name)

        gen_time_ms = (time.time() - start_t) * 1000.0
        semantic_data["enhancement_time_ms"] = round(gen_time_ms, 2)

        # 3. Store into cache
        self.cache.put(clean_name, semantic_data)
        return semantic_data

    def _generate_with_llm(self, object_name: str) -> Dict[str, str]:
        """Generate structured visual attributes using HuggingFace model."""
        prompt = build_semantic_query(object_name)
        messages = [
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=False
            )
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return parse_structured_response(response_text, default_name=object_name)

    def _generate_with_ontology(self, object_name: str) -> Dict[str, str]:
        """Generate structured visual attributes using semantic composition heuristics."""
        key = object_name.lower().strip()
        # Direct ontology match
        if key in ONTOLOGY_PRIORS:
            prior = ONTOLOGY_PRIORS[key]
            combined = f"{object_name}, characterized by {prior['morphology']}, {prior['material']}, with visible {prior['salient_parts']}"
            return {
                "name": object_name,
                "morphology": prior["morphology"],
                "material": prior["material"],
                "salient_parts": prior["salient_parts"],
                "context": prior["context"],
                "combined_prompt": combined
            }

        # Fuzzy / compositional inference for arbitrary query
        words = key.split()
        adj = words[:-1]
        noun = words[-1] if words else key

        morphology = f"characteristic form and distinctive geometry of {key}"
        material = f"natural or synthetic surface texture of {noun}"
        salient_parts = f"identifying boundary contours and structural components of {key}"
        context = f"natural or urban environment where {noun} is typically situated"

        combined = f"{object_name}, with {morphology}, distinctive {material}, and salient {salient_parts}"
        return {
            "name": object_name,
            "morphology": morphology,
            "material": material,
            "salient_parts": salient_parts,
            "context": context,
            "combined_prompt": combined
        }
