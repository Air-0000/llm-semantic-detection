"""Structured Prompt Template Engineering for Zero-Shot Semantic Augmentation.

Transforms bare object names (e.g. 'police car', 'solar panel') into structured,
attribute-rich visual semantic descriptions covering morphology, material,
distinctive parts, and spatial context priors.
"""

from typing import Dict, Optional


STRUCTURED_SYSTEM_PROMPT = (
    "You are an expert computer vision semanticist. Given an object name, output a concise "
    "visual specification in four dimensions: Morphology (shape/boundaries), Material/Texture, "
    "Salient Parts (distinct features), and Context (typical surroundings). Keep response under 50 words."
)

PROMPT_TEMPLATE = """Object: {object_name}
Provide visual characteristics for open-vocabulary detection:
Morphology:
Material:
Salient Parts:
Context:"""


def build_semantic_query(object_name: str) -> str:
    """Build standardized structured prompt for LLM semantic enhancement."""
    return PROMPT_TEMPLATE.format(object_name=object_name.strip())


def parse_structured_response(text: str, default_name: str) -> Dict[str, str]:
    """Parse LLM output text into structured semantic attribute dictionary."""
    result = {
        "name": default_name,
        "morphology": "",
        "material": "",
        "salient_parts": "",
        "context": "",
        "combined_prompt": ""
    }

    lines = text.strip().split("\n")
    for line in lines:
        line_clean = line.strip()
        lower = line_clean.lower()
        if lower.startswith("morphology:"):
            result["morphology"] = line_clean[len("morphology:"):].strip()
        elif lower.startswith("material:"):
            result["material"] = line_clean[len("material:"):].strip()
        elif lower.startswith("salient parts:"):
            result["salient_parts"] = line_clean[len("salient parts:"):].strip()
        elif lower.startswith("context:"):
            result["context"] = line_clean[len("context:"):].strip()

    # Form unified enriched visual prompt for open-vocabulary encoder
    parts = []
    if result["morphology"]:
        parts.append(result["morphology"])
    if result["material"]:
        parts.append(result["material"])
    if result["salient_parts"]:
        parts.append(result["salient_parts"])

    if parts:
        result["combined_prompt"] = f"{default_name}, characterized by {', '.join(parts)}"
    else:
        result["combined_prompt"] = text.strip() if text.strip() else default_name

    return result
