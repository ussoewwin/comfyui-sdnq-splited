"""
Model Registry and Catalog

Phase 2: This module will contain:
- Hardcoded catalog of known SDNQ models from Disty0 collection
- Model metadata (type, quant level, size, VRAM requirements)
- Detection of locally installed models
- Model recommendation based on available VRAM
"""

from typing import Dict, List, Optional


# Catalog of pre-quantized SDNQ models
SDNQ_MODEL_CATALOG = {
    "FLUX.1-dev-qint8": {
        "repo_id": "Disty0/FLUX.1-dev-qint8",
        "type": "FLUX",
        "quant_level": "int8",
        "vram_required": "~12 GB",
        "quality": "~99%",
        "description": "FLUX.1-dev quantized to 8-bit"
    },
    "FLUX.1-dev-qint4": {
        "repo_id": "Disty0/FLUX.1-dev-qint4",
        "type": "FLUX",
        "quant_level": "uint4",
        "vram_required": "~6 GB",
        "quality": "~95%",
        "description": "FLUX.1-dev quantized to 4-bit (extreme VRAM savings)"
    },
    "SD3.5-Large-qint8": {
        "repo_id": "Disty0/stable-diffusion-3.5-large-qint8",
        "type": "SD3.5",
        "quant_level": "int8",
        "vram_required": "~10 GB",
        "quality": "~99%",
        "description": "Stable Diffusion 3.5 Large quantized to 8-bit"
    },
    "SDXL-base-qint8": {
        "repo_id": "Disty0/stable-diffusion-xl-base-1.0-qint8",
        "type": "SDXL",
        "quant_level": "int8",
        "vram_required": "~6 GB",
        "quality": "~99%",
        "description": "SDXL Base 1.0 quantized to 8-bit"
    },
}


def get_model_catalog() -> Dict[str, Dict]:
    """
    Get the catalog of available SDNQ models.

    Returns:
        Dictionary of model metadata
    """
    return SDNQ_MODEL_CATALOG


def get_model_names() -> List[str]:
    """
    Get list of available model names.

    Returns:
        List of model names
    """
    return list(SDNQ_MODEL_CATALOG.keys())


def get_model_info(model_name: str) -> Optional[Dict]:
    """
    Get metadata for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Model metadata dictionary or None if not found
    """
    return SDNQ_MODEL_CATALOG.get(model_name)


# TODO: Phase 2 functions to implement:
# - detect_local_models(path): Scan for locally installed SDNQ models
# - recommend_model(vram_gb): Suggest models based on available VRAM
# - refresh_catalog(): Update catalog from HuggingFace API
