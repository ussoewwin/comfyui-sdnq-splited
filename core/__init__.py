"""
SDNQ Core Package

Contains core functionality for SDNQ integration with ComfyUI:
- Model registry and catalog
- HuggingFace Hub downloader
- ComfyUI type wrappers (MODEL, CLIP, VAE)
- Configuration helpers
"""

from .wrapper import wrap_pipeline_components
from .config import get_dtype_from_string

__all__ = ['wrap_pipeline_components', 'get_dtype_from_string']
