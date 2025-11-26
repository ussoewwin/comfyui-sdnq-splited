"""
ComfyUI-SDNQ: Custom node pack for loading SDNQ quantized models in ComfyUI

SDNQ (SD.Next Quantization) is developed by Disty0
Repository: https://github.com/Disty0/sdnq
Pre-quantized models: https://huggingface.co/collections/Disty0/sdnq

This node pack enables loading SDNQ models with 50-75% VRAM savings while maintaining quality.
"""

from .nodes.loader import SDNQModelLoader

# V1 API - Node Registration
NODE_CLASS_MAPPINGS = {
    "SDNQModelLoader": SDNQModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDNQModelLoader": "SDNQ Model Loader",
}

# Web directory for custom UI (optional, for future use)
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
