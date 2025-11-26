"""
SDNQ Model Loader Node

Main node for loading pre-quantized SDNQ models in ComfyUI workflows.
"""

import os
import torch
from typing import Tuple

# Import SDNQ config to register quantization methods with diffusers
from sdnq import SDNQConfig
from sdnq.loader import apply_sdnq_options_to_model
from sdnq.common import use_torch_compile as triton_is_available

import diffusers

from ..core.wrapper import wrap_pipeline_components
from ..core.config import get_dtype_from_string, get_device_map


class SDNQModelLoader:
    """
    Load SDNQ (SD.Next Quantization) quantized models.

    SDNQ provides 50-75% VRAM savings while maintaining quality,
    enabling large models like FLUX and SD3.5 on consumer hardware.

    Supports:
    - Local SDNQ models
    - HuggingFace Hub models (Disty0 collection)
    - Multiple quantization levels (int8, int6, uint4, etc.)
    - Optional Triton acceleration
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to SDNQ model or HuggingFace repo ID (e.g., Disty0/FLUX.1-dev-qint8)"
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16"
                }),
                "use_quantized_matmul": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use quantized matrix multiplication (requires Triton)"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable model CPU offloading to save VRAM"
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "loaders/SDNQ"
    DESCRIPTION = "Load SDNQ quantized models for significant VRAM savings. Developed by Disty0."

    def load_model(
        self,
        model_path: str,
        dtype: str,
        use_quantized_matmul: bool = True,
        cpu_offload: bool = True,
        device: str = "auto"
    ) -> Tuple:
        """
        Load an SDNQ quantized model and return ComfyUI-compatible components.

        Args:
            model_path: Path to local model or HuggingFace repo ID
            dtype: Data type for model weights
            use_quantized_matmul: Enable Triton quantized matmul optimization
            cpu_offload: Enable model CPU offloading
            device: Device placement strategy

        Returns:
            Tuple of (MODEL, CLIP, VAE) wrappers compatible with ComfyUI

        Raises:
            ValueError: If model_path is empty or invalid
            RuntimeError: If model loading fails
        """
        if not model_path or model_path.strip() == "":
            raise ValueError(
                "model_path cannot be empty. Provide either:\n"
                "- Local path: /path/to/model\n"
                "- HuggingFace repo: Disty0/FLUX.1-dev-qint8"
            )

        model_path = model_path.strip()

        print(f"\n{'='*60}")
        print(f"SDNQ Model Loader")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Dtype: {dtype}")
        print(f"Quantized MatMul: {use_quantized_matmul}")
        print(f"CPU Offload: {cpu_offload}")
        print(f"Device: {device}")

        # Convert dtype string to torch dtype
        torch_dtype = get_dtype_from_string(dtype)

        # Determine if loading from local path or HuggingFace
        is_local = os.path.exists(model_path)

        print(f"Loading from: {'Local path' if is_local else 'HuggingFace Hub'}")

        try:
            # Load pipeline with SDNQ support
            # The SDNQConfig import above registers SDNQ into diffusers
            print("Loading model pipeline...")

            pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                local_files_only=is_local,
            )

            print(f"Pipeline loaded: {type(pipeline).__name__}")

            # Apply quantized matmul optimization if requested and available
            if use_quantized_matmul:
                if triton_is_available:
                    print("Applying Triton quantized matmul optimization...")
                    if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                        pipeline.transformer = apply_sdnq_options_to_model(
                            pipeline.transformer,
                            use_quantized_matmul=True
                        )
                    elif hasattr(pipeline, 'unet') and pipeline.unet is not None:
                        pipeline.unet = apply_sdnq_options_to_model(
                            pipeline.unet,
                            use_quantized_matmul=True
                        )
                else:
                    print("Warning: Triton not available, skipping quantized matmul optimization")

            # Apply CPU offloading if requested
            if cpu_offload:
                print("Enabling model CPU offload...")
                pipeline.enable_model_cpu_offload()

            # Wrap pipeline components for ComfyUI compatibility
            print("Wrapping pipeline components for ComfyUI...")
            model_wrapper, clip_wrapper, vae_wrapper = wrap_pipeline_components(pipeline)

            print(f"{'='*60}")
            print("✓ Model loaded successfully!")
            print(f"{'='*60}\n")

            return (model_wrapper, clip_wrapper, vae_wrapper)

        except Exception as e:
            print(f"\n{'='*60}")
            print("✗ Model loading failed!")
            print(f"{'='*60}")
            print(f"Error: {str(e)}")
            print(f"\nTroubleshooting:")
            print(f"1. Verify the model path is correct")
            print(f"2. For HuggingFace models, check internet connection")
            print(f"3. Ensure the model is SDNQ-quantized")
            print(f"4. Check that required dependencies are installed")
            print(f"{'='*60}\n")
            raise RuntimeError(f"Failed to load SDNQ model: {str(e)}") from e
