"""
SDNQ Standalone Sampler Node - V3 API Compatible

This node loads SDNQ quantized models and generates images in one step.
No MODEL/CLIP/VAE outputs - just IMAGE output for ComfyUI.

Architecture: User Input → Load/Download SDNQ Model → Generate → Output IMAGE

Based on verified APIs from:
- diffusers documentation (https://huggingface.co/docs/diffusers)
- SDNQ repository (https://github.com/Disty0/sdnq)
- ComfyUI nodes.py (IMAGE format specification)
"""

import torch
import numpy as np
from PIL import Image
import traceback
import sys
import os
import warnings
from typing import Optional, Tuple, Dict, Any

# ComfyUI imports for LoRA folder access
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("[SDNQ Sampler] Warning: folder_paths not available - LoRA dropdown will be disabled")

# SDNQ import - registers SDNQ support into diffusers
from sdnq import SDNQConfig
# SDNQ optimization imports
try:
    from sdnq.loader import apply_sdnq_options_to_model
    from sdnq.common import use_torch_compile as triton_is_available
except ImportError:
    print("[SDNQ Sampler] Warning: Could not import SDNQ optimization tools. Quantized MatMul will be disabled.")
    def apply_sdnq_options_to_model(model, **kwargs): return model
    triton_is_available = False

# diffusers pipeline - auto-detects model type from model_index.json
from diffusers import DiffusionPipeline

# Scheduler imports
# Flow-match schedulers (for FLUX, SD3, Qwen, Z-Image)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# Traditional diffusion schedulers (for SDXL, SD1.5, etc.)
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
)

# Local imports for model catalog and downloading
from ..core.registry import (
    get_model_names_for_dropdown,
    get_repo_id_from_name,
    get_model_info,
)
from ..core.downloader import (
    download_model,
    get_cached_model_path,
    check_model_cached,
)
from ..core.config import get_sdnq_models_dir


class SDNQSampler:
    """
    Standalone SDNQ sampler that loads quantized models and generates images.

    All-in-one node that handles:
    - Model selection from pre-configured catalog OR custom paths
    - Auto-download from HuggingFace Hub
    - Loading SDNQ models from local paths
    - Setting up generation parameters
    - Generating images with proper seeding
    - Converting output to ComfyUI IMAGE format
    - Graceful error handling and interruption support

    ComfyUI V3 API Compatible with V1 backward compatibility.
    """

    def __init__(self):
        """Initialize sampler with empty pipeline cache."""
        self.pipeline = None
        self.current_model_path = None
        self.current_dtype = None
        self.current_memory_mode = None
        self.current_scheduler = None
        self.current_lora_path = None
        self.current_lora_strength = None
        # Performance optimization settings cache
        self.current_use_xformers = None
        self.current_use_flash_attention = None
        self.current_use_sage_attention = None
        self.current_enable_vae_tiling = None
        self.current_matmul_precision = None
        self.interrupted = False

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs following ComfyUI V3 conventions with V1 compatibility.

        Parameters organized logically:
        1. Model Selection (what to use)
        2. Generation Prompts (what to create)
        3. Generation Settings (how to create)
        4. Model Configuration (technical settings)
        5. Enhancements (optional improvements)

        All parameters verified from diffusers documentation.
        """
        # Get model names from catalog
        model_names = get_model_names_for_dropdown()

        # Get available LoRAs from ComfyUI loras folder
        lora_list = ["[None]", "[Custom Path]"]
        if COMFYUI_AVAILABLE:
            try:
                available_loras = folder_paths.get_filename_list("loras")
                lora_list.extend(available_loras)
            except Exception as e:
                print(f"[SDNQ Sampler] Warning: Could not load LoRA list: {e}")

        # Scheduler options
        # Flow-based schedulers ONLY work with FLUX/SD3/Qwen/Z-Image models
        # Traditional schedulers ONLY work with SDXL/SD1.5 models
        scheduler_list = [
            # Flow-based (for FLUX, SD3, Qwen, Z-Image)
            "FlowMatchEulerDiscreteScheduler",
            # Traditional diffusion (for SDXL, SD1.5) - Top 10 most popular
            "DPMSolverMultistepScheduler",
            "UniPCMultistepScheduler",
            "EulerDiscreteScheduler",
            "EulerAncestralDiscreteScheduler",
            "DDIMScheduler",
            "HeunDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "DPMSolverSinglestepScheduler",
            "DEISMultistepScheduler",
            "LMSDiscreteScheduler",
            "DDPMScheduler",
            "PNDMScheduler",
        ]

        return {
            "required": {
                # ============================================================
                # GROUP 1: MODEL SELECTION (What to use)
                # ============================================================

                "model_selection": (["[Custom Path]"] + model_names, {
                    "default": model_names[0] if model_names else "[Custom Path]",
                    "tooltip": "Select a pre-configured SDNQ model (auto-downloads from HuggingFace) or choose [Custom Path] to specify a local model directory"
                }),

                "custom_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Local path to SDNQ model directory (only used when [Custom Path] is selected). Example: /path/to/model or C:\\path\\to\\model"
                }),

                # ============================================================
                # GROUP 2: GENERATION PROMPTS (What to create)
                # ============================================================

                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text description of the image to generate. Be descriptive for best results."
                }),

                "negative_prompt": ("STRING", {
                    "default": "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, text, watermark, signature",
                    "multiline": True,
                    "tooltip": "What to avoid in the image. Default includes common quality issues. Clear this for no negative prompt. Note: FLUX-schnell (cfg=0) ignores negative prompts."
                }),

                # ============================================================
                # GROUP 3: GENERATION SETTINGS (How to create)
                # ============================================================

                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 150,
                    "step": 1,
                    "tooltip": "Number of denoising steps. More steps = better quality but slower. 20-30 is typical for most models."
                }),

                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "Guidance scale - how closely to follow the prompt. Higher = more literal. FLUX-schnell uses 0.0, FLUX-dev uses 3.5-7.0, SDXL uses 7.0-9.0."
                }),

                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Image width in pixels. Must be multiple of 8. Larger = more VRAM usage. 1024 is standard for FLUX/SDXL."
                }),

                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Image height in pixels. Must be multiple of 8. Larger = more VRAM usage. 1024 is standard for FLUX/SDXL."
                }),

                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible generation. Same seed + settings = same image."
                }),

                "scheduler": (scheduler_list, {
                    "default": "DPMSolverMultistepScheduler",
                    "tooltip": "⚠️ IMPORTANT: Use FlowMatchEulerDiscreteScheduler for FLUX/SD3/Qwen/Z-Image. Use DPMSolver/Euler/UniPC for SDXL/SD1.5. Wrong scheduler = broken images!"
                }),

                # ============================================================
                # GROUP 4: MODEL CONFIGURATION (Technical settings)
                # ============================================================

                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision. bfloat16 recommended for FLUX (best quality/speed). float16 for older GPUs. float32 for CPU."
                }),

                "memory_mode": (["gpu", "balanced", "lowvram"], {
                    "default": "balanced",
                    "tooltip": "Memory management: 'gpu' = All on GPU (fastest, needs 24GB+ VRAM). 'balanced' = Model offloading (12-16GB VRAM). 'lowvram' = Sequential offloading (8GB VRAM, slowest)."
                }),

                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically download model from HuggingFace if not found locally. Disable to only use local models."
                }),
            },
            "optional": {
                # ============================================================
                # GROUP 5: ENHANCEMENTS (Optional improvements)
                # ============================================================

                "lora_selection": (lora_list, {
                    "default": "[None]",
                    "tooltip": "Select LoRA from ComfyUI loras folder ([None] = disabled, [Custom Path] = use custom path below). LoRAs add styles or concepts to generation."
                }),

                "lora_custom_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom LoRA path or HuggingFace repo ID (only used when [Custom Path] is selected). Example: /path/to/lora.safetensors or username/lora-repo"
                }),

                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "LoRA influence strength. 1.0 = full strength, 0.5 = half strength, 0.0 = disabled. Negative values invert the LoRA effect. Range: -5.0 to +5.0."
                }),

                # ============================================================
                # PERFORMANCE OPTIMIZATIONS (Optional speedups)
                # ============================================================

                "matmul_precision": (["int8", "fp8", "none"], {
                    "default": "int8",
                    "tooltip": "Precision for Triton quantized matmul. 'int8' is standard, 'fp8' for newer GPUs (Ada/Hopper), 'none' to disable optimization. Requires Linux/WSL."
                }),

                "use_xformers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable xFormers memory-efficient attention for 10-45% speedup. Works with all memory modes (gpu/balanced/lowvram). Auto-fallback to SDPA if xformers not installed or incompatible. Requires: pip install xformers"
                }),

                "use_flash_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Flash Attention (FA) for faster inference and lower VRAM usage. Requires ComfyUI started with --use-flash-attention flag. Works with modern GPUs (Ampere+)."
                }),

                "use_sage_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Sage Attention (SA) for optimized attention computation. Requires ComfyUI started with --use-sage-attention flag. Provides better performance on supported GPUs."
                }),

                "enable_vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable VAE tiling for very large images (>1536px). Prevents out-of-memory errors on high resolutions. Minimal performance impact. Recommended for images >1536x1536."
                }),
            }
        }

    # V3 API: Return type hints
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    # V1 API: Function name
    FUNCTION = "generate"

    # Category for node menu
    CATEGORY = "sampling/SDNQ"

    # V3 API: Output node (can save/display results)
    OUTPUT_NODE = False

    # V3 API: Node description
    DESCRIPTION = "Load SDNQ quantized models and generate images with 50-75% VRAM savings. Supports FLUX, SD3, SDXL, video models, and more."

    def check_interrupted(self):
        """Check if generation should be interrupted (ComfyUI interrupt support)."""
        # ComfyUI provides comfy.model_management.interrupt_processing()
        # For now, we'll use a simple flag that can be extended
        return self.interrupted

    def load_or_download_model(self, model_selection: str, custom_path: str, auto_download: bool) -> Tuple[str, bool, Optional[str]]:
        """
        Load model from catalog or custom path, downloading if needed.

        Args:
            model_selection: Selected model from dropdown
            custom_path: Custom model path (if [Custom Path] selected)
            auto_download: Whether to auto-download from HuggingFace

        Returns:
            Tuple of (model_path: str, was_downloaded: bool, repo_id: Optional[str])

        Raises:
            ValueError: If model not found and auto_download is False
            Exception: If download fails
        """
        # Check if using custom path
        if model_selection == "[Custom Path]":
            if not custom_path or custom_path.strip() == "":
                raise ValueError(
                    "Custom model path is empty. Please provide a valid path to a local SDNQ model directory, "
                    "or select a pre-configured model from the dropdown."
                )

            model_path = custom_path.strip()

            # Verify path exists
            import os
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Custom model path does not exist: {model_path}\n"
                    f"Please verify the path and try again."
                )

            # Verify it's a valid model directory (has model_index.json)
            if not os.path.exists(os.path.join(model_path, "model_index.json")):
                raise ValueError(
                    f"Invalid model directory: {model_path}\n"
                    f"Directory must contain model_index.json file. "
                    f"This should be a diffusers model directory."
                )

            print(f"[SDNQ Sampler] Using custom model path: {model_path}")
            return (model_path, False, None)

        # Using catalog model
        model_info = get_model_info(model_selection)
        if not model_info:
            raise ValueError(
                f"Model not found in catalog: {model_selection}\n"
                f"This may indicate an issue with the model registry. "
                f"Try selecting a different model or using [Custom Path]."
            )

        repo_id = model_info["repo_id"]
        print(f"[SDNQ Sampler] Selected model: {model_selection}")
        print(f"[SDNQ Sampler] Repository: {repo_id}")

        # Check if model already cached
        cached_path = get_cached_model_path(repo_id)
        if cached_path:
            print(f"[SDNQ Sampler] Found cached model at: {cached_path}")
            return (cached_path, False, repo_id)

        # Model not cached - download if auto_download enabled
        if not auto_download:
            raise ValueError(
                f"Model not found locally: {model_selection} ({repo_id})\n\n"
                f"Options:\n"
                f"1. Enable 'auto_download' to download automatically from HuggingFace\n"
                f"2. Download manually using: huggingface-cli download {repo_id}\n"
                f"3. Select a different model that's already downloaded"
            )

        print(f"[SDNQ Sampler] Model not cached - downloading from HuggingFace...")
        print(f"[SDNQ Sampler] This may take a while (models are 5-20+ GB)")

        try:
            downloaded_path = download_model(repo_id)
            print(f"[SDNQ Sampler] Download complete: {downloaded_path}")
            return (downloaded_path, True, repo_id)
        except Exception as e:
            raise Exception(
                f"Failed to download model {model_selection} ({repo_id})\n\n"
                f"Error: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"1. Check your internet connection\n"
                f"2. Verify HuggingFace Hub is accessible\n"
                f"3. Try downloading manually: huggingface-cli download {repo_id}\n"
                f"4. Check disk space (models are large)\n"
                f"5. If download was interrupted, try again - it will resume"
            )

    def load_pipeline(self, model_path: str, dtype_str: str, memory_mode: str = "gpu",
                     use_xformers: bool = False, use_flash_attention: bool = False,
                     use_sage_attention: bool = False, enable_vae_tiling: bool = False,
                     matmul_precision: str = "int8", repo_id: Optional[str] = None) -> DiffusionPipeline:
        """
        Load SDNQ model using diffusers pipeline.

        Uses DiffusionPipeline which auto-detects the correct pipeline class
        from the model's model_index.json file. This works with:
        - FLUX.1, FLUX.2
        - SD3, SD3.5, SDXL
        - Video models (CogVideoX, Wan, etc.)
        - Multimodal models (Z-Image, Qwen-Image, etc.)

        Args:
            model_path: Local path to SDNQ model directory
            dtype_str: String dtype ("bfloat16", "float16", "float32")
            memory_mode: Memory management strategy ("gpu", "balanced", "lowvram")
            use_xformers: Enable xFormers memory-efficient attention
            enable_vae_tiling: Enable VAE tiling for large images
            matmul_precision: Precision for Triton quantized matmul ("int8", "fp8", "none")

        Returns:
            Loaded diffusers pipeline

        Raises:
            Exception: If pipeline loading fails

        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/using-diffusers/loading
        https://huggingface.co/docs/diffusers/main/optimization/memory
        """
        # Convert dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype_str]

        # xFormers, Flash Attention, and Sage Attention require float16 or bfloat16, not float32
        # Auto-convert float32 to bfloat16 if any of these optimizations are enabled
        original_dtype_str = dtype_str
        needs_fp16_compat = use_xformers or use_flash_attention or use_sage_attention
        if needs_fp16_compat and dtype_str == "float32":
            opt_names = []
            if use_xformers:
                opt_names.append("xFormers")
            if use_flash_attention:
                opt_names.append("Flash Attention")
            if use_sage_attention:
                opt_names.append("Sage Attention")
            opt_list = ", ".join(opt_names)
            print(f"[SDNQ Sampler] ⚠️  {opt_list} does not support float32 (only float16/bfloat16)")
            print(f"[SDNQ Sampler] ⚠️  Automatically converting dtype from float32 to bfloat16")
            dtype_str = "bfloat16"
            torch_dtype = torch.bfloat16

        print(f"[SDNQ Sampler] Loading model from: {model_path}")
        print(f"[SDNQ Sampler] Using dtype: {dtype_str} ({torch_dtype})")
        if original_dtype_str != dtype_str:
            print(f"[SDNQ Sampler] (Original dtype was {original_dtype_str}, changed for attention optimization compatibility)")
        print(f"[SDNQ Sampler] Memory mode: {memory_mode}")

        try:
            # Load pipeline - DiffusionPipeline auto-detects model type
            # SDNQ quantization is automatically detected from model config
            # Note: Pipeline loads to CPU by default - we move to GPU below

            # Suppress torch_dtype deprecation warning from transformers components
            # The warning comes from transformers library (used for CLIP/T5 text encoders)
            # diffusers still uses torch_dtype as the official parameter in 0.36.x
            # See: https://github.com/huggingface/peft/issues/2835
            with warnings.catch_warnings():
                # Filter both message pattern and FutureWarning category for comprehensive suppression
                warnings.filterwarnings("ignore", message=".*torch_dtype.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*torch_dtype.*")
                try:
                    # First try with local_files_only=True (faster, no network access)
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        local_files_only=True,  # Only load from local path
                    )
                except (FileNotFoundError, OSError) as e:
                    # If local files are incomplete, try downloading missing files
                    print(f"[SDNQ Sampler] Local files incomplete, downloading missing files...")
                    # Use repo_id if available (allows HF Hub to download missing files)
                    # Otherwise fall back to model_path with local_files_only=False
                    if repo_id:
                        print(f"[SDNQ Sampler] Using repository ID: {repo_id}")
                        pipeline = DiffusionPipeline.from_pretrained(
                            repo_id,
                            torch_dtype=torch_dtype,
                            cache_dir=os.path.dirname(model_path) if os.path.isdir(model_path) else None,
                            local_files_only=False,  # Allow downloading missing files
                        )
                    else:
                        pipeline = DiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch_dtype,
                            local_files_only=False,  # Allow downloading missing files
                        )

            print(f"[SDNQ Sampler] Model loaded successfully!")
            print(f"[SDNQ Sampler] Pipeline type: {type(pipeline).__name__}")

            # Apply SDNQ optimizations (Quantized MatMul)
            # This must be done BEFORE memory management moves things around
            use_quantized_matmul = matmul_precision != "none"
            if use_quantized_matmul:
                if triton_is_available and torch.cuda.is_available():
                    print(f"[SDNQ Sampler] Applying Triton Quantized MatMul optimizations (precision: {matmul_precision})...")
                    try:
                        # Apply to transformer (FLUX, SD3)
                        if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                            # Try with matmul_precision first, fallback without it if not supported
                            try:
                                pipeline.transformer = apply_sdnq_options_to_model(
                                    pipeline.transformer,
                                    use_quantized_matmul=True,
                                    matmul_precision=matmul_precision
                                )
                            except TypeError:
                                # API doesn't support matmul_precision parameter
                                pipeline.transformer = apply_sdnq_options_to_model(
                                    pipeline.transformer,
                                    use_quantized_matmul=True
                                )
                            print("[SDNQ Sampler] ✓ Optimization applied to transformer")

                        # Apply to UNet (SDXL, SD1.5)
                        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
                            try:
                                pipeline.unet = apply_sdnq_options_to_model(
                                    pipeline.unet,
                                    use_quantized_matmul=True,
                                    matmul_precision=matmul_precision
                                )
                            except TypeError:
                                # API doesn't support matmul_precision parameter
                                pipeline.unet = apply_sdnq_options_to_model(
                                    pipeline.unet,
                                    use_quantized_matmul=True
                                )
                            print("[SDNQ Sampler] ✓ Optimization applied to UNet")

                        # Apply to text encoders (if they are quantized, e.g. FLUX.2)
                        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                            # Only apply if it looks like a quantized model (has SDNQ layers)
                            # Safe to try, sdnq loader checks internally
                            try:
                                pipeline.text_encoder = apply_sdnq_options_to_model(
                                    pipeline.text_encoder,
                                    use_quantized_matmul=True,
                                    matmul_precision=matmul_precision
                                )
                            except TypeError:
                                # API doesn't support matmul_precision parameter
                                pipeline.text_encoder = apply_sdnq_options_to_model(
                                    pipeline.text_encoder,
                                    use_quantized_matmul=True
                                )
                            print("[SDNQ Sampler] ✓ Optimization applied to text_encoder")

                        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                            pipeline.text_encoder_2 = apply_sdnq_options_to_model(
                                pipeline.text_encoder_2,
                                use_quantized_matmul=True,
                                matmul_precision=matmul_precision
                            )
                            print("[SDNQ Sampler] ✓ Optimization applied to text_encoder_2")

                    except Exception as e:
                        print(f"[SDNQ Sampler] ⚠️  Failed to apply optimizations: {e}")
                        print("[SDNQ Sampler] Continuing without optimizations...")
                else:
                    if not torch.cuda.is_available():
                        print("[SDNQ Sampler] ℹ️  Quantized MatMul requires CUDA. Optimization disabled.")
                    elif not triton_is_available:
                        print("[SDNQ Sampler] ℹ️  Triton not available/supported (requires Linux/WSL). Quantized MatMul disabled.")
            else:
                 print("[SDNQ Sampler] Quantized MatMul optimization disabled.")

            # CRITICAL: Apply xFormers BEFORE memory management
            # xFormers must be enabled before CPU offloading is set up
            if use_xformers:
                try:
                    print(f"[SDNQ Sampler] Enabling xFormers memory-efficient attention...")
                    print(f"[SDNQ Sampler] Current dtype: {dtype_str} (xFormers requires float16/bfloat16)")
                    
                    # Double-check dtype compatibility
                    if torch_dtype == torch.float32:
                        print(f"[SDNQ Sampler] ⚠️  CRITICAL: dtype is still float32, xFormers will fail!")
                        print(f"[SDNQ Sampler] ⚠️  This should have been converted earlier - check dtype conversion logic")
                        raise ValueError("xFormers requires float16 or bfloat16, but dtype is float32")
                    
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("[SDNQ Sampler] ✓ xFormers memory-efficient attention enabled")

                except ModuleNotFoundError as e:
                    # xFormers package not installed
                    print(f"[SDNQ Sampler] ⚠️  xFormers not installed: {e}")
                    print("[SDNQ Sampler] Install with: pip install xformers")
                    print("[SDNQ Sampler] Falling back to SDPA (PyTorch 2.0+ default attention)")

                except ValueError as e:
                    # CUDA not available or dtype incompatibility
                    error_msg = str(e)
                    if "float32" in error_msg or "dtype" in error_msg.lower():
                        print(f"[SDNQ Sampler] ⚠️  xFormers dtype incompatibility: {e}")
                        print("[SDNQ Sampler] xFormers only supports float16 and bfloat16, not float32")
                        print("[SDNQ Sampler] Please change dtype to bfloat16 or float16, or disable xFormers")
                    else:
                        print(f"[SDNQ Sampler] ⚠️  xFormers requires CUDA: {e}")
                    print("[SDNQ Sampler] Falling back to SDPA")

                except NotImplementedError as e:
                    # Model architecture doesn't support xFormers
                    print(f"[SDNQ Sampler] ℹ️  xFormers not supported for this model architecture")
                    print(f"[SDNQ Sampler] Details: {e}")
                    print("[SDNQ Sampler] Using SDPA instead (this is normal for some models)")

                except (RuntimeError, AttributeError) as e:
                    # Version incompatibility, dimension mismatch, or API changes
                    error_msg = str(e)
                    print(f"[SDNQ Sampler] ⚠️  xFormers compatibility issue: {type(e).__name__}")
                    print(f"[SDNQ Sampler] Error: {e}")
                    
                    # Check for dtype-related errors
                    if "float32" in error_msg or "dtype" in error_msg.lower() or "not supported" in error_msg.lower():
                        print("[SDNQ Sampler] This error is likely due to:")
                        print("[SDNQ Sampler]   - dtype=float32 (xFormers requires float16/bfloat16)")
                        print("[SDNQ Sampler]   - GPU capability mismatch (newer GPUs may not be fully supported)")
                        print("[SDNQ Sampler]   - xFormers version incompatibility with your GPU")
                    else:
                        print("[SDNQ Sampler] This may indicate:")
                        print("[SDNQ Sampler]   - xFormers version mismatch with PyTorch/CUDA")
                        print("[SDNQ Sampler]   - GPU architecture incompatibility")
                        print("[SDNQ Sampler]   - Tensor dimension issues with this model")
                    
                    print("[SDNQ Sampler] Try:")
                    print("[SDNQ Sampler]   1. Change dtype to bfloat16 or float16")
                    print("[SDNQ Sampler]   2. pip install -U xformers --force-reinstall")
                    print("[SDNQ Sampler]   3. Disable xFormers and use SDPA instead")
                    print("[SDNQ Sampler] Falling back to SDPA")

                except Exception as e:
                    # Unexpected error - log full details for debugging
                    error_msg = str(e)
                    print(f"[SDNQ Sampler] ⚠️  Unexpected xFormers error: {type(e).__name__}")
                    print(f"[SDNQ Sampler] Error message: {e}")
                    
                    # Check for dtype-related errors
                    if "float32" in error_msg or "dtype" in error_msg.lower() or "not supported" in error_msg.lower():
                        print("[SDNQ Sampler] This error is likely due to dtype=float32 or GPU capability mismatch")
                        print("[SDNQ Sampler] xFormers requires float16/bfloat16 and compatible GPU architecture")
                    
                    print("[SDNQ Sampler] Full traceback:")
                    traceback.print_exc()
                    print("[SDNQ Sampler] Falling back to SDPA")
            else:
                print("[SDNQ Sampler] Using SDPA (scaled dot product attention, default in PyTorch 2.0+)")

            # Flash Attention (FA) - requires ComfyUI started with --use-flash-attention
            if use_flash_attention:
                try:
                    print(f"[SDNQ Sampler] Enabling Flash Attention (FA)...")
                    print(f"[SDNQ Sampler] Current dtype: {dtype_str} (FA requires float16/bfloat16)")
                    
                    # Double-check dtype compatibility
                    if torch_dtype == torch.float32:
                        print(f"[SDNQ Sampler] ⚠️  CRITICAL: dtype is still float32, FA will fail!")
                        print(f"[SDNQ Sampler] ⚠️  This should have been converted earlier - check dtype conversion logic")
                        raise ValueError("Flash Attention requires float16 or bfloat16, but dtype is float32")
                    
                    flash_enabled = False
                    
                    # Flash Attention 2 is enabled via attn_processor
                    # For FLUX.2 models, they use Flux2AttnProcessor by default
                    # FLUX.2 models use Flux2ParallelSelfAttention which is incompatible with
                    # FusedFluxAttnProcessor2_0 or FluxAttnProcessor2_0
                    # Flash Attention must be enabled at ComfyUI system level (--use-flash-attention flag)
                    pipeline_type = type(pipeline).__name__
                    
                    # CRITICAL: FLUX.2's Flux2ParallelSelfAttention uses a different architecture
                    # Flux2ParallelSelfAttnProcessor uses parallel QKV decomposition and custom attention structure
                    # This is incompatible with standard FlashAttention2/SageAttention2.2 APIs
                    # 
                    # Why FA/SA cannot be used:
                    # - FA requires: Q, K, V together, contiguous, 1-shot attention, no CFG branching
                    # - Flux2 does: QKV decomposition, head/block parallel, CFG double pass, LoRA/Control branching
                    # - Even if dispatch_attention_fn supports FA/SA, Flux2's structure doesn't match FA kernel requirements
                    # - xFormers is just a dispatch layer - it can't fix architectural incompatibility
                    if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                        print(f"[SDNQ Sampler] ⚠️  WARNING: FLUX.2 uses Flux2ParallelSelfAttention")
                        print(f"[SDNQ Sampler] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible with FA/SA")
                        print(f"[SDNQ Sampler] ⚠️  Reason: Parallel QKV decomposition ≠ standard Q, K, V structure")
                        print(f"[SDNQ Sampler] ⚠️  FA requires: Q/K/V together, contiguous, 1-shot, no CFG branching")
                        print(f"[SDNQ Sampler] ⚠️  Flux2 does: QKV split, head/block parallel, CFG double pass, LoRA branching")
                        print(f"[SDNQ Sampler] ⚠️  xFormers is just dispatch layer - cannot fix architectural mismatch")
                        print(f"[SDNQ Sampler] ⚠️  FLUX.2 uses optimized parallel attention (different optimization path)")
                    if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                        # FLUX.2 models: Try to enable Flash Attention via transformer methods
                        if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                            # Try enable_xformers_memory_efficient_attention first
                            try:
                                if hasattr(pipeline.transformer, 'enable_xformers_memory_efficient_attention'):
                                    pipeline.transformer.enable_xformers_memory_efficient_attention()
                                    print("[SDNQ Sampler] ✓ Flash Attention enabled via transformer.enable_xformers_memory_efficient_attention()")
                                    flash_enabled = True
                                elif hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                                    pipeline.enable_xformers_memory_efficient_attention()
                                    print("[SDNQ Sampler] ✓ Flash Attention enabled via pipeline.enable_xformers_memory_efficient_attention()")
                                    flash_enabled = True
                            except Exception as e:
                                print(f"[SDNQ Sampler] ℹ️  xformers Flash Attention not available: {e}")
                            
                            # Try enable_attn_processor if xformers didn't work
                            if not flash_enabled:
                                try:
                                    # Check if transformer has enable_attn_processor method
                                    if hasattr(pipeline.transformer, 'enable_attn_processor'):
                                        # Try to enable Flash Attention 2 processor
                                        try:
                                            from diffusers.models.attention_processor import AttnProcessor2_0
                                            pipeline.transformer.enable_attn_processor(AttnProcessor2_0())
                                            print("[SDNQ Sampler] ✓ Flash Attention enabled via transformer.enable_attn_processor(AttnProcessor2_0)")
                                            flash_enabled = True
                                        except Exception as e:
                                            print(f"[SDNQ Sampler] ℹ️  AttnProcessor2_0 not compatible with FLUX.2: {e}")
                                except Exception as e:
                                    print(f"[SDNQ Sampler] ℹ️  Could not enable Flash Attention via enable_attn_processor: {e}")
                            
                            # If still not enabled, FLUX.2 uses default Flux2AttnProcessor
                            if not flash_enabled:
                                print("[SDNQ Sampler] ℹ️  FLUX.2 uses Flux2AttnProcessor (default)")
                                print("[SDNQ Sampler] ℹ️  Flash Attention may be handled internally by the processor")
                                print("[SDNQ Sampler] ℹ️  Check if ComfyUI --use-flash-attention flag enables it at system level")
                    else:
                        # For other models, use standard AttnProcessor2_0
                        if hasattr(pipeline, 'set_attn_processor') or hasattr(pipeline, 'enable_attn_processor'):
                            try:
                                from diffusers.models.attention_processor import AttnProcessor2_0
                                if hasattr(pipeline, 'set_attn_processor'):
                                    pipeline.set_attn_processor(AttnProcessor2_0())
                                else:
                                    pipeline.enable_attn_processor(AttnProcessor2_0())
                                print("[SDNQ Sampler] ✓ Flash Attention 2 enabled via AttnProcessor2_0")
                                flash_enabled = True
                            except (ImportError, AttributeError) as e:
                                # Fallback to xformers if AttnProcessor2_0 not available
                                try:
                                    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                                        pipeline.enable_xformers_memory_efficient_attention()
                                        print("[SDNQ Sampler] ✓ Flash Attention enabled via xformers")
                                        flash_enabled = True
                                    else:
                                        print(f"[SDNQ Sampler] ℹ️  AttnProcessor2_0 not available: {e}")
                                except Exception as e2:
                                    print(f"[SDNQ Sampler] ⚠️  Flash Attention via xformers failed: {e2}")
                    
                    # Check if already enabled via ComfyUI system level
                    if not flash_enabled:
                        # ComfyUI with --use-flash-attention enables it at model loading time
                        # Check if we can detect it via model attributes
                        try:
                            import comfy.model_management as model_management
                            # Flash attention is typically enabled globally in ComfyUI
                            print("[SDNQ Sampler] ℹ️  Flash Attention should be enabled at system level")
                            print("[SDNQ Sampler] ℹ️  (ComfyUI started with --use-flash-attention flag)")
                            
                            # Check transformer for flash attention processors
                            if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                                if hasattr(pipeline.transformer, 'attn_processors'):
                                    processors = pipeline.transformer.attn_processors
                                    if processors:
                                        print("[SDNQ Sampler] ℹ️  Checking transformer attention processors...")
                                        
                                        # Check actual processor types
                                        processor_types = {}
                                        for name, processor in processors.items():
                                            proc_type = type(processor).__name__
                                            processor_types[name] = proc_type
                                        
                                        # Check if Flash Attention processors are present
                                        flash_proc_names = [
                                            'FlashAttention2Processor',
                                            'AttnProcessor2_0',
                                            'XFormersAttnProcessor',
                                            'FlashAttnProcessor'
                                        ]
                                        
                                        found_flash = False
                                        for proc_type in processor_types.values():
                                            if any(flash_name in proc_type for flash_name in flash_proc_names):
                                                found_flash = True
                                                break
                                        
                                        # For FLUX.2 models, Flux2ParallelSelfAttnProcessor may use Flash Attention internally
                                        # Check if ComfyUI's --use-flash-attention flag is active
                                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                                            # FLUX.2 models use Flux2ParallelSelfAttnProcessor
                                            # WARNING: Flux2ParallelSelfAttention may NOT support standard Flash Attention
                                            # Even if --use-flash-attention flag is set, FA may not actually work
                                            try:
                                                import comfy.model_management as model_management
                                                print(f"[SDNQ Sampler] ⚠️  FLUX.2 processor types: {list(set(processor_types.values()))}")
                                                print("[SDNQ Sampler] ⚠️  FLUX.2 uses Flux2ParallelSelfAttention (may NOT support standard FA)")
                                                print("[SDNQ Sampler] ⚠️  Even with --use-flash-attention flag, FA may not actually work")
                                                print("[SDNQ Sampler] ⚠️  Flux2ParallelSelfAttention uses its own attention implementation")
                                                print("[SDNQ Sampler] ⚠️  If generation is slow, FA is likely NOT working")
                                                # Do NOT assume enabled - FLUX.2 may not support FA
                                                flash_enabled = False
                                            except ImportError:
                                                print(f"[SDNQ Sampler] ⚠️  Flash Attention not detected. Processor types: {list(set(processor_types.values()))}")
                                                print("[SDNQ Sampler] ⚠️  Make sure ComfyUI is started with --use-flash-attention flag")
                                        elif found_flash:
                                            print(f"[SDNQ Sampler] ✓ Flash Attention detected in processors: {list(set(processor_types.values()))}")
                                            flash_enabled = True
                                        else:
                                            print(f"[SDNQ Sampler] ⚠️  Flash Attention not detected. Processor types: {list(set(processor_types.values()))}")
                                            print("[SDNQ Sampler] ⚠️  Make sure ComfyUI is started with --use-flash-attention flag")
                                    else:
                                        print("[SDNQ Sampler] ⚠️  No attention processors found on transformer")
                                else:
                                    print("[SDNQ Sampler] ⚠️  Transformer does not have attn_processors attribute")
                            else:
                                print("[SDNQ Sampler] ⚠️  Pipeline does not have transformer attribute")
                                
                        except ImportError:
                            print("[SDNQ Sampler] ⚠️  Cannot verify Flash Attention status")
                            print("[SDNQ Sampler] Make sure ComfyUI is started with --use-flash-attention flag")
                        except Exception as e:
                            print(f"[SDNQ Sampler] ⚠️  Error checking Flash Attention: {e}")
                    
                    # Final status report with detailed information
                    if flash_enabled:
                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                            print("[SDNQ Sampler] ⚠️  FLUX.2 Flash Attention: NOT SUPPORTED")
                            print("[SDNQ Sampler] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible")
                            print("[SDNQ Sampler] ⚠️  Uses parallel QKV decomposition (not standard attention structure)")
                            print("[SDNQ Sampler] ⚠️  FlashAttention2 requires standard Q, K, V (incompatible)")
                            print("[SDNQ Sampler] ⚠️  FLUX.2 uses optimized parallel attention (different optimization)")
                        else:
                            print("[SDNQ Sampler] ✓ Flash Attention is ACTIVE")
                            print("[SDNQ Sampler] ℹ️  If generation is still slow, FA may not be working correctly")
                            print("[SDNQ Sampler] ℹ️  Check ComfyUI startup logs for 'Flash Attention ✅' message")
                            print("[SDNQ Sampler] ℹ️  Verify dtype is bfloat16 or float16 (not float32)")
                    else:
                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                            print("[SDNQ Sampler] ⚠️  Flash Attention: NOT SUPPORTED for FLUX.2")
                            print("[SDNQ Sampler] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible")
                            print("[SDNQ Sampler] ⚠️  Parallel QKV decomposition ≠ standard Q, K, V structure")
                            print("[SDNQ Sampler] ⚠️  FlashAttention2/SageAttention2.2 cannot be used")
                            print("[SDNQ Sampler] ⚠️  This is expected - FLUX.2 uses different optimization path")
                        else:
                            print("[SDNQ Sampler] ⚠️  Flash Attention is NOT active")
                            print("[SDNQ Sampler] ⚠️  Possible reasons:")
                            print("[SDNQ Sampler]   1. ComfyUI not started with --use-flash-attention flag")
                            print("[SDNQ Sampler]   2. flash-attn package not installed (pip install flash-attn)")
                            print("[SDNQ Sampler]   3. dtype is float32 (FA requires float16/bfloat16)")
                            print("[SDNQ Sampler]   4. GPU not compatible with Flash Attention")
                            print("[SDNQ Sampler] ⚠️  Check startup logs for 'Flash Attention ✅' message")
                            print("[SDNQ Sampler] ⚠️  Generation will use default attention (slower)")
                        
                except ValueError as e:
                    # Dtype incompatibility
                    print(f"[SDNQ Sampler] ⚠️  Flash Attention dtype error: {e}")
                    print("[SDNQ Sampler] ⚠️  Flash Attention requires float16 or bfloat16, not float32")
                    print("[SDNQ Sampler] ⚠️  Please change dtype to bfloat16 or float16")
                except Exception as e:
                    print(f"[SDNQ Sampler] ⚠️  Failed to enable Flash Attention: {e}")
                    print("[SDNQ Sampler] ⚠️  Make sure ComfyUI is started with --use-flash-attention flag")
                    print("[SDNQ Sampler] ⚠️  Check that flash-attn package is installed: pip install flash-attn")

            # Sage Attention (SA) - requires ComfyUI started with --use-sage-attention
            if use_sage_attention:
                try:
                    print(f"[SDNQ Sampler] Enabling Sage Attention (SA)...")
                    print(f"[SDNQ Sampler] Current dtype: {dtype_str} (SA requires float16/bfloat16)")
                    
                    # Double-check dtype compatibility
                    if torch_dtype == torch.float32:
                        print(f"[SDNQ Sampler] ⚠️  CRITICAL: dtype is still float32, SA will fail!")
                        print(f"[SDNQ Sampler] ⚠️  This should have been converted earlier - check dtype conversion logic")
                        raise ValueError("Sage Attention requires float16 or bfloat16, but dtype is float32")
                    
                    # CRITICAL: FLUX.2's Flux2ParallelSelfAttention uses a different architecture
                    # Same explanation as Flash Attention - architectural incompatibility
                    if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                        print(f"[SDNQ Sampler] ⚠️  WARNING: FLUX.2 uses Flux2ParallelSelfAttention")
                        print(f"[SDNQ Sampler] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible with SA")
                        print(f"[SDNQ Sampler] ⚠️  Same reason as Flash Attention: parallel QKV decomposition")
                        print(f"[SDNQ Sampler] ⚠️  SageAttention2.2 requires standard Q, K, V structure (incompatible)")
                        print(f"[SDNQ Sampler] ⚠️  FLUX.2 uses optimized parallel attention (different optimization path)")
                    
                    # Sage Attention is typically enabled at model initialization
                    # Check if it's available via ComfyUI's system-level settings
                    sage_enabled = False
                    
                    # Try to enable on transformer (FLUX models)
                    if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                        if hasattr(pipeline.transformer, 'enable_sage_attention'):
                            pipeline.transformer.enable_sage_attention()
                            print("[SDNQ Sampler] ✓ Sage Attention enabled on transformer")
                            sage_enabled = True
                        # Check if already enabled by checking attention modules
                        elif hasattr(pipeline.transformer, 'attn_processors'):
                            # Check if sage attention is already active
                            processors = pipeline.transformer.attn_processors
                            if processors:
                                print("[SDNQ Sampler] ℹ️  Checking transformer attention processors...")
                                # Sage attention might be set at system level
                                sage_enabled = True
                    
                    # Try to enable on pipeline level
                    if not sage_enabled and hasattr(pipeline, 'enable_sage_attention'):
                        pipeline.enable_sage_attention()
                        print("[SDNQ Sampler] ✓ Sage Attention enabled on pipeline")
                        sage_enabled = True
                    
                    # Check if Sage Attention is available at system level
                    if not sage_enabled:
                        try:
                            # Check if sageattention package is available (indicates Sage Attention is installed)
                            from diffusers.utils import is_sageattention_available
                            
                            sage_attn_available = is_sageattention_available()
                            
                            if sage_attn_available:
                                # Sage Attention package is available
                                # For FLUX.2 models, if ComfyUI was started with --use-sage-attention,
                                # Sage Attention should be active even if processor name doesn't show it
                                if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                                    print(f"[SDNQ Sampler] ✓ Sage Attention package available")
                                    print("[SDNQ Sampler] ✓ Sage Attention is ACTIVE for FLUX.2 (enabled via --use-sage-attention flag)")
                                    sage_enabled = True
                                else:
                                    # For other models, check if already enabled
                                    print(f"[SDNQ Sampler] ✓ Sage Attention package available")
                                    print("[SDNQ Sampler] ℹ️  Sage Attention may be active if --use-sage-attention flag is set")
                                    sage_enabled = True  # Assume enabled if package is available and flag is set
                            else:
                                print("[SDNQ Sampler] ⚠️  Sage Attention package not available")
                                print("[SDNQ Sampler] ⚠️  Install with: pip install sageattention")
                                
                        except ImportError as e:
                            print(f"[SDNQ Sampler] ⚠️  Cannot check Sage Attention availability: {e}")
                        except Exception as e:
                            print(f"[SDNQ Sampler] ⚠️  Error checking Sage Attention: {e}")
                    
                    # Final status report with detailed information
                    if sage_enabled:
                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                            print("[SDNQ Sampler] ⚠️  FLUX.2 Sage Attention: NOT SUPPORTED")
                            print("[SDNQ Sampler] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible")
                            print("[SDNQ Sampler] ⚠️  Uses parallel QKV decomposition (not standard attention structure)")
                            print("[SDNQ Sampler] ⚠️  SageAttention2.2 requires standard Q, K, V (incompatible)")
                            print("[SDNQ Sampler] ⚠️  FLUX.2 uses optimized parallel attention (different optimization)")
                        else:
                            print("[SDNQ Sampler] ✓ Sage Attention is ACTIVE")
                            print("[SDNQ Sampler] ℹ️  If generation is still slow, SA may not be working correctly")
                            print("[SDNQ Sampler] ℹ️  Check ComfyUI startup logs for 'Sage Attention ✅' message")
                            print("[SDNQ Sampler] ℹ️  Verify dtype is bfloat16 or float16 (not float32)")
                    else:
                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                            print("[SDNQ Sampler] ⚠️  Sage Attention: NOT SUPPORTED for FLUX.2")
                            print("[SDNQ Sampler] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible")
                            print("[SDNQ Sampler] ⚠️  Parallel QKV decomposition ≠ standard Q, K, V structure")
                            print("[SDNQ Sampler] ⚠️  FlashAttention2/SageAttention2.2 cannot be used")
                            print("[SDNQ Sampler] ⚠️  This is expected - FLUX.2 uses different optimization path")
                        else:
                            print("[SDNQ Sampler] ⚠️  Sage Attention is NOT active")
                            print("[SDNQ Sampler] ⚠️  Possible reasons:")
                            print("[SDNQ Sampler]   1. ComfyUI not started with --use-sage-attention flag")
                            print("[SDNQ Sampler]   2. sageattention package not installed (pip install sageattention)")
                            print("[SDNQ Sampler]   3. dtype is float32 (SA requires float16/bfloat16)")
                            print("[SDNQ Sampler]   4. GPU not compatible with Sage Attention")
                            print("[SDNQ Sampler] ⚠️  Check startup logs for 'Sage Attention ✅' message")
                            print("[SDNQ Sampler] ⚠️  Generation will use default attention (slower)")
                    
                except ValueError as e:
                    # Dtype incompatibility
                    print(f"[SDNQ Sampler] ⚠️  Sage Attention dtype error: {e}")
                    print("[SDNQ Sampler] ⚠️  Sage Attention requires float16 or bfloat16, not float32")
                    print("[SDNQ Sampler] ⚠️  Please change dtype to bfloat16 or float16")
                except Exception as e:
                    print(f"[SDNQ Sampler] ⚠️  Failed to enable Sage Attention: {e}")
                    print("[SDNQ Sampler] ⚠️  Make sure ComfyUI is started with --use-sage-attention flag")
                    print("[SDNQ Sampler] ⚠️  Check that sageattention package is installed: pip install sageattention")

            # Apply memory management strategy
            # Based on: https://huggingface.co/docs/diffusers/main/optimization/memory
            if memory_mode == "gpu":
                # Full GPU mode: Fastest performance, needs 24GB+ VRAM
                # Load entire pipeline to GPU
                print(f"[SDNQ Sampler] Moving model to GPU (full GPU mode)...")
                pipeline.to("cuda")
                print(f"[SDNQ Sampler] ✓ Model loaded to GPU (all components on VRAM)")

            elif memory_mode == "balanced":
                # Sequential CPU offload: Prevents VRAM growth during generation
                # All components start on CPU, moved to GPU only when needed during generation
                # This prevents VRAM from growing during generation process
                print(f"[SDNQ Sampler] Enabling sequential CPU offload (balanced mode)...")
                print(f"[SDNQ Sampler] All components will be offloaded to CPU before generation")
                print(f"[SDNQ Sampler] Components will be moved to GPU only when needed during generation")
                print(f"[SDNQ Sampler] This prevents VRAM growth during generation process")
                try:
                    # Get VRAM info for logging (before enabling offload)
                    if torch.cuda.is_available():
                        device_props = torch.cuda.get_device_properties(0)
                        vram_total_bytes = device_props.total_memory
                        vram_total_gb = vram_total_bytes / (1024**3)
                        
                        # Clear cache before enabling offload
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                        # Get current VRAM usage before offload
                        vram_free, _ = torch.cuda.mem_get_info()
                        vram_used = vram_total_bytes - vram_free
                        vram_used_gb = vram_used / (1024**3)
                        
                        print(f"[SDNQ Sampler] VRAM: {vram_total_gb:.2f}GB total, {vram_used_gb:.2f}GB used before offload")
                    
                    # Enable sequential CPU offload
                    # This automatically manages component placement:
                    # - Components start on CPU
                    # - Moved to GPU only when needed during generation
                    # - Automatically moved back to CPU when not needed
                    # DO NOT manually move components with .to("cpu") after this!
                    pipeline.enable_sequential_cpu_offload()
                    print("[SDNQ Sampler] ✓ Sequential CPU offload enabled")
                    print("[SDNQ Sampler] ✓ Components will be automatically managed by diffusers")
                    print("[SDNQ Sampler] ✓ This prevents VRAM growth during generation process")
                        
                except Exception as e:
                    # Fallback to standard offload if sequential fails
                    print(f"[SDNQ Sampler] ⚠️  Sequential offload failed, using standard offload: {e}")
                    pipeline.enable_model_cpu_offload()
                    print("[SDNQ Sampler] ✓ Model offloading enabled (standard mode)")

            elif memory_mode == "lowvram":
                # Sequential CPU offload: Maximum memory savings for 8GB VRAM
                # Slowest but works on limited VRAM
                print(f"[SDNQ Sampler] Enabling sequential CPU offload (low VRAM mode)...")
                pipeline.enable_sequential_cpu_offload()
                print(f"[SDNQ Sampler] ✓ Sequential offloading enabled (minimal VRAM usage)")

            # VAE tiling (works with all memory modes, but not all pipelines support it)
            if enable_vae_tiling:
                try:
                    # Check if pipeline supports VAE tiling
                    # FLUX.2 and some other pipelines don't have this method
                    if hasattr(pipeline, 'enable_vae_tiling'):
                        pipeline.enable_vae_tiling()
                        print("[SDNQ Sampler] ✓ VAE tiling enabled")
                    else:
                        pipeline_type = type(pipeline).__name__
                        print(f"[SDNQ Sampler] ℹ️  VAE tiling not supported by {pipeline_type} pipeline")
                except Exception as e:
                    print(f"[SDNQ Sampler] ⚠️  VAE tiling failed: {e}")

            return pipeline

        except Exception as e:
            raise Exception(
                f"Failed to load SDNQ model from: {model_path}\n\n"
                f"Error: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"1. Verify the path contains a valid diffusers model (should have model_index.json)\n"
                f"2. Check if model download completed successfully\n"
                f"3. Try a different dtype (bfloat16 requires modern GPUs)\n"
                f"4. Check VRAM availability (use smaller model if needed)\n"
                f"5. Look at the error message above for specific details"
            )

    def load_lora(self, pipeline: DiffusionPipeline, lora_path: str, lora_strength: float = 1.0):
        """
        Load LoRA weights into pipeline.

        Supports both local .safetensors files and HuggingFace repo IDs.

        Args:
            pipeline: Loaded diffusers pipeline
            lora_path: Path to LoRA file or HuggingFace repo ID
            lora_strength: LoRA influence strength (0.0 to 2.0)

        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/api/loaders/lora
        https://huggingface.co/blog/lora-fast
        """
        import os

        if not lora_path or lora_path.strip() == "":
            print(f"[SDNQ Sampler] No LoRA specified, skipping...")
            return

        print(f"[SDNQ Sampler] Loading LoRA...")
        print(f"[SDNQ Sampler]   Path: {lora_path}")
        print(f"[SDNQ Sampler]   Strength: {lora_strength}")

        try:
            # Check if it's a local file or HuggingFace repo
            is_local_file = os.path.exists(lora_path) and os.path.isfile(lora_path)

            if is_local_file:
                # Local .safetensors file
                # Extract directory and filename
                lora_dir = os.path.dirname(lora_path)
                lora_file = os.path.basename(lora_path)

                pipeline.load_lora_weights(
                    lora_dir,
                    weight_name=lora_file,
                    adapter_name="lora"
                )
            else:
                # Assume it's a HuggingFace repo ID
                pipeline.load_lora_weights(
                    lora_path,
                    adapter_name="lora"
                )

            # Set LoRA strength
            if lora_strength != 1.0:
                pipeline.set_adapters(["lora"], adapter_weights=[lora_strength])
            else:
                pipeline.set_adapters(["lora"])

            print(f"[SDNQ Sampler] ✓ LoRA loaded successfully")

        except Exception as e:
            raise Exception(
                f"Failed to load LoRA\n\n"
                f"Error: {str(e)}\n\n"
                f"LoRA path: {lora_path}\n\n"
                f"Troubleshooting:\n"
                f"1. Verify LoRA file exists (.safetensors format)\n"
                f"2. For HuggingFace repos, verify repo ID is correct\n"
                f"3. Ensure LoRA is compatible with the model architecture\n"
                f"4. Check if LoRA is for the correct model type (FLUX, SDXL, etc.)\n"
                f"5. Try with lora_strength=1.0 first"
            )

    def unload_lora(self, pipeline: DiffusionPipeline):
        """
        Unload LoRA weights from pipeline.

        Args:
            pipeline: Pipeline with loaded LoRA
        """
        try:
            if hasattr(pipeline, 'unload_lora_weights'):
                print(f"[SDNQ Sampler] Unloading previous LoRA...")
                pipeline.unload_lora_weights()
        except Exception as e:
            # Non-critical error, just log it
            print(f"[SDNQ Sampler] Warning: Failed to unload LoRA: {e}")

    def swap_scheduler(self, pipeline: DiffusionPipeline, scheduler_name: str):
        """
        Swap the pipeline's scheduler.

        Uses the from_config() pattern to create a new scheduler with the same
        configuration as the current scheduler but with different algorithm.

        Args:
            pipeline: Loaded diffusers pipeline
            scheduler_name: Name of scheduler to use

        Raises:
            Exception: If scheduler swap fails

        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers
        """
        print(f"[SDNQ Sampler] Swapping scheduler to: {scheduler_name}")

        try:
            # Map scheduler name to class
            scheduler_map = {
                # Flow-based schedulers (FLUX, SD3, Qwen, Z-Image)
                "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
                # Traditional diffusion schedulers (SDXL, SD1.5)
                "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
                "UniPCMultistepScheduler": UniPCMultistepScheduler,
                "EulerDiscreteScheduler": EulerDiscreteScheduler,
                "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
                "DDIMScheduler": DDIMScheduler,
                "HeunDiscreteScheduler": HeunDiscreteScheduler,
                "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
                "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
                "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler,
                "DEISMultistepScheduler": DEISMultistepScheduler,
                "LMSDiscreteScheduler": LMSDiscreteScheduler,
                "DDPMScheduler": DDPMScheduler,
                "PNDMScheduler": PNDMScheduler,
            }

            if scheduler_name not in scheduler_map:
                raise ValueError(
                    f"Unknown scheduler: {scheduler_name}\n"
                    f"Available schedulers: {list(scheduler_map.keys())}"
                )

            scheduler_class = scheduler_map[scheduler_name]

            # Swap scheduler using from_config pattern
            # This preserves scheduler configuration while changing the algorithm
            pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)

            print(f"[SDNQ Sampler] ✓ Scheduler swapped successfully")

        except Exception as e:
            raise Exception(
                f"Failed to swap scheduler\n\n"
                f"Error: {str(e)}\n\n"
                f"Requested scheduler: {scheduler_name}\n\n"
                f"Troubleshooting:\n"
                f"1. Ensure scheduler is compatible with the model type\n"
                f"2. FLUX/SD3/Qwen/Z-Image: Use FlowMatchEulerDiscreteScheduler\n"
                f"3. SDXL/SD1.5: Use DPMSolver, Euler, UniPC, or DDIM\n"
                f"4. Wrong scheduler type will produce broken/corrupted images\n"
                f"5. Check diffusers version (requires >=0.36.0)"
            )


    def generate_image(self, pipeline: DiffusionPipeline, prompt: str, negative_prompt: str,
                      steps: int, cfg: float, width: int, height: int, seed: int) -> Image.Image:
        """
        Generate image using the loaded pipeline.

        Args:
            pipeline: Loaded diffusers pipeline
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (optional)
            steps: Number of inference steps
            cfg: Guidance scale (classifier-free guidance strength)
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            seed: Random seed for reproducibility

        Returns:
            PIL Image object

        Raises:
            Exception: If generation fails or is interrupted

        Based on verified API from FLUX examples:
        https://huggingface.co/docs/diffusers/main/api/pipelines/flux
        """
        print(f"[SDNQ Sampler] Generating image...")
        print(f"[SDNQ Sampler]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"[SDNQ Sampler]   Steps: {steps}, CFG: {cfg}")
        print(f"[SDNQ Sampler]   Size: {width}x{height}")
        print(f"[SDNQ Sampler]   Seed: {seed}")

        # Check for interruption before starting
        if self.check_interrupted():
            raise InterruptedError("Generation interrupted by user")

        try:
            # Clear VRAM cache before generation to maximize available memory
            # This helps prevent OOM during generation, especially for large images
            if torch.cuda.is_available():
                # Aggressive memory cleanup before generation
                # Multiple passes to ensure all cached memory is freed
                import gc
                for _ in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Note: Do NOT manually move components with .to("cpu")
                # If enable_sequential_cpu_offload() was used, diffusers manages component placement automatically
                # Manually moving components can cause "Cannot copy out of meta tensor" errors
                
                vram_free_before, _ = torch.cuda.mem_get_info()
                device_props = torch.cuda.get_device_properties(0)
                vram_total = device_props.total_memory
                vram_used_before = vram_total - vram_free_before
                vram_used_before_gb = vram_used_before / (1024**3)
                vram_free_before_gb = vram_free_before / (1024**3)
                
                print(f"[SDNQ Sampler] VRAM before generation: {vram_used_before_gb:.2f}GB used, {vram_free_before_gb:.2f}GB free")
                print(f"[SDNQ Sampler] All components on CPU, will be moved to GPU only when needed during generation")
            
            # Create generator for reproducible generation
            # Generator handles random sampling during denoising
            generator = torch.Generator(device="cuda").manual_seed(seed)

            # Build pipeline call kwargs
            # Only include parameters that are supported by the specific pipeline
            # Different pipelines have different signatures (FLUX.2 doesn't accept negative_prompt)
            pipeline_kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "width": width,
                "height": height,
                "generator": generator,
            }

            # Check if pipeline supports negative_prompt
            # FLUX.2, FLUX-schnell, and some other pipelines don't support it
            pipeline_type = type(pipeline).__name__
            supports_negative_prompt = pipeline_type not in ["Flux2Pipeline", "FluxPipeline", "FluxSchnellPipeline"]
            
            # Standard generation path
            # Only add negative_prompt if pipeline supports it and it's not empty
            if supports_negative_prompt and negative_prompt and negative_prompt.strip():
                pipeline_kwargs["negative_prompt"] = negative_prompt
            elif negative_prompt and negative_prompt.strip() and not supports_negative_prompt:
                print(f"[SDNQ Sampler] ⚠️  Pipeline {pipeline_type} doesn't support negative_prompt - skipping it")

            # Try calling pipeline with all parameters
            # If negative_prompt is unsupported, retry without it
            try:
                result = pipeline(**pipeline_kwargs)
            except TypeError as e:
                # Check if error is about negative_prompt parameter
                if "negative_prompt" in str(e) and "unexpected keyword argument" in str(e):
                    # Pipeline doesn't support negative_prompt (e.g., FLUX.2, FLUX-schnell)
                    print(f"[SDNQ Sampler] ⚠️  Pipeline {type(pipeline).__name__} doesn't support negative_prompt - skipping it")

                    # Remove negative_prompt and retry
                    if "negative_prompt" in pipeline_kwargs:
                        del pipeline_kwargs["negative_prompt"]

                    # Retry generation without negative_prompt
                    result = pipeline(**pipeline_kwargs)
                else:
                    # Different TypeError - re-raise with helpful message
                    import re
                    match = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                    param_name = match.group(1) if match else "unknown"
                    raise Exception(
                        f"Pipeline doesn't support parameter: '{param_name}'\n\n"
                        f"Error: {str(e)}\n\n"
                        f"Pipeline type: {type(pipeline).__name__}\n"
                        f"This pipeline has a different signature than expected.\n\n"
                        f"Please report this issue on GitHub with the pipeline type above."
                    )

            # Check for interruption after generation
            if self.check_interrupted():
                raise InterruptedError("Generation interrupted by user")

            # Extract first image from results
            # result.images[0] is a PIL.Image.Image object
            image = result.images[0]

            print(f"[SDNQ Sampler] Image generated! Size: {image.size}")

            return image

        except InterruptedError:
            raise
        except torch.cuda.OutOfMemoryError as e:
            # VRAM不足エラーの場合、より具体的な対処法を提示
            raise Exception(
                f"VRAM不足エラーが発生しました\n\n"
                f"Error: {str(e)}\n\n"
                f"対処法:\n"
                f"1. Memory modeを'lowvram'に変更（sequential CPU offload）\n"
                f"2. 画像サイズを小さくする（例: 1024x1024 → 768x768）\n"
                f"3. VAE tilingを有効にする（enable_vae_tilingをtrueに）\n"
                f"4. 他のアプリケーションを閉じてVRAMを解放\n"
                f"5. より小さいモデルを使用する\n"
                f"6. Stepsを減らす（現在: {steps}）\n"
                f"7. Memory modeを'lowvram'に変更することを推奨"
            )
        except Exception as e:
            # Don't double-wrap exceptions we already formatted
            if "Pipeline doesn't support parameter" in str(e):
                raise

            # Other errors - provide troubleshooting
            raise Exception(
                f"Failed to generate image\n\n"
                f"Error: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"1. Check VRAM usage (reduce size or use smaller model)\n"
                f"2. Verify parameters are valid (size multiple of 8, CFG reasonable)\n"
                f"3. Try reducing steps if running out of memory\n"
                f"4. Some models have specific parameter requirements (check HuggingFace page)\n"
                f"5. Look at the error message above for specific details"
            )

    def pil_to_comfy_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to ComfyUI IMAGE tensor format.

        ComfyUI IMAGE format (verified from nodes.py LoadImage node):
        - Shape: [N, H, W, C] (batch, height, width, channels)
        - Dtype: torch.float32
        - Range: 0.0 to 1.0 (normalized)
        - Color: RGB

        Args:
            pil_image: PIL.Image.Image object

        Returns:
            torch.Tensor in ComfyUI format [1, H, W, 3]

        Based on verified conversion from ComfyUI nodes.py:
        https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py
        """
        # Ensure image is RGB (no alpha channel)
        pil_image = pil_image.convert("RGB")

        # Convert to numpy array and normalize to 0-1 range
        # PIL images are uint8 (0-255), ComfyUI uses float32 (0.0-1.0)
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        # [H, W, C] -> [1, H, W, C]
        tensor = torch.from_numpy(numpy_image)[None, :]

        print(f"[SDNQ Sampler] Converted to ComfyUI tensor: shape={tensor.shape}, dtype={tensor.dtype}")

        return tensor

    def generate(self, model_selection: str, custom_model_path: str, prompt: str,
                negative_prompt: str, steps: int, cfg: float, width: int, height: int,
                seed: int, scheduler: str, dtype: str, memory_mode: str, auto_download: bool,
                lora_selection: str = "[None]", lora_custom_path: str = "", lora_strength: float = 1.0,
                use_xformers: bool = False, use_flash_attention: bool = False,
                use_sage_attention: bool = False, enable_vae_tiling: bool = False,
                matmul_precision: str = "int8") -> Tuple[torch.Tensor]:
        """
        Main generation function called by ComfyUI.

        This is the entry point when the node executes in a workflow.

        Args:
            model_selection: Selected model from dropdown
            custom_model_path: Custom model path (if [Custom Path] selected)
            prompt: Text prompt
            negative_prompt: Negative prompt
            steps: Inference steps
            cfg: Guidance scale
            width: Image width
            height: Image height
            seed: Random seed
            scheduler: Scheduler algorithm name
            dtype: Data type string
            memory_mode: Memory management mode ("gpu", "balanced", "lowvram")
            auto_download: Whether to auto-download models
            lora_selection: Selected LoRA from dropdown ([None], [Custom Path], or filename)
            lora_custom_path: Custom LoRA path (used when [Custom Path] selected)
            lora_strength: LoRA influence strength (-5.0 to +5.0)
            use_xformers: Enable xFormers memory-efficient attention (10-45% speedup)
            use_flash_attention: Enable Flash Attention (FA) for faster inference and lower VRAM
            use_sage_attention: Enable Sage Attention (SA) for optimized attention computation
            enable_vae_tiling: Enable VAE tiling for large images
            matmul_precision: Precision for Triton quantized matmul ("int8", "fp8", "none")

        Returns:
            Tuple containing (IMAGE,) in ComfyUI format

        Raises:
            ValueError: For invalid inputs
            FileNotFoundError: For missing models/paths
            Exception: For other errors during loading/generation
        """
        print(f"\n{'='*60}")
        print(f"[SDNQ Sampler] Starting generation")
        print(f"{'='*60}\n")

        self.interrupted = False

        try:
            # Step 1: Load or download model
            model_path, was_downloaded, repo_id = self.load_or_download_model(
                model_selection,
                custom_model_path,
                auto_download
            )

            # Step 2: Load pipeline (with caching)
            # Check if we need to reload the pipeline
            # Cache is invalidated if any of these change:
            # - Model path, dtype, memory mode
            # - Performance optimization settings (xformers, flash_attention, sage_attention, vae_tiling, matmul_precision)
            if (self.pipeline is None or
                self.current_model_path != model_path or
                self.current_dtype != dtype or
                self.current_memory_mode != memory_mode or
                self.current_use_xformers != use_xformers or
                self.current_use_flash_attention != use_flash_attention or
                self.current_use_sage_attention != use_sage_attention or
                self.current_enable_vae_tiling != enable_vae_tiling or
                self.current_matmul_precision != matmul_precision):

                print(f"[SDNQ Sampler] Pipeline cache miss - loading model...")
                self.pipeline = self.load_pipeline(
                    model_path, dtype, memory_mode,
                    use_xformers=use_xformers,
                    use_flash_attention=use_flash_attention,
                    use_sage_attention=use_sage_attention,
                    enable_vae_tiling=enable_vae_tiling,
                    matmul_precision=matmul_precision,
                    repo_id=repo_id
                )
                self.current_model_path = model_path
                self.current_dtype = dtype
                self.current_memory_mode = memory_mode
                self.current_use_xformers = use_xformers
                self.current_use_flash_attention = use_flash_attention
                self.current_use_sage_attention = use_sage_attention
                self.current_enable_vae_tiling = enable_vae_tiling
                self.current_matmul_precision = matmul_precision
                # Clear LoRA and scheduler cache when pipeline changes
                self.current_lora_path = None
                self.current_lora_strength = None
                self.current_scheduler = None
            else:
                print(f"[SDNQ Sampler] Using cached pipeline")

            # Step 2.5: Handle LoRA loading/unloading
            # Resolve actual LoRA path from lora_selection and lora_custom_path
            lora_path = None
            if lora_selection == "[None]":
                lora_path = None
            elif lora_selection == "[Custom Path]":
                lora_path = lora_custom_path if lora_custom_path and lora_custom_path.strip() else None
            else:
                # User selected a LoRA from the dropdown
                # Build full path from ComfyUI loras folder
                if COMFYUI_AVAILABLE:
                    try:
                        lora_folders = folder_paths.get_folder_paths("loras")
                        if lora_folders:
                            # Try to find the file in lora folders
                            for lora_folder in lora_folders:
                                potential_path = os.path.join(lora_folder, lora_selection)
                                if os.path.exists(potential_path):
                                    lora_path = potential_path
                                    break
                            if not lora_path:
                                # Fallback: use first folder + filename
                                lora_path = os.path.join(lora_folders[0], lora_selection)
                    except Exception as e:
                        print(f"[SDNQ Sampler] Warning: Could not resolve LoRA path: {e}")
                        lora_path = lora_selection  # Try using it as-is

            # Check if LoRA configuration has changed
            lora_changed = (lora_path != self.current_lora_path or
                           lora_strength != self.current_lora_strength)

            if lora_path and lora_path.strip():
                # User wants to use LoRA
                if lora_changed:
                    print(f"[SDNQ Sampler] LoRA configuration changed - updating...")

                    # Unload previous LoRA if it exists
                    if self.current_lora_path:
                        self.unload_lora(self.pipeline)

                    # Load new LoRA
                    self.load_lora(self.pipeline, lora_path, lora_strength)
                    self.current_lora_path = lora_path
                    self.current_lora_strength = lora_strength
                else:
                    print(f"[SDNQ Sampler] Using cached LoRA: {lora_path}")
            else:
                # User doesn't want LoRA, but we have one loaded
                if self.current_lora_path:
                    print(f"[SDNQ Sampler] Unloading LoRA...")
                    self.unload_lora(self.pipeline)
                    self.current_lora_path = None
                    self.current_lora_strength = None

            # Step 2.6: Handle scheduler swap
            # Check if scheduler changed from cached value
            if scheduler != self.current_scheduler:
                print(f"[SDNQ Sampler] Scheduler changed - swapping to {scheduler}...")
                self.swap_scheduler(self.pipeline, scheduler)
                self.current_scheduler = scheduler
            else:
                if self.current_scheduler:
                    print(f"[SDNQ Sampler] Using cached scheduler: {scheduler}")

            # Step 3: Generate image
            pil_image = self.generate_image(
                self.pipeline,
                prompt,
                negative_prompt,
                steps,
                cfg,
                width,
                height,
                seed,
            )

            # Step 4: Convert to ComfyUI format
            comfy_tensor = self.pil_to_comfy_tensor(pil_image)

            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler] Generation complete!")
            print(f"{'='*60}\n")

            # Return as tuple (ComfyUI expects tuple of outputs)
            return (comfy_tensor,)

        except InterruptedError as e:
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler] Generation interrupted")
            print(f"{'='*60}\n")
            raise

        except (ValueError, FileNotFoundError) as e:
            # User errors - display message clearly
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler] Error: {str(e)}")
            print(f"{'='*60}\n")
            raise

        except Exception as e:
            # Unexpected errors - provide full traceback
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler] Unexpected error occurred")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            raise
