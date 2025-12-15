"""
SDNQ Model Loader Node - V3 API Compatible

This node loads SDNQ quantized models and outputs a MODEL (DiffusionPipeline).
Separated from sampler for better modularity.

Architecture: Model Selection → Download (if needed) → Load Pipeline → Output MODEL
"""

import torch
import os
import warnings
import traceback
from typing import Optional, Tuple

# SDNQ import - registers SDNQ support into diffusers
from sdnq import SDNQConfig
# SDNQ optimization imports
try:
    from sdnq.loader import apply_sdnq_options_to_model
    from sdnq.common import use_torch_compile as triton_is_available
except ImportError:
    print("[SDNQ Model Loader] Warning: Could not import SDNQ optimization tools. Quantized MatMul will be disabled.")
    def apply_sdnq_options_to_model(model, **kwargs): return model
    triton_is_available = False

# diffusers pipeline - auto-detects model type from model_index.json
from diffusers import DiffusionPipeline

# Local imports for model catalog and downloading
from ..core.registry import (
    get_model_names_for_dropdown,
    get_model_info,
)
from ..core.downloader import (
    download_model,
    get_cached_model_path,
)


class ComfyVAEWrapper:
    """
    Wrapper for diffusers VAE to make it compatible with ComfyUI's VAEEncode node.
    
    ComfyUI's VAEEncode expects:
    - Input: [N, H, W, C] format (NHWC)
    - Output: [N, C, H, W] format (NCHW) latent tensor
    
    diffusers VAE expects:
    - Input: [N, C, H, W] format (NCHW)
    - Output: LatentDict with "latent_dist" or direct tensor
    """
    def __init__(self, vae):
        # Convert VAE to float32 for ComfyUI compatibility
        # ComfyUI's VAEEncode expects float32 input, so VAE should also be float32
        self.vae = vae.to(dtype=torch.float32)
    
    def encode(self, pixels):
        """
        Encode pixels to latent space.
        
        Args:
            pixels: Tensor in [N, H, W, C] format (ComfyUI format)
        
        Returns:
            Tensor in [N, C, H, W] format (latent space)
        """
        # Convert from [N, H, W, C] to [N, C, H, W]
        # pixels is already in [N, H, W, C] format from ComfyUI
        if len(pixels.shape) == 4:
            # [N, H, W, C] -> [N, C, H, W]
            pixels_nchw = pixels.permute(0, 3, 1, 2)
        else:
            pixels_nchw = pixels
        
        # Ensure pixels are in the correct dtype (float32)
        pixels_nchw = pixels_nchw.to(dtype=torch.float32)
        
        # Call diffusers VAE encode method
        # diffusers VAE.encode() returns a LatentDict or tensor
        result = self.vae.encode(pixels_nchw)
        
        # Extract latent tensor from result
        if isinstance(result, dict):
            # Some VAEs return a dict with "latent_dist" or "latents"
            if "latent_dist" in result:
                # Sample from the distribution
                latent = result["latent_dist"].sample()
            elif "latents" in result:
                latent = result["latents"]
            else:
                # Try to get the first tensor value
                latent = next(iter(result.values()))
        else:
            latent = result
        
        # Return in [N, C, H, W] format (ComfyUI expects this)
        return latent
    
    def decode(self, samples):
        """
        Decode latent to pixels.
        
        Args:
            samples: Tensor in [N, C, H, W] format (latent space)
        
        Returns:
            Tensor in [N, H, W, C] format (ComfyUI format)
        """
        # Call diffusers VAE decode method
        result = self.vae.decode(samples)
        
        # Extract pixels from result
        if isinstance(result, dict):
            if "sample" in result:
                pixels = result["sample"]
            else:
                pixels = next(iter(result.values()))
        else:
            pixels = result
        
        # Convert from [N, C, H, W] to [N, H, W, C]
        if len(pixels.shape) == 4:
            # [N, C, H, W] -> [N, H, W, C]
            pixels_nhwc = pixels.permute(0, 2, 3, 1)
        else:
            pixels_nhwc = pixels
        
        # Normalize to [0, 1] range if needed
        pixels_nhwc = pixels_nhwc.clamp(0, 1)
        
        return pixels_nhwc


class SDNQModelLoader:
    """
    SDNQ model loader that handles downloading and loading quantized models.
    
    Outputs a DiffusionPipeline (MODEL) that can be used by sampler nodes.
    """

    def __init__(self):
        """Initialize loader with empty pipeline cache."""
        self.pipeline = None
        self.current_model_path = None
        self.current_dtype = None
        self.current_memory_mode = None
        self.current_use_xformers = None
        self.current_use_flash_attention = None
        self.current_use_sage_attention = None
        self.current_enable_vae_tiling = None
        self.current_matmul_precision = None

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        # Get model names from catalog
        model_names = get_model_names_for_dropdown()

        return {
            "required": {
                "model_selection": (["[Custom Path]"] + model_names, {
                    "default": model_names[0] if model_names else "[Custom Path]",
                    "tooltip": "Select a pre-configured SDNQ model (auto-downloads from HuggingFace) or choose [Custom Path] to specify a local model directory"
                }),

                "custom_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Local path to SDNQ model directory (only used when [Custom Path] is selected). Example: /path/to/model or C:\\path\\to\\model"
                }),

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
    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")

    # V1 API: Function name
    FUNCTION = "load_model"

    # Category for node menu
    CATEGORY = "loaders/SDNQ"

    # V3 API: Output node (can save/display results)
    OUTPUT_NODE = False

    # V3 API: Node description
    DESCRIPTION = "Load SDNQ quantized models with 50-75% VRAM savings. Supports FLUX, SD3, SDXL, video models, and more."

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

            print(f"[SDNQ Model Loader] Using custom model path: {model_path}")
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
        print(f"[SDNQ Model Loader] Selected model: {model_selection}")
        print(f"[SDNQ Model Loader] Repository: {repo_id}")

        # Check if model already cached
        cached_path = get_cached_model_path(repo_id)
        if cached_path:
            print(f"[SDNQ Model Loader] Found cached model at: {cached_path}")
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

        print(f"[SDNQ Model Loader] Model not cached - downloading from HuggingFace...")
        print(f"[SDNQ Model Loader] This may take a while (models are 5-20+ GB)")

        try:
            downloaded_path = download_model(repo_id)
            print(f"[SDNQ Model Loader] Download complete: {downloaded_path}")
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
                     matmul_precision: str = "int8", repo_id: Optional[str] = None):
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
            print(f"[SDNQ Model Loader] ⚠️  {opt_list} does not support float32 (only float16/bfloat16)")
            print(f"[SDNQ Model Loader] ⚠️  Automatically converting dtype from float32 to bfloat16")
            dtype_str = "bfloat16"
            torch_dtype = torch.bfloat16

        print(f"[SDNQ Model Loader] Loading model from: {model_path}")
        print(f"[SDNQ Model Loader] Using dtype: {dtype_str} ({torch_dtype})")
        if original_dtype_str != dtype_str:
            print(f"[SDNQ Model Loader] (Original dtype was {original_dtype_str}, changed for attention optimization compatibility)")
        print(f"[SDNQ Model Loader] Memory mode: {memory_mode}")

        try:
            # Load pipeline - DiffusionPipeline auto-detects model type
            # SDNQ quantization is automatically detected from model config
            # Note: Pipeline loads to CPU by default - we move to GPU below

            # Suppress torch_dtype deprecation warning from transformers components
            with warnings.catch_warnings():
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
                    print(f"[SDNQ Model Loader] Local files incomplete, downloading missing files...")
                    # Use repo_id if available (allows HF Hub to download missing files)
                    # Otherwise fall back to model_path with local_files_only=False
                    if repo_id:
                        print(f"[SDNQ Model Loader] Using repository ID: {repo_id}")
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

            print(f"[SDNQ Model Loader] Model loaded successfully!")
            print(f"[SDNQ Model Loader] Pipeline type: {type(pipeline).__name__}")

            # Ensure VAE is float32 to avoid dtype mismatch during decode (bfloat16 input vs float bias)
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                try:
                    pipeline.vae = pipeline.vae.to(dtype=torch.float32)
                    print("[SDNQ Model Loader] ✓ VAE converted to float32 to match bias dtype")
                except Exception as e:
                    print(f"[SDNQ Model Loader] ⚠️  VAE float32 conversion failed: {e}")

            # Apply SDNQ optimizations (Quantized MatMul)
            # This must be done BEFORE memory management moves things around
            use_quantized_matmul = matmul_precision != "none"
            if use_quantized_matmul:
                if triton_is_available and torch.cuda.is_available():
                    print(f"[SDNQ Model Loader] Applying Triton Quantized MatMul optimizations (precision: {matmul_precision})...")
                    try:
                        # Apply to transformer (FLUX, SD3)
                        if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                            try:
                                pipeline.transformer = apply_sdnq_options_to_model(
                                    pipeline.transformer,
                                    use_quantized_matmul=True,
                                    matmul_precision=matmul_precision
                                )
                            except TypeError:
                                pipeline.transformer = apply_sdnq_options_to_model(
                                    pipeline.transformer,
                                    use_quantized_matmul=True
                                )
                            print("[SDNQ Model Loader] ✓ Optimization applied to transformer")

                        # Apply to UNet (SDXL, SD1.5)
                        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
                            try:
                                pipeline.unet = apply_sdnq_options_to_model(
                                    pipeline.unet,
                                    use_quantized_matmul=True,
                                    matmul_precision=matmul_precision
                                )
                            except TypeError:
                                pipeline.unet = apply_sdnq_options_to_model(
                                    pipeline.unet,
                                    use_quantized_matmul=True
                                )
                            print("[SDNQ Model Loader] ✓ Optimization applied to UNet")

                        # Apply to text encoders (if they are quantized, e.g. FLUX.2)
                        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                            try:
                                pipeline.text_encoder = apply_sdnq_options_to_model(
                                    pipeline.text_encoder,
                                    use_quantized_matmul=True,
                                    matmul_precision=matmul_precision
                                )
                            except TypeError:
                                pipeline.text_encoder = apply_sdnq_options_to_model(
                                    pipeline.text_encoder,
                                    use_quantized_matmul=True
                                )
                            print("[SDNQ Model Loader] ✓ Optimization applied to text_encoder")

                        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                            pipeline.text_encoder_2 = apply_sdnq_options_to_model(
                                pipeline.text_encoder_2,
                                use_quantized_matmul=True,
                                matmul_precision=matmul_precision
                            )
                            print("[SDNQ Model Loader] ✓ Optimization applied to text_encoder_2")

                    except Exception as e:
                        print(f"[SDNQ Model Loader] ⚠️  Failed to apply optimizations: {e}")
                        print("[SDNQ Model Loader] Continuing without optimizations...")
                else:
                    if not torch.cuda.is_available():
                        print("[SDNQ Model Loader] ℹ️  Quantized MatMul requires CUDA. Optimization disabled.")
                    elif not triton_is_available:
                        print("[SDNQ Model Loader] ℹ️  Triton not available/supported (requires Linux/WSL). Quantized MatMul disabled.")
            else:
                 print("[SDNQ Model Loader] Quantized MatMul optimization disabled.")

            # CRITICAL: Apply xFormers BEFORE memory management
            # xFormers must be enabled before CPU offloading is set up
            if use_xformers:
                try:
                    print(f"[SDNQ Model Loader] Enabling xFormers memory-efficient attention...")
                    print(f"[SDNQ Model Loader] Current dtype: {dtype_str} (xFormers requires float16/bfloat16)")
                    
                    # Double-check dtype compatibility
                    if torch_dtype == torch.float32:
                        print(f"[SDNQ Model Loader] ⚠️  CRITICAL: dtype is still float32, xFormers will fail!")
                        print(f"[SDNQ Model Loader] ⚠️  This should have been converted earlier - check dtype conversion logic")
                        raise ValueError("xFormers requires float16 or bfloat16, but dtype is float32")
                    
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("[SDNQ Model Loader] ✓ xFormers memory-efficient attention enabled")

                except ModuleNotFoundError as e:
                    # xFormers package not installed
                    print(f"[SDNQ Model Loader] ⚠️  xFormers not installed: {e}")
                    print("[SDNQ Model Loader] Install with: pip install xformers")
                    print("[SDNQ Model Loader] Falling back to SDPA (PyTorch 2.0+ default attention)")

                except ValueError as e:
                    # CUDA not available or dtype incompatibility
                    error_msg = str(e)
                    if "float32" in error_msg or "dtype" in error_msg.lower():
                        print(f"[SDNQ Model Loader] ⚠️  xFormers dtype incompatibility: {e}")
                        print("[SDNQ Model Loader] xFormers only supports float16 and bfloat16, not float32")
                        print("[SDNQ Model Loader] Please change dtype to bfloat16 or float16, or disable xFormers")
                    else:
                        print(f"[SDNQ Model Loader] ⚠️  xFormers requires CUDA: {e}")
                    print("[SDNQ Model Loader] Falling back to SDPA")

                except NotImplementedError as e:
                    # Model architecture doesn't support xFormers
                    print(f"[SDNQ Model Loader] ℹ️  xFormers not supported for this model architecture")
                    print(f"[SDNQ Model Loader] Details: {e}")
                    print("[SDNQ Model Loader] Using SDPA instead (this is normal for some models)")

                except (RuntimeError, AttributeError) as e:
                    # Version incompatibility, dimension mismatch, or API changes
                    error_msg = str(e)
                    print(f"[SDNQ Model Loader] ⚠️  xFormers compatibility issue: {type(e).__name__}")
                    print(f"[SDNQ Model Loader] Error: {e}")
                    
                    # Check for dtype-related errors
                    if "float32" in error_msg or "dtype" in error_msg.lower() or "not supported" in error_msg.lower():
                        print("[SDNQ Model Loader] This error is likely due to:")
                        print("[SDNQ Model Loader]   - dtype=float32 (xFormers requires float16/bfloat16)")
                        print("[SDNQ Model Loader]   - GPU capability mismatch (newer GPUs may not be fully supported)")
                        print("[SDNQ Model Loader]   - xFormers version incompatibility with your GPU")
                    else:
                        print("[SDNQ Model Loader] This may indicate:")
                        print("[SDNQ Model Loader]   - xFormers version mismatch with PyTorch/CUDA")
                        print("[SDNQ Model Loader]   - GPU architecture incompatibility")
                        print("[SDNQ Model Loader]   - Tensor dimension issues with this model")
                    
                    print("[SDNQ Model Loader] Try:")
                    print("[SDNQ Model Loader]   1. Change dtype to bfloat16 or float16")
                    print("[SDNQ Model Loader]   2. pip install -U xformers --force-reinstall")
                    print("[SDNQ Model Loader]   3. Disable xFormers and use SDPA instead")
                    print("[SDNQ Model Loader] Falling back to SDPA")

                except Exception as e:
                    # Unexpected error - log full details for debugging
                    error_msg = str(e)
                    print(f"[SDNQ Model Loader] ⚠️  Unexpected xFormers error: {type(e).__name__}")
                    print(f"[SDNQ Model Loader] Error message: {e}")
                    
                    # Check for dtype-related errors
                    if "float32" in error_msg or "dtype" in error_msg.lower() or "not supported" in error_msg.lower():
                        print("[SDNQ Model Loader] This error is likely due to dtype=float32 or GPU capability mismatch")
                        print("[SDNQ Model Loader] xFormers requires float16/bfloat16 and compatible GPU architecture")
                    
                    print("[SDNQ Model Loader] Full traceback:")
                    traceback.print_exc()
                    print("[SDNQ Model Loader] Falling back to SDPA")
            else:
                print("[SDNQ Model Loader] Using SDPA (scaled dot product attention, default in PyTorch 2.0+)")

            # Flash Attention (FA) - requires ComfyUI started with --use-flash-attention
            if use_flash_attention:
                try:
                    print(f"[SDNQ Model Loader] Enabling Flash Attention (FA)...")
                    print(f"[SDNQ Model Loader] Current dtype: {dtype_str} (FA requires float16/bfloat16)")
                    
                    # Double-check dtype compatibility
                    if torch_dtype == torch.float32:
                        print(f"[SDNQ Model Loader] ⚠️  CRITICAL: dtype is still float32, FA will fail!")
                        print(f"[SDNQ Model Loader] ⚠️  This should have been converted earlier - check dtype conversion logic")
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
                        print(f"[SDNQ Model Loader] ⚠️  WARNING: FLUX.2 uses Flux2ParallelSelfAttention")
                        print(f"[SDNQ Model Loader] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible with FA/SA")
                        print(f"[SDNQ Model Loader] ⚠️  Reason: Parallel QKV decomposition ≠ standard Q, K, V structure")
                        print(f"[SDNQ Model Loader] ⚠️  FA requires: Q/K/V together, contiguous, 1-shot, no CFG branching")
                        print(f"[SDNQ Model Loader] ⚠️  Flux2 does: QKV split, head/block parallel, CFG double pass, LoRA branching")
                        print(f"[SDNQ Model Loader] ⚠️  xFormers is just dispatch layer - cannot fix architectural mismatch")
                        print(f"[SDNQ Model Loader] ⚠️  FLUX.2 uses optimized parallel attention (different optimization path)")
                    if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                        # FLUX.2 models: Try to enable Flash Attention via transformer methods
                        if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                            # Try enable_xformers_memory_efficient_attention first
                            try:
                                if hasattr(pipeline.transformer, 'enable_xformers_memory_efficient_attention'):
                                    pipeline.transformer.enable_xformers_memory_efficient_attention()
                                    print("[SDNQ Model Loader] ✓ Flash Attention enabled via transformer.enable_xformers_memory_efficient_attention()")
                                    flash_enabled = True
                                elif hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                                    pipeline.enable_xformers_memory_efficient_attention()
                                    print("[SDNQ Model Loader] ✓ Flash Attention enabled via pipeline.enable_xformers_memory_efficient_attention()")
                                    flash_enabled = True
                            except Exception as e:
                                print(f"[SDNQ Model Loader] ℹ️  xformers Flash Attention not available: {e}")
                            
                            # Try enable_attn_processor if xformers didn't work
                            if not flash_enabled:
                                try:
                                    # Check if transformer has enable_attn_processor method
                                    if hasattr(pipeline.transformer, 'enable_attn_processor'):
                                        # Try to enable Flash Attention 2 processor
                                        try:
                                            from diffusers.models.attention_processor import AttnProcessor2_0
                                            pipeline.transformer.enable_attn_processor(AttnProcessor2_0())
                                            print("[SDNQ Model Loader] ✓ Flash Attention enabled via transformer.enable_attn_processor(AttnProcessor2_0)")
                                            flash_enabled = True
                                        except Exception as e:
                                            print(f"[SDNQ Model Loader] ℹ️  AttnProcessor2_0 not compatible with FLUX.2: {e}")
                                except Exception as e:
                                    print(f"[SDNQ Model Loader] ℹ️  Could not enable Flash Attention via enable_attn_processor: {e}")
                            
                            # If still not enabled, FLUX.2 uses default Flux2AttnProcessor
                            if not flash_enabled:
                                print("[SDNQ Model Loader] ℹ️  FLUX.2 uses Flux2AttnProcessor (default)")
                                print("[SDNQ Model Loader] ℹ️  Flash Attention may be handled internally by the processor")
                                print("[SDNQ Model Loader] ℹ️  Check if ComfyUI --use-flash-attention flag enables it at system level")
                    else:
                        # For other models, use standard AttnProcessor2_0
                        if hasattr(pipeline, 'set_attn_processor') or hasattr(pipeline, 'enable_attn_processor'):
                            try:
                                from diffusers.models.attention_processor import AttnProcessor2_0
                                if hasattr(pipeline, 'set_attn_processor'):
                                    pipeline.set_attn_processor(AttnProcessor2_0())
                                else:
                                    pipeline.enable_attn_processor(AttnProcessor2_0())
                                print("[SDNQ Model Loader] ✓ Flash Attention 2 enabled via AttnProcessor2_0")
                                flash_enabled = True
                            except (ImportError, AttributeError) as e:
                                # Fallback to xformers if AttnProcessor2_0 not available
                                try:
                                    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                                        pipeline.enable_xformers_memory_efficient_attention()
                                        print("[SDNQ Model Loader] ✓ Flash Attention enabled via xformers")
                                        flash_enabled = True
                                    else:
                                        print(f"[SDNQ Model Loader] ℹ️  AttnProcessor2_0 not available: {e}")
                                except Exception as e2:
                                    print(f"[SDNQ Model Loader] ⚠️  Flash Attention via xformers failed: {e2}")
                    
                    # Check if already enabled via ComfyUI system level
                    if not flash_enabled:
                        # ComfyUI with --use-flash-attention enables it at model loading time
                        # Check if we can detect it via model attributes
                        try:
                            import comfy.model_management as model_management
                            # Flash attention is typically enabled globally in ComfyUI
                            print("[SDNQ Model Loader] ℹ️  Flash Attention should be enabled at system level")
                            print("[SDNQ Model Loader] ℹ️  (ComfyUI started with --use-flash-attention flag)")
                            
                            # Check transformer for flash attention processors
                            if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                                if hasattr(pipeline.transformer, 'attn_processors'):
                                    processors = pipeline.transformer.attn_processors
                                    if processors:
                                        print("[SDNQ Model Loader] ℹ️  Checking transformer attention processors...")
                                        
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
                                                print(f"[SDNQ Model Loader] ⚠️  FLUX.2 processor types: {list(set(processor_types.values()))}")
                                                print("[SDNQ Model Loader] ⚠️  FLUX.2 uses Flux2ParallelSelfAttention (may NOT support standard FA)")
                                                print("[SDNQ Model Loader] ⚠️  Even with --use-flash-attention flag, FA may not actually work")
                                                print("[SDNQ Model Loader] ⚠️  Flux2ParallelSelfAttention uses its own attention implementation")
                                                print("[SDNQ Model Loader] ⚠️  If generation is slow, FA is likely NOT working")
                                                # Do NOT assume enabled - FLUX.2 may not support FA
                                                flash_enabled = False
                                            except ImportError:
                                                print(f"[SDNQ Model Loader] ⚠️  Flash Attention not detected. Processor types: {list(set(processor_types.values()))}")
                                                print("[SDNQ Model Loader] ⚠️  Make sure ComfyUI is started with --use-flash-attention flag")
                                        elif found_flash:
                                            print(f"[SDNQ Model Loader] ✓ Flash Attention detected in processors: {list(set(processor_types.values()))}")
                                            flash_enabled = True
                                        else:
                                            print(f"[SDNQ Model Loader] ⚠️  Flash Attention not detected. Processor types: {list(set(processor_types.values()))}")
                                            print("[SDNQ Model Loader] ⚠️  Make sure ComfyUI is started with --use-flash-attention flag")
                                    else:
                                        print("[SDNQ Model Loader] ⚠️  No attention processors found on transformer")
                                else:
                                    print("[SDNQ Model Loader] ⚠️  Transformer does not have attn_processors attribute")
                            else:
                                print("[SDNQ Model Loader] ⚠️  Pipeline does not have transformer attribute")
                                
                        except ImportError:
                            print("[SDNQ Model Loader] ⚠️  Cannot verify Flash Attention status")
                            print("[SDNQ Model Loader] Make sure ComfyUI is started with --use-flash-attention flag")
                        except Exception as e:
                            print(f"[SDNQ Model Loader] ⚠️  Error checking Flash Attention: {e}")
                    
                    # Final status report with detailed information
                    if flash_enabled:
                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                            print("[SDNQ Model Loader] ⚠️  FLUX.2 Flash Attention: NOT SUPPORTED")
                            print("[SDNQ Model Loader] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible")
                            print("[SDNQ Model Loader] ⚠️  Uses parallel QKV decomposition (not standard attention structure)")
                            print("[SDNQ Model Loader] ⚠️  FlashAttention2 requires standard Q, K, V (incompatible)")
                            print("[SDNQ Model Loader] ⚠️  FLUX.2 uses optimized parallel attention (different optimization)")
                        else:
                            print("[SDNQ Model Loader] ✓ Flash Attention is ACTIVE")
                            print("[SDNQ Model Loader] ℹ️  If generation is still slow, FA may not be working correctly")
                            print("[SDNQ Model Loader] ℹ️  Check ComfyUI startup logs for 'Flash Attention ✅' message")
                            print("[SDNQ Model Loader] ℹ️  Verify dtype is bfloat16 or float16 (not float32)")
                    else:
                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                            print("[SDNQ Model Loader] ⚠️  Flash Attention: NOT SUPPORTED for FLUX.2")
                            print("[SDNQ Model Loader] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible")
                            print("[SDNQ Model Loader] ⚠️  Parallel QKV decomposition ≠ standard Q, K, V structure")
                            print("[SDNQ Model Loader] ⚠️  FlashAttention2/SageAttention2.2 cannot be used")
                            print("[SDNQ Model Loader] ⚠️  This is expected - FLUX.2 uses different optimization path")
                        else:
                            print("[SDNQ Model Loader] ⚠️  Flash Attention is NOT active")
                            print("[SDNQ Model Loader] ⚠️  Possible reasons:")
                            print("[SDNQ Model Loader]   1. ComfyUI not started with --use-flash-attention flag")
                            print("[SDNQ Model Loader]   2. flash-attn package not installed (pip install flash-attn)")
                            print("[SDNQ Model Loader]   3. dtype is float32 (FA requires float16/bfloat16)")
                            print("[SDNQ Model Loader]   4. GPU not compatible with Flash Attention")
                            print("[SDNQ Model Loader] ⚠️  Check startup logs for 'Flash Attention ✅' message")
                            print("[SDNQ Model Loader] ⚠️  Generation will use default attention (slower)")
                        
                except ValueError as e:
                    # Dtype incompatibility
                    print(f"[SDNQ Model Loader] ⚠️  Flash Attention dtype error: {e}")
                    print("[SDNQ Model Loader] ⚠️  Flash Attention requires float16 or bfloat16, not float32")
                    print("[SDNQ Model Loader] ⚠️  Please change dtype to bfloat16 or float16")
                except Exception as e:
                    print(f"[SDNQ Model Loader] ⚠️  Failed to enable Flash Attention: {e}")
                    print("[SDNQ Model Loader] ⚠️  Make sure ComfyUI is started with --use-flash-attention flag")
                    print("[SDNQ Model Loader] ⚠️  Check that flash-attn package is installed: pip install flash-attn")

            # Sage Attention (SA) - requires ComfyUI started with --use-sage-attention
            if use_sage_attention:
                try:
                    print(f"[SDNQ Model Loader] Enabling Sage Attention (SA)...")
                    print(f"[SDNQ Model Loader] Current dtype: {dtype_str} (SA requires float16/bfloat16)")
                    
                    # Double-check dtype compatibility
                    if torch_dtype == torch.float32:
                        print(f"[SDNQ Model Loader] ⚠️  CRITICAL: dtype is still float32, SA will fail!")
                        print(f"[SDNQ Model Loader] ⚠️  This should have been converted earlier - check dtype conversion logic")
                        raise ValueError("Sage Attention requires float16 or bfloat16, but dtype is float32")
                    
                    # CRITICAL: FLUX.2's Flux2ParallelSelfAttention uses a different architecture
                    # Same explanation as Flash Attention - architectural incompatibility
                    pipeline_type = type(pipeline).__name__
                    if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                        print(f"[SDNQ Model Loader] ⚠️  WARNING: FLUX.2 uses Flux2ParallelSelfAttention")
                        print(f"[SDNQ Model Loader] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible with SA")
                        print(f"[SDNQ Model Loader] ⚠️  Same reason as Flash Attention: parallel QKV decomposition")
                        print(f"[SDNQ Model Loader] ⚠️  SageAttention2.2 requires standard Q, K, V structure (incompatible)")
                        print(f"[SDNQ Model Loader] ⚠️  FLUX.2 uses optimized parallel attention (different optimization path)")
                    
                    # Sage Attention is typically enabled at model initialization
                    # Check if it's available via ComfyUI's system-level settings
                    sage_enabled = False
                    
                    # Try to enable on transformer (FLUX models)
                    if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                        if hasattr(pipeline.transformer, 'enable_sage_attention'):
                            pipeline.transformer.enable_sage_attention()
                            print("[SDNQ Model Loader] ✓ Sage Attention enabled on transformer")
                            sage_enabled = True
                        # Check if already enabled by checking attention modules
                        elif hasattr(pipeline.transformer, 'attn_processors'):
                            # Check if sage attention is already active
                            processors = pipeline.transformer.attn_processors
                            if processors:
                                print("[SDNQ Model Loader] ℹ️  Checking transformer attention processors...")
                                # Sage attention might be set at system level
                                sage_enabled = True
                    
                    # Try to enable on pipeline level
                    if not sage_enabled and hasattr(pipeline, 'enable_sage_attention'):
                        pipeline.enable_sage_attention()
                        print("[SDNQ Model Loader] ✓ Sage Attention enabled on pipeline")
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
                                    print(f"[SDNQ Model Loader] ✓ Sage Attention package available")
                                    print("[SDNQ Model Loader] ✓ Sage Attention is ACTIVE for FLUX.2 (enabled via --use-sage-attention flag)")
                                    sage_enabled = True
                                else:
                                    # For other models, check if already enabled
                                    print(f"[SDNQ Model Loader] ✓ Sage Attention package available")
                                    print("[SDNQ Model Loader] ℹ️  Sage Attention may be active if --use-sage-attention flag is set")
                                    sage_enabled = True  # Assume enabled if package is available and flag is set
                            else:
                                print("[SDNQ Model Loader] ⚠️  Sage Attention package not available")
                                print("[SDNQ Model Loader] ⚠️  Install with: pip install sageattention")
                                
                        except ImportError as e:
                            print(f"[SDNQ Model Loader] ⚠️  Cannot check Sage Attention availability: {e}")
                        except Exception as e:
                            print(f"[SDNQ Model Loader] ⚠️  Error checking Sage Attention: {e}")
                    
                    # Final status report with detailed information
                    if sage_enabled:
                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                            print("[SDNQ Model Loader] ⚠️  FLUX.2 Sage Attention: NOT SUPPORTED")
                            print("[SDNQ Model Loader] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible")
                            print("[SDNQ Model Loader] ⚠️  Uses parallel QKV decomposition (not standard attention structure)")
                            print("[SDNQ Model Loader] ⚠️  SageAttention2.2 requires standard Q, K, V (incompatible)")
                            print("[SDNQ Model Loader] ⚠️  FLUX.2 uses optimized parallel attention (different optimization)")
                        else:
                            print("[SDNQ Model Loader] ✓ Sage Attention is ACTIVE")
                            print("[SDNQ Model Loader] ℹ️  If generation is still slow, SA may not be working correctly")
                            print("[SDNQ Model Loader] ℹ️  Check ComfyUI startup logs for 'Sage Attention ✅' message")
                            print("[SDNQ Model Loader] ℹ️  Verify dtype is bfloat16 or float16 (not float32)")
                    else:
                        if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                            print("[SDNQ Model Loader] ⚠️  Sage Attention: NOT SUPPORTED for FLUX.2")
                            print("[SDNQ Model Loader] ⚠️  Flux2ParallelSelfAttnProcessor architecture is incompatible")
                            print("[SDNQ Model Loader] ⚠️  Parallel QKV decomposition ≠ standard Q, K, V structure")
                            print("[SDNQ Model Loader] ⚠️  FlashAttention2/SageAttention2.2 cannot be used")
                            print("[SDNQ Model Loader] ⚠️  This is expected - FLUX.2 uses different optimization path")
                        else:
                            print("[SDNQ Model Loader] ⚠️  Sage Attention is NOT active")
                            print("[SDNQ Model Loader] ⚠️  Possible reasons:")
                            print("[SDNQ Model Loader]   1. ComfyUI not started with --use-sage-attention flag")
                            print("[SDNQ Model Loader]   2. sageattention package not installed (pip install sageattention)")
                            print("[SDNQ Model Loader]   3. dtype is float32 (SA requires float16/bfloat16)")
                            print("[SDNQ Model Loader]   4. GPU not compatible with Sage Attention")
                            print("[SDNQ Model Loader] ⚠️  Check startup logs for 'Sage Attention ✅' message")
                            print("[SDNQ Model Loader] ⚠️  Generation will use default attention (slower)")
                    
                except ValueError as e:
                    # Dtype incompatibility
                    print(f"[SDNQ Model Loader] ⚠️  Sage Attention dtype error: {e}")
                    print("[SDNQ Model Loader] ⚠️  Sage Attention requires float16 or bfloat16, not float32")
                    print("[SDNQ Model Loader] ⚠️  Please change dtype to bfloat16 or float16")
                except Exception as e:
                    print(f"[SDNQ Model Loader] ⚠️  Failed to enable Sage Attention: {e}")
                    print("[SDNQ Model Loader] ⚠️  Make sure ComfyUI is started with --use-sage-attention flag")
                    print("[SDNQ Model Loader] ⚠️  Check that sageattention package is installed: pip install sageattention")

            # Apply memory management strategy
            # Based on: https://huggingface.co/docs/diffusers/main/optimization/memory
            if memory_mode == "gpu":
                # Full GPU mode: Fastest performance, needs 24GB+ VRAM
                # Load entire pipeline to GPU
                print(f"[SDNQ Model Loader] Moving model to GPU (full GPU mode)...")
                pipeline.to("cuda")
                print(f"[SDNQ Model Loader] ✓ Model loaded to GPU (all components on VRAM)")

            elif memory_mode == "balanced":
                # Sequential CPU offload: Prevents VRAM growth during generation
                # All components start on CPU, moved to GPU only when needed during generation
                # This prevents VRAM from growing during generation process
                print(f"[SDNQ Model Loader] Enabling sequential CPU offload (balanced mode)...")
                print(f"[SDNQ Model Loader] All components will be offloaded to CPU before generation")
                print(f"[SDNQ Model Loader] Components will be moved to GPU only when needed during generation")
                print(f"[SDNQ Model Loader] This prevents VRAM growth during generation process")
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
                        
                        print(f"[SDNQ Model Loader] VRAM: {vram_total_gb:.2f}GB total, {vram_used_gb:.2f}GB used before offload")
                    
                    # Enable sequential CPU offload
                    # This automatically manages component placement:
                    # - Components start on CPU
                    # - Moved to GPU only when needed during generation
                    # - Automatically moved back to CPU when not needed
                    # DO NOT manually move components with .to("cpu") after this!
                    pipeline.enable_sequential_cpu_offload()
                    print("[SDNQ Model Loader] ✓ Sequential CPU offload enabled")
                    print("[SDNQ Model Loader] ✓ Components will be automatically managed by diffusers")
                    print("[SDNQ Model Loader] ✓ This prevents VRAM growth during generation process")
                        
                except Exception as e:
                    # Fallback to standard offload if sequential fails
                    print(f"[SDNQ Model Loader] ⚠️  Sequential offload failed, using standard offload: {e}")
                    pipeline.enable_model_cpu_offload()
                    print("[SDNQ Model Loader] ✓ Model offloading enabled (standard mode)")

            elif memory_mode == "lowvram":
                # Sequential CPU offload: Maximum memory savings for 8GB VRAM
                # Slowest but works on limited VRAM
                print(f"[SDNQ Model Loader] Enabling sequential CPU offload (low VRAM mode)...")
                pipeline.enable_sequential_cpu_offload()
                print(f"[SDNQ Model Loader] ✓ Sequential offloading enabled (minimal VRAM usage)")

            # VAE tiling (works with all memory modes, but not all pipelines support it)
            if enable_vae_tiling:
                try:
                    # Check if pipeline supports VAE tiling
                    # FLUX.2 and some other pipelines don't have this method
                    if hasattr(pipeline, 'enable_vae_tiling'):
                        pipeline.enable_vae_tiling()
                        print("[SDNQ Model Loader] ✓ VAE tiling enabled")
                    else:
                        pipeline_type = type(pipeline).__name__
                        print(f"[SDNQ Model Loader] ℹ️  VAE tiling not supported by {pipeline_type} pipeline")
                except Exception as e:
                    print(f"[SDNQ Model Loader] ⚠️  VAE tiling failed: {e}")

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

    def load_model(self, model_selection: str, custom_model_path: str, dtype: str, memory_mode: str,
                   auto_download: bool = True, matmul_precision: str = "int8",
                   use_xformers: bool = False, use_flash_attention: bool = False,
                   use_sage_attention: bool = False, enable_vae_tiling: bool = False):
        """
        Main function to load model - called by ComfyUI.
        
        Returns:
            Tuple containing (pipeline, vae) for MODEL and VAE outputs
        """
        # Check if we can reuse cached pipeline
        model_path, was_downloaded, repo_id = self.load_or_download_model(
            model_selection, custom_model_path, auto_download
        )

        # Check if pipeline can be reused
        if (self.pipeline is not None and
            self.current_model_path == model_path and
            self.current_dtype == dtype and
            self.current_memory_mode == memory_mode and
            self.current_use_xformers == use_xformers and
            self.current_use_flash_attention == use_flash_attention and
            self.current_use_sage_attention == use_sage_attention and
            self.current_enable_vae_tiling == enable_vae_tiling and
            self.current_matmul_precision == matmul_precision):
            print("[SDNQ Model Loader] Reusing cached pipeline")
            # Extract VAE from pipeline and wrap for ComfyUI compatibility
            vae = self.pipeline.vae if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None else None
            if vae is not None:
                # Wrap VAE to handle ComfyUI's [N, H, W, C] input format
                vae = ComfyVAEWrapper(vae)
            return (self.pipeline, vae)

        # Load new pipeline
        self.pipeline = self.load_pipeline(
            model_path, dtype, memory_mode,
            use_xformers, use_flash_attention, use_sage_attention,
            enable_vae_tiling, matmul_precision, repo_id
        )

        # Update cache
        self.current_model_path = model_path
        self.current_dtype = dtype
        self.current_memory_mode = memory_mode
        self.current_use_xformers = use_xformers
        self.current_use_flash_attention = use_flash_attention
        self.current_use_sage_attention = use_sage_attention
        self.current_enable_vae_tiling = enable_vae_tiling
        self.current_matmul_precision = matmul_precision

        # Extract VAE from pipeline and wrap for ComfyUI compatibility
        vae = self.pipeline.vae if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None else None
        if vae is None:
            print("[SDNQ Model Loader] ⚠️  Warning: Pipeline does not have VAE component")
        else:
            # Wrap VAE to handle ComfyUI's [N, H, W, C] input format
            vae = ComfyVAEWrapper(vae)
            print("[SDNQ Model Loader] ✓ VAE wrapped for ComfyUI compatibility")
        
        return (self.pipeline, vae)

