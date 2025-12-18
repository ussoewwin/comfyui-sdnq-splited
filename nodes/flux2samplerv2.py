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
from typing import Tuple, Optional
import secrets
import inspect

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
# Flux2 pipelines ONLY support FlowMatchEulerDiscreteScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# Local imports for LoRA folder access (if needed)


class Flux2SDNQSamplerV2:
    """
    Flux2-optimized SDNQ sampler that loads quantized models and generates images.

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
        """Initialize sampler."""
        self.current_scheduler = None
        self.current_lora_path = None
        self.current_lora_strength = None
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
        # Scheduler options
        # Flux2 pipelines ONLY support FlowMatchEulerDiscreteScheduler
        scheduler_list = [
            "FlowMatchEulerDiscreteScheduler",
        ]

        return {
            "required": {
                # ============================================================
                # GROUP 1: MODEL INPUT (from SDNQModelLoader)
                # ============================================================

                "model": ("MODEL", {
                    "tooltip": "SDNQ model pipeline from SDNQ Model Loader node"
                }),

                # ============================================================
                # GROUP 2: GENERATION PROMPTS (What to create)
                # ============================================================

                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text description of the image to generate. Be descriptive for best results."
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

                "latent_image": ("LATENT", {
                    "tooltip": "Latent image input. Width and height are extracted from this latent tensor."
                }),

                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "The random seed used for creating the noise. Seed=0 will randomize each run. Seed>=1 will use the specified value."
                }),

                "scheduler": (scheduler_list, {
                    "default": "FlowMatchEulerDiscreteScheduler",
                    "tooltip": "FlowMatchEulerDiscreteScheduler is the only scheduler supported for Flux2 pipelines."
                }),

                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising strength. 1.0 = full denoising, lower values = less denoising. ComfyUI standard parameter matching KSampler."
                }),
            },
            "optional": {}
        }

    # V3 API: Return type hints
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    # V1 API: Function name
    FUNCTION = "generate"

    # Category for node menu
    CATEGORY = "sampling/SDNQ/Flux2"

    # V3 API: Output node (can save/display results)
    OUTPUT_NODE = False

    # V3 API: Node description
    DESCRIPTION = "Generate images using SDNQ Flux2 models. Optimized for Flux2Pipeline. Connect SDNQ Model Loader to provide the model."

    def check_interrupted(self):
        """Check if generation should be interrupted (ComfyUI interrupt support)."""
        # ComfyUI provides comfy.model_management.interrupt_processing()
        # For now, we'll use a simple flag that can be extended
        return self.interrupted

    def swap_scheduler(self, pipeline, scheduler_name: str):
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
        print(f"[Flux2 SDNQ Sampler V2] Swapping scheduler to: {scheduler_name}")

        try:
            # Map scheduler name to class
            # Flux2 pipelines ONLY support FlowMatchEulerDiscreteScheduler
            scheduler_map = {
                "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
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

            print(f"[Flux2 SDNQ Sampler V2] ✓ Scheduler swapped successfully")

        except Exception as e:
            raise Exception(
                f"Failed to swap scheduler\n\n"
                f"Error: {str(e)}\n\n"
                f"Requested scheduler: {scheduler_name}\n\n"
                f"Troubleshooting:\n"
                f"1. Flux2 pipelines ONLY support FlowMatchEulerDiscreteScheduler\n"
                f"2. Ensure you are using FlowMatchEulerDiscreteScheduler\n"
                f"3. Wrong scheduler type will produce broken/corrupted images\n"
                f"4. Check diffusers version (requires >=0.36.0)"
            )

    def generate_image(self, pipeline: DiffusionPipeline, prompt: str,
                      steps: int, cfg: float, denoise: float, latent_image: dict, seed: int) -> Image.Image:
        """
        Generate image using the loaded pipeline.

        Args:
            pipeline: Loaded diffusers pipeline
            prompt: Text prompt for generation
            steps: Number of inference steps
            cfg: Guidance scale (classifier-free guidance strength)
            denoise: Denoising strength (0.0-1.0)
            latent_image: Latent image dict with "samples" tensor
            seed: Random seed for reproducibility

        Returns:
            PIL Image object

        Raises:
            Exception: If generation fails or is interrupted

        Based on verified API from FLUX examples:
        https://huggingface.co/docs/diffusers/main/api/pipelines/flux
        """
        # Extract width and height from latent image
        if "samples" not in latent_image:
            raise ValueError("latent_image must contain 'samples' key")

        pipeline_type = type(pipeline).__name__
        is_flux_family = pipeline_type in ["Flux2Pipeline", "FluxPipeline", "FluxSchnellPipeline"]
        call_params = set()
        try:
            call_params = set(inspect.signature(pipeline.__call__).parameters.keys())
        except Exception:
            call_params = set()
        supports_image_arg = ("image" in call_params)

        # For non-FLUX pipelines, if VAE is provided via latent, apply it to pipeline to keep encode/decode consistent.
        if (not is_flux_family) and isinstance(latent_image, dict) and latent_image.get("vae") is not None:
            try:
                pipeline.vae = latent_image["vae"]
            except Exception:
                pass

        samples = latent_image["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError("latent_image['samples'] must be a torch.Tensor")

        # Latent shape: [batch, channels, height, width]
        # For diffusers, we need actual image dimensions
        # FLUX models typically use 8x downsampling, so multiply by 8
        if len(samples.shape) != 4:
            raise ValueError(f"latent_image['samples'] must have shape [batch, channels, height, width], got {samples.shape}")

        batch, channels, latent_height, latent_width = samples.shape
        if latent_width <= 0 or latent_height <= 0:
            raise ValueError(f"Invalid latent dimensions: {latent_width}x{latent_height}")

        # Calculate image dimensions from latent
        # Most models (FLUX, SDXL, SD1.5) use 8x downsampling, but check pipeline's vae_scale_factor if available
        vae_scale_factor = getattr(pipeline, "vae_scale_factor", None)
        if vae_scale_factor is None and hasattr(pipeline, "vae") and hasattr(pipeline.vae, "scale_factor"):
            vae_scale_factor = pipeline.vae.scale_factor
        if vae_scale_factor is None:
            # Default to 8x for most models (FLUX, SDXL, SD1.5)
            vae_scale_factor = 8
        else:
            vae_scale_factor = int(vae_scale_factor)
        
        width = latent_width * vae_scale_factor
        height = latent_height * vae_scale_factor
        
        print(f"[Flux2 SDNQ Sampler V2] Generating image...")
        print(f"[Flux2 SDNQ Sampler V2]   Pipeline type: {pipeline_type}")
        print(f"[Flux2 SDNQ Sampler V2]   Is Flux family: {is_flux_family}")
        print(f"[Flux2 SDNQ Sampler V2]   Supports image arg: {supports_image_arg}")
        print(f"[Flux2 SDNQ Sampler V2]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"[Flux2 SDNQ Sampler V2]   Steps: {steps}, CFG: {cfg}, Denoise: {denoise}")
        print(f"[Flux2 SDNQ Sampler V2]   Latent size: {latent_width}x{latent_height} -> Image size: {width}x{height} (scale_factor={vae_scale_factor})")
        print(f"[Flux2 SDNQ Sampler V2]   Seed: {seed}")

        # Check for interruption before starting
        if self.check_interrupted():
            raise InterruptedError("Generation interrupted by user")

        try:
            # Clear VRAM cache before generation to maximize available memory
            if torch.cuda.is_available():
                import gc
                for _ in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                vram_free_before, _ = torch.cuda.mem_get_info()
                device_props = torch.cuda.get_device_properties(0)
                vram_total = device_props.total_memory
                vram_used_before = vram_total - vram_free_before
                vram_used_before_gb = vram_used_before / (1024**3)
                vram_free_before_gb = vram_free_before / (1024**3)
                
                print(f"[Flux2 SDNQ Sampler V2] VRAM before generation: {vram_used_before_gb:.2f}GB used, {vram_free_before_gb:.2f}GB free")
            
            # Create generator for reproducible generation (match pipeline execution device when possible)
            generator_device = getattr(pipeline, "_execution_device", None)
            if generator_device is None and hasattr(pipeline, "device"):
                generator_device = pipeline.device
            if generator_device is None:
                generator_device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            
            # Build pipeline call kwargs
            pipeline_kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "width": width,
                "height": height,
                "generator": generator,
            }

            # i2i handling
            # - FLUX family pipelines: use `image=` conditioning (PIL). For Flux2 img2img we also initialize `latents`
            #   from the input image and use `sigmas` to control denoise strength.
            # - Other pipelines: pass `latents=` and `strength=` when available.
            strength = None
            try:
                strength = float(denoise)
                if strength < 0.0:
                    strength = 0.0
                elif strength > 1.0:
                    strength = 1.0
            except Exception:
                strength = None

            def _tensor_to_pil_rgb(t: torch.Tensor) -> Image.Image:
                # Expect ComfyUI IMAGE tensor: [N,H,W,C] float 0..1 (or -1..1), C>=3
                t0 = t
                if t0.dim() == 4:
                    t0 = t0[0]
                if t0.dim() != 3:
                    raise ValueError(f"unexpected tensor shape for image: {tuple(t.shape)}")
                # NHWC
                if t0.shape[-1] >= 3:
                    x = t0[:, :, :3]
                # NCHW -> HWC
                elif t0.shape[0] >= 3:
                    x = t0[:3, :, :].permute(1, 2, 0)
                else:
                    raise ValueError(f"unexpected channel layout for image: {tuple(t0.shape)}")
                x = x.detach().cpu().to(torch.float32)
                # Normalize if looks like [-1,1]
                try:
                    mn = float(x.min().item())
                    mx = float(x.max().item())
                    if mn < 0.0 and mx <= 1.0:
                        x = (x + 1.0) / 2.0
                except Exception:
                    pass
                x = torch.clamp(x, 0.0, 1.0).numpy()
                x = (x * 255.0).round().astype(np.uint8)
                return Image.fromarray(x, mode="RGB")

            def _decode_latents_to_pil(latents: torch.Tensor) -> Optional[Image.Image]:
                # Prefer latent-provided VAE; fallback to pipeline.vae
                vae_obj = None
                if isinstance(latent_image, dict) and latent_image.get("vae") is not None:
                    vae_obj = latent_image.get("vae")
                elif hasattr(pipeline, "vae") and pipeline.vae is not None:
                    vae_obj = pipeline.vae
                if vae_obj is None or not hasattr(vae_obj, "decode"):
                    return None
                try:
                    decoded = vae_obj.decode(latents)
                except Exception:
                    # Some wrappers keep the actual VAE at .vae
                    try:
                        if hasattr(vae_obj, "vae") and hasattr(vae_obj.vae, "decode"):
                            decoded = vae_obj.vae.decode(latents)
                        else:
                            return None
                    except Exception:
                        return None

                # Handle common return shapes/containers
                if isinstance(decoded, dict):
                    for k in ("pixels", "images", "image", "samples"):
                        if k in decoded:
                            decoded = decoded[k]
                            break
                    else:
                        decoded = next(iter(decoded.values()))
                if hasattr(decoded, "sample"):
                    try:
                        decoded = decoded.sample
                    except Exception:
                        pass
                if isinstance(decoded, (list, tuple)) and len(decoded) > 0:
                    decoded = decoded[0]
                if not isinstance(decoded, torch.Tensor):
                    return None
                return _tensor_to_pil_rgb(decoded)

            # i2i: if pipeline supports `image`, prefer that (more reliable than `latents` across diffusers).
            # - For Flux family, `image` is effectively required for i2i.
            # - For non-Flux img2img pipelines, `image` + `strength` is standard.
            pil_cond = None
            if isinstance(latent_image, dict) and latent_image.get("pixels") is not None:
                try:
                    px = latent_image["pixels"]
                    if isinstance(px, torch.Tensor):
                        pil_cond = _tensor_to_pil_rgb(px)
                except Exception as e:
                    print(f"[Flux2 SDNQ Sampler V2] Warning: failed to convert pixels to PIL: {e}")
            if pil_cond is None and isinstance(latent_image, dict) and latent_image.get("vae") is not None:
                try:
                    pil_cond = _decode_latents_to_pil(samples)
                except Exception as e:
                    print(f"[Flux2 SDNQ Sampler V2] Warning: failed to decode latents to PIL: {e}")

            # For non-Flux pipelines: prefer `image` + `strength` over `latents` for i2i
            # For Flux pipelines: will handle `image` and `latents` separately below
            if supports_image_arg and pil_cond is not None and (("vae" in latent_image) or (latent_image.get("pixels") is not None)):
                # For non-Flux: use `image` + `strength` (standard img2img API)
                if not is_flux_family:
                    print(f"[Flux2 SDNQ Sampler V2]   i2i mode: Non-Flux pipeline - using `image` + `strength`")
                    pipeline_kwargs["image"] = pil_cond
                    # When conditioning on an init image, keep width/height consistent with that image.
                    try:
                        pipeline_kwargs["width"] = int(pil_cond.size[0])
                        pipeline_kwargs["height"] = int(pil_cond.size[1])
                    except Exception:
                        pass
                # For Flux: set `image` first, will be conditionally removed if latents are initialized
                else:
                    print(f"[Flux2 SDNQ Sampler V2]   i2i mode: Flux pipeline - setting `image`, will initialize latents below")
                    pipeline_kwargs["image"] = pil_cond
                    # When conditioning on an init image, keep width/height consistent with that image.
                    # Some pipelines (notably Flux) can behave incorrectly if width/height disagree with the provided image.
                    try:
                        pipeline_kwargs["width"] = int(pil_cond.size[0])
                        pipeline_kwargs["height"] = int(pil_cond.size[1])
                    except Exception:
                        pass
                # Flux2 img2img: start from the encoded image latents (not pure noise), then add noise according to denoise.
                # This is required to avoid "complete noise" results where the init image isn't actually used as a starting point.
                flux_latent_init_ok = False
                if is_flux_family and pipeline_type in ["Flux2Pipeline", "FluxPipeline"] and strength is not None:
                    print(f"[Flux2 SDNQ Sampler V2]   Flux i2i: Initializing latents from input image (strength={strength})")
                    try:
                        # Build a sigma schedule that keeps the user-requested step count stable.
                        # In flow-matching, sigma directly controls the mixing ratio:
                        #   x_t = sigma * noise + (1 - sigma) * x0
                        req_steps = int(steps)
                        if req_steps < 1:
                            req_steps = 1
                        # Use 0.0 terminal sigma so low denoise can truly stay close to the init image.
                        # (Using 1/steps creates a "noise floor" that can make denoise feel inverted at low step counts.)
                        sigma_end = 0.0
                        sigma_start = float(strength)
                        if sigma_start < sigma_end:
                            sigma_start = sigma_end
                        if sigma_start > 1.0:
                            sigma_start = 1.0
                        sigmas = np.linspace(sigma_start, sigma_end, req_steps, dtype=np.float32).tolist()
                        pipeline_kwargs["sigmas"] = sigmas

                        # Prepare image tensor exactly like the pipeline does
                        # Note: For i2i mode, preserve input image size (do not resize to 1024x1024)
                        img_w, img_h = pil_cond.size

                        multiple_of = int(getattr(pipeline, "vae_scale_factor", 16)) * 2
                        if multiple_of > 0:
                            img_w = (img_w // multiple_of) * multiple_of
                            img_h = (img_h // multiple_of) * multiple_of
                        if img_w <= 0 or img_h <= 0:
                            img_w, img_h = pil_cond.size

                        image_tensor = pipeline.image_processor.preprocess(
                            pil_cond, height=img_h, width=img_w, resize_mode="crop"
                        )
                        image_tensor = image_tensor.to(device=generator_device, dtype=pipeline.vae.dtype)

                        # Encode to Flux2 latent space (unpacked shape [B, 128, H', W'])
                        x0 = pipeline._encode_vae_image(image=image_tensor, generator=generator)

                        # Set scheduler timesteps once so we can add noise at the correct starting sigma
                        from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps, compute_empirical_mu

                        # image_seq_len equals packed latent sequence length (H' * W')
                        token_h = max(1, int(img_h // multiple_of))
                        token_w = max(1, int(img_w // multiple_of))
                        image_seq_len = int(token_h * token_w)
                        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=req_steps)

                        timesteps, _ = retrieve_timesteps(
                            pipeline.scheduler, req_steps, generator_device, sigmas=sigmas, mu=mu
                        )
                        t0 = timesteps[0].expand(x0.shape[0]).to(device=x0.device)
                        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
                        x_t = pipeline.scheduler.scale_noise(sample=x0, timestep=t0, noise=noise)

                        # Pass img2img starting latents (pipeline will pack internally)
                        pipeline_kwargs["latents"] = x_t
                        flux_latent_init_ok = True
                        print(f"[Flux2 SDNQ Sampler V2]   Flux i2i: Latents initialized successfully, will remove `image` arg")
                    except Exception as e:
                        print(f"[Flux2 SDNQ Sampler V2] Warning: Flux img2img latent init failed, falling back to conditioning-only: {e}")
                # IMPORTANT:
                # If we successfully initialize `latents` from the input image, do NOT also pass `image=`.
                # Flux2 treats `image` as additional reference conditioning tokens; keeping it makes denoise appear
                # "stuck" (0.2 and 0.8 look similar) because the reference conditioning dominates.
                if flux_latent_init_ok:
                    pipeline_kwargs.pop("image", None)
                    print(f"[Flux2 SDNQ Sampler V2]   Flux i2i: Removed `image` arg (using latents only)")
            
            # For non-Flux pipelines: add `strength` parameter for i2i
            # Note: `image` was already added above if `supports_image_arg` and `pil_cond` are available
            if (not is_flux_family) and strength is not None and ("strength" in call_params):
                print(f"[Flux2 SDNQ Sampler V2]   Non-Flux i2i: Adding `strength`={strength}")
                pipeline_kwargs["strength"] = strength
            
            # Fallback path for non-Flux pipelines that don't support `image` but accept `latents` (rare)
            # Only use this if `image` was not already set above
            if (not is_flux_family) and ("image" not in pipeline_kwargs) and isinstance(latent_image, dict) and latent_image.get("vae") is not None:
                print(f"[Flux2 SDNQ Sampler V2]   Non-Flux i2i: Fallback path - using `latents` directly (image arg not supported)")
                try:
                    init_latents = samples
                    target_dtype = None
                    for attr in ("unet", "transformer"):
                        m = getattr(pipeline, attr, None)
                        if m is not None and hasattr(m, "dtype"):
                            target_dtype = m.dtype
                            break
                    if target_dtype is not None and init_latents.dtype != target_dtype:
                        init_latents = init_latents.to(dtype=target_dtype)
                    init_latents = init_latents.to(device=generator_device)
                    if "latents" in call_params:
                        pipeline_kwargs["latents"] = init_latents
                    if strength is not None and ("strength" in call_params):
                        pipeline_kwargs["strength"] = strength
                except Exception as e:
                    print(f"[Flux2 SDNQ Sampler V2] Warning: failed to set i2i latents/strength: {e}")

            # Note: Flux2 pipelines do not support negative_prompt, so it is not included in pipeline_kwargs

            # Patch VAE.decode to enforce float32 input (avoid bfloat16/float bias mismatch in FLUX VAE)
            # Note: This patch is applied to all pipelines, but is primarily needed for Flux
            original_vae_decode = None
            if hasattr(pipeline, "vae") and pipeline.vae is not None and hasattr(pipeline.vae, "decode"):
                try:
                    original_vae_decode = pipeline.vae.decode
                    def patched_decode(z, *args, **kwargs):
                        return original_vae_decode(z.float(), *args, **kwargs)
                    pipeline.vae.decode = patched_decode
                    if is_flux_family:
                        print("[Flux2 SDNQ Sampler V2] Patched VAE.decode to force float32 input (Flux pipeline)")
                    else:
                        print("[Flux2 SDNQ Sampler V2] Patched VAE.decode to force float32 input (non-Flux pipeline)")
                except Exception as e:
                    print(f"[Flux2 SDNQ Sampler V2] Warning: Could not patch VAE.decode: {e}")

            # Patch retrieve_timesteps for FLUX pipelines:
            # - remove `mu` when scheduler doesn't support it
            # (denoise for Flux img2img is handled via `sigmas` + `latents` initialization above)
            original_retrieve_timesteps = None
            if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
                try:
                    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps
                    import diffusers.pipelines.flux2.pipeline_flux2 as flux2_module
                    
                    # Check if current scheduler supports mu parameter
                    scheduler_supports_mu = isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)

                    # Save original function
                    original_retrieve_timesteps = retrieve_timesteps

                    def patched_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, **kwargs):
                        if not scheduler_supports_mu:
                            kwargs.pop("mu", None)
                        return original_retrieve_timesteps(
                            scheduler, num_inference_steps, device, timesteps=timesteps, **kwargs
                        )

                    flux2_module.retrieve_timesteps = patched_retrieve_timesteps
                    if not scheduler_supports_mu:
                        print(f"[Flux2 SDNQ Sampler V2] Patched retrieve_timesteps to remove mu parameter for {type(pipeline.scheduler).__name__}")
                    # (Flux denoise handling is done via `sigmas` + `latents` initialization above.)
                except Exception as e:
                    print(f"[Flux2 SDNQ Sampler V2] Warning: Could not patch retrieve_timesteps: {e}")

            # Try calling pipeline
            try:
                try:
                    result = pipeline(**pipeline_kwargs)
                except TypeError as e:
                    # Some pipelines may not accept latents/strength; retry without them.
                    if "latents" in str(e) and "unexpected keyword argument" in str(e):
                        if "latents" in pipeline_kwargs:
                            del pipeline_kwargs["latents"]
                        if "strength" in pipeline_kwargs:
                            del pipeline_kwargs["strength"]
                        result = pipeline(**pipeline_kwargs)
                    elif "strength" in str(e) and "unexpected keyword argument" in str(e):
                        if "strength" in pipeline_kwargs:
                            del pipeline_kwargs["strength"]
                        result = pipeline(**pipeline_kwargs)
                    elif ("width" in str(e) or "height" in str(e)) and "unexpected keyword argument" in str(e):
                        # Some img2img pipelines infer size from `image` and don't accept width/height.
                        pipeline_kwargs.pop("width", None)
                        pipeline_kwargs.pop("height", None)
                        result = pipeline(**pipeline_kwargs)
                    else:
                        raise

                # Check for interruption after generation
                if self.check_interrupted():
                    raise InterruptedError("Generation interrupted by user")

                # Extract first image from results
                image = result.images[0]

                print(f"[Flux2 SDNQ Sampler V2] Image generated! Size: {image.size}")

                return image
            finally:
                # Restore original retrieve_timesteps if patched
                if original_retrieve_timesteps is not None:
                    try:
                        import diffusers.pipelines.flux2.pipeline_flux2 as flux2_module
                        flux2_module.retrieve_timesteps = original_retrieve_timesteps
                        print(f"[Flux2 SDNQ Sampler V2] Restored original retrieve_timesteps")
                    except Exception as e:
                        print(f"[Flux2 SDNQ Sampler V2] Warning: Could not restore retrieve_timesteps: {e}")
                # Restore VAE.decode if patched
                if original_vae_decode is not None:
                    try:
                        pipeline.vae.decode = original_vae_decode
                        print("[Flux2 SDNQ Sampler V2] Restored original VAE.decode")
                    except Exception as e:
                        print(f"[Flux2 SDNQ Sampler V2] Warning: Could not restore VAE.decode: {e}")

        except InterruptedError:
            raise
        except torch.cuda.OutOfMemoryError as e:
            raise Exception(
                f"VRAM out of memory error occurred\n\n"
                f"Error: {str(e)}\n\n"
                f"Solutions:\n"
                f"1. Change memory mode to 'lowvram' (sequential CPU offload)\n"
                f"2. Reduce image size (e.g., 1024x1024 → 768x768)\n"
                f"3. Enable VAE tiling (set enable_vae_tiling to true)\n"
                f"4. Close other applications to free VRAM\n"
                f"5. Use a smaller model\n"
                f"6. Reduce steps (current: {steps})\n"
                f"7. Recommended: Change memory mode to 'lowvram'"
            )
        except Exception as e:
            if "Pipeline doesn't support parameter" in str(e):
                raise

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

        print(f"[Flux2 SDNQ Sampler V2] Converted to ComfyUI tensor: shape={tensor.shape}, dtype={tensor.dtype}")

        return tensor

    def generate(self, model, prompt: str,
                steps: int, cfg: float, denoise: float, latent_image: dict,
                seed: int, scheduler: str) -> Tuple[torch.Tensor]:
        """
        Main generation function called by ComfyUI.

        This is the entry point when the node executes in a workflow.

        Args:
            model: DiffusionPipeline from SDNQModelLoader (or SDNQLoraLoader if LoRA is needed)
            prompt: Text prompt
            steps: Inference steps
            cfg: Guidance scale
            denoise: Denoising strength (0.0-1.0)
            latent_image: Latent image dict with "samples" tensor
            seed: Random seed
            scheduler: Scheduler algorithm name

        Returns:
            Tuple containing (IMAGE,) in ComfyUI format

        Raises:
            ValueError: For invalid inputs
            Exception: For other errors during generation
        """
        print(f"\n{'='*60}")
        print(f"[Flux2 SDNQ Sampler V2] Starting generation")
        print(f"{'='*60}\n")

        self.interrupted = False

        try:
            # Use provided pipeline from MODEL input
            pipeline = model

            # Handle scheduler swap
            if scheduler != self.current_scheduler:
                print(f"[Flux2 SDNQ Sampler V2] Scheduler changed - swapping to {scheduler}...")
                self.swap_scheduler(pipeline, scheduler)
                self.current_scheduler = scheduler

            # Seed handling:
            # - seed==0: randomize every run
            # - seed>=1: use the specified seed value
            in_seed = int(seed)
            if in_seed == 0:
                run_seed = secrets.randbits(64)
                if run_seed == 0:
                    run_seed = 1
                print(f"[Flux2 SDNQ Sampler V2] Seed mode: AUTO(0)  input=0  run={run_seed}")
            else:
                run_seed = in_seed
                print(f"[Flux2 SDNQ Sampler V2] Seed mode: FIXED  input={in_seed}  run={run_seed}")

            # Generate image
            pil_image = self.generate_image(
                pipeline,
                prompt,
                steps,
                cfg,
                denoise,
                latent_image,
                run_seed,
            )

            # Convert to ComfyUI format
            comfy_tensor = self.pil_to_comfy_tensor(pil_image)

            print(f"\n{'='*60}")
            print(f"[Flux2 SDNQ Sampler V2] Generation complete!")
            print(f"{'='*60}\n")

            # ComfyUI 0.4+ prefers dict return with "result" and optional "ui".
            # IMPORTANT: 
            # - seed==0: keep widget value at 0 so user can keep getting random seeds (don't update UI)
            # - seed>=1: keep the user-specified value (don't update UI to avoid changing what user set)
            return {"result": (comfy_tensor,)}

        except InterruptedError:
            print(f"\n{'='*60}")
            print(f"[Flux2 SDNQ Sampler V2] Generation interrupted")
            print(f"{'='*60}\n")
            raise

        except (ValueError, FileNotFoundError) as e:
            print(f"\n{'='*60}")
            print(f"[Flux2 SDNQ Sampler V2] Error: {str(e)}")
            print(f"{'='*60}\n")
            raise

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[Flux2 SDNQ Sampler V2] Unexpected error occurred")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            raise
