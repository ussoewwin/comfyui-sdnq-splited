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
from typing import Tuple

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

# Local imports for LoRA folder access (if needed)


class SDNQSamplerV2:
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

                "latent_image": ("LATENT", {
                    "tooltip": "Latent image input. Width and height are extracted from this latent tensor."
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
            },
            "optional": {}
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
    DESCRIPTION = "Generate images using SDNQ models. Connect SDNQ Model Loader to provide the model."

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
        print(f"[SDNQ Sampler V2] Swapping scheduler to: {scheduler_name}")

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

            print(f"[SDNQ Sampler V2] ✓ Scheduler swapped successfully")

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
                      steps: int, cfg: float, latent_image: dict, seed: int) -> Image.Image:
        """
        Generate image using the loaded pipeline.

        Args:
            pipeline: Loaded diffusers pipeline
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (optional)
            steps: Number of inference steps
            cfg: Guidance scale (classifier-free guidance strength)
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

        # Calculate image dimensions from latent (8x downsampling for FLUX/SDXL models)
        width = latent_width * 8
        height = latent_height * 8
        
        print(f"[SDNQ Sampler V2] Generating image...")
        print(f"[SDNQ Sampler V2]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"[SDNQ Sampler V2]   Steps: {steps}, CFG: {cfg}")
        print(f"[SDNQ Sampler V2]   Latent size: {latent_width}x{latent_height} -> Image size: {width}x{height}")
        print(f"[SDNQ Sampler V2]   Seed: {seed}")

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
                
                print(f"[SDNQ Sampler V2] VRAM before generation: {vram_used_before_gb:.2f}GB used, {vram_free_before_gb:.2f}GB free")
            
            # Create generator for reproducible generation
            generator = torch.Generator(device="cuda").manual_seed(seed)

            # Build pipeline call kwargs
            pipeline_kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "width": width,
                "height": height,
                "generator": generator,
            }

            # Check if pipeline supports negative_prompt
            pipeline_type = type(pipeline).__name__
            supports_negative_prompt = pipeline_type not in ["Flux2Pipeline", "FluxPipeline", "FluxSchnellPipeline"]
            
            if supports_negative_prompt and negative_prompt and negative_prompt.strip():
                pipeline_kwargs["negative_prompt"] = negative_prompt
            elif negative_prompt and negative_prompt.strip() and not supports_negative_prompt:
                print(f"[SDNQ Sampler V2] ⚠️  Pipeline {pipeline_type} doesn't support negative_prompt - skipping it")

            # Try calling pipeline
            try:
                result = pipeline(**pipeline_kwargs)
            except TypeError as e:
                if "negative_prompt" in str(e) and "unexpected keyword argument" in str(e):
                    print(f"[SDNQ Sampler V2] ⚠️  Pipeline {type(pipeline).__name__} doesn't support negative_prompt - skipping it")
                    if "negative_prompt" in pipeline_kwargs:
                        del pipeline_kwargs["negative_prompt"]
                    result = pipeline(**pipeline_kwargs)
                else:
                    raise

            # Check for interruption after generation
            if self.check_interrupted():
                raise InterruptedError("Generation interrupted by user")

            # Extract first image from results
            image = result.images[0]

            print(f"[SDNQ Sampler V2] Image generated! Size: {image.size}")

            return image

        except InterruptedError:
            raise
        except torch.cuda.OutOfMemoryError as e:
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

        print(f"[SDNQ Sampler V2] Converted to ComfyUI tensor: shape={tensor.shape}, dtype={tensor.dtype}")

        return tensor

    def generate(self, model, prompt: str,
                negative_prompt: str, steps: int, cfg: float, latent_image: dict,
                seed: int, scheduler: str) -> Tuple[torch.Tensor]:
        """
        Main generation function called by ComfyUI.

        This is the entry point when the node executes in a workflow.

        Args:
            model: DiffusionPipeline from SDNQModelLoader (or SDNQLoraLoader if LoRA is needed)
            prompt: Text prompt
            negative_prompt: Negative prompt
            steps: Inference steps
            cfg: Guidance scale
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
        print(f"[SDNQ Sampler V2] Starting generation")
        print(f"{'='*60}\n")

        self.interrupted = False

        try:
            # Use provided pipeline from MODEL input
            pipeline = model

            # Handle scheduler swap
            if scheduler != self.current_scheduler:
                print(f"[SDNQ Sampler V2] Scheduler changed - swapping to {scheduler}...")
                self.swap_scheduler(pipeline, scheduler)
                self.current_scheduler = scheduler

            # Generate image
            pil_image = self.generate_image(
                pipeline,
                prompt,
                negative_prompt,
                steps,
                cfg,
                latent_image,
                seed,
            )

            # Convert to ComfyUI format
            comfy_tensor = self.pil_to_comfy_tensor(pil_image)

            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler V2] Generation complete!")
            print(f"{'='*60}\n")

            return (comfy_tensor,)

        except InterruptedError as e:
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler V2] Generation interrupted")
            print(f"{'='*60}\n")
            raise

        except (ValueError, FileNotFoundError) as e:
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler V2] Error: {str(e)}")
            print(f"{'='*60}\n")
            raise

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler V2] Unexpected error occurred")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            raise
