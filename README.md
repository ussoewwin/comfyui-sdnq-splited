# ComfyUI-SDNQ-Splited

> **This repository is a fork of [EnragedAntelope/comfyui-sdnq](https://github.com/EnragedAntelope/comfyui-sdnq)**

## Acknowledgments

We would like to express our deepest gratitude to **EnragedAntelope**, the creator of the original [comfyui-sdnq](https://github.com/EnragedAntelope/comfyui-sdnq) repository. This modular node structure would not have been possible without their foundational work. In particular, we are especially grateful for their development of the dedicated scheduler implementation, which has been instrumental in enabling this fork's split-node architecture. Thank you for your excellent work and for making this project possible.

---

**Load and run SDNQ quantized models in ComfyUI with 50-75% VRAM savings!**

This custom node pack enables running [SDNQ (SD.Next Quantization)](https://github.com/Disty0/sdnq) models directly in ComfyUI. Run large models like FLUX.2, FLUX.1, SD3.5, and more on consumer hardware with significantly reduced VRAM requirements while maintaining quality.

> **SDNQ is developed by [Disty0](https://github.com/Disty0)** - this node pack provides ComfyUI integration.

## Modular Node Structure

This fork provides a **modular node structure with split functionality**. The following nodes are implemented:

- **SDNQ Model Loader**: Dedicated node for loading models
- **SDNQ LoRA Loader**: Dedicated node for loading LoRAs
- **SDNQ VAE Encode**: Dedicated node for encoding images to latent space (compatible with diffusers VAE)
- **SDNQ Sampler V2**: Dedicated node for image generation (general models)
- **Flux2 SDNQ Sampler V2**: Dedicated node for image generation (Flux2-optimized)

This allows you to use SDNQ models with the same workflow structure as standard ComfyUI workflows (Model Load → LoRA Apply → Sampling).

---

## Features

- **🔀 Modular Node Structure**: Functionality split into separate nodes (Model Loader, LoRA Loader, Sampler) - compatible with standard ComfyUI workflows
- **📦 Model Catalog**: 30+ pre-configured SDNQ models with auto-download (note: at the moment, development is focused on FLUX.2 compatibility)
- **💾 Smart Caching**: Download once, use forever
- **🚀 VRAM Savings**: 50-75% memory reduction with quantization
- **⚡ Performance Optimizations**: Optional xFormers, Flash Attention (FA), Sage Attention (SA), VAE tiling, SDPA (automatic)
- **🎯 LoRA Support**: Load LoRAs from ComfyUI loras folder via dedicated loader node
- **📅 Multi-Scheduler**: 14 schedulers (FLUX/SD3 flow-match + traditional diffusion)
- **🔧 Memory Modes**: GPU (fastest), balanced (12-16GB VRAM), lowvram (8GB VRAM)

---

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ussoewwin/comfyui-sdnq-splited.git
cd comfyui-sdnq-splited
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## Quick Start

### Using Split Nodes (Recommended - Modular Workflow)

1. Add **SDNQ Model Loader** node (under `loaders/SDNQ`)
2. Add **SDNQ LoRA Loader** node (optional, under `loaders/SDNQ`)
3. Add **SDNQ VAE Encode** node (under `latent/SDNQ`) for image-to-image workflows (optional)
4. Add **SDNQ Sampler V2** node (under `sampling/SDNQ`) or **Flux2 SDNQ Sampler V2** node (under `sampling/SDNQ/Flux2`) for Flux2 models
5. Connect Model Loader → LoRA Loader → (VAE Encode) → Sampler
6. Select model from dropdown (auto-downloads on first use)
7. Enter your prompt and click Queue Prompt

---

## Node Reference

### SDNQ Model Loader

**Category**: `loaders/SDNQ`

**Main Parameters**:
- `model_selection`: Dropdown with 30+ pre-configured SDNQ models
- `custom_model_path`: For local models or custom HuggingFace repos
- `memory_mode`:
  - `gpu` = Full GPU (fastest, 24GB+ VRAM required)
  - `balanced` = CPU offloading (12-16GB VRAM)
  - `lowvram` = Sequential offloading (8GB VRAM, slowest)
- `dtype`: bfloat16 (recommended), float16, or float32

**Outputs**: `MODEL` (connects to SDNQ Sampler V2 or other SDNQ nodes)

---

### SDNQ LoRA Loader

**Category**: `loaders/SDNQ`

**Main Parameters**:
- `lora_selection`: Dropdown from ComfyUI loras folder
- `lora_custom_path`: Custom LoRA path or HuggingFace repo
- `lora_strength`: -5.0 to +5.0 (1.0 = full strength)
- `model`: Input from SDNQ Model Loader

**Outputs**: `MODEL` (connects to SDNQ Sampler V2)

---

### SDNQ Sampler V2

**Category**: `sampling/SDNQ`

**Main Parameters**:
- `model`: Input from SDNQ Model Loader or SDNQ LoRA Loader
- `prompt` / `negative_prompt`: What to create / what to avoid
- `steps`, `cfg`, `width`, `height`, `seed`: Standard generation controls
- `scheduler`: FlowMatchEulerDiscreteScheduler (FLUX/SD3) or traditional samplers

**Performance Optimizations** (optional):
- `use_xformers`: Memory-efficient attention (safe to try, auto-fallback to SDPA)
- `use_flash_attention`: Flash Attention (FA) for faster inference and lower VRAM
- `use_sage_attention`: Sage Attention (SA) for optimized attention computation
- `enable_vae_tiling`: For large images >1536px (prevents OOM)
- SDPA (Scaled Dot Product Attention): Always active - automatic PyTorch 2.0+ optimization

**Outputs**: `IMAGE` (connects to SaveImage, Preview, etc.)

---

### Flux2 SDNQ Sampler V2

**Category**: `sampling/SDNQ/Flux2`

**Purpose**: Flux2 models (FLUX.2-dev, FLUX.1-dev, etc.) with specialized optimizations for Flow Matching architecture.

**Main Parameters**:
- `model`: Input from SDNQ Model Loader or SDNQ LoRA Loader (must be Flux2 pipeline)
- `prompt`: Text prompt for generation
- `steps`, `cfg`, `seed`: Standard generation controls
- `latent_image`: Latent input from SDNQ VAE Encode (supports i2i workflows)
- `denoise`: Denoising strength (0.0-1.0) - controls initial noise level via sigma schedule
- `scheduler`: FlowMatchEulerDiscreteScheduler (only supported scheduler for Flux2)

**Flux2-Specific Optimizations**:
- **Flow Matching Support**: Specialized implementation for Flux2's Flow Matching architecture
- **Advanced i2i Processing**: 
  - Initializes latents from input image using `pipeline._encode_vae_image()`
  - Uses sigma schedule (`sigmas`) to control denoise strength accurately
  - Properly handles `compute_empirical_mu` and `retrieve_timesteps` for Flux2
- **VAE Compatibility**: Patches VAE.decode to force float32 input (prevents dtype mismatches with Flux2 VAE)
- **Accurate Denoise Control**: Unlike standard samplers, maintains full step count while adjusting initial noise level via sigma schedule

**When to Use**:
- **Recommended** for all Flux2 models (FLUX.2-dev, FLUX.1-dev, FLUX.1-schnell, etc.)
- Provides better i2i (image-to-image) results with Flux2 compared to generic SDNQ Sampler V2
- More accurate denoise control for Flux2's Flow Matching architecture

**Outputs**: `IMAGE` (connects to SaveImage, Preview, etc.)

**Note**: This node is specifically optimized for Flux2 pipelines. For other models (SDXL, SD1.5, Qwen, etc.), use **SDNQ Sampler V2** instead.

---

## Available Models

30+ pre-configured models including:
- **FLUX**: FLUX.1-dev, FLUX.1-schnell, FLUX.2-dev, FLUX.1-Krea, FLUX.1-Kontext
- **Qwen**: Qwen-Image variants (Edit, Lightning, Turbo)
- **SD3/SDXL**: SD3-Medium, SD3.5-Large, NoobAI-XL variants
- **Others**: Z-Image-Turbo, Chroma1-HD, HunyuanImage3, Video models

Most available in uint4 (max VRAM savings) or int8 (best quality). Browse: https://huggingface.co/collections/Disty0/sdnq

**Note**: While the models listed above are theoretically supported, **at the moment**, this repository is being developed specifically for FLUX.2. Other models have not been tested and their functionality is not guaranteed at this time. If you encounter issues with non-FLUX.2 models, please be aware that current development focus is on FLUX.2 compatibility.

---

## Performance Tips

**For All Memory Modes**:
- SDPA (Scaled Dot Product Attention) is always active - automatic PyTorch 2.0+ optimization
- Enable the xFormers option in the UI (safe to try)
- Enable the Flash Attention (FA) option in the UI - faster inference and lower VRAM
- Enable the Sage Attention (SA) option in the UI - optimized attention computation
- Use `enable_vae_tiling=True` for large images (>1536px) to prevent OOM

**Scheduler Selection**:
- FLUX/SD3/Qwen/Z-Image: Use `FlowMatchEulerDiscreteScheduler`
- SDXL/SD1.5: Use `DPMSolverMultistepScheduler`, `EulerDiscreteScheduler`, or `UniPCMultistepScheduler`
- Wrong scheduler = broken images!

---

## Model Storage

Downloaded models are stored in:
- **Location**: `ComfyUI/models/diffusers/sdnq/`
- **Format**: Standard diffusers format

Models are cached automatically - download once, use forever!

---

## Troubleshooting

### xFormers Not Working

If you see "xFormers not available" but have it installed:
- This is usually fine - the node automatically falls back to SDPA (PyTorch 2.0+ default)
- SDPA provides good performance without xFormers
- If xFormers is incompatible with your GPU/model, fallback is automatic

### Performance is Slow

**Balanced/lowvram modes**: Inherently slower due to CPU↔GPU data movement. Options:
- Enable the xFormers option in the UI (if compatible)
- Enable the Flash Attention (FA) option in the UI
- Enable the Sage Attention (SA) option in the UI
- SDPA is always active for automatic optimization
- Upgrade to more VRAM for full GPU mode
- Use smaller model (uint4 vs int8)

### Out of Memory

1. Use lower memory mode (gpu → balanced → lowvram)
2. Use more aggressive quantization (uint4 instead of int8)
3. Reduce resolution or batch size
4. Enable `enable_vae_tiling=True` for large images

### Model Loading Fails

1. Check internet connection (for auto-download)
2. Verify repo ID is correct for custom models
3. For local models, ensure path points to directory (not a file)
4. Check model is actually SDNQ-quantized (from Disty0's collection)

---

## Technical Information

### Standard KSampler vs SDNQSamplerV2: Complete Technical Explanation

This section provides a comprehensive technical explanation of the fundamental differences between the standard ComfyUI KSampler and SDNQSamplerV2 implementations, covering architecture, denoise control mechanisms, image-to-image processing, and Flux2-specific implementations.

#### 1. Overview and Fundamental Architectural Differences

##### 1.1 Standard KSampler Architecture

The standard ComfyUI KSampler directly manipulates ComfyUI's internal model representation (`ModelPatcher`, `CLIP`, `VAE`).

**Processing Flow:**
```
Input LATENT → VAE Decode → Image Tensor → Add Noise → VAE Encode → Diffusion Process → VAE Decode → Output IMAGE
```

**Characteristics:**
- Uses ComfyUI's internal APIs (`sampling_function`, `model_function`)
- Optimized for traditional diffusion models (SDXL, SD1.5)
- The `denoise` parameter functions as **step count reduction**
  - `effective_steps = steps * denoise`
  - Example: `steps=20, denoise=0.5` → processes with 10 steps

##### 1.2 SDNQSamplerV2 Architecture

SDNQSamplerV2 directly uses diffusers library pipelines.

**Processing Flow:**
```
Input LATENT → Pipeline Detection → Flux2/Non-Flux Branch → Pipeline-Specific Processing → diffusers.__call__() → Output IMAGE
```

**Characteristics:**
- Directly calls diffusers' `DiffusionPipeline`
- Supports both Flow Matching models (Flux2) and traditional diffusion models
- The `denoise` parameter interpretation **fundamentally differs** by pipeline type

#### 2. Denoise Strength Control Mechanism

##### 2.1 Standard KSampler Denoise Implementation

**Implementation Logic:**
```python
# Standard KSampler (conceptual implementation)
effective_steps = int(steps * denoise)
# denoise=0.5, steps=20 → effective_steps=10

# Only the first 10 steps are executed
# The remaining 10 steps are skipped
```

**Operation Principle:**
- `denoise` controls the ratio of processing steps
- `denoise=1.0`: Execute all steps (complete denoising)
- `denoise=0.5`: Execute only half the steps (closer to original image)
- Step reduction leaves denoising incomplete, preserving original image characteristics

**Constraints:**
- Quality may degrade with fewer steps
- Insufficient denoising at low `denoise` values

##### 2.2 SDNQSamplerV2 Denoise Implementation (Non-Flux Pipelines)

**Implementation Logic:**
```python
# SDNQSamplerV2 - Non-Flux pipeline (nodes/samplerv2.py:607-609)
if (not is_flux_family) and strength is not None and ("strength" in call_params):
    pipeline_kwargs["strength"] = strength  # strength = denoise
```

**Operation Principle:**
- Uses diffusers' standard `strength` parameter
- `strength` controls the initial noise level
  - `strength=1.0`: Start from complete noise (text-to-image equivalent)
  - `strength=0.5`: Start from intermediate noise level
  - `strength=0.2`: Start from state close to original image
- **Step count remains unchanged**; only the initial noise level is adjusted

**Differences from Standard KSampler:**
- Step count always remains `steps` (not reduced)
- Initial noise level adjustment enables smoother denoise control
- Less quality degradation

##### 2.3 SDNQSamplerV2 Denoise Implementation (Flux2 Pipeline)

**Implementation Logic:**
```python
# SDNQSamplerV2 - Flux2 pipeline (nodes/samplerv2.py:530-548)
if is_flux_family and pipeline_type in ["Flux2Pipeline", "FluxPipeline"] and strength is not None:
    # sigma_end = 0.0 fixed (important!)
    sigma_end = 0.0
    sigma_start = float(strength)  # Use denoise value directly
    
    # Generate sigma array
    sigmas = np.linspace(sigma_start, sigma_end, req_steps, dtype=np.float32).tolist()
    pipeline_kwargs["sigmas"] = sigmas
```

**Flow Matching Basic Principle:**

In Flow Matching, sigma (noise level) directly controls the mixing ratio:
```
x_t = sigma * noise + (1 - sigma) * x0
```

- `sigma=1.0`: Complete noise (`x_t = noise`)
- `sigma=0.5`: 50:50 mix of noise and original image
- `sigma=0.0`: Original image itself (`x_t = x0`)

**Importance of `sigma_end = 0.0`:**

From code comments (nodes/samplerv2.py:539-540):
```python
# Use 0.0 terminal sigma so low denoise can truly stay close to the init image.
# (Using 1/steps creates a "noise floor" that can make denoise feel inverted at low step counts.)
sigma_end = 0.0
```

**Why 0.0 is Important:**
- If `sigma_end` is `1/steps` (e.g., 0.05), 5% noise remains at the end
- This prevents fully approaching the original image even at low `denoise` values
- `sigma_end=0.0` enables **complete convergence** to the original image

**Fundamental Differences from Standard KSampler:**
1. Step count is not changed: `req_steps = int(steps)` (not `steps * denoise`)
2. Control via sigma schedule: `denoise` value directly becomes `sigma_start`
3. Mathematically accurate mixing: Based on Flow Matching formula

#### 3. Image-to-Image (i2i) Processing Implementation

##### 3.1 Standard KSampler i2i Processing

**Processing Flow:**
```
Input IMAGE → VAE Encode → LATENT Acquisition
→ Add Noise (according to denoise) → Diffusion Process → VAE Decode → Output IMAGE
```

**Implementation Characteristics:**
- Uses ComfyUI's internal VAE
- Encodes input image into latent space
- Adds noise to start diffusion process
- Adjusts noise amount according to `denoise` value

**Constraints:**
- Depends on ComfyUI's model structure
- Does not account for differences by pipeline type

##### 3.2 SDNQSamplerV2 i2i Processing (Non-Flux Pipelines)

**Implementation Logic:**
```python
# nodes/samplerv2.py:507-515, 607-609
if not is_flux_family:
    pipeline_kwargs["image"] = pil_cond
    # ... width/height adjustment ...
    
# Later:
if (not is_flux_family) and strength is not None and ("strength" in call_params):
    pipeline_kwargs["strength"] = strength  # denoise value
```

**Operation Principle:**
- Passes input image as `pipeline_kwargs["image"]`
- Controls noise level with `strength` parameter
- Diffusers pipeline handles noise addition internally

**Differences from Standard KSampler:**
- Uses diffusers' standard API
- Pipeline handles noise addition internally

##### 3.3 SDNQSamplerV2 i2i Processing (Flux2 Pipeline) - Core Implementation

**Complete Implementation Logic:**

**Step 1: Generate sigma schedule** (nodes/samplerv2.py:533-548)
```python
sigma_end = 0.0
sigma_start = float(strength)  # denoise value
sigmas = np.linspace(sigma_start, sigma_end, req_steps, dtype=np.float32).tolist()
pipeline_kwargs["sigmas"] = sigmas
```

**Step 2: Image preprocessing** (nodes/samplerv2.py:550-569)
```python
# Prepare image tensor exactly like the pipeline does
image_tensor = pipeline.image_processor.preprocess(
    pil_cond, height=img_h, width=img_w, resize_mode="crop"
)
image_tensor = image_tensor.to(device=generator_device, dtype=pipeline.vae.dtype)
```

**Step 3: Encode to Flux2 latent space** (nodes/samplerv2.py:572-573)
```python
# Encode to Flux2 latent space (unpacked shape [B, 128, H', W'])
x0 = pipeline._encode_vae_image(image=image_tensor, generator=generator)
```

**Step 4: Initialize latents with noise** (nodes/samplerv2.py:575-589)
```python
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
```

**Step 5: Remove image argument** (nodes/samplerv2.py:597-603)
```python
# IMPORTANT:
# If we successfully initialize `latents` from the input image, do NOT also pass `image=`.
# Flux2 treats `image` as additional reference conditioning tokens; keeping it makes denoise appear
# "stuck" (0.2 and 0.8 look similar) because the reference conditioning dominates.
if flux_latent_init_ok:
    pipeline_kwargs.pop("image", None)
```

**Why This Implementation is Critical:**

1. **Manual latent initialization**: Achieves true i2i behavior by starting from encoded image latents (`x0`), not random noise
2. **Sigma schedule control**: Precise denoise control via sigma mixing ratio
3. **Image argument removal**: Prevents reference conditioning from overriding denoise effect
4. **Complete convergence**: `sigma_end=0.0` enables fully approaching original image at low denoise values

#### 4. Technical Challenges and Solutions

##### 4.1 Challenge 1: "Complete Noise" Output

**Problem:**
- Running i2i with Flux2 produces complete noise images
- `denoise` value doesn't work

**Cause:**
- Flux2's `image` parameter functions as conditioning tokens
- If initial latent remains random noise, i2i doesn't occur

**Solution:**
1. Encode input image to obtain `x0` (nodes/samplerv2.py:572-573)
2. Mix noise according to `denoise` value to create `x_t` (nodes/samplerv2.py:587-589)
3. Pass `x_t` to `pipeline_kwargs["latents"]` (nodes/samplerv2.py:592)
4. Remove `image` argument (nodes/samplerv2.py:601-603)

##### 4.2 Challenge 2: Weak Denoise Value Effect

**Problem:**
- `denoise=0.2` and `denoise=0.8` show almost no difference
- Reference conditioning is too strong

**Cause:**
- Keeping `image` argument causes reference conditioning to override denoise effect

**Solution:**
- When initial latent is successfully prepared, remove `image` argument (nodes/samplerv2.py:601-603)
- This prioritizes initial latent i2i effect

##### 4.3 Challenge 3: Not Approaching Original Image at Low Denoise

**Problem:**
- Even at `denoise=0.2`, results deviate from original image

**Cause:**
- If `sigma_end` is `1/steps` (e.g., 0.05), a noise floor remains

**Solution:**
- Fix `sigma_end = 0.0` (nodes/samplerv2.py:541)
- This enables complete convergence to original image

##### 4.4 Challenge 4: Quality Degradation from Step Reduction

**Problem:**
- Standard KSampler's `effective_steps = steps * denoise` reduces step count
- Quality degrades at low `denoise` values

**Solution:**
- SDNQSamplerV2 does not change step count (always uses `num_inference_steps: steps`)
- `denoise` value only controls initial noise level (sigma)
- This enables denoise control while maintaining quality

#### 5. Implementation Consistency and Design Decisions

##### 5.1 Branching by Pipeline Type

**Design Decision:**
```python
# nodes/samplerv2.py:311-318
pipeline_type = type(pipeline).__name__
is_flux_family = pipeline_type in ["Flux2Pipeline", "FluxPipeline", "FluxSchnellPipeline"]
call_params = set(inspect.signature(pipeline.__call__).parameters.keys())
supports_image_arg = ("image" in call_params)
```

**Why Needed:**
- Flux2 and non-Flux have fundamentally different processing
- Runtime pipeline type detection selects appropriate processing

##### 5.2 Input Format Flexibility

**Design Decision:**
```python
# nodes/samplerv2.py:486-501
pil_cond = None
# Pattern 1: pixels directly provided
if isinstance(latent_image, dict) and latent_image.get("pixels") is not None:
    pil_cond = _tensor_to_pil_rgb(px)
# Pattern 2: decode from latent
if pil_cond is None and isinstance(latent_image, dict) and latent_image.get("vae") is not None:
    pil_cond = _decode_latents_to_pil(samples)
```

**Why Needed:**
- ComfyUI's `SDNQVAEEncode` provides `pixels` (optimal)
- Traditional VAE encode nodes only provide `latent`
- Supporting both formats ensures flexibility

##### 5.3 Step Count Preservation

**Design Decision:**
```python
# nodes/samplerv2.py:394
"num_inference_steps": steps,  # Not changed by denoise value
```

**Difference from Standard KSampler:**
- Standard: `effective_steps = steps * denoise` (step reduction)
- SDNQSamplerV2: Uses `steps` as-is (step preservation)

**Reasons:**
1. Quality maintenance: Preserving step count maintains quality
2. Accurate denoise control: Only controls initial noise level (sigma)
3. Flow Matching consistency: Flow Matching controls via sigma schedule

##### 5.4 VAE.decode Patch for Flux2 Compatibility

**Implementation:**
```python
# nodes/samplerv2.py:642-656
original_vae_decode = None
if hasattr(pipeline, "vae") and pipeline.vae is not None and hasattr(pipeline.vae, "decode"):
    try:
        original_vae_decode = pipeline.vae.decode
        def patched_decode(z, *args, **kwargs):
            return original_vae_decode(z.float(), *args, **kwargs)
        pipeline.vae.decode = patched_decode
```

**Why Needed:**
- Flux2 VAE may have dtype mismatches (bfloat16/float bias issues)
- Forces float32 input to prevent errors
- Primarily needed for Flux pipelines, but applied to all for consistency

##### 5.5 retrieve_timesteps Patch for Flux2

**Implementation:**
```python
# nodes/samplerv2.py:658-685
if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps
    import diffusers.pipelines.flux2.pipeline_flux2 as flux2_module
    
    scheduler_supports_mu = isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    
    def patched_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, **kwargs):
        if not scheduler_supports_mu:
            kwargs.pop("mu", None)
        return original_retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps=timesteps, **kwargs
        )
    
    flux2_module.retrieve_timesteps = patched_retrieve_timesteps
```

**Why Needed:**
- Some schedulers don't support `mu` parameter
- Removes `mu` when scheduler doesn't support it
- Flux denoise handling is done via `sigmas` + `latents` initialization

#### 6. Key Advantages of SDNQSamplerV2

1. **Quality Preservation**: Step count is not reduced, maintaining image quality even at low denoise values
2. **Accurate Denoise Control**: Mathematically precise control via sigma schedule (Flux2) or strength parameter (non-Flux)
3. **Flux2 Compatibility**: Complete support for Flux2's unique Flow Matching architecture
4. **Flexibility**: Supports various pipeline types through runtime detection and appropriate processing selection
5. **True i2i Behavior**: Manual latent initialization for Flux2 ensures actual image-to-image transformation, not just reference conditioning

#### 7. Code References

Key implementation locations in `nodes/samplerv2.py`:

- **Pipeline Type Detection**: Lines 311-318
- **Denoise Control (Flux2)**: Lines 530-548
- **Initial Latent Preparation (Flux2)**: Lines 550-593
- **Image Argument Removal**: Lines 597-603
- **Denoise Control (Non-Flux)**: Lines 605-609
- **Fallback Path (Non-Flux)**: Lines 611-631
- **VAE.decode Patch**: Lines 642-656
- **retrieve_timesteps Patch**: Lines 658-685
- **Error Handling**: Lines 687-710

See also: [COMPARISON_TABLE.md](md/COMPARISON_TABLE.md) for a summary comparison table.

### Standard ComfyUI LoRA Loader vs SDNQ LoRA Loader Comparison

A comprehensive technical explanation comparing the standard ComfyUI LoRA Loader and SDNQ LoRA Loader, covering architecture differences (ModelPatcher vs DiffusionPipeline), LoRA application mechanisms (patches vs adapters), multiple LoRA processing methods (sequential vs parallel), and detailed implementation comparisons.

See: [LoRA_Loader_Comparison_Standard_vs_SDNQ_EN.md](md/LoRA_Loader_Comparison_Standard_vs_SDNQ_EN.md)

---

## Contributing

Contributions welcome! Please:
1. Follow existing code style
2. Test with multiple model types
3. Update documentation for new features

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)

This project integrates with [SDNQ by Disty0](https://github.com/Disty0/sdnq).

---

## Credits

### Original Repository
- **Original ComfyUI-SDNQ**: [EnragedAntelope/comfyui-sdnq](https://github.com/EnragedAntelope/comfyui-sdnq)
- This repository is a fork with modular node structure (split functionality)

### SDNQ - SD.Next Quantization Engine
- **Author**: Disty0
- **Repository**: https://github.com/Disty0/sdnq
- **Pre-quantized models**: https://huggingface.co/collections/Disty0/sdnq

This node pack provides ComfyUI integration for SDNQ. All quantization technology is developed and maintained by Disty0.
