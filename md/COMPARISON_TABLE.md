# Standard KSampler vs SDNQSamplerV2: Complete Technical Explanation of i2i and Denoise Implementation

## Table of Contents

1. [Overview and Fundamental Architectural Differences](#overview-and-fundamental-architectural-differences)
2. [Denoise Strength Control Mechanism](#denoise-strength-control-mechanism)
3. [Image-to-Image (i2i) Processing Implementation](#image-to-image-i2i-processing-implementation)
4. [Flux2-Specific Special Implementation](#flux2-specific-special-implementation)
5. [Non-Flux Pipeline Processing](#non-flux-pipeline-processing)
6. [Technical Challenges and Solutions](#technical-challenges-and-solutions)
7. [Implementation Consistency and Design Decisions](#implementation-consistency-and-design-decisions)

---

## 1. Overview and Fundamental Architectural Differences

### 1.1 Standard KSampler Architecture

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

### 1.2 SDNQSamplerV2 Architecture

SDNQSamplerV2 directly uses diffusers library pipelines.

**Processing Flow:**
```
Input LATENT → Pipeline Detection → Flux2/Non-Flux Branch → Pipeline-Specific Processing → diffusers.__call__() → Output IMAGE
```

**Characteristics:**
- Directly calls diffusers' `DiffusionPipeline`
- Supports both Flow Matching models (Flux2) and traditional diffusion models
- The `denoise` parameter interpretation **fundamentally differs** by pipeline type

---

## 2. Denoise Strength Control Mechanism

### 2.1 Standard KSampler Denoise Implementation

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

### 2.2 SDNQSamplerV2 Denoise Implementation (Non-Flux Pipelines)

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

### 2.3 SDNQSamplerV2 Denoise Implementation (Flux2 Pipeline)

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

---

## 3. Image-to-Image (i2i) Processing Implementation

### 3.1 Standard KSampler i2i Processing

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

### 3.2 SDNQSamplerV2 i2i Processing (Non-Flux Pipelines)

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

### 3.3 SDNQSamplerV2 i2i Processing (Flux2 Pipeline) - Core Implementation

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

---

## 4. Flux2-Specific Special Implementation

### 4.1 Flow Matching Architecture Understanding

Flux2 uses Flow Matching, which is fundamentally different from traditional diffusion models:

**Traditional Diffusion:**
- Starts from noise and denoises step by step
- Denoise strength controlled by step reduction or noise level

**Flow Matching:**
- Uses continuous flow from noise to image
- Sigma directly controls mixing ratio: `x_t = sigma * noise + (1 - sigma) * x0`
- Requires sigma schedule for proper control

### 4.2 Sigma Schedule Implementation

**Critical Implementation Details:**

```python
# nodes/samplerv2.py:541-547
sigma_end = 0.0  # Fixed - critical for low denoise convergence
sigma_start = float(strength)  # denoise value (0.0-1.0)
sigmas = np.linspace(sigma_start, sigma_end, req_steps, dtype=np.float32).tolist()
pipeline_kwargs["sigmas"] = sigmas
```

**Why `sigma_end = 0.0` is Essential:**

- If `sigma_end = 1/steps` (e.g., 0.05 for 20 steps), a "noise floor" remains
- At low denoise values (e.g., 0.2), results cannot fully approach the original image
- `sigma_end = 0.0` allows complete convergence to the original image

### 4.3 Latent Initialization for Flux2 i2i

**Complete Process:**

1. **Image Preprocessing**: Resize and normalize to pipeline requirements
2. **VAE Encoding**: Encode to Flux2 latent space using `pipeline._encode_vae_image()`
3. **Timestep Calculation**: Use `compute_empirical_mu()` and `retrieve_timesteps()` with sigma schedule
4. **Noise Mixing**: Apply `scale_noise()` to mix original latents (`x0`) with noise according to sigma
5. **Latent Passing**: Pass mixed latents (`x_t`) to pipeline as starting point

**Code Reference:** nodes/samplerv2.py:550-592

---

## 5. Non-Flux Pipeline Processing

### 5.1 Standard img2img API

For non-Flux pipelines (SDXL, SD1.5, etc.), SDNQSamplerV2 uses diffusers' standard img2img API:

```python
# nodes/samplerv2.py:507-515, 607-609
if not is_flux_family:
    pipeline_kwargs["image"] = pil_cond
    pipeline_kwargs["strength"] = strength  # denoise value
```

**Operation:**
- `image`: Input image (PIL Image)
- `strength`: Controls initial noise level (0.0-1.0)
- Pipeline handles noise addition internally

### 5.2 Error Handling

**Implementation Logic:**
```python
# nodes/samplerv2.py:687-710
try:
    result = pipeline(**pipeline_kwargs)
except TypeError as e:
    # Retry if latents/strength not supported
    if "latents" in str(e) and "unexpected keyword argument" in str(e):
        if "latents" in pipeline_kwargs:
            del pipeline_kwargs["latents"]
        if "strength" in pipeline_kwargs:
            del pipeline_kwargs["strength"]
        result = pipeline(**pipeline_kwargs)
    # If width/height not supported
    elif ("width" in str(e) or "height" in str(e)) and "unexpected keyword argument" in str(e):
        pipeline_kwargs.pop("width", None)
        pipeline_kwargs.pop("height", None)
        result = pipeline(**pipeline_kwargs)
    else:
        raise
```

**Why Needed:**
- Diffusers pipelines have different APIs by type
- Flexible error handling supports various pipelines

---

## 6. Technical Challenges and Solutions

### 6.1 Challenge 1: "Complete Noise" Output

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

### 6.2 Challenge 2: Weak Denoise Value Effect

**Problem:**
- `denoise=0.2` and `denoise=0.8` show almost no difference
- Reference conditioning is too strong

**Cause:**
- Keeping `image` argument causes reference conditioning to override denoise effect

**Solution:**
- When initial latent is successfully prepared, remove `image` argument (nodes/samplerv2.py:601-603)
- This prioritizes initial latent i2i effect

### 6.3 Challenge 3: Not Approaching Original Image at Low Denoise

**Problem:**
- Even at `denoise=0.2`, results deviate from original image

**Cause:**
- If `sigma_end` is `1/steps` (e.g., 0.05), a noise floor remains

**Solution:**
- Fix `sigma_end = 0.0` (nodes/samplerv2.py:541)
- This enables complete convergence to original image

### 6.4 Challenge 4: Quality Degradation from Step Reduction

**Problem:**
- Standard KSampler's `effective_steps = steps * denoise` reduces step count
- Quality degrades at low `denoise` values

**Solution:**
- SDNQSamplerV2 does not change step count (always uses `num_inference_steps: steps`)
- `denoise` value only controls initial noise level (sigma)
- This enables denoise control while maintaining quality

---

## 7. Implementation Consistency and Design Decisions

### 7.1 Branching by Pipeline Type

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

### 7.2 Input Format Flexibility

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

### 7.3 Step Count Preservation

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

### 7.4 VAE.decode Patch for Flux2 Compatibility

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

### 7.5 retrieve_timesteps Patch for Flux2

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

### 7.6 Multi-Layer Error Handling

**Design Decision:**
```python
# nodes/samplerv2.py:595-596
except Exception as e:
    print(f"[SDNQ Sampler V2] Warning: Flux img2img latent init failed, falling back to conditioning-only: {e}")
```

**Why Needed:**
- Fallback needed if Flux2 initial latent preparation fails
- Continues with `image` conditioning only (not full i2i but avoids error)

---

## 8. Summary: Complete Technical Consistency of Implementation

### 8.1 Fundamental Differences from Standard KSampler

| Item | Standard KSampler | SDNQSamplerV2 |
|------|------------------|---------------|
| **Architecture** | ComfyUI Internal API | diffusers Pipeline |
| **Denoise Control** | Step Count Reduction | Initial Noise Level Adjustment |
| **i2i Processing** | VAE Encode/Decode | Pipeline-Specific Processing |
| **Flux2 Support** | None | Full Support (Special Implementation) |
| **Step Count** | `steps * denoise` | `steps` (unchanged) |

### 8.2 Importance of Flux2-Specific Implementation

1. Manual initial latent creation: Achieves true i2i
2. `sigma_end = 0.0`: Enables complete original image convergence
3. `image` argument removal: Makes denoise effect function correctly
4. Explicit sigma schedule control: Based on Flow Matching principles

### 8.3 Design Consistency

- Branching by pipeline type: Selects appropriate processing for Flux2 and non-Flux
- Flexible input formats: Supports both `pixels` and `latent`
- Multi-layer defense: Error handling and fallback
- Mathematical accuracy: Implementation based on Flow Matching principles

### 8.4 Technical Advantages

1. Quality maintenance: Quality preserved by not reducing step count
2. Accurate denoise control: Mathematically accurate control via sigma schedule
3. Full Flux2 support: Fully understands and appropriately handles Flux2's special behavior
4. Flexibility: Supports various pipeline types

---

## 9. Code References

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

---

This implementation enables SDNQSamplerV2 to provide functionality beyond the standard KSampler, achieving accurate, high-quality i2i and denoise control, especially for Flux2 models.
