# Complete Explanation of the 1024 Fixed Size Issue

## Problem Overview

In Flux2 SDNQ Sampler V2's i2i (image-to-image) mode, the output was always fixed to 1024×1024 regardless of the input image size.

## Root Cause: Automatic Resize Processing

The cause was an automatic image resize processing that existed in lines 496-502 of `flux2samplerv2.py`.

### Problematic Code (Before Fix)

```python
# Prepare image tensor exactly like the pipeline does
img_w, img_h = pil_cond.size
if img_w * img_h > 1024 * 1024:
    try:
        pil_cond = pipeline.image_processor._resize_to_target_area(pil_cond, 1024 * 1024)
        pipeline_kwargs["image"] = pil_cond
        img_w, img_h = pil_cond.size
    except Exception:
        pass

multiple_of = int(getattr(pipeline, "vae_scale_factor", 16)) * 2
```

**File:** `ComfyUI/custom_nodes/comfyui-sdnq-splited/nodes/flux2samplerv2.py`  
**Lines:** 494-503

## What Was Wrong

### 1. Size Check
- **Condition:** `if img_w * img_h > 1024 * 1024:`
- This detected when the pixel count exceeded 1,048,576 pixels (1024×1024).

### 2. Forced Resize
- **Method:** `pipeline.image_processor._resize_to_target_area(pil_cond, 1024 * 1024)`
- This forcibly resized the image to a target area of 1,048,576 pixels.
- **Example:** A 2048×2048 image (4,194,304 pixels) → resized to approximately 1024×1024
- Aspect ratio was not preserved, so dimensions changed.

### 3. Size Overwrite
- After resizing, `img_w, img_h = pil_cond.size` overwrote the dimension variables with the resized values.

### 4. Execution Context
- This processing was located inside the conditional block at line 474 (i2i mode for Flux pipelines):
  - `if is_flux_family and pipeline_type in ["Flux2Pipeline", "FluxPipeline"] and strength is not None:`
- **It only executed when:**
  - i2i mode (`strength is not None`) AND
  - Flux pipeline type

## Why This Processing Existed

This was likely intended to reduce VRAM usage by limiting image size. However, for i2i mode, it's necessary to preserve the input image size, and this unconditional resizing was problematic.

## Fix Applied

Removed the resize processing and changed the code to preserve the input image size.

### Fixed Code (After Fix)

```python
# Prepare image tensor exactly like the pipeline does
# Note: For i2i mode, preserve input image size (do not resize to 1024x1024)
img_w, img_h = pil_cond.size

multiple_of = int(getattr(pipeline, "vae_scale_factor", 16)) * 2
if multiple_of > 0:
    img_w = (img_w // multiple_of) * multiple_of
    img_h = (img_h // multiple_of) * multiple_of
if img_w <= 0 or img_h <= 0:
    img_w, img_h = pil_cond.size
```

**Changes Made:**
1. Removed the size check and `_resize_to_target_area` call
2. Changed to use the input image size (`pil_cond.size`) as-is
3. Only adjusted to meet VAE's `multiple_of` requirement

## Impact of the Fix

### t2i Mode: No Impact
- This processing only runs inside the conditional block at line 474 (i2i mode for Flux pipelines)
- In t2i mode, `strength is None`, so this block is not executed
- t2i mode sets `width`/`height` at lines 339-346 and uses them as-is

### i2i Mode: Input Image Size Preserved
- Input images of any size (e.g., 2048×2048) are now processed as-is
- Only adjusted to meet VAE's `multiple_of` requirement (maintains compatibility)

## Processing Flow (After Fix)

1. **Input Image Conversion** (lines 433-445)
   - Convert input image to PIL format

2. **i2i Mode Detection** (line 449)
   - Determine if i2i mode is active

3. **Flux Pipeline Check** (line 474)
   - Check if it's a Flux pipeline

4. **Image Size Acquisition** (line 496)
   - `img_w, img_h = pil_cond.size` (no resize performed)

5. **VAE Requirement Adjustment** (lines 498-503)
   - Adjust to values divisible by `multiple_of`

6. **Image Tensor Preprocessing** (lines 505-507)
   - `pipeline.image_processor.preprocess()` for preprocessing

7. **VAE Encoding** (line 511)
   - Encode to latent space

8. **Noise Addition and Latent Setting** (lines 525-530)
   - Add noise to create starting latents

## Code Location

- **File:** `ComfyUI/custom_nodes/comfyui-sdnq-splited/nodes/flux2samplerv2.py`
- **Fix Location:** Lines 494-503 (removed resize processing)
- **Related Conditional:** Line 474 (i2i mode for Flux pipeline)

## Conclusion

The 1024×1024 fixed size issue in i2i mode was caused by an unconditional resize processing. By removing this processing, input image sizes are now preserved, and larger sizes like 2048×2048 are correctly processed. This fix does not affect t2i mode.
