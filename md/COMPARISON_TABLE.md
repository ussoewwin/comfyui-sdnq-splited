# Standard KSampler vs SDNQSamplerV2: Implementation Comparison

## Overview

This document provides a side-by-side comparison of the standard ComfyUI KSampler and SDNQSamplerV2 implementations, highlighting key architectural and functional differences.

## Comparison Table

| Item | Standard KSampler | SDNQSamplerV2 |
|------|------------------|---------------|
| **Architecture** | ComfyUI Internal API | diffusers Pipeline |
| **Denoise Control** | Step Count Reduction | Initial Noise Level Adjustment |
| **i2i Processing** | VAE Encode/Decode | Pipeline-Specific Processing |
| **Flux2 Support** | None | Full Support (Special Implementation) |
| **Step Count** | `steps * denoise` | `steps` (unchanged) |

## Detailed Explanations

### Architecture

- **Standard KSampler**: Uses ComfyUI's internal model representation (`ModelPatcher`, `CLIP`, `VAE`) and internal APIs (`sampling_function`, `model_function`).
- **SDNQSamplerV2**: Directly uses diffusers library's `DiffusionPipeline`, enabling support for both Flow Matching models (Flux2) and traditional diffusion models.

### Denoise Control

- **Standard KSampler**: Reduces the number of processing steps proportionally (`effective_steps = steps * denoise`). For example, `steps=20, denoise=0.5` results in 10 steps being executed.
- **SDNQSamplerV2**: Maintains the full step count but adjusts the initial noise level. For Flux2, this is controlled via sigma schedule; for non-Flux pipelines, via the `strength` parameter.

### Image-to-Image (i2i) Processing

- **Standard KSampler**: Uses ComfyUI's internal VAE for encoding/decoding, with noise addition handled internally.
- **SDNQSamplerV2**: Uses pipeline-specific processing. For Flux2, manually initializes latents from the input image; for non-Flux pipelines, uses standard `image` + `strength` API.

### Flux2 Support

- **Standard KSampler**: No support for Flux2 or Flow Matching models.
- **SDNQSamplerV2**: Full support with special implementation including:
  - Manual latent initialization from input images
  - Sigma schedule control (`sigma_end = 0.0` for complete convergence)
  - Conditional `image` argument removal to prevent reference conditioning override

### Step Count Behavior

- **Standard KSampler**: Step count is reduced based on denoise value, which can lead to quality degradation at low denoise values.
- **SDNQSamplerV2**: Step count remains constant, ensuring quality is maintained while denoise strength is controlled via initial noise level.

## Key Advantages of SDNQSamplerV2

1. **Quality Preservation**: Step count is not reduced, maintaining image quality even at low denoise values.
2. **Accurate Denoise Control**: Mathematically precise control via sigma schedule (Flux2) or strength parameter (non-Flux).
3. **Flux2 Compatibility**: Complete support for Flux2's unique Flow Matching architecture.
4. **Flexibility**: Supports various pipeline types through runtime detection and appropriate processing selection.

## Technical Notes

- SDNQSamplerV2 uses dynamic pipeline type detection to select the appropriate processing path.
- For Flux2, the implementation includes special handling to ensure true i2i behavior rather than just reference conditioning.
- The `sigma_end = 0.0` setting is crucial for Flux2 to enable complete convergence to the original image at low denoise values.

