# ComfyUI-SDNQ

**Load SDNQ quantized models in ComfyUI with 50-75% VRAM savings!**

This custom node pack enables loading [SDNQ (SD.Next Quantization)](https://github.com/Disty0/sdnq) models in ComfyUI workflows. Run large models like FLUX.1 and SD3.5 on consumer hardware with significantly reduced VRAM requirements while maintaining image quality.

> **SDNQ is developed by [Disty0](https://github.com/Disty0)** - this node pack provides ComfyUI integration.
> See [CREDITS.md](CREDITS.md) for full attribution.

---

## Features

- **🚀 Massive VRAM Savings**: 50-75% reduction in memory usage
- **🎨 Quality Maintained**: Minimal to no degradation in output quality
- **⚡ Multiple Quant Levels**: Support for int8, int6, uint4, and more
- **🔌 Drop-in Compatibility**: Works with standard ComfyUI nodes (KSampler, etc.)
- **🌐 HuggingFace Integration**: Load pre-quantized models directly from the hub
- **🏃 Triton Acceleration**: Optional quantized matrix multiplication speedup

---

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for "SDNQ" in the manager
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/EnragedAntelope/comfyui-sdnq.git
cd comfyui-sdnq
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## Quick Start

### 1. Get an SDNQ Model

**Pre-quantized models** from Disty0's collection:
- Browse: https://huggingface.co/collections/Disty0/sdnq
- Popular choices:
  - `Disty0/FLUX.1-dev-qint8` (FLUX.1 - 8-bit)
  - `Disty0/FLUX.1-dev-qint4` (FLUX.1 - 4-bit, extreme savings)
  - `Disty0/stable-diffusion-3.5-large-qint8` (SD3.5 Large)
  - `Disty0/stable-diffusion-xl-base-1.0-qint8` (SDXL)

### 2. Using the Node

1. Add the **SDNQ Model Loader** node (under `loaders/SDNQ`)
2. Enter either:
   - **HuggingFace repo ID**: `Disty0/FLUX.1-dev-qint8`
   - **Local path**: `/path/to/downloaded/model`
3. Configure settings:
   - **dtype**: `bfloat16` (recommended), `float16`, or `float32`
   - **use_quantized_matmul**: Enable Triton optimization (if available)
   - **cpu_offload**: Save even more VRAM by offloading to CPU
4. Connect outputs to standard ComfyUI nodes:
   - `MODEL` → KSampler
   - `CLIP` → CLIP Text Encode
   - `VAE` → VAE Decode

### 3. Example Workflow

```
SDNQ Model Loader
├─ model: Disty0/FLUX.1-dev-qint8
├─ dtype: bfloat16
└─ use_quantized_matmul: ✓

     ↓ (MODEL)

  KSampler ← (your prompts, seeds, etc.)

     ↓ (LATENT)

  VAE Decode ← (VAE from SDNQ loader)

     ↓ (IMAGE)

 Save Image
```

---

## Node Reference

### SDNQ Model Loader

**Category**: `loaders/SDNQ`

**Inputs**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_path | STRING | "" | HuggingFace repo ID or local path |
| dtype | CHOICE | bfloat16 | Weight data type (bfloat16/float16/float32) |
| use_quantized_matmul | BOOLEAN | True | Enable Triton quantized matmul (faster inference) |
| cpu_offload | BOOLEAN | True | Offload model to CPU when not in use |
| device | CHOICE | auto | Device placement (auto/cuda/cpu) |

**Outputs**:
- `MODEL`: Quantized diffusion model (compatible with KSampler)
- `CLIP`: Text encoder (compatible with CLIP Text Encode)
- `VAE`: Variational autoencoder (compatible with VAE Decode/Encode)

---

## Performance Comparison

Typical VRAM usage for FLUX.1-dev on RTX 4090:

| Version | VRAM Usage | Quality |
|---------|------------|---------|
| Full fp16 | ~24 GB | 100% (reference) |
| SDNQ int8 | ~12 GB | ~99% |
| SDNQ int6 | ~9 GB | ~97% |
| SDNQ uint4 | ~6 GB | ~95% |

*Measurements may vary based on resolution, batch size, and system configuration.*

---

## Model Storage

Models are cached following Diffusers/HuggingFace Hub conventions:
- **HuggingFace downloads**: `~/.cache/huggingface/hub/`
- **Recommended local path**: `ComfyUI/models/diffusers/sdnq/`

You can move models from the cache to your ComfyUI models folder and reference them by path to avoid re-downloading.

---

## Troubleshooting

### "Triton not available" Warning

Triton is an optional optimization. The model will work without it, just slightly slower.

To enable Triton (Linux/WSL only):
```bash
pip install triton
```

Triton is not available on native Windows. Use WSL2 for Triton support.

### Out of Memory Errors

1. Enable **cpu_offload** in the node settings
2. Use a more aggressive quantization (int6 or uint4)
3. Reduce batch size or resolution
4. Close other GPU applications

### Model Loading Fails

1. Check internet connection (for HuggingFace models)
2. Verify the repo ID is correct
3. For local models, ensure the path points to the model directory (not a file)
4. Check that the model is actually SDNQ-quantized (from Disty0's collection)

### "Pipeline missing transformer/unet" Error

The model may not be in the expected diffusers format. SDNQ models should have a standard diffusers directory structure with `model_index.json`.

---

## Quantizing Your Own Models

**Coming in Phase 2**: Support for quantizing existing checkpoints to SDNQ format.

For now, use the [sdnq](https://github.com/Disty0/sdnq) package directly or use pre-quantized models from the [Disty0 collection](https://huggingface.co/collections/Disty0/sdnq).

---

## Development Status

### Phase 1 (Current): ✅ Complete
- [x] Basic SDNQ model loading
- [x] Local and HuggingFace Hub support
- [x] ComfyUI type compatibility (MODEL, CLIP, VAE)
- [x] Triton optimization support
- [x] CPU offloading

### Phase 2 (Planned):
- [ ] Model catalog with dropdown selection
- [ ] Automatic model downloading with progress bar
- [ ] Checkpoint quantization node
- [ ] LoRA support with SDNQ models

### Phase 3 (Future):
- [ ] Memory usage reporting
- [ ] Advanced optimization options
- [ ] Video model support (Wan2.2, etc.)

---

## Contributing

Contributions welcome! Please:
1. Follow the existing code style
2. Test with multiple model types (FLUX, SD3, SDXL)
3. Update documentation for new features

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)

This project integrates with [SDNQ by Disty0](https://github.com/Disty0/sdnq). Please respect the upstream project's license.

---

## Links

- **This Repository**: https://github.com/EnragedAntelope/comfyui-sdnq
- **SDNQ Engine**: https://github.com/Disty0/sdnq
- **Pre-quantized Models**: https://huggingface.co/collections/Disty0/sdnq
- **SDNQ Documentation**: https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI

---

**Made possible by [Disty0's SDNQ](https://github.com/Disty0/sdnq)** - bringing large models to consumer hardware!
