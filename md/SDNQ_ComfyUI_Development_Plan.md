# SDNQ ComfyUI Node Pack - Development Plan & Feasibility Analysis

## Executive Summary

**Project**: Create a ComfyUI custom node pack that enables loading and using SDNQ (SD.Next Quantization) quantized models within ComfyUI workflows.

**Verdict**: ✅ **FEASIBLE WITHOUT MONKEYPATCHING**

SDNQ integrates cleanly through the standard `diffusers` API. Simply importing `SDNQConfig` from the `sdnq` package registers the quantization method into diffusers/transformers, enabling transparent loading of pre-quantized models via `from_pretrained()`. No ComfyUI core modifications are required.

---

## Research Findings

### 1. SDNQ Architecture Analysis

**Source Repository**: https://github.com/Disty0/sdnq

**Key Design Principles**:
- SDNQ registers into `diffusers` and `transformers` via import side-effects
- Pre-quantized models follow standard diffusers directory structure
- Loading is transparent: `from sdnq import SDNQConfig` + `AutoModel.from_pretrained(path)`
- Supports int8/int6/int4/uint4 quantization with optional SVD compression
- Cross-platform: works on CUDA, ROCm, Intel Arc, CPU
- Optional Triton acceleration via `torch.compile`

**Model Structure** (standard diffusers format):
```
model_name/
├── model_index.json          # Pipeline configuration
├── transformer/ or unet/     # Main diffusion model
│   ├── config.json
│   └── *.safetensors
├── text_encoder/             # CLIP/T5 text encoder
├── vae/                      # Variational Autoencoder
├── tokenizer/                # Text tokenizer
└── scheduler/                # Noise scheduler config
```

**Pre-quantized Models Available** (https://huggingface.co/collections/Disty0/sdnq):
- FLUX.1-dev, FLUX.2-dev (various quant levels)
- SD3.5 Large/Medium
- SDXL (NoobAI, etc.)
- Wan2.2 I2V/T2V video models
- Qwen Image Edit models

### 2. ComfyUI Integration Points

**Current Diffusers Loading Pattern** (from ComfyUI-DiffusersLoader):
- Models stored in `ComfyUI/models/diffusers/`
- Standard diffusers `from_pretrained()` API
- NODE_CLASS_MAPPINGS registration (V1 API)

**V3 Node API** (for future-proofing):
- Uses `comfy_api.latest` imports
- `ComfyExtension` class with `comfy_entrypoint()`
- Stateless class methods for execution
- Better type hints and schema definitions

**Model Folder Integration**:
- Use existing `models/diffusers/` folder for SDNQ models
- Support `extra_model_paths.yaml` for custom paths
- Leverage ComfyUI's model management infrastructure

### 3. No Monkeypatching Required

**Evidence**:
1. SDNQ works via import registration - no code injection
2. Models use standard safetensors format
3. Loading uses unmodified `diffusers.from_pretrained()`
4. Memory management through diffusers' `enable_model_cpu_offload()`
5. Triton optimization is opt-in and handled by SDNQ internally

**Integration Pattern**:
```python
from sdnq import SDNQConfig  # This registers SDNQ into diffusers
import diffusers

# Pre-quantized models load transparently
pipe = diffusers.FluxPipeline.from_pretrained(
    "Disty0/FLUX.1-dev-qint8",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
```

---

## Architecture Design

### Node Pack Structure

```
ComfyUI-SDNQ/
├── __init__.py                    # V1 + V3 dual compatibility entry point
├── nodes/
│   ├── __init__.py
│   ├── loader.py                  # SDNQ Model Loader node
│   ├── quantizer.py               # Checkpoint Quantization node
│   └── utils.py                   # Shared utilities
├── core/
│   ├── __init__.py
│   ├── model_registry.py          # Pre-quantized model catalog
│   ├── downloader.py              # HuggingFace Hub integration
│   └── config.py                  # SDNQ configuration helpers
├── requirements.txt
├── install.py                     # ComfyUI Manager install script
├── pyproject.toml                 # Modern Python packaging
├── LICENSE                        # Apache 2.0 (matching SDNQ models)
├── README.md
├── CREDITS.md                     # Attribution to Disty0
└── context.md                     # Running project context (Claude Code)
```

### Node Definitions

#### 1. SDNQModelLoader
**Purpose**: Load pre-quantized SDNQ models from local storage or HuggingFace

**Inputs**:
- `model_source` (dropdown): "Local", "HuggingFace"
- `model_name` (dynamic): List of available models based on source
- `hf_repo_id` (string, optional): Custom HuggingFace repo ID
- `dtype` (dropdown): "bfloat16", "float16", "float32"
- `use_quantized_matmul` (boolean): Enable INT8/FP8 matmul acceleration
- `device_map` (dropdown): "auto", "cuda", "cpu"
- `cpu_offload` (boolean): Enable model CPU offloading

**Outputs**:
- `MODEL` (ComfyUI MODEL type)
- `CLIP` (ComfyUI CLIP type)
- `VAE` (ComfyUI VAE type)

#### 2. SDNQCheckpointQuantizer
**Purpose**: Quantize existing checkpoints to SDNQ format

**Inputs**:
- `checkpoint` (file selector): Source checkpoint
- `quant_type` (dropdown): "int8", "int6", "uint4", "float8_e4m3fn"
- `use_svd` (boolean): Enable SVD compression
- `svd_rank` (int): SVD rank (default 32)
- `group_size` (int): Quantization group size (0=auto)
- `output_name` (string): Output model name

**Outputs**:
- `save_path` (string): Path to saved quantized model

#### 3. SDNQModelCatalog
**Purpose**: Display and manage available pre-quantized models

**Inputs**:
- `refresh` (button): Refresh model list

**Outputs**:
- `catalog` (list): Available models with metadata

---

## Implementation Plan

### Phase 1: Foundation (Week 1)
1. **Project Setup**
   - Initialize git repository with proper structure
   - Create pyproject.toml with dependencies
   - Set up basic V1 node registration
   - Create context.md for ongoing development

2. **Core Infrastructure**
   - Implement model registry for tracking available SDNQ models
   - Create HuggingFace Hub integration for downloads
   - Implement model path resolution (local + extra_model_paths.yaml)

3. **Basic Loader Node**
   - SDNQModelLoader with local file support
   - Output compatible ComfyUI MODEL/CLIP/VAE types
   - Error handling and logging

### Phase 2: Full Features (Week 2)
4. **HuggingFace Integration**
   - Catalog of pre-quantized models from Disty0 collection
   - Auto-download functionality
   - Progress tracking and caching

5. **Quantization Node**
   - SDNQCheckpointQuantizer implementation
   - Support for all SDNQ quant types
   - SVD compression option
   - Save to ComfyUI models folder

6. **Advanced Options**
   - Triton/torch.compile optimization toggle
   - Memory management options
   - Device placement controls

### Phase 3: Polish & Compatibility (Week 3)
7. **V3 API Compatibility**
   - Add V3 schema definitions alongside V1
   - Future-proof for async execution

8. **Testing & Documentation**
   - Test with multiple model types (FLUX, SD3, SDXL)
   - Example workflows
   - Comprehensive README

9. **Upstream Sync Strategy**
   - Git submodule or dependency-only approach for sdnq
   - Version pinning strategy
   - Changelog monitoring for breaking changes

---

## Technical Specifications

### Dependencies

```
# requirements.txt
sdnq @ git+https://github.com/Disty0/sdnq.git
diffusers>=0.30.0
transformers>=4.40.0
huggingface-hub>=0.20.0
safetensors>=0.4.0
torch>=2.0.0
accelerate>=0.25.0
```

### Model Storage Strategy

SDNQ models will be stored in:
```
ComfyUI/models/diffusers/sdnq/
├── FLUX.1-dev-qint8/
├── SD3.5-Large-uint4-svd-r32/
└── ...
```

This leverages existing ComfyUI diffusers model infrastructure while keeping SDNQ models organized in a dedicated subfolder.

### ComfyUI Type Mapping

| SDNQ Component | ComfyUI Type | Conversion Strategy |
|----------------|--------------|---------------------|
| Pipeline.transformer/unet | MODEL | Wrap in ComfyUI ModelPatcher |
| Pipeline.text_encoder | CLIP | Create ComfyUI CLIP wrapper |
| Pipeline.vae | VAE | Create ComfyUI VAE wrapper |
| Scheduler | (internal) | Used during sampling |

### Memory Management

SDNQ models support several memory strategies that map to ComfyUI patterns:
- `enable_model_cpu_offload()` → Similar to ComfyUI's model offloading
- `enable_sequential_cpu_offload()` → For very low VRAM scenarios
- Triton quantized matmul → Automatic if Triton available

---

## Upstream Sync Strategy

### Recommended Approach: Dependency-Only

**Rationale**: SDNQ is a pip-installable package. Using it as a dependency rather than embedding source code ensures:
- Automatic access to bug fixes and new model support
- No code duplication
- Clear licensing boundaries
- Minimal maintenance burden

**Implementation**:
```python
# requirements.txt - Pin to release tag or commit for stability
sdnq @ git+https://github.com/Disty0/sdnq.git@v1.0.0
```

### Version Monitoring

1. Watch Disty0/sdnq releases via GitHub notifications
2. Test new versions before updating pin
3. Document breaking changes in CHANGELOG.md

### Contributing Back

If modifications are needed:
1. Fork sdnq repository
2. Submit PR to upstream
3. Temporarily use fork in requirements.txt
4. Revert to upstream once PR merged

---

## Licensing & Attribution

### SDNQ License
The `sdnq` package appears to be permissively licensed (models reference Apache 2.0).

### Required Attribution
Create `CREDITS.md`:
```markdown
# Credits

## SDNQ - SD.Next Quantization Engine
- Author: Disty0
- Repository: https://github.com/Disty0/sdnq
- License: [Include actual license]

This node pack provides ComfyUI integration for SDNQ. 
All quantization technology is developed and maintained by Disty0.

Pre-quantized models: https://huggingface.co/collections/Disty0/sdnq
```

### Node Pack License
Recommend Apache 2.0 for compatibility with both SDNQ and ComfyUI ecosystem.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SDNQ API changes | Medium | High | Pin versions, monitor releases |
| ComfyUI V3 breaking changes | Low | Medium | Maintain dual V1/V3 compatibility |
| Memory issues with large models | Medium | Medium | Document requirements, add guards |
| HuggingFace rate limits | Low | Low | Implement caching, respect limits |
| Triton unavailable on Windows | Known | Low | Graceful fallback, document setup |

---

## Success Metrics

1. **Functional**: Load and use any model from Disty0/sdnq collection
2. **Performance**: Memory usage matches SD.Next benchmarks
3. **Compatibility**: Works with standard ComfyUI workflows
4. **Maintainability**: Updates require only dependency version bumps
5. **Adoption**: Positive community feedback on HuggingFace/GitHub

---

## Conclusion

This project is **highly feasible** because:
1. SDNQ provides clean API integration via diffusers
2. No ComfyUI core modifications required
3. Existing diffusers loader patterns provide proven templates
4. Pre-quantized models are immediately available
5. Upstream sync is handled via pip dependency management

The architecture leverages existing infrastructure while adding significant value for users with limited VRAM who want to run large models like FLUX and SD3.5.
