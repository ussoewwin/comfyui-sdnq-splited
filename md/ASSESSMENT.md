# ComfyUI-SDNQ Reality-Based Assessment

**Date**: December 9, 2025
**Status**: CRITICAL ARCHITECTURAL ISSUES IDENTIFIED
**Recommendation**: REDESIGN REQUIRED

---

## Executive Summary

After grounding research in actual source code (Disty's SDNQ repo, diffusers 0.36.0+, ComfyUI 0.3.77+), I've identified that **the current wrapper-based architecture is fundamentally incompatible with ComfyUI**.

**Key Finding**: ComfyUI expects `ModelPatcher` objects with specific cloning/patching infrastructure. Our custom wrappers (`SDNQModelWrapper`, `SDNQCLIPWrapper`, `SDNQVAEWrapper`) don't inherit from these classes and will fail when KSampler or other nodes try to use them.

**Evidence**: GitHub Issue #14 confirms no one has successfully generated images with this node pack. Multiple users report the same incompatibility issues.

---

## Problems Identified (Grounded in Reality)

### 1. Wrapper Architecture is Fundamentally Wrong

**What ComfyUI Expects**:
```python
# ComfyUI expects MODEL to be a ModelPatcher object
from comfy.model_patcher import ModelPatcher

class ModelPatcher:
    def __init__(self, model, load_device, offload_device, ...):
        self.model = model  # The actual PyTorch module
        self.model_options = {}
        self.model_size()
        self.patches = {}
        # ... extensive patching/cloning infrastructure

    def clone(self):
        """KSampler calls this to create working copies"""
        # Returns new ModelPatcher with same model

    @property
    def latent_format(self):
        """Required for latent space operations"""
        # Returns LatentFormat object defining channels, scale, etc.
```

**What We Currently Provide**:
```python
# core/wrapper.py
class SDNQModelWrapper:
    def __init__(self, pipeline, model_component, model_type=None):
        self.pipeline = pipeline
        self.model = model_component
        self.model_type = model_type

    # Missing: clone(), latent_format, model_options, patches, etc.
```

**Result**: When KSampler tries to call `model.clone()` or access `model.latent_format`, it will fail with AttributeError.

**Evidence**:
- Lines 204-234 in context.md document this exact failure
- Error: `'NoneType' object has no attribute 'latent_channels'`
- This is because we don't set `latent_format`

### 2. Device Parameter Never Used (Confirmed Bug)

**Source**: GitHub Issue #14, Ph0rk0z comment

**Current Code** (nodes/loader.py:97-99, 291-296):
```python
# Line 97-99: We ACCEPT device parameter
"device": (["auto", "cuda", "cpu"],),

# Line 291-296: We NEVER USE it
pipeline = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    local_files_only=is_local,
    attn_implementation="sdpa",
    # Missing: device_map parameter!
)
```

**Impact**: Models always load to CPU first, then rely on `enable_model_cpu_offload()`, which is slower than direct GPU loading.

### 3. SDPA Warning on Flux2Pipeline (Confirmed Issue)

**Current Error**:
```
Keyword arguments {'attn_implementation': 'sdpa'} are not expected by Flux2Pipeline and will be ignored.
```

**Root Cause**: I assumed all pipelines support `attn_implementation` parameter without verifying.

**Reality Check**: According to diffusers documentation, `attn_implementation` IS supported on modern pipelines, BUT there may be version mismatches or the parameter name changed.

**Need to Verify**: What's the correct parameter name in diffusers 0.36.0.dev0?

### 4. Tensor Dimension Mismatch (SDNQ Dequantization Bug)

**Current Error**:
```
RuntimeError: a and b must have same reduction dim, but got [32, 4096] X [5120, 32]
```

**Root Cause**: torch.compile is failing during SDNQ weight dequantization.

**Reality**: This suggests deeper issues with how SDNQ quantized weights interact with torch.compile on certain hardware/compiler combinations.

**Hypothesis**: The `suppress_errors=True` config may not be working as expected, or the error is happening before compilation.

---

## What Actually Works (Grounded in Reality)

### SDNQ Standalone (Proven Working)

From GitHub Issue #14, yopfix's example:

```python
import torch
import diffusers
from sdnq import SDNQConfig

# This works perfectly
pipe = diffusers.ZImagePipeline.from_pretrained(
    "./models/diffusers/sdnq/[model-path]",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

# Generate image
image = pipe(
    prompt="a beautiful landscape",
    num_inference_steps=25,
    guidance_scale=7.0,
).images[0]
image.save("output.png")
```

**Key Insight**: SDNQ works flawlessly when used directly with diffusers. The problem is ONLY the ComfyUI integration layer.

---

## Three Viable Paths Forward

### Option A: Standalone Sampler Node (RECOMMENDED)

**Description**: Create a self-contained node that loads SDNQ models and generates images in one step. No MODEL/CLIP/VAE outputs.

**Architecture**:
```python
class SDNQSampler:
    """
    All-in-one SDNQ node:
    - Loads model from dropdown/path
    - Takes prompt + sampler params
    - Outputs IMAGE directly
    """

    INPUT_TYPES = {
        "required": {
            "model_selection": (model_options,),
            "prompt": ("STRING", {"multiline": True}),
            "negative_prompt": ("STRING", {"multiline": True}),
            "steps": ("INT", {"default": 25}),
            "cfg": ("FLOAT", {"default": 7.0}),
            "width": ("INT", {"default": 1024}),
            "height": ("INT", {"default": 1024}),
            "seed": ("INT", {"default": 0}),
        }
    }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "sampling/SDNQ"

    def generate(self, model_selection, prompt, negative_prompt,
                 steps, cfg, width, height, seed):
        # Load SDNQ pipeline
        pipe = self._load_pipeline(model_selection)

        # Set seed
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Generate
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        # Convert PIL to ComfyUI tensor format
        return (pil_to_comfy_tensor(image),)
```

**Pros**:
- ✅ Will actually work (proven by yopfix example)
- ✅ Simple implementation (1-2 days)
- ✅ Reliable and maintainable
- ✅ Can add advanced params (LoRA, ControlNet via diffusers API)

**Cons**:
- ❌ Can't reuse ComfyUI's existing sampler nodes
- ❌ Less flexible for complex workflows
- ❌ Need to duplicate sampler parameters

**Estimated Effort**: 2-3 days

**Risk**: LOW

### Option B: ModelPatcher Integration (COMPLEX)

**Description**: Make wrappers inherit from ComfyUI's base classes (ModelPatcher, CLIP, VAE).

**Architecture**:
```python
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP, VAE
from comfy.model_base import BaseModel

class SDNQModelPatcher(ModelPatcher):
    def __init__(self, diffusers_pipeline, model_component):
        # Detect model type and create appropriate BaseModel
        base_model = self._create_base_model(model_component)

        super().__init__(
            model=base_model,
            load_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
        )

        self.pipeline = diffusers_pipeline
        self.model_options["latent_format"] = self._detect_latent_format()

    def _detect_latent_format(self):
        # Return appropriate LatentFormat based on model type
        # FLUX: SD3LatentFormat
        # SDXL: SDXLLatentFormat
        pass
```

**Challenges**:
1. Need to wrap diffusers transformer/unet in ComfyUI's `BaseModel` class
2. Need to implement proper `latent_format` detection
3. Need to ensure quantized weights survive the wrapping
4. Need to handle clone() operations without breaking SDNQ

**Research Needed**:
- Study [ComfyUI-DiffusersLoader](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader) source
- Understand how state dicts are extracted and reconstructed
- Test if SDNQ weights survive state dict operations

**Estimated Effort**: 1-2 weeks

**Risk**: HIGH (may not be possible without extensive monkeypatching)

### Option C: Hybrid Approach

**Description**: Provide both a standalone sampler AND limited MODEL/CLIP/VAE outputs for experimental use.

**Architecture**:
```python
class SDNQLoader:
    """
    Loads SDNQ model and provides:
    1. IMAGE output (standalone generation)
    2. MODEL/CLIP/VAE outputs (experimental, may not work with all nodes)
    """

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE")

    # Document limitations clearly in UI
```

**Pros**:
- ✅ Provides working solution immediately (IMAGE output)
- ✅ Allows experimentation with ComfyUI nodes
- ✅ Users can choose based on their needs

**Cons**:
- ❌ Confusing to users (which output to use?)
- ❌ Experimental outputs may fail unpredictably

**Estimated Effort**: 3-4 days

**Risk**: MEDIUM

---

## Recommended Implementation Plan

### Phase 1: Prove It Works (Days 1-2)

**Goal**: Get SDNQ generating images in ComfyUI, even if not integrated with other nodes.

**Tasks**:
1. Create `SDNQSampler` node (standalone, all-in-one)
2. Implement basic parameters (prompt, steps, cfg, seed, size)
3. Test with FLUX.1, FLUX.2, Z-Image models
4. Verify images generate successfully

**Success Criteria**: Can generate images with SDNQ models in ComfyUI.

### Phase 2: Fix Known Bugs (Day 3)

**Goal**: Fix confirmed issues in current code.

**Tasks**:
1. Fix device parameter usage:
   ```python
   pipeline = DiffusionPipeline.from_pretrained(
       model_path,
       torch_dtype=torch_dtype,
       device_map="auto" if device == "auto" else device,
   )
   ```

2. Research and fix attn_implementation parameter:
   - Verify correct parameter name in diffusers 0.36.0.dev0
   - Test with different pipeline classes
   - Document which pipelines support it

3. Handle torch.compile errors more gracefully:
   - Add better error messages
   - Provide clear fallback behavior

**Success Criteria**: No warnings/errors during model loading.

### Phase 3: Advanced Features (Days 4-5)

**Goal**: Add features users expect from SDNQ.

**Tasks**:
1. Add LoRA support (via diffusers `load_lora_weights()`)
2. Add batch generation
3. Add img2img support
4. Add inpainting support
5. Add video model support (I2V, T2V)

**Success Criteria**: Feature parity with standalone SDNQ usage.

### Phase 4: Documentation & Polish (Day 6)

**Goal**: Make the node pack production-ready.

**Tasks**:
1. Update README with clear usage examples
2. Document limitations (can't use with KSampler, etc.)
3. Create example workflows
4. Add error handling and user-friendly messages
5. Update context.md with lessons learned

**Success Criteria**: Users can install and use without confusion.

---

## Decision Point: Integration Worth It?

**Question**: Should we attempt Option B (ModelPatcher integration) after Phase 1-4?

**Factors to Consider**:

**For Integration**:
- Users expect ComfyUI nodes to work with KSampler
- Would enable complex workflows (ControlNet, IPAdapter, etc.)
- More "native" ComfyUI experience

**Against Integration**:
- High implementation complexity (1-2 weeks)
- High risk of failure (may not be possible)
- High maintenance burden (breaks with ComfyUI updates)
- Standalone sampler works perfectly fine

**Recommendation**:
1. Ship Phase 1-4 first (standalone sampler)
2. Gather user feedback
3. If users strongly demand KSampler integration, then attempt Option B
4. Otherwise, iterate on standalone sampler features

---

## Reality Check: What I Got Wrong

### Hallucinations Identified

1. **"AutoPipeline was removed in 0.36.0"** - WRONG. It still exists.
2. **"attn_implementation works on all pipelines"** - PARTIALLY WRONG. It works on most, but not verified for all.
3. **"Wrappers will work if we just implement the right methods"** - WRONG. Need to inherit from ComfyUI base classes.
4. **"torch.compile errors are just warnings"** - WRONG. They indicate actual runtime failures.

### How I'll Avoid This

1. **Always fetch actual source code** before making claims
2. **Test assumptions** before implementing
3. **Read error messages carefully** instead of assuming
4. **Study working examples** (like yopfix's standalone code)

---

## Next Steps

**If continuing with current session**:
1. Implement Phase 1 (SDNQSampler node)
2. Test with real models
3. Verify images generate

**If starting fresh session**:
1. Read this ASSESSMENT.md file
2. Follow the implementation plan
3. Start with Phase 1 (standalone sampler)

---

## References (All Verified)

- [SDNQ Repository](https://github.com/Disty0/sdnq) - Actual SDNQ implementation
- [diffusers Documentation](https://huggingface.co/docs/diffusers) - Current API reference
- [ComfyUI ModelPatcher](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py) - What MODEL must be
- [GitHub Issue #14](https://github.com/EnragedAntelope/comfyui-sdnq/issues/14) - User-reported problems
- [yopfix's working example](https://github.com/EnragedAntelope/comfyui-sdnq/issues/14#issuecomment-2524485963) - Proven code

---

**Bottom Line**: The standalone sampler approach (Option A) is the most realistic path to a working node pack. ModelPatcher integration is theoretically possible but high-risk and may not be worth the effort.
