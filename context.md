# context.md - ComfyUI-SDNQ Development Context

> **IMPORTANT**: Update this file after every significant change or discovery!

## Project Status

**Current Phase**: Not Started
**Last Updated**: [UPDATE THIS]
**Overall Progress**: 0%

---

## Quick Reference

### Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Test in ComfyUI (restart server after changes)
# Node should appear under loaders/SDNQ

# Run a quick SDNQ test
python -c "from sdnq import SDNQConfig; print('SDNQ imported successfully')"
```

### Key Files
- `CLAUDE.md` - Development guide (read-only reference)
- `nodes/loader.py` - Main loader node
- `core/wrapper.py` - ComfyUI type wrappers

---

## Completed Tasks

- [ ] Phase 1: Project setup
- [ ] Phase 1: Core wrapper implementation
- [ ] Phase 1: Basic loader node
- [ ] Phase 1: Test with real model
- [ ] Phase 2: Model registry
- [ ] Phase 2: HuggingFace downloader
- [ ] Phase 2: Catalog dropdown
- [ ] Phase 3: Quantization node
- [ ] Phase 3: V3 API schemas

---

## Current Blockers

*None yet - project not started*

---

## Lessons Learned

*Document discoveries here as you work*

### ComfyUI Type System
- [What did you learn about MODEL, CLIP, VAE types?]

### SDNQ Integration
- [What worked? What didn't?]

### Diffusers Pipeline
- [How do pipeline components map to ComfyUI?]

---

## Code Snippets to Remember

```python
# Add useful code patterns you discover
```

---

## Future TODOs

- [ ] Support for LoRA with SDNQ models
- [ ] Batch quantization of multiple checkpoints
- [ ] Memory usage reporting
- [ ] Integration with ComfyUI Manager for one-click install

---

## Testing Notes

### Models Tested
| Model | Version | Status | Notes |
|-------|---------|--------|-------|
| FLUX.1-dev-qint8 | - | ❓ | Not tested yet |

### Environments Tested
| OS | Python | PyTorch | CUDA | Status |
|----|--------|---------|------|--------|
| - | - | - | - | Not tested yet |

---

## Session Log

### [DATE] - Session 1
**Goal**: [What you planned to do]
**Achieved**: [What you actually did]
**Issues**: [Problems encountered]
**Next**: [What to do next session]

---

*Remember: This file is your memory between sessions. Future you will thank present you for detailed notes!*
