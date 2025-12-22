import torch
import logging
import copy

logger = logging.getLogger(__name__)


class Flux2SDNQTorchCompile:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode (may conflict with accelerate hooks)"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
                    "single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                },
                "optional": {
                    "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                    "force_parameter_static_shapes": ("BOOLEAN", {"default": True, "tooltip": "torch._dynamo.config.force_parameter_static_shapes"}),
                }
                }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "SDNQ/torchcompile"
    EXPERIMENTAL = True
    DISPLAY_NAME = "Flux2 SDNQ TorchCompile"

    def patch(self, model, backend, mode, fullgraph, single_blocks, double_blocks, dynamic, dynamo_cache_size_limit=64, force_parameter_static_shapes=True):
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        torch._dynamo.config.force_parameter_static_shapes = force_parameter_static_shapes

        # Skip accelerate.hooks during torch._dynamo tracing to avoid AttributeError
        # This prevents torch._dynamo from tracing accelerate hooks that cause 'dict' object has no attribute 'node' errors
        # Note: This does NOT disable accelerate functionality, only prevents dynamo from tracing it
        # Save original state and restore after compilation to avoid affecting other code
        import torch._dynamo.config as dynamo_config
        original_skipfiles = None
        original_skip_files = None
        skipfiles_modified = False
        skip_files_modified = False
        
        if hasattr(dynamo_config, 'skipfiles'):
            original_skipfiles = copy.copy(dynamo_config.skipfiles)
            if isinstance(dynamo_config.skipfiles, set):
                if "accelerate.hooks" not in dynamo_config.skipfiles:
                    dynamo_config.skipfiles.add("accelerate.hooks")
                    skipfiles_modified = True
            elif isinstance(dynamo_config.skipfiles, list):
                if "accelerate.hooks" not in dynamo_config.skipfiles:
                    dynamo_config.skipfiles.append("accelerate.hooks")
                    skipfiles_modified = True
        # Also try skip_files (alternative attribute name in some torch versions)
        if hasattr(dynamo_config, 'skip_files'):
            original_skip_files = copy.copy(dynamo_config.skip_files)
            if isinstance(dynamo_config.skip_files, set):
                if "accelerate.hooks" not in dynamo_config.skip_files:
                    dynamo_config.skip_files.add("accelerate.hooks")
                    skip_files_modified = True
            elif isinstance(dynamo_config.skip_files, list):
                if "accelerate.hooks" not in dynamo_config.skip_files:
                    dynamo_config.skip_files.append("accelerate.hooks")
                    skip_files_modified = True

        try:
            # Check if model is a DiffusionPipeline (from comfyui-sdnq-splited)
            is_pipeline = False
            pipeline = None
            
            # Check if model is a diffusers pipeline
            try:
                from diffusers import DiffusionPipeline
                if isinstance(model, DiffusionPipeline):
                    is_pipeline = True
                    pipeline = model
                    logger.info(f"[Flux2SDNQTorchCompile] Detected DiffusionPipeline: {type(pipeline).__name__}")
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"[Flux2SDNQTorchCompile] Not a DiffusionPipeline: {e}")
            
            if is_pipeline:
                # DiffusionPipeline (Flux2Pipeline from SDNQ): compile transformer.transformer_blocks directly
                if not hasattr(pipeline, 'transformer') or pipeline.transformer is None:
                    raise RuntimeError("Pipeline does not have transformer attribute. This node only supports Flux models.")
                
                transformer = pipeline.transformer
                
                # Check if transformer has transformer_blocks (Flux2 architecture)
                if hasattr(transformer, 'transformer_blocks'):
                    transformer_blocks = transformer.transformer_blocks
                    if transformer_blocks is not None and len(transformer_blocks) > 0:
                        logger.info(f"[Flux2SDNQTorchCompile] Found {len(transformer_blocks)} transformer blocks in Flux2Pipeline")
                        
                        # Compile entire transformer module (more efficient than individual blocks)
                        # Check if transformer is already compiled
                        transformer_compiled = hasattr(transformer, '_orig_mod') or (hasattr(transformer, 'forward') and hasattr(transformer.forward, '_orig_mod'))
                        
                        if single_blocks or double_blocks:  # Either flag enables compilation
                            if not transformer_compiled:
                                logger.info(f"[Flux2SDNQTorchCompile] Compiling entire transformer module (recommended for best performance)")
                                logger.info(f"[Flux2SDNQTorchCompile] Compilation settings: backend={backend}, mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")
                                try:
                                    compiled_transformer = torch.compile(
                                        transformer,
                                        backend=backend,
                                        mode=mode,
                                        dynamic=dynamic,
                                        fullgraph=fullgraph
                                    )
                                    # Update the pipeline's transformer reference
                                    pipeline.transformer = compiled_transformer
                                    
                                    # Verify compilation succeeded
                                    if hasattr(compiled_transformer, '_orig_mod') or (hasattr(compiled_transformer, 'forward') and hasattr(compiled_transformer.forward, '_orig_mod')):
                                        logger.info(f"[Flux2SDNQTorchCompile] Successfully compiled entire transformer module - compilation verified")
                                    else:
                                        logger.warning(f"[Flux2SDNQTorchCompile] Compilation completed but _orig_mod not found - may not be effective")
                                    
                                except Exception as e:
                                    logger.error(f"[Flux2SDNQTorchCompile] Failed to compile entire transformer: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise RuntimeError(f"Failed to compile entire transformer: {str(e)}")
                            else:
                                logger.info("[Flux2SDNQTorchCompile] Transformer is already compiled, skipping recompilation")
                        else:
                            logger.warning("[Flux2SDNQTorchCompile] Both single_blocks and double_blocks are disabled. No blocks will be compiled.")
                    else:
                        raise RuntimeError("transformer.transformer_blocks is empty or None")
                else:
                    raise RuntimeError("transformer does not have transformer_blocks attribute. This may not be a Flux2 model.")
                
                # Return the pipeline directly (no clone needed)
                return (pipeline,)
            
            else:
                # Standard ComfyUI ModelPatcher: use clone() and get_model_object
                from comfy_api.torch_helpers import set_torch_compile_wrapper
                m = model.clone()
                diffusion_model = m.get_model_object("diffusion_model")
                
                compile_key_list = []
                
                # Check if this is a Nunchaku Flux2 model (ComfyFluxWrapper)
                # Nunchaku models have ComfyFluxWrapper with model.transformer_blocks
                is_nunchaku = False
                nunchaku_model = None
                
                # Check if diffusion_model is ComfyFluxWrapper (has 'model' attribute with transformer_blocks)
                if hasattr(diffusion_model, 'model') and hasattr(diffusion_model.model, 'transformer_blocks'):
                    try:
                        # Try to access transformer_blocks to confirm it's a Nunchaku model
                        transformer_blocks = diffusion_model.model.transformer_blocks
                        if transformer_blocks is not None and len(transformer_blocks) > 0:
                            is_nunchaku = True
                            nunchaku_model = diffusion_model.model
                            logger.info(f"[Flux2SDNQTorchCompile] Detected Nunchaku Flux2 model with {len(transformer_blocks)} transformer blocks")
                    except Exception as e:
                        logger.debug(f"[Flux2SDNQTorchCompile] Not a Nunchaku model (transformer_blocks check failed): {e}")
                
                if is_nunchaku:
                    # Nunchaku Flux2 model: compile entire transformer model directly
                    if single_blocks or double_blocks:  # Either flag enables compilation for Nunchaku
                        # Check if transformer model is already compiled
                        nunchaku_compiled = hasattr(nunchaku_model, '_orig_mod') or (hasattr(nunchaku_model, 'forward') and hasattr(nunchaku_model.forward, '_orig_mod'))
                        
                        if not nunchaku_compiled:
                            logger.info(f"[Flux2SDNQTorchCompile] Compiling entire Nunchaku transformer model (recommended for best performance)")
                            logger.info(f"[Flux2SDNQTorchCompile] Compilation settings: backend={backend}, mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")
                            try:
                                compiled_nunchaku = torch.compile(
                                    nunchaku_model,
                                    backend=backend,
                                    mode=mode,
                                    dynamic=dynamic,
                                    fullgraph=fullgraph
                                )
                                # Update the diffusion_model's model reference
                                diffusion_model.model = compiled_nunchaku
                                
                                # Verify compilation succeeded
                                if hasattr(compiled_nunchaku, '_orig_mod') or (hasattr(compiled_nunchaku, 'forward') and hasattr(compiled_nunchaku.forward, '_orig_mod')):
                                    logger.info(f"[Flux2SDNQTorchCompile] Successfully compiled entire Nunchaku transformer model - compilation verified")
                                else:
                                    logger.warning(f"[Flux2SDNQTorchCompile] Compilation completed but _orig_mod not found - may not be effective")
                                
                            except Exception as e:
                                logger.error(f"[Flux2SDNQTorchCompile] Failed to compile Nunchaku transformer model: {e}")
                                import traceback
                                traceback.print_exc()
                                raise RuntimeError(f"Failed to compile Nunchaku transformer model: {str(e)}")
                        else:
                            logger.info("[Flux2SDNQTorchCompile] Nunchaku transformer model is already compiled, skipping recompilation")
                    else:
                        logger.warning("[Flux2SDNQTorchCompile] Both single_blocks and double_blocks are disabled. No compilation will be performed.")
                else:
                    # Standard ComfyUI Flux model: use set_torch_compile_wrapper for compatibility
                    if double_blocks:
                        if hasattr(diffusion_model, 'double_blocks'):
                            for i, block in enumerate(diffusion_model.double_blocks):
                                print(f"[Flux2SDNQTorchCompile] Adding double block {i} to compile list")
                                compile_key_list.append(f"diffusion_model.double_blocks.{i}")
                        else:
                            logger.warning("[Flux2SDNQTorchCompile] double_blocks not found in diffusion_model")
                    
                    if single_blocks:
                        if hasattr(diffusion_model, 'single_blocks'):
                            for i, block in enumerate(diffusion_model.single_blocks):
                                print(f"[Flux2SDNQTorchCompile] Adding single block {i} to compile list")
                                compile_key_list.append(f"diffusion_model.single_blocks.{i}")
                        else:
                            logger.warning("[Flux2SDNQTorchCompile] single_blocks not found in diffusion_model")

                    if len(compile_key_list) == 0:
                        raise RuntimeError("No blocks found to compile. Check if model is a Flux model and blocks are accessible.")

                    logger.info(f"[Flux2SDNQTorchCompile] Compiling {len(compile_key_list)} blocks using set_torch_compile_wrapper")
                    set_torch_compile_wrapper(model=m, keys=compile_key_list, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
                
                return (m,)
                
        except Exception as e:
            logger.error(f"[Flux2SDNQTorchCompile] Failed to compile model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to compile model: {str(e)}")
        finally:
            # Restore original skipfiles state to avoid affecting other code
            if skipfiles_modified and original_skipfiles is not None:
                dynamo_config.skipfiles = original_skipfiles
            if skip_files_modified and original_skip_files is not None:
                dynamo_config.skip_files = original_skip_files
        # rest of the layers that are not patched
        # diffusion_model.final_layer = torch.compile(diffusion_model.final_layer, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.guidance_in = torch.compile(diffusion_model.guidance_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.img_in = torch.compile(diffusion_model.img_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.time_in = torch.compile(diffusion_model.time_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.txt_in = torch.compile(diffusion_model.txt_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.vector_in = torch.compile(diffusion_model.vector_in, mode=mode, fullgraph=fullgraph, backend=backend)

