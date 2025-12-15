"""
SDNQ LoRA Loader Node - V3 API Compatible

This node loads LoRA weights into SDNQ pipelines (diffusers DiffusionPipeline).
Separated from sampler for better modularity.

Architecture: MODEL Input → Load LoRA → Output MODEL (with LoRA applied)
"""

import os
from typing import Tuple

# ComfyUI imports for LoRA folder access
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("[SDNQ LoRA Loader] Warning: folder_paths not available - LoRA dropdown will be disabled")

# diffusers pipeline type hint
from diffusers import DiffusionPipeline


class SDNQLoraLoader:
    """
    SDNQ LoRA loader that applies LoRA weights to diffusers pipelines.
    
    Uses diffusers standard API (pipeline.load_lora_weights, set_adapters).
    Compatible with FLUX, SDXL, SD3, and other diffusers-based models.
    """

    def __init__(self):
        """Initialize loader."""
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs following ComfyUI V3 conventions with V1 compatibility.
        Supports up to 10 LoRA slots for stacking multiple LoRAs.
        """
        # Get available LoRAs from ComfyUI loras folder
        loras = ["None"]
        if COMFYUI_AVAILABLE:
            try:
                available_loras = folder_paths.get_filename_list("loras")
                loras.extend(available_loras)
            except Exception as e:
                print(f"[SDNQ LoRA Loader] Warning: Could not load LoRA list: {e}")

        # Build inputs for 10 LoRA slots (lora_name_1, lora_wt_1, ... lora_name_10, lora_wt_10)
        inputs = {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "SDNQ model pipeline from SDNQ Model Loader node"
                }),
            },
            "optional": {}
        }
        
        # Add 10 LoRA slots as optional (JS will control visibility)
        for i in range(1, 11):
            inputs["optional"][f"lora_name_{i}"] = (loras, {"tooltip": f"LoRA {i} filename"})
            inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})
        
        return inputs

    # V3 API: Return type hints
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    # V1 API: Function name
    FUNCTION = "load_lora"

    # Category for node menu
    CATEGORY = "loaders/SDNQ"

    # V3 API: Output node (can save/display results)
    OUTPUT_NODE = False

    # V3 API: Node description
    DESCRIPTION = "Load LoRA weights into SDNQ model pipeline. Uses diffusers standard API."

    def _resolve_lora_path(self, lora_selection: str) -> str:
        """
        Resolve actual LoRA path from selection.
        
        Args:
            lora_selection: Selected LoRA from dropdown (None or filename)
            
        Returns:
            Resolved LoRA path or None if no LoRA selected
        """
        if not lora_selection or lora_selection == "None":
            return None
        
        # User selected a LoRA from the dropdown
        if COMFYUI_AVAILABLE:
            try:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_selection)
                return lora_path
            except Exception as e:
                print(f"[SDNQ LoRA Loader] Warning: Could not resolve LoRA path: {e}")
                return None
        return None

    def _load_lora_weights(self, pipeline: DiffusionPipeline, lora_path: str, lora_strength: float = 1.0):
        """
        Load LoRA weights into pipeline.
        
        Supports both local .safetensors files and HuggingFace repo IDs.
        
        Args:
            pipeline: Loaded diffusers pipeline
            lora_path: Path to LoRA file or HuggingFace repo ID
            lora_strength: LoRA influence strength (0.0 to 2.0)
            
        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/api/loaders/lora
        https://huggingface.co/blog/lora-fast
        """
        if not lora_path or lora_path.strip() == "":
            print(f"[SDNQ LoRA Loader] No LoRA specified, skipping...")
            return

        print(f"[SDNQ LoRA Loader] Loading LoRA...")
        print(f"[SDNQ LoRA Loader]   Path: {lora_path}")
        print(f"[SDNQ LoRA Loader]   Strength: {lora_strength}")

        try:
            # Check if it's a local file or HuggingFace repo
            is_local_file = os.path.exists(lora_path) and os.path.isfile(lora_path)

            if is_local_file:
                # Local .safetensors file
                # Extract directory and filename
                lora_dir = os.path.dirname(lora_path)
                lora_file = os.path.basename(lora_path)

                pipeline.load_lora_weights(
                    lora_dir,
                    weight_name=lora_file,
                    adapter_name="lora"
                )
            else:
                # Assume it's a HuggingFace repo ID
                pipeline.load_lora_weights(
                    lora_path,
                    adapter_name="lora"
                )

            # Set LoRA strength
            if lora_strength != 1.0:
                pipeline.set_adapters(["lora"], adapter_weights=[lora_strength])
            else:
                pipeline.set_adapters(["lora"])

            print(f"[SDNQ LoRA Loader] ✓ LoRA loaded successfully")

        except Exception as e:
            raise Exception(
                f"Failed to load LoRA\n\n"
                f"Error: {str(e)}\n\n"
                f"LoRA path: {lora_path}\n\n"
                f"Troubleshooting:\n"
                f"1. Verify LoRA file exists (.safetensors format)\n"
                f"2. For HuggingFace repos, verify repo ID is correct\n"
                f"3. Ensure LoRA is compatible with the model architecture\n"
                f"4. Check if LoRA is for the correct model type (FLUX, SDXL, etc.)\n"
                f"5. Try with lora_strength=1.0 first"
            )

    def load_lora(self, model: DiffusionPipeline, **kwargs) -> Tuple[DiffusionPipeline]:
        """
        Main function called by ComfyUI.
        
        Supports up to 10 LoRA slots (lora_name_1, lora_wt_1, ... lora_name_10, lora_wt_10).
        Multiple LoRAs are loaded sequentially and combined.
        
        Args:
            model: DiffusionPipeline from SDNQModelLoader
            **kwargs: LoRA parameters (lora_name_1, lora_wt_1, ... lora_name_10, lora_wt_10)
            
        Returns:
            Tuple containing (MODEL,) with LoRAs applied
        """
        # Process up to 10 LoRA slots
        lora_adapters = []
        lora_weights = []
        
        for i in range(1, 11):
            lora_name_key = f"lora_name_{i}"
            lora_wt_key = f"lora_wt_{i}"
            
            if lora_name_key not in kwargs or lora_wt_key not in kwargs:
                continue
            
            lora_selection = kwargs.get(lora_name_key)
            lora_strength = kwargs.get(lora_wt_key, 1.0)
            
            # Skip if no LoRA selected or strength is 0
            if not lora_selection or lora_selection == "None" or abs(lora_strength) < 1e-5:
                continue
            
            # Resolve LoRA path
            lora_path = self._resolve_lora_path(lora_selection)
            
            if lora_path and lora_path.strip():
                try:
                    # Check if it's a local file or HuggingFace repo
                    is_local_file = os.path.exists(lora_path) and os.path.isfile(lora_path)
                    
                    adapter_name = f"lora_{i}"
                    
                    if is_local_file:
                        # Local .safetensors file
                        lora_dir = os.path.dirname(lora_path)
                        lora_file = os.path.basename(lora_path)
                        
                        model.load_lora_weights(
                            lora_dir,
                            weight_name=lora_file,
                            adapter_name=adapter_name
                        )
                    else:
                        # Assume it's a HuggingFace repo ID
                        model.load_lora_weights(
                            lora_path,
                            adapter_name=adapter_name
                        )
                    
                    lora_adapters.append(adapter_name)
                    lora_weights.append(lora_strength)
                    
                    print(f"[SDNQ LoRA Loader] ✓ LoRA {i} loaded: {lora_selection} (strength: {lora_strength})")
                    
                except Exception as e:
                    print(f"[SDNQ LoRA Loader] ⚠️  Failed to load LoRA {i} ({lora_selection}): {e}")
                    continue
        
        # Set all adapters with their weights
        if lora_adapters:
            model.set_adapters(lora_adapters, adapter_weights=lora_weights)
            print(f"[SDNQ LoRA Loader] ✓ {len(lora_adapters)} LoRA(s) applied to pipeline")
        else:
            print(f"[SDNQ LoRA Loader] No LoRAs to load")
        
        return (model,)

