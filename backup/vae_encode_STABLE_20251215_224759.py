"""
SDNQ VAEEncode Node - V3 API Compatible

Custom VAEEncode node for diffusers VAE compatibility.
Handles format conversion between ComfyUI and diffusers.
"""

import torch
import numpy as np
from typing import Tuple


class SDNQVAEEncode:
    """
    VAEEncode node for SDNQ diffusers VAE.
    
    Handles:
    - Input format conversion: [N, H, W, C] (ComfyUI) -> [N, C, H, W] (diffusers)
    - Output format: [N, C, H, W] latent tensor
    - dtype conversion: float32 for compatibility
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        return {
            "required": {
                "pixels": ("IMAGE", {
                    "tooltip": "Image tensor in ComfyUI format [N, H, W, C]"
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE model (diffusers VAE wrapped in ComfyVAEWrapper or standard ComfyUI VAE)"
                }),
            }
        }
    
    # V3 API: Return type hints
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    
    # V1 API: Function name
    FUNCTION = "encode"
    
    # Category for node menu
    CATEGORY = "latent/SDNQ"
    
    # V3 API: Output node (can save/display results)
    OUTPUT_NODE = False
    
    # V3 API: Node description
    DESCRIPTION = "Encode image to latent space using SDNQ VAE. Compatible with diffusers VAE."
    
    def encode(self, pixels: torch.Tensor, vae) -> Tuple[dict]:
        """
        Encode pixels to latent space.
        
        Args:
            pixels: Image tensor in ComfyUI format [N, H, W, C] (float32, 0.0-1.0)
            vae: VAE model (ComfyVAEWrapper or standard ComfyUI VAE)
        
        Returns:
            Tuple containing dict with "samples" key (latent tensor [N, C, H, W])
        """
        # Patch VAE.decode once to force float32 input (avoid bfloat16/float bias mismatch at decode)
        if hasattr(vae, "decode") and not getattr(vae, "_sdnq_decode_patched", False):
            try:
                original_decode = vae.decode

                def decode_float32(z, *args, **kwargs):
                    return original_decode(z.float(), *args, **kwargs)

                vae.decode = decode_float32
                vae._sdnq_decode_patched = True
            except Exception:
                pass
        # Also patch underlying diffusers VAE if wrapped
        if hasattr(vae, "vae") and hasattr(vae.vae, "decode") and not getattr(vae.vae, "_sdnq_decode_patched", False):
            try:
                original_decode_inner = vae.vae.decode

                def decode_float32_inner(z, *args, **kwargs):
                    return original_decode_inner(z.float(), *args, **kwargs)

                vae.vae.decode = decode_float32_inner
                vae.vae._sdnq_decode_patched = True
            except Exception:
                pass

        # Extract RGB channels (ignore alpha if present)
        pixels_rgb = pixels[:, :, :, :3]
        
        # Check if VAE is wrapped (has encode method that handles format conversion)
        if hasattr(vae, 'encode'):
            # Use VAE's encode method (handles format conversion internally)
            latent = vae.encode(pixels_rgb)
        else:
            # Fallback: assume it's a diffusers VAE and convert format manually
            # Convert from [N, H, W, C] to [N, C, H, W]
            pixels_nchw = pixels_rgb.permute(0, 3, 1, 2)
            
            # Ensure float32 dtype
            pixels_nchw = pixels_nchw.to(dtype=torch.float32)
            
            # Call diffusers VAE encode
            result = vae.encode(pixels_nchw)
            
            # Extract latent tensor from result
            if isinstance(result, dict):
                if "latent_dist" in result:
                    latent = result["latent_dist"].sample()
                elif "latents" in result:
                    latent = result["latents"]
                else:
                    latent = next(iter(result.values()))
            else:
                latent = result
        
        # Return in ComfyUI LATENT format
        # Also include original pixels (RGB) to enable pipelines that consume `image=` conditioning directly (e.g. Flux2Pipeline).
        return ({"samples": latent, "vae": vae, "pixels": pixels_rgb},)

