"""
Optimized Attention Processor for FLUX.2

CRITICAL: This processor uses dispatch_attention_fn (NOT self-implemented softmax).
Flux2 is trained with dispatch_attention_fn's internal corrections, so replacing
attention computation breaks Flux2's learned attention distribution and causes noise.

Speed optimization should be done via CFG fusion (sampler side), NOT by replacing attention.
"""

import torch
import torch
from typing import Optional


class OptimizedFlux2ParallelSelfAttnProcessor:
    """
    Optimized version of Flux2ParallelSelfAttnProcessor.
    
    CRITICAL: This processor uses dispatch_attention_fn for correct attention semantics.
    Self-implemented softmax breaks Flux2's learned attention distribution.
    
    This processor acts as a wrapper that ensures proper processor replacement
    while maintaining dispatch_attention_fn for correct attention semantics.
    Speed optimization should be done via CFG fusion (sampler side).
    """
    
    _attention_backend = None
    _parallel_config = None
    
    def __init__(self, use_blockwise_softmax: bool = True, block_size: int = 64):
        """
        Initialize optimized processor.
        
        NOTE: use_blockwise_softmax and block_size are kept for API compatibility
        but are not used. This processor always uses dispatch_attention_fn.
        
        Args:
            use_blockwise_softmax: Unused (kept for API compatibility)
            block_size: Unused (kept for API compatibility)
        """
        # Parameters are kept for API compatibility but not used
        pass
    
    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask: Optional = None,
        image_rotary_emb: Optional = None,
    ):
        """
        Optimized attention computation for FLUX.2 parallel self-attention.
        
        This is a drop-in replacement for Flux2ParallelSelfAttnProcessor
        that uses dispatch_attention_fn for correct attention semantics.
        """
        # Import here to avoid circular imports
        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        
        # Parallel in (QKV + MLP in) projection
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        
        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)
        
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        
        # CRITICAL: Flux2 requires dispatch_attention_fn for correct attention semantics
        # Self-implemented softmax breaks Flux2's learned attention distribution
        # Flux2 is trained with dispatch_attention_fn's internal corrections (scale, accumulation, stabilization)
        # Using dispatch_attention_fn ensures:
        # 1. Correct attention output distribution (matches learned weights)
        # 2. CFG compatibility (attention differences are preserved)
        # 3. Internal corrections (Flash/xformers compatible normalization)
        # 
        # Speed optimization should be done via CFG fusion (sampler side), NOT by replacing attention
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        
        # Handle the feedforward (FF) logic
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        
        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)
        
        return hidden_states
