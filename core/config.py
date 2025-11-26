"""
Configuration helpers for SDNQ integration
"""

import torch
from typing import Optional


def get_dtype_from_string(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.

    Args:
        dtype_str: String representation ("bfloat16", "float16", "float32")

    Returns:
        torch.dtype object

    Raises:
        ValueError: If dtype_str is not supported
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. "
            f"Supported values: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def get_device_map(device_str: str) -> Optional[str]:
    """
    Convert device string to device map for model loading.

    Args:
        device_str: Device selection ("auto", "cuda", "cpu")

    Returns:
        Device map string or None
    """
    device_map = {
        "auto": "auto",
        "cuda": "cuda",
        "cpu": "cpu",
    }

    return device_map.get(device_str, "auto")
