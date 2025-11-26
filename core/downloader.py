"""
HuggingFace Hub Downloader

Phase 2: This module will contain:
- Download management using huggingface_hub
- Progress callbacks for UI integration
- Caching and resume support
- Bandwidth throttling options
"""

from typing import Optional, Callable
from huggingface_hub import snapshot_download, hf_hub_download


def download_model(
    repo_id: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    force_download: bool = False
) -> str:
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Disty0/FLUX.1-dev-qint8")
        cache_dir: Optional custom cache directory
        progress_callback: Optional callback function for progress updates
        force_download: Force re-download even if cached

    Returns:
        Path to the downloaded model directory

    TODO: Implement in Phase 2
    """
    # Placeholder implementation
    print(f"Downloading {repo_id} from HuggingFace Hub...")

    # Use snapshot_download to get the entire model
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        force_download=force_download,
        # TODO: Add progress callback integration
    )

    return local_path


def check_model_cached(repo_id: str, cache_dir: Optional[str] = None) -> bool:
    """
    Check if a model is already cached locally.

    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Optional custom cache directory

    Returns:
        True if model is cached, False otherwise

    TODO: Implement in Phase 2
    """
    # Placeholder - always return False for now
    return False


def get_model_size(repo_id: str) -> Optional[int]:
    """
    Get the size of a model in bytes before downloading.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Size in bytes or None if unavailable

    TODO: Implement in Phase 2 using HF API
    """
    return None


# TODO: Phase 2 features to implement:
# - Progress bar integration with ComfyUI
# - Download queue management for multiple models
# - Bandwidth throttling
# - Resume interrupted downloads
# - Verify model integrity after download
