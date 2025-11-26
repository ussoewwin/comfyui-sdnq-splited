"""
HuggingFace Hub Downloader

Handles downloading SDNQ models from HuggingFace Hub with progress tracking,
caching, and resume support.
"""

import os
import time
from typing import Optional, Callable, Dict
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, try_to_load_from_cache, model_info


class DownloadProgress:
    """Track and display download progress"""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.start_time = time.time()
        self.last_update = time.time()
        self.total_files = 0
        self.completed_files = 0

    def update(self, filename: str = "", current: int = 0, total: int = 0):
        """Update progress"""
        now = time.time()
        # Only print every 2 seconds to avoid spam
        if now - self.last_update > 2.0:
            elapsed = now - self.start_time
            if total > 0:
                percent = (current / total) * 100
                speed_mb = (current / elapsed) / (1024 * 1024) if elapsed > 0 else 0
                print(f"  Downloading {filename}: {percent:.1f}% ({speed_mb:.1f} MB/s)")
            else:
                print(f"  Downloading {filename}...")
            self.last_update = now


def download_model(
    repo_id: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    force_download: bool = False,
    max_workers: int = 8
) -> str:
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Disty0/FLUX.1-dev-qint8")
        cache_dir: Optional custom cache directory (default: HF cache)
        progress_callback: Optional callback function for progress updates
        force_download: Force re-download even if cached
        max_workers: Number of parallel download threads

    Returns:
        Path to the downloaded model directory

    Raises:
        Exception: If download fails
    """
    print(f"\n{'='*60}")
    print(f"Downloading SDNQ Model from HuggingFace Hub")
    print(f"{'='*60}")
    print(f"Repository: {repo_id}")

    # Check if already cached first (unless force_download)
    if not force_download:
        try:
            cached_path = try_to_load_from_cache(
                repo_id=repo_id,
                filename="model_index.json",
                cache_dir=cache_dir
            )
            if cached_path and cached_path != "_not_found_":
                model_dir = os.path.dirname(cached_path)
                print(f"✓ Model already cached at: {model_dir}")
                print(f"{'='*60}\n")
                return model_dir
        except Exception:
            pass

    print("Fetching model info...")
    try:
        info = model_info(repo_id)
        # Estimate size from siblings
        total_size = sum(getattr(sibling, 'size', 0) or 0 for sibling in info.siblings)
        size_gb = total_size / (1024**3)
        print(f"Model size: ~{size_gb:.1f} GB")
        print(f"Files: {len(info.siblings)}")
    except Exception as e:
        print(f"Warning: Could not fetch model info: {e}")
        print("Proceeding with download...")

    print("\nStarting download (this may take a while)...")
    print("Downloads are cached and can be resumed if interrupted.\n")

    progress = DownloadProgress(repo_id)
    start_time = time.time()

    try:
        # Download entire model using snapshot_download
        # This handles caching, resume, and parallel downloads automatically
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
            max_workers=max_workers,
            # HuggingFace Hub handles progress internally
        )

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print(f"\n{'='*60}")
        print(f"✓ Download complete!")
        print(f"{'='*60}")
        print(f"Time: {minutes}m {seconds}s")
        print(f"Location: {local_path}")
        print(f"{'='*60}\n")

        return local_path

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Download failed!")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"1. Check internet connection")
        print(f"2. Verify repo ID: {repo_id}")
        print(f"3. Check HuggingFace Hub status")
        print(f"4. Try again - downloads can resume")
        print(f"{'='*60}\n")
        raise


def check_model_cached(repo_id: str, cache_dir: Optional[str] = None) -> bool:
    """
    Check if a model is already cached locally.

    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Optional custom cache directory

    Returns:
        True if model is cached, False otherwise
    """
    try:
        cached_path = try_to_load_from_cache(
            repo_id=repo_id,
            filename="model_index.json",
            cache_dir=cache_dir
        )
        return cached_path is not None and cached_path != "_not_found_"
    except Exception:
        return False


def get_cached_model_path(repo_id: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """
    Get the path to a cached model.

    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Optional custom cache directory

    Returns:
        Path to cached model directory or None if not cached
    """
    try:
        cached_path = try_to_load_from_cache(
            repo_id=repo_id,
            filename="model_index.json",
            cache_dir=cache_dir
        )
        if cached_path and cached_path != "_not_found_":
            return os.path.dirname(cached_path)
    except Exception:
        pass
    return None


def get_model_size_estimate(repo_id: str) -> Optional[Dict[str, any]]:
    """
    Get estimated size of a model before downloading.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Dictionary with size info or None if unavailable
    """
    try:
        info = model_info(repo_id)
        total_size = sum(getattr(sibling, 'size', 0) or 0 for sibling in info.siblings)

        return {
            "total_bytes": total_size,
            "total_gb": total_size / (1024**3),
            "total_mb": total_size / (1024**2),
            "num_files": len(info.siblings),
        }
    except Exception:
        return None


def download_model_with_status(
    repo_id: str,
    cache_dir: Optional[str] = None
) -> Dict[str, any]:
    """
    Download a model and return detailed status.

    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Optional custom cache directory

    Returns:
        Dictionary with download status and path
    """
    status = {
        "success": False,
        "path": None,
        "was_cached": False,
        "error": None
    }

    try:
        # Check if already cached
        was_cached = check_model_cached(repo_id, cache_dir)
        status["was_cached"] = was_cached

        # Download (or get from cache)
        path = download_model(repo_id, cache_dir, force_download=False)

        status["success"] = True
        status["path"] = path

    except Exception as e:
        status["error"] = str(e)

    return status
