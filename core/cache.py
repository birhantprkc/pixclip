"""
core/cache.py
PixClip local disk caching module.
Manages temp/ directory structure, parameter hashing, image saving/loading,
and automatic cache cleanup.
"""

from __future__ import annotations
import os
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict

from core.params import AdjustmentParams

# Base temp directory is in the project root
TEMP_DIR = Path(__file__).parent.parent / "temp"

THUMB_DIR = TEMP_DIR / "thumbnails"
PREVIEW_DIR = TEMP_DIR / "previews"
EDITED_DIR = TEMP_DIR / "edited"
METADATA_DIR = TEMP_DIR / "metadata"


def init_cache() -> None:
    """Initialize the cache directories and clean up all previous temp files."""
    for d in [THUMB_DIR, PREVIEW_DIR, EDITED_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    clear_all_cache()


def clear_all_cache() -> None:
    """Clear all cached files (thumbnails, previews, edits, metadata) for a clean startup."""
    for folder in [THUMB_DIR, PREVIEW_DIR, EDITED_DIR, METADATA_DIR]:
        if folder.exists():
            for f in folder.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                except Exception:
                    pass


def get_params_hash(params: AdjustmentParams) -> str:
    """
    Generate a deterministic MD5 hash of AdjustmentParams.
    """
    d = params.to_dict()
    # Sort keys to ensure deterministic JSON representation
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ── Thumbnails ────────────────────────────────────────────────────────────

def get_cached_thumbnail(image_id: str) -> Optional[np.ndarray]:
    """Retrieve the cached thumbnail for an image, or None if it doesn't exist."""
    path = THUMB_DIR / f"{image_id}.jpg"
    if path.exists():
        try:
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception:
            return None
    return None


def save_cached_thumbnail(image_id: str, arr: np.ndarray) -> bool:
    """Save the thumbnail image to the disk cache."""
    try:
        path = THUMB_DIR / f"{image_id}.jpg"
        cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception:
        return False


# ── Previews ──────────────────────────────────────────────────────────────

def get_cached_preview(image_id: str, params: AdjustmentParams) -> Optional[np.ndarray]:
    """Retrieve a cached preview if it matches the current parameters."""
    h = get_params_hash(params)
    path = PREVIEW_DIR / f"{image_id}_{h}.jpg"
    if path.exists():
        try:
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception:
            return None
    return None


def save_cached_preview(image_id: str, params: AdjustmentParams, arr: np.ndarray) -> bool:
    """Save the rendered preview image to the disk cache."""
    try:
        h = get_params_hash(params)
        path = PREVIEW_DIR / f"{image_id}_{h}.jpg"
        cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception:
        return False


# ── Edited Outputs ────────────────────────────────────────────────────────

def get_cached_edited(image_id: str, params: AdjustmentParams) -> Optional[np.ndarray]:
    """Retrieve a cached high-resolution edited image."""
    h = get_params_hash(params)
    path = EDITED_DIR / f"{image_id}_{h}.jpg"
    if path.exists():
        try:
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception:
            return None
    return None


def save_cached_edited(image_id: str, params: AdjustmentParams, arr: np.ndarray) -> bool:
    """Save a high-resolution edited image to the disk cache."""
    try:
        h = get_params_hash(params)
        path = EDITED_DIR / f"{image_id}_{h}.jpg"
        cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception:
        return False


# ── Project State Metadata ────────────────────────────────────────────────

def save_project_metadata(state_dict: dict) -> bool:
    """Save the project state metadata to state.json."""
    try:
        path = METADATA_DIR / "state.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        return True
    except Exception:
        return False


def load_project_metadata() -> Optional[dict]:
    """Load the project state metadata if it exists."""
    path = METADATA_DIR / "state.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ── General Cache Housekeeping ────────────────────────────────────────────

def clean_old_caches(max_files: int = 500) -> None:
    """
    Clean up older files in the previews directory if the count exceeds max_files
    using Least Recently Used (LRU) / oldest creation time.
    """
    try:
        files = sorted(PREVIEW_DIR.glob("*.jpg"), key=os.path.getmtime)
        if len(files) > max_files:
            for f in files[:-max_files]:
                f.unlink()
    except Exception:
        pass
