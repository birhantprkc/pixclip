"""
core/params.py
Core data structures for PixClip — AdjustmentParams, ImageRecord, ProjectState.
Supports a three-level parameter merge: Global → Group → Individual.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


@dataclass
class AdjustmentParams:
    """
    All adjustment parameters for a single image or scope level.
    Ranges mirror the Xiaomi/Redmi editor style (-100 to +100 or 0 to 100).
    """
    # ── Clarity & Sharpness (highest priority) ──────────────────────────────
    clarity: float = 0.0       # 0 → 100  (Local Contrast Enhancement strength)
    sharpness: float = 0.0     # 0 → 100  (Unsharp Mask strength)

    # ── Light & Exposure Controls ────────────────────────────────────────────
    exposure: float = 0.0      # -100 → +100  (EV mapped to ±3 stops)
    brightness: float = 0.0    # -100 → +100  (additive luminance offset)
    contrast: float = 0.0      # -100 → +100  (S-curve pivot at 0.5)
    lightness: float = 0.0     # -100 → +100  (LAB L* channel direct shift)
    highlights: float = 0.0    # -100 → +100  (luminosity-masked bright region)
    shadows: float = 0.0       # -100 → +100  (luminosity-masked dark region)
    light_range: float = 0.0   # -100 → +100  (white point control)
    dark_range: float = 0.0    # -100 → +100  (black point control)

    # ── Style Filter ─────────────────────────────────────────────────────────
    filter_id: str = "none"
    filter_intensity: float = 1.0  # 0.0 → 1.0

    def is_default(self) -> bool:
        """Returns True if all values are at their neutral/default state."""
        default = AdjustmentParams()
        return (
            self.clarity == default.clarity and
            self.sharpness == default.sharpness and
            self.exposure == default.exposure and
            self.brightness == default.brightness and
            self.contrast == default.contrast and
            self.lightness == default.lightness and
            self.highlights == default.highlights and
            self.shadows == default.shadows and
            self.light_range == default.light_range and
            self.dark_range == default.dark_range and
            self.filter_id == default.filter_id
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdjustmentParams":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def copy(self) -> "AdjustmentParams":
        return AdjustmentParams(**asdict(self))


def merge_params(base: AdjustmentParams, override: AdjustmentParams) -> AdjustmentParams:
    """
    Merge two param sets. override values replace base values only when
    they differ from the default (neutral) state — allowing selective overrides.
    The filter is always taken from the most-specific non-'none' level.
    """
    default = AdjustmentParams()
    result = base.copy()

    for f in AdjustmentParams.__dataclass_fields__:
        if f in ("filter_id", "filter_intensity"):
            continue
        override_val = getattr(override, f)
        default_val = getattr(default, f)
        if override_val != default_val:
            setattr(result, f, override_val)

    if override.filter_id != "none":
        result.filter_id = override.filter_id
        result.filter_intensity = override.filter_intensity

    return result


@dataclass
class ImageRecord:
    """Represents a single imported image with its own adjustment state."""
    path: Path
    id: str = ""
    group_id: Optional[str] = None
    params: AdjustmentParams = field(default_factory=AdjustmentParams)

    def __post_init__(self):
        if not self.id:
            # Generate deterministic ID based on path hash
            import hashlib
            self.id = hashlib.md5(str(self.path.resolve()).encode("utf-8")).hexdigest()

    # Runtime-only fields (not serialised to presets)
    _original: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _preview_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _thumbnail: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _preview_source: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _preview_source_resolution: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def original(self) -> Optional[np.ndarray]:
        return self._original

    @original.setter
    def original(self, value: Optional[np.ndarray]) -> None:
        self._original = value
        self._preview_cache = None  # invalidate cache
        self._preview_source = None
        self._preview_source_resolution = None

    @property
    def preview_cache(self) -> Optional[np.ndarray]:
        return self._preview_cache

    @preview_cache.setter
    def preview_cache(self, value: Optional[np.ndarray]) -> None:
        self._preview_cache = value

    @property
    def thumbnail(self) -> Optional[np.ndarray]:
        return self._thumbnail

    @thumbnail.setter
    def thumbnail(self, value: Optional[np.ndarray]) -> None:
        self._thumbnail = value

    def invalidate_cache(self) -> None:
        self._preview_cache = None
        self._preview_source = None
        self._preview_source_resolution = None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "id": self.id,
            "group_id": self.group_id,
            "params": self.params.to_dict()
        }

    @classmethod
    def from_dict(cls, d: dict) -> ImageRecord:
        record = cls(
            path=Path(d["path"]),
            id=d["id"],
            group_id=d.get("group_id")
        )
        record.params = AdjustmentParams.from_dict(d["params"])
        return record


@dataclass
class VideoRecord:
    """Represents a single imported video with its own adjustment state."""
    path: Path
    id: str = ""
    group_id: Optional[str] = None
    params: AdjustmentParams = field(default_factory=AdjustmentParams)

    # Video metadata
    duration: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if not self.id:
            # Generate deterministic ID based on path hash
            import hashlib
            self.id = hashlib.md5(str(self.path.resolve()).encode("utf-8")).hexdigest()

    # Runtime-only fields
    _thumbnail: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def thumbnail(self) -> Optional[np.ndarray]:
        return self._thumbnail

    @thumbnail.setter
    def thumbnail(self, value: Optional[np.ndarray]) -> None:
        self._thumbnail = value

    def invalidate_cache(self) -> None:
        pass  # Subclass custom caching logic as needed

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "id": self.id,
            "group_id": self.group_id,
            "params": self.params.to_dict(),
            "duration": self.duration,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VideoRecord":
        record = cls(
            path=Path(d["path"]),
            id=d["id"],
            group_id=d.get("group_id"),
            duration=d.get("duration", 0.0),
            fps=d.get("fps", 0.0),
            frame_count=d.get("frame_count", 0),
            width=d.get("width", 0),
            height=d.get("height", 0)
        )
        record.params = AdjustmentParams.from_dict(d["params"])
        return record


class ProjectState:
    """
    Holds the entire editing session: images, videos, global/group/individual params,
    and saved presets. Implements the three-level merge hierarchy.
    """

    def __init__(self):
        self.images: List[ImageRecord] = []
        self.videos: List[VideoRecord] = []
        self.global_params: AdjustmentParams = AdjustmentParams()
        self.group_params: Dict[str, AdjustmentParams] = {}
        self.presets: Dict[str, AdjustmentParams] = {}
        self._selected_ids: List[str] = []
        self._active_id: Optional[str] = None
        self._active_video_id: Optional[str] = None

    # ── Image management ──────────────────────────────────────────────────────

    def add_image(self, record: ImageRecord) -> None:
        self.images.append(record)
        if self._active_id is None:
            self._active_id = record.id

    def remove_image(self, image_id: str) -> None:
        self.images = [img for img in self.images if img.id != image_id]
        if self._active_id == image_id:
            self._active_id = self.images[0].id if self.images else None

    def get_image(self, image_id: str) -> Optional[ImageRecord]:
        for img in self.images:
            if img.id == image_id:
                return img
        return None

    @property
    def active_image(self) -> Optional[ImageRecord]:
        return self.get_image(self._active_id) if self._active_id else None

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    @active_id.setter
    def active_id(self, value: Optional[str]) -> None:
        self._active_id = value

    @property
    def selected_ids(self) -> List[str]:
        return self._selected_ids

    @selected_ids.setter
    def selected_ids(self, value: List[str]) -> None:
        self._selected_ids = value

    # ── Video management ──────────────────────────────────────────────────────

    def add_video(self, record: VideoRecord) -> None:
        self.videos.append(record)
        if self._active_video_id is None:
            self._active_video_id = record.id

    def remove_video(self, video_id: str) -> None:
        self.videos = [vid for vid in self.videos if vid.id != video_id]
        if self._active_video_id == video_id:
            self._active_video_id = self.videos[0].id if self.videos else None

    def get_video(self, video_id: str) -> Optional[VideoRecord]:
        for vid in self.videos:
            if vid.id == video_id:
                return vid
        return None

    @property
    def active_video(self) -> Optional[VideoRecord]:
        return self.get_video(self._active_video_id) if self._active_video_id else None

    @property
    def active_video_id(self) -> Optional[str]:
        return self._active_video_id

    @active_video_id.setter
    def active_video_id(self, value: Optional[str]) -> None:
        self._active_video_id = value

    # ── Param resolution (Global → Group → Individual) ───────────────────────

    def resolved_params(self, image_id: str) -> AdjustmentParams:
        """Merge global → group → individual params for a specific image or video."""
        img = self.get_image(image_id)
        if img is None:
            img = self.get_video(image_id)
        if img is None:
            return AdjustmentParams()

        result = self.global_params.copy()

        if img.group_id and img.group_id in self.group_params:
            result = merge_params(result, self.group_params[img.group_id])

        result = merge_params(result, img.params)
        return result

    # ── Scope-aware setters ───────────────────────────────────────────────────

    def set_param_global(self, param: str, value) -> None:
        setattr(self.global_params, param, value)
        for img in self.images:
            img.invalidate_cache()
        for vid in self.videos:
            vid.invalidate_cache()

    def set_param_group(self, group_id: str, param: str, value) -> None:
        if group_id not in self.group_params:
            self.group_params[group_id] = AdjustmentParams()
        setattr(self.group_params[group_id], param, value)
        for img in self.images:
            if img.group_id == group_id:
                img.invalidate_cache()
        for vid in self.videos:
            if vid.group_id == group_id:
                vid.invalidate_cache()

    def set_param_individual(self, image_id: str, param: str, value) -> None:
        img = self.get_image(image_id)
        if img is None:
            img = self.get_video(image_id)
        if img:
            setattr(img.params, param, value)
            img.invalidate_cache()

    # ── Group management ──────────────────────────────────────────────────────

    def create_group(self, group_id: str) -> None:
        if group_id not in self.group_params:
            self.group_params[group_id] = AdjustmentParams()

    def assign_to_group(self, image_id: str, group_id: Optional[str]) -> None:
        img = self.get_image(image_id)
        if img is None:
            img = self.get_video(image_id)
        if img:
            img.group_id = group_id
            img.invalidate_cache()

    def get_groups(self) -> List[str]:
        return list(self.group_params.keys())

    # ── Preset management ─────────────────────────────────────────────────────

    def save_preset(self, name: str, params: AdjustmentParams) -> None:
        self.presets[name] = params.copy()

    def load_preset(self, name: str) -> Optional[AdjustmentParams]:
        return self.presets.get(name)

    def delete_preset(self, name: str) -> None:
        self.presets.pop(name, None)

    def invalidate_all_caches(self) -> None:
        for img in self.images:
            img.invalidate_cache()
        for vid in self.videos:
            vid.invalidate_cache()

    def to_dict(self) -> dict:
        return {
            "images": [img.to_dict() for img in self.images],
            "videos": [vid.to_dict() for vid in self.videos],
            "global_params": self.global_params.to_dict(),
            "group_params": {k: v.to_dict() for k, v in self.group_params.items()},
            "presets": {k: v.to_dict() for k, v in self.presets.items()},
            "active_id": self._active_id,
            "selected_ids": self._selected_ids,
            "active_video_id": self._active_video_id
        }

    def from_dict(self, d: dict) -> None:
        self.images = [ImageRecord.from_dict(img) for img in d.get("images", [])]
        self.videos = [VideoRecord.from_dict(vid) for vid in d.get("videos", [])]
        self.global_params = AdjustmentParams.from_dict(d.get("global_params", {}))
        self.group_params = {k: AdjustmentParams.from_dict(v) for k, v in d.get("group_params", {}).items()}
        self.presets = {k: AdjustmentParams.from_dict(v) for k, v in d.get("presets", {}).items()}
        self._active_id = d.get("active_id")
        self._selected_ids = d.get("selected_ids", [])
        self._active_video_id = d.get("active_video_id")
