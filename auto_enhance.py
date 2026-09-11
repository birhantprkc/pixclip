"""
auto_enhance.py
Auto-enhance analysis module for PixClip.

Analyzes photos and videos to produce an EditProfile with automatically-chosen
values for all major adjustment parameters.  The profile is then translated
into AdjustmentParams by the UI layer (see mapping constants below).

Public API
----------
    auto_enhance_photo(path)              -> EditProfile
    auto_enhance_video_profile(path, n)   -> (EditProfile, dict[video_info])
    apply_to_image(img, profile)          -> np.ndarray
    apply_to_video(in_path, out_path, profile)

    EditProfile.save(path)
    EditProfile.load(path)

    ProfileLibrary(path)
        .add(name, profile)
        .get(name)            -> Optional[EditProfile]
        .list_names()         -> list[str]
        .delete(name)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── EditProfile ───────────────────────────────────────────────────────────────

@dataclass
class EditProfile:
    """
    All auto-enhance parameters.  Ranges:
      sharpness           0 – 100
      clarity             0 – 100
      contrast          -100 – +100
      vibrance          -100 – +100
      exposure          -100 – +100
      shadow_recovery     0 – 100
      highlight_recovery  0 – 100
      denoise             0 – 100
      warmth            -100 – +100
      tint              -100 – +100
      filter_style        str  (matches AdjustmentParams.filter_id, e.g. "none")
      filter_intensity    0.0 – 1.0
    """
    sharpness: float = 0.0
    clarity: float = 0.0
    contrast: float = 0.0
    vibrance: float = 0.0
    exposure: float = 0.0
    shadow_recovery: float = 0.0
    highlight_recovery: float = 0.0
    denoise: float = 0.0
    warmth: float = 0.0
    tint: float = 0.0
    filter_style: str = "none"
    filter_intensity: float = 1.0

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EditProfile":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def save(self, path) -> None:
        """Persist this profile to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "EditProfile":
        """Load an EditProfile from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ── Heuristic analysis helpers ────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _analyze_frame(frame_bgr: np.ndarray) -> dict:
    """
    Extract perceptual metrics from a single BGR frame.
    Returns a dict with raw measurements used by the profile builder.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {}

    # Convert to LAB for perceptual metrics.
    # In OpenCV, 8-bit images converted to LAB have:
    # L in [0, 255] (representing L* * 255 / 100)
    # A in [0, 255] (representing a* + 128)
    # B in [0, 255] (representing b* + 128)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)   # Scale to standard perceptual L* [0, 100]
    A = lab[:, :, 1] - 128.0             # Centered at 0
    B = lab[:, :, 2] - 128.0             # Centered at 0

    # ── Luminance statistics (0–100 scale) ─────────────────────────────────
    mean_L   = float(np.mean(L))
    median_L = float(np.median(L))
    std_L    = float(np.std(L))
    p5_L     = float(np.percentile(L, 5))
    p95_L    = float(np.percentile(L, 95))

    # Accurate shadow/highlight clipping fractions
    shadow_clip    = float(np.mean(L < 5.0))    # fraction crushed to black (<5%)
    highlight_clip = float(np.mean(L > 95.0))   # fraction blown to white (>95%)

    # ── Color metrics ─────────────────────────────────────────────────────
    # Standard chroma in LAB space
    chroma      = np.sqrt(A**2 + B**2)
    mean_chroma = float(np.mean(chroma))

    # Evaluate color cast on near-neutral midtones to avoid being fooled
    # by large saturated areas (e.g. green trees, red clothes, blue water)
    neutral_mask = (chroma < 14.0) & (L > 15.0) & (L < 88.0)
    if np.mean(neutral_mask) > 0.01:
        cast_a = float(np.mean(A[neutral_mask]))
        cast_b = float(np.mean(B[neutral_mask]))
    else:
        cast_a = float(np.mean(A)) * 0.2
        cast_b = float(np.mean(B)) * 0.2

    # ── Sharpness (Laplacian variance on grayscale) ───────────────────────
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── Noise estimation (Median Absolute Deviation on fine Laplacian) ────
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=1)
    noise_std = float(np.median(np.abs(lap)) / 0.6745)

    return {
        "mean_L": mean_L,
        "median_L": median_L,
        "std_L": std_L,
        "p5_L": p5_L,
        "p95_L": p95_L,
        "shadow_clip": shadow_clip,
        "highlight_clip": highlight_clip,
        "mean_chroma": mean_chroma,
        "cast_a": cast_a,
        "cast_b": cast_b,
        "lap_var": lap_var,
        "noise_std": noise_std,
    }


def _metrics_to_profile(m: dict) -> EditProfile:
    """
    Convert raw frame metrics into EditProfile values using perceptual heuristics.
    All output values are in the EditProfile's own ranges (documented on EditProfile).
    """
    if not m:
        return EditProfile()

    mean_L         = m.get("mean_L", 50.0)
    median_L       = m.get("median_L", mean_L)
    std_L          = m.get("std_L", 25.0)
    p5_L           = m.get("p5_L", 10.0)
    p95_L          = m.get("p95_L", 90.0)
    shadow_clip    = m.get("shadow_clip", 0.0)
    highlight_clip = m.get("highlight_clip", 0.0)
    mean_chroma    = m.get("mean_chroma", 20.0)
    cast_a         = m.get("cast_a", 0.0)
    cast_b         = m.get("cast_b", 0.0)
    lap_var        = m.get("lap_var", 300.0)
    noise_std      = m.get("noise_std", 5.0)

    # ── Exposure ──────────────────────────────────────────────────────────
    # Intelligent exposure balancing that avoids underexposing sunny outdoor
    # scenes while gently rescuing dark/dim images.
    if median_L < 42.0:
        raw_exp = (48.0 - median_L) * 0.7
        lift_limit = max(0.0, (100.0 - p95_L) * 1.5)
        exposure = min(raw_exp, lift_limit)
    elif median_L > 62.0 and (p95_L > 92.0 or highlight_clip > 0.03):
        # Noticeably overexposed/blown — pull down exposure gently
        exposure = -_clamp((median_L - 55.0) * 0.5, 2.0, 20.0)
    else:
        # Well-balanced scene — subtle nudge only
        exposure = _clamp((50.0 - median_L) * 0.25, -4.0, 4.0)

    # ── Shadow recovery ───────────────────────────────────────────────────
    # If deep shadows exist, lift them to reveal rich texture
    if p5_L < 15.0 or shadow_clip > 0.005:
        shadow_recovery = _clamp((15.0 - p5_L) * 2.2 + shadow_clip * 150.0, 5.0, 40.0)
    else:
        shadow_recovery = 0.0

    # ── Highlight recovery ────────────────────────────────────────────────
    # If bright highlights exist (blown skies, reflections), recover them
    if p95_L > 90.0 or highlight_clip > 0.02:
        highlight_recovery = _clamp((p95_L - 88.0) * 2.0 + highlight_clip * 100.0, 5.0, 40.0)
    else:
        highlight_recovery = 0.0

    # ── Contrast ──────────────────────────────────────────────────────────
    # Low std_L → flat/hazy → boost contrast; high std_L → already wide range
    if std_L < 22.0:
        contrast = _clamp((24.0 - std_L) * 1.2, 0.0, 20.0)
    elif std_L > 32.0:
        contrast = _clamp((28.0 - std_L) * 0.6, -8.0, 0.0)
    else:
        contrast = _clamp((26.0 - std_L) * 0.6, -4.0, 6.0)

    # ── Vibrance (colour richness) ────────────────────────────────────────
    # Standard chroma is ~12–25. Provide a healthy, natural boost.
    if mean_chroma < 22.0:
        vibrance = _clamp((22.0 - mean_chroma) * 1.8, 5.0, 30.0)
    else:
        vibrance = _clamp((26.0 - mean_chroma) * 0.8, -5.0, 10.0)

    # ── Sharpness ─────────────────────────────────────────────────────────
    # lap_var on 800px resized frame: >800 is sharp, <200 is soft
    if lap_var < 200.0:
        sharpness = _clamp(35.0 - (lap_var / 200.0) * 10.0, 25.0, 35.0)
    elif lap_var > 800.0:
        sharpness = 20.0
    else:
        sharpness = _clamp(30.0 - (lap_var - 200.0) / 600.0 * 10.0, 20.0, 30.0)

    # ── Clarity (local contrast / micro-contrast) ─────────────────────────
    clarity = _clamp(sharpness * 0.85, 15.0, 28.0)

    # ── Denoise ───────────────────────────────────────────────────────────
    denoise = _clamp((noise_std - 6.0) * 2.5, 0.0, 30.0) if noise_std > 6.0 else 0.0

    # ── Warmth (temperature) & Tint (color cast compensation) ─────────────
    warmth = _clamp(-cast_b * 1.5, -20.0, 20.0) if abs(cast_b) > 2.5 else 0.0
    tint   = _clamp(-cast_a * 1.5, -15.0, 15.0) if abs(cast_a) > 2.0 else 0.0

    # Subtle filter style if vibrance boost is beneficial
    filter_style = "vivid-1" if vibrance > 10.0 else "none"
    filter_intensity = round(min(0.45, vibrance / 60.0), 2) if filter_style != "none" else 1.0

    return EditProfile(
        sharpness=round(sharpness, 1),
        clarity=round(clarity, 1),
        contrast=round(contrast, 1),
        vibrance=round(vibrance, 1),
        exposure=round(exposure, 1),
        shadow_recovery=round(shadow_recovery, 1),
        highlight_recovery=round(highlight_recovery, 1),
        denoise=round(denoise, 1),
        warmth=round(warmth, 1),
        tint=round(tint, 1),
        filter_style=filter_style,
        filter_intensity=filter_intensity,
    )


# ── Public Analysis Functions ─────────────────────────────────────────────────

def auto_enhance_photo(path) -> EditProfile:
    """
    Analyze a photo and return a recommended EditProfile.

    Parameters
    ----------
    path : str | Path
        Path to the input image file (any format supported by cv2.imread).

    Returns
    -------
    EditProfile
        Automatically-chosen adjustment values.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Down-sample for fast analysis (max 800 px on longest side)
    h, w = img.shape[:2]
    scale = min(1.0, 800.0 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)

    metrics = _analyze_frame(img)
    return _metrics_to_profile(metrics)


def auto_enhance_video_profile(path, num_samples: int = 30) -> tuple:
    """
    Analyze a video by sampling ``num_samples`` evenly-spaced frames and
    averaging their metrics into a single EditProfile.

    Parameters
    ----------
    path : str | Path
        Path to the video file.
    num_samples : int
        Number of frames to sample (default 30).

    Returns
    -------
    (EditProfile, dict)
        Profile and a video_info dict with keys: width, height, fps,
        frame_count, duration.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps

    video_info = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": total_frames,
        "duration": duration,
    }

    if total_frames <= 0:
        cap.release()
        return EditProfile(), video_info

    # Choose evenly-spaced sample indices, skipping first/last 5 % (often black)
    margin = max(1, int(total_frames * 0.05))
    indices = np.linspace(margin, max(margin, total_frames - margin - 1),
                          num=min(num_samples, total_frames),
                          dtype=int)

    accumulated: dict[str, list] = {}
    count = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Down-sample for speed
        h_f, w_f = frame.shape[:2]
        sc = min(1.0, 480.0 / max(h_f, w_f))
        if sc < 1.0:
            frame = cv2.resize(frame, (max(1, int(w_f * sc)), max(1, int(h_f * sc))),
                               interpolation=cv2.INTER_AREA)

        m = _analyze_frame(frame)
        for key, val in m.items():
            accumulated.setdefault(key, []).append(val)
        count += 1

    cap.release()

    if count == 0:
        return EditProfile(), video_info

    # Average metrics across sampled frames
    avg_metrics = {k: float(np.mean(v)) for k, v in accumulated.items()}
    profile = _metrics_to_profile(avg_metrics)
    return profile, video_info


# ── Apply Functions ───────────────────────────────────────────────────────────

def _apply_profile_to_bgr(img: np.ndarray, profile: EditProfile) -> np.ndarray:
    """
    Apply an EditProfile's adjustments to a BGR uint8 numpy array.
    Returns a new uint8 BGR array.
    """
    out = img.astype(np.float32) / 255.0

    # ── Exposure (multiplicative EV shift matching pipeline) ──────────────
    if profile.exposure != 0.0:
        ev = (profile.exposure / 100.0) * 2.0          # map ±100 → ±2 stops
        out = out * (2.0 ** ev)

    # ── Contrast (S-curve around 0.5 matching pipeline) ───────────────────
    if profile.contrast != 0.0:
        c_factor = 1.0 + (profile.contrast / 100.0) * 1.5
        out = (out - 0.5) * c_factor + 0.5

    # ── Vibrance (perceptual saturation boost protecting neutral tones) ───
    if profile.vibrance != 0.0:
        clipped = np.clip(out, 0.0, 1.0)
        hsv = cv2.cvtColor((clipped * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        s_norm = hsv[:, :, 1] / 255.0
        # Multiplicative curve: boosts muted colors, leaves neutral grays/whites untouched
        v_strength = profile.vibrance / 100.0
        s_boosted = s_norm * (1.0 + v_strength * (1.0 - s_norm))
        hsv[:, :, 1] = np.clip(s_boosted * 255.0, 0.0, 255.0)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # ── Shadow recovery (luminosity-masked shadow lift) ───────────────────
    if profile.shadow_recovery > 0.0:
        luminance = (0.299 * out[:, :, 2] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 0])
        sh_mask = np.clip((0.5 - luminance) * 2.0, 0.0, 1.0)[:, :, np.newaxis]
        sh_adj = (profile.shadow_recovery / 100.0) * 0.30
        out = out + sh_mask * sh_adj

    # ── Highlight recovery (luminosity-masked highlight compression) ──────
    if profile.highlight_recovery > 0.0:
        luminance = (0.299 * out[:, :, 2] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 0])
        hi_mask = np.clip((luminance - 0.5) * 2.0, 0.0, 1.0)[:, :, np.newaxis]
        hi_adj = (profile.highlight_recovery / 100.0) * 0.30
        out = out - hi_mask * hi_adj

    # ── Warmth & Tint (LAB color space) ───────────────────────────────────
    if profile.warmth != 0.0 or profile.tint != 0.0:
        clipped = np.clip(out, 0.0, 1.0)
        lab = cv2.cvtColor((clipped * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        if profile.warmth != 0.0:
            shift_b = (profile.warmth / 100.0) * 15.0
            lab[:, :, 2] = np.clip(lab[:, :, 2] + shift_b, 0, 255)
        if profile.tint != 0.0:
            shift_a = (profile.tint / 100.0) * 12.0
            lab[:, :, 1] = np.clip(lab[:, :, 1] + shift_a, 0, 255)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    # ── Denoise (edge-preserving bilateral filter) ─────────────────────────
    if profile.denoise > 8.0:
        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        sigma_c = float(profile.denoise * 0.6)
        denoised = cv2.bilateralFilter(out_u8, d=5, sigmaColor=sigma_c, sigmaSpace=5.0).astype(np.float32) / 255.0
        blend = min(1.0, profile.denoise / 70.0)
        out = out * (1.0 - blend) + denoised * blend

    # ── Clarity (local contrast via structure-preserving bilateral on L) ──
    if profile.clarity > 0.0:
        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        lab = cv2.cvtColor(out_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0] / 255.0
        blurred_l = cv2.bilateralFilter(L, d=0, sigmaColor=0.12, sigmaSpace=16.0)
        detail = L - blurred_l
        c_amp = (profile.clarity / 100.0) * 2.0
        lab[:, :, 0] = np.clip(L + detail * c_amp, 0.0, 1.0) * 255.0
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    # ── Sharpness (thresholded unsharp mask) ──────────────────────────────
    if profile.sharpness > 0.0:
        blurred_s = cv2.GaussianBlur(out, (0, 0), 1.0)
        amount = (profile.sharpness / 100.0) * 1.2
        diff = np.abs(out - blurred_s)
        mask = (diff > 0.015).astype(np.float32)
        sharpened = out + amount * (out - blurred_s)
        out = out * (1.0 - mask) + sharpened * mask

    # ── Filter style (if set) ─────────────────────────────────────────────
    if profile.filter_style != "none":
        try:
            from core.filters import apply_filter
            out = apply_filter(out, profile.filter_style, profile.filter_intensity)
        except Exception:
            pass

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def apply_to_image(img: np.ndarray, profile: EditProfile) -> np.ndarray:
    """
    Apply an EditProfile to a BGR uint8 numpy array.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR, uint8).
    profile : EditProfile
        Profile to apply.

    Returns
    -------
    np.ndarray
        Processed image (BGR, uint8).
    """
    return _apply_profile_to_bgr(img, profile)


def apply_to_video(in_path, out_path, profile: EditProfile,
                   progress_callback=None) -> None:
    """
    Apply an EditProfile to every frame of a video and write the result.

    Parameters
    ----------
    in_path : str | Path
        Source video path.
    out_path : str | Path
        Destination video path (MP4 / H.264).
    profile : EditProfile
        Profile to apply to each frame.
    progress_callback : callable(current_frame, total_frames), optional
        Called after each frame.
    """
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = _apply_profile_to_bgr(frame, profile)
            writer.write(processed)
            idx += 1
            if progress_callback:
                progress_callback(idx, total)
    finally:
        cap.release()
        writer.release()


# ── ProfileLibrary ────────────────────────────────────────────────────────────

class ProfileLibrary:
    """
    A persistent, JSON-backed store of named EditProfiles.

    Usage
    -----
        lib = ProfileLibrary(Path.home() / ".pixclip" / "profiles.json")
        lib.add("My Warm Look", profile)
        profile = lib.get("My Warm Look")
        names   = lib.list_names()
        lib.delete("My Warm Look")
    """

    def __init__(self, path):
        self._path = Path(path)
        self._data: dict[str, dict] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(self, name: str, profile: EditProfile) -> None:
        """Save (or overwrite) a named profile."""
        self._data[name] = profile.to_dict()
        self._save()

    def get(self, name: str) -> Optional[EditProfile]:
        """Return the named profile, or None if not found."""
        entry = self._data.get(name)
        if entry is None:
            return None
        try:
            return EditProfile.from_dict(entry)
        except Exception:
            return None

    def list_names(self) -> list:
        """Return all saved profile names, sorted alphabetically."""
        return sorted(self._data.keys())

    def delete(self, name: str) -> bool:
        """Delete a named profile.  Returns True if it existed."""
        if name in self._data:
            del self._data[name]
            self._save()
            return True
        return False

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a profile.  Returns True on success."""
        if old_name not in self._data or not new_name:
            return False
        self._data[new_name] = self._data.pop(old_name)
        self._save()
        return True
