"""
core/pipeline.py
The stateless image processing pipeline for PixClip.

Each function operates on numpy arrays and is designed to be composable.
The master process_frame() function is the single entry point — identical
signature for both still images and video frames.

All internal processing uses float32 in [0, 1] range for precision.
LAB color space is used for perceptual operations (clarity, lightness).
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional

from core.params import AdjustmentParams


# ── Individual Adjustment Functions ──────────────────────────────────────────

def apply_exposure(img: np.ndarray, exposure: float) -> np.ndarray:
    """
    Exposure adjustment via EV (exposure value) stops.
    exposure: -100 to +100, mapped to -2 to +2 EV stops.
    Reference value: -47 → ~-0.94 EV (moderate darkening to cut haze).
    """
    if exposure == 0.0:
        return img
    ev = (exposure / 100.0) * 2.0
    factor = 2.0 ** ev
    return np.clip(img * factor, 0.0, 1.0)


def apply_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """
    Brightness: additive offset. Contrast: S-curve pivot at 0.5.
    Reference: brightness=-14, contrast=10.
    """
    if brightness == 0.0 and contrast == 0.0:
        return img
    if brightness != 0.0:
        b = (brightness / 100.0) * 0.5
        img = img + b
    if contrast != 0.0:
        # contrast=10 → factor=1.15 (subtle lift)
        c_factor = 1.0 + (contrast / 100.0) * 1.5
        img = (img - 0.5) * c_factor + 0.5
    return np.clip(img, 0.0, 1.0)


def apply_tonal_range(img: np.ndarray, light_range: float, dark_range: float) -> np.ndarray:
    """
    Controls white and black points — compressing or expanding the tonal range.
    light_range > 0: pull white point inward (protect highlights).
    dark_range > 0: lift black point (open up shadows).
    Reference: light_range=30, dark_range=5.
    """
    if light_range == 0.0 and dark_range == 0.0:
        return img
    white_point = 1.0 - (light_range / 100.0) * 0.30
    black_point = (dark_range / 100.0) * 0.12
    white_point = max(white_point, black_point + 0.01)
    img = (img - black_point) / (white_point - black_point)
    return np.clip(img, 0.0, 1.0)


def apply_highlights_shadows(img: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    """
    Luminosity-masked highlights and shadows control.
    Highlights only affect pixels above 0.5 luminance.
    Shadows only affect pixels below 0.5 luminance.
    Reference: highlights=24 (moderate lift), shadows=-1 (near neutral).
    """
    if highlights == 0.0 and shadows == 0.0:
        return img

    luminance = (0.299 * img[:, :, 2] +
                 0.587 * img[:, :, 1] +
                 0.114 * img[:, :, 0])  # BGR order

    if highlights != 0.0:
        hi_mask = np.clip((luminance - 0.5) * 2.0, 0.0, 1.0)[:, :, np.newaxis]
        hi_adj = (highlights / 100.0) * 0.30
        img = img + hi_mask * hi_adj

    if shadows != 0.0:
        sh_mask = np.clip((0.5 - luminance) * 2.0, 0.0, 1.0)[:, :, np.newaxis]
        sh_adj = (shadows / 100.0) * 0.30
        img = img + sh_mask * sh_adj

    return np.clip(img, 0.0, 1.0)


def _to_lab(img_float: np.ndarray) -> np.ndarray:
    """
    Convert float32 BGR [0,1] → LAB float32.
    OpenCV LAB: L in [0,255] (maps to [0,100]), A/B in [0,255] (neutral=128).
    We store L in [0,100], A/B in [-127, 127] for intuitive arithmetic.
    """
    img_u8 = (np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    # L: [0,255] → [0,100]
    lab[:, :, 0] = lab[:, :, 0] * (100.0 / 255.0)
    # A, B: [0,255] → [-127, 128]  (OpenCV offset: 0→-128, 128→0, 255→127)
    lab[:, :, 1] = lab[:, :, 1] - 128.0
    lab[:, :, 2] = lab[:, :, 2] - 128.0
    return lab


def _from_lab(lab: np.ndarray) -> np.ndarray:
    """Convert LAB float32 (L:[0,100], A/B:[-127,128]) → BGR float32 [0,1]."""
    out = lab.copy()
    # L: [0,100] → [0,255]
    out[:, :, 0] = np.clip(out[:, :, 0] * (255.0 / 100.0), 0.0, 255.0)
    # A, B: [-127,128] → [0,255]
    out[:, :, 1] = np.clip(out[:, :, 1] + 128.0, 0.0, 255.0)
    out[:, :, 2] = np.clip(out[:, :, 2] + 128.0, 0.0, 255.0)
    bgr = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return bgr.astype(np.float32) / 255.0


def apply_lightness(lab: np.ndarray, lightness: float) -> np.ndarray:
    """
    Direct L* channel shift in LAB space — perceptually uniform.
    lightness ±100 maps to ±50 L* units (half the full [0,100] range).
    Reference: lightness=58 → +29 L* — significant luminance recovery.
    """
    if lightness == 0.0:
        return lab
    delta_L = (lightness / 100.0) * 50.0
    lab[:, :, 0] = np.clip(lab[:, :, 0] + delta_L, 0.0, 100.0)
    return lab


def apply_clarity(lab: np.ndarray, clarity: float) -> np.ndarray:
    """
    Local Contrast Enhancement (LCE) operating on the L* channel only.
    Uses a structure-preserving bilateral filter to isolate detail layers,
    then amplifies the mid-frequency detail proportionally to clarity strength.

    clarity: 0–100 scale (as stored in AdjustmentParams).
    Amplification: clarity=100 → ~2.9x the natural detail layer.
    """
    if clarity <= 0.0:
        return lab

    # clarity is on a 0–100 scale directly
    c = clarity

    L_norm = (lab[:, :, 0] / 100.0).astype(np.float32)

    # Bilateral filter: structure-preserving blur that respects edges
    # sigmaColor=0.12 keeps edges sharp; sigmaSpace scales with strength
    sigma_space = 12.0 + (c / 100.0) * 18.0  # 12 → 30 across the full range
    blurred = cv2.bilateralFilter(L_norm, d=0, sigmaColor=0.12, sigmaSpace=sigma_space)

    # Detail layer = high-frequency + mid-frequency structure
    detail = L_norm - blurred

    # Amplification starts at 0 (no effect) and scales to 2.9 at full strength.
    # IMPORTANT: using (c/100)*X rather than 1.0+(c/100)*X avoids the jump
    # that occurred on the very first slider step in the previous formula.
    amplification = (c / 100.0) * 2.9

    L_enhanced = np.clip(L_norm + detail * amplification, 0.0, 1.0)
    lab[:, :, 0] = L_enhanced * 100.0
    return lab



def apply_sharpness(img: np.ndarray, sharpness: float) -> np.ndarray:
    """
    Unsharp Masking (USM) for fine edge enhancement.
    Uses a noise threshold to avoid sharpening grain/JPEG artifacts.

    sharpness: 0–100 scale (as stored in AdjustmentParams).
    Reference: sharpness=34 → subtle but definite edge crispness.
    """
    if sharpness <= 0.0:
        return img

    # sharpness is on a 0–100 scale directly
    # amount: sharpness=34 → ~0.51, sharpness=100 → ~1.5
    s = sharpness
    amount = (s / 100.0) * 1.5
    radius = 0.8  # tight radius for fine detail only

    blurred = cv2.GaussianBlur(img, (0, 0), radius)

    # USM: original + amount * (original - blurred)
    sharpened = img + amount * (img - blurred)

    # Noise threshold mask: suppress sharpening in near-flat regions
    # diff and mask are [H,W,3] (same shape as img) — no newaxis needed
    diff = np.abs(img - blurred)
    threshold = 0.015  # ~4/255 in normalized float
    mask = (diff > threshold).astype(np.float32)

    # Smooth mask edges to avoid haloing — operates per-channel
    mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
    mask = np.clip(mask, 0.0, 1.0)

    result = img * (1.0 - mask) + sharpened * mask
    return np.clip(result, 0.0, 1.0)


# ── Master Pipeline ───────────────────────────────────────────────────────────

def process_frame(
    frame_bgr: np.ndarray,
    params: AdjustmentParams,
    filter_lut_fn=None,
) -> np.ndarray:
    """
    The stateless master processing pipeline.

    Input:  uint8 BGR numpy array (OpenCV native format)
    Output: uint8 BGR numpy array

    This function is video-ready: pass frames from cv2.VideoCapture.read()
    directly. The internal pipeline is identical for stills and video.

    Args:
        frame_bgr:    uint8 BGR image [H, W, 3]
        params:       AdjustmentParams with all adjustment values
        filter_lut_fn: Optional callable(img_float, filter_id, intensity) → img_float
                       Injected to avoid circular imports with filters module.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    # Normalise to float32 [0, 1] for all operations
    img = frame_bgr.astype(np.float32) / 255.0

    # ── Stage 1: Exposure (multiplicative — must come first) ────────────────
    img = apply_exposure(img, params.exposure)

    # ── Stage 2: Brightness & Contrast ─────────────────────────────────────
    img = apply_brightness_contrast(img, params.brightness, params.contrast)

    # ── Stage 3: Tonal Range (white/black point) ────────────────────────────
    img = apply_tonal_range(img, params.light_range, params.dark_range)

    # ── Stage 4: Highlights & Shadows (luminosity-masked) ──────────────────
    img = apply_highlights_shadows(img, params.highlights, params.shadows)

    # ── Stage 5: Convert to LAB for perceptual operations ──────────────────
    lab = _to_lab(img)

    # ── Stage 6: Lightness (L* direct shift) ───────────────────────────────
    lab = apply_lightness(lab, params.lightness)

    # ── Stage 7: Clarity (LCE on L* channel — the signature effect) ────────
    lab = apply_clarity(lab, params.clarity)

    # ── Stage 8: Convert back to BGR float32 ───────────────────────────────
    img = _from_lab(lab)

    # ── Stage 9: Sharpness (USM on full RGB) ───────────────────────────────
    img = apply_sharpness(img, params.sharpness)

    # ── Stage 10: Style Filter (LUT — injected from filters module) ────────
    if filter_lut_fn is not None and params.filter_id != "none":
        img = filter_lut_fn(img, params.filter_id, params.filter_intensity)

    # Return as uint8 BGR
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def process_frame_preview(
    frame_bgr: np.ndarray,
    params: AdjustmentParams,
    scale: float = 0.5,
    filter_lut_fn=None,
) -> np.ndarray:
    """
    Downscaled preview variant for real-time UI updates.
    Processes at `scale` resolution for speed, returns at full display size.
    """
    if frame_bgr is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    processed = process_frame(small, params, filter_lut_fn)
    # Return at preview scale (MainWindow will handle display scaling)
    return processed
