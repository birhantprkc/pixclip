"""
core/filters.py
Curated filter system for PixClip — 20 stylistic presets across 5 families.

Each filter is defined as a programmatic tone-curve + HSL adjustment bundle.
Filters are applied as 3D LUT interpolation for speed.
Variants (e.g., Vivid-1 through Vivid-4) are intensity-scaled versions.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Callable, Optional, Tuple
from dataclasses import dataclass

# LUT resolution: 32x32x32 is the standard balance of quality vs. memory
LUT_SIZE = 32


@dataclass
class FilterDefinition:
    id: str
    name: str
    family: str
    description: str
    # Tone curve control points: list of (input, output) in [0, 1]
    r_curve: Optional[list] = None  # Per-channel curves
    g_curve: Optional[list] = None
    b_curve: Optional[list] = None
    rgb_curve: Optional[list] = None  # Applied to all channels first
    # HSL adjustments: hue_shift (degrees), saturation_mult, lightness_add
    hue_shift: float = 0.0
    saturation_mult: float = 1.0
    lightness_add: float = 0.0
    # Shadow/highlight tinting
    shadow_tint: Optional[Tuple[float, float, float]] = None   # BGR [0,1] additive
    highlight_tint: Optional[Tuple[float, float, float]] = None
    tint_strength: float = 0.08


# ── Filter Catalog ────────────────────────────────────────────────────────────

FILTER_DEFINITIONS: Dict[str, FilterDefinition] = {}


def _register_family(definitions: list[FilterDefinition]):
    for d in definitions:
        FILTER_DEFINITIONS[d.id] = d


# ── 1. Vivid Family — Saturated, punchy, boosted contrast ────────────────────
_register_family([
    FilterDefinition(
        id="vivid-1", name="Vibrant Boost", family="Vivid",
        description="Mild saturation boost with lifted contrast",
        rgb_curve=[(0, 0), (0.25, 0.22), (0.5, 0.52), (0.75, 0.80), (1, 1)],
        saturation_mult=1.25, lightness_add=0.02,
    ),
    FilterDefinition(
        id="vivid-2", name="Velvia Chrome", family="Vivid",
        description="Bold saturation, deeper shadows",
        rgb_curve=[(0, 0), (0.2, 0.16), (0.5, 0.54), (0.8, 0.85), (1, 1)],
        saturation_mult=1.45, lightness_add=0.0,
    ),
    FilterDefinition(
        id="vivid-3", name="Deep Vibrancy", family="Vivid",
        description="High saturation, strong S-curve",
        rgb_curve=[(0, 0), (0.15, 0.10), (0.5, 0.56), (0.85, 0.92), (1, 1)],
        saturation_mult=1.65, lightness_add=-0.02,
    ),
    FilterDefinition(
        id="vivid-4", name="Vivid Punch", family="Vivid",
        description="Maximum vibrancy — cinematic saturation",
        rgb_curve=[(0, 0), (0.1, 0.06), (0.5, 0.58), (0.9, 0.96), (1, 1)],
        saturation_mult=1.85, lightness_add=-0.03,
    ),
])

# ── 2. Aesthetic Family — Muted pastels, film-like softness ──────────────────
_register_family([
    FilterDefinition(
        id="aesthetic-1", name="Soft Pastel", family="Aesthetic",
        description="Soft muted tones, lifted blacks",
        rgb_curve=[(0, 0.06), (0.3, 0.32), (0.7, 0.72), (1, 0.96)],
        saturation_mult=0.82, lightness_add=0.04,
        shadow_tint=(0.10, 0.08, 0.06), tint_strength=0.06,
    ),
    FilterDefinition(
        id="aesthetic-2", name="Warm Matte", family="Aesthetic",
        description="Warm pastel, slightly faded",
        rgb_curve=[(0, 0.08), (0.4, 0.38), (0.8, 0.78), (1, 0.93)],
        saturation_mult=0.72, lightness_add=0.05,
        shadow_tint=(0.12, 0.09, 0.06), highlight_tint=(0.05, 0.03, 0.02), tint_strength=0.07,
    ),
    FilterDefinition(
        id="aesthetic-3", name="Cool Desaturated", family="Aesthetic",
        description="Cool muted tones, desaturated",
        rgb_curve=[(0, 0.07), (0.35, 0.33), (0.75, 0.74), (1, 0.94)],
        saturation_mult=0.65, lightness_add=0.03,
        shadow_tint=(0.06, 0.08, 0.10), tint_strength=0.05,
    ),
    FilterDefinition(
        id="aesthetic-4", name="Faded Dream", family="Aesthetic",
        description="Dreamy fade — maximum softness",
        rgb_curve=[(0, 0.10), (0.5, 0.50), (1, 0.90)],
        saturation_mult=0.55, lightness_add=0.06,
        shadow_tint=(0.08, 0.07, 0.06), highlight_tint=(0.04, 0.04, 0.05), tint_strength=0.08,
    ),
])

# ── 3. Film Family — Grain-aware vintage tone curves ─────────────────────────
_register_family([
    FilterDefinition(
        id="film-1", name="Kodak Gold", family="Film",
        description="Classic Kodak warm tones",
        rgb_curve=[(0, 0.02), (0.25, 0.28), (0.5, 0.53), (0.75, 0.76), (1, 0.98)],
        r_curve=[(0, 0.04), (0.5, 0.56), (1, 1.0)],
        b_curve=[(0, 0.0), (0.5, 0.47), (1, 0.94)],
        saturation_mult=0.90, lightness_add=0.01,
    ),
    FilterDefinition(
        id="film-2", name="Fuji Chrome", family="Film",
        description="Fuji cool cross-processed",
        rgb_curve=[(0, 0.01), (0.3, 0.27), (0.7, 0.74), (1, 0.99)],
        r_curve=[(0, 0.0), (0.5, 0.48), (1, 0.95)],
        b_curve=[(0, 0.03), (0.5, 0.54), (1, 1.0)],
        saturation_mult=0.88, lightness_add=0.0,
    ),
    FilterDefinition(
        id="film-3", name="Noir Classic", family="Film",
        description="High contrast B&W-leaning film",
        rgb_curve=[(0, 0.0), (0.2, 0.14), (0.5, 0.52), (0.8, 0.87), (1, 1.0)],
        saturation_mult=0.70, lightness_add=-0.01,
    ),
    FilterDefinition(
        id="film-4", name="Expired Retro", family="Film",
        description="Expired film — fade and grain",
        rgb_curve=[(0, 0.08), (0.3, 0.32), (0.7, 0.70), (1, 0.90)],
        r_curve=[(0, 0.06), (0.5, 0.52), (1, 0.92)],
        g_curve=[(0, 0.05), (0.5, 0.50), (1, 0.93)],
        b_curve=[(0, 0.03), (0.5, 0.48), (1, 0.90)],
        saturation_mult=0.68, lightness_add=0.04,
    ),
])

# ── 4. Cinematic Family — Teal/Orange Hollywood grade ────────────────────────
_register_family([
    FilterDefinition(
        id="cinematic-1", name="Teal Shadows", family="Cinematic",
        description="Mild teal shadows, warm highlights",
        rgb_curve=[(0, 0), (0.2, 0.17), (0.5, 0.52), (0.8, 0.84), (1, 1)],
        saturation_mult=0.95,
        shadow_tint=(0.12, 0.08, 0.02), highlight_tint=(0.02, 0.04, 0.08), tint_strength=0.10,
    ),
    FilterDefinition(
        id="cinematic-2", name="Teal & Orange", family="Cinematic",
        description="Strong teal/orange split grade",
        rgb_curve=[(0, 0), (0.15, 0.12), (0.5, 0.53), (0.85, 0.88), (1, 1)],
        saturation_mult=1.05,
        shadow_tint=(0.14, 0.10, 0.02), highlight_tint=(0.02, 0.06, 0.12), tint_strength=0.13,
    ),
    FilterDefinition(
        id="cinematic-3", name="Muted Hollywood", family="Cinematic",
        description="Desaturated blockbuster look",
        rgb_curve=[(0, 0), (0.1, 0.08), (0.5, 0.54), (0.9, 0.93), (1, 1)],
        saturation_mult=0.80,
        shadow_tint=(0.10, 0.12, 0.04), highlight_tint=(0.03, 0.05, 0.10), tint_strength=0.12,
    ),
    FilterDefinition(
        id="cinematic-4", name="Dark Thriller", family="Cinematic",
        description="Dark thriller — crushed blacks, teal cast",
        rgb_curve=[(0, 0), (0.1, 0.06), (0.5, 0.55), (0.9, 0.95), (1, 1)],
        saturation_mult=0.88,
        shadow_tint=(0.08, 0.14, 0.06), highlight_tint=(0.01, 0.04, 0.08), tint_strength=0.15,
        lightness_add=-0.04,
    ),
])

# ── 5. Fog / Atmospheric Family ────────────────────────────────────────────────
_register_family([
    FilterDefinition(
        id="fog-1", name="Atmospheric Haze", family="Fog",
        description="Lifted atmospheric haze",
        rgb_curve=[(0, 0.10), (0.4, 0.42), (0.8, 0.80), (1, 0.92)],
        saturation_mult=0.75, lightness_add=0.08,
    ),
    FilterDefinition(
        id="fog-2", name="Blue Mist", family="Fog",
        description="Dense morning mist — cool blue cast",
        rgb_curve=[(0, 0.12), (0.5, 0.52), (1, 0.90)],
        saturation_mult=0.65, lightness_add=0.10,
        shadow_tint=(0.08, 0.08, 0.14), tint_strength=0.07,
    ),
    FilterDefinition(
        id="fog-3", name="Golden Haze", family="Fog",
        description="Golden fog — warm dreamy haze",
        rgb_curve=[(0, 0.08), (0.5, 0.50), (1, 0.91)],
        saturation_mult=0.70, lightness_add=0.07,
        shadow_tint=(0.06, 0.05, 0.03), highlight_tint=(0.07, 0.05, 0.02), tint_strength=0.08,
    ),
    FilterDefinition(
        id="fog-4", name="Deep Perspective", family="Fog",
        description="Deep atmospheric perspective",
        rgb_curve=[(0, 0.06), (0.3, 0.30), (0.7, 0.70), (1, 0.94)],
        saturation_mult=0.60, lightness_add=0.05,
        shadow_tint=(0.05, 0.08, 0.12), tint_strength=0.06,
    ),
])

# ── 6. Natural Family — Realistic, clean, portrait/landscape boosts ───────────
_register_family([
    FilterDefinition(
        id="natural-1", name="Clean Portrait", family="Natural",
        description="Lighter skin tones with warm highlights",
        rgb_curve=[(0, 0), (0.25, 0.22), (0.75, 0.78), (1, 1)],
        r_curve=[(0, 0), (0.5, 0.52), (1, 1)],
        saturation_mult=1.05, lightness_add=0.02,
    ),
    FilterDefinition(
        id="natural-2", name="Landscape", family="Natural",
        description="Vibrant foliage greens and blue skies",
        rgb_curve=[(0, 0), (0.2, 0.16), (0.5, 0.5), (0.8, 0.84), (1, 1)],
        g_curve=[(0, 0), (0.5, 0.53), (1, 1)],
        b_curve=[(0, 0), (0.5, 0.52), (1, 1)],
        saturation_mult=1.15,
    ),
    FilterDefinition(
        id="natural-3", name="Fine Art", family="Natural",
        description="Neutral classic tones, cool shadows, warm highlights",
        rgb_curve=[(0, 0.02), (0.3, 0.28), (0.7, 0.72), (1, 0.98)],
        shadow_tint=(0.04, 0.05, 0.06), highlight_tint=(0.06, 0.05, 0.04),
        tint_strength=0.05, saturation_mult=0.95,
    ),
    FilterDefinition(
        id="natural-4", name="High Dynamic", family="Natural",
        description="Compressed contrast with open shadows",
        rgb_curve=[(0, 0.05), (0.25, 0.30), (0.75, 0.70), (1, 0.95)],
        saturation_mult=1.02, lightness_add=0.01,
    ),
])

# ── 7. Warm Family — Sunny, peach, and glowing tones ─────────────────────────
_register_family([
    FilterDefinition(
        id="warm-1", name="Golden Hour", family="Warm",
        description="Sunny ambient glow, warm orange accents",
        r_curve=[(0, 0), (0.5, 0.55), (1, 1)],
        g_curve=[(0, 0), (0.5, 0.52), (1, 1)],
        b_curve=[(0, 0), (0.5, 0.45), (1, 1)],
        saturation_mult=1.10, lightness_add=0.02,
    ),
    FilterDefinition(
        id="warm-2", name="Creamy Warm", family="Warm",
        description="Warm pastels with faded cream shadows",
        rgb_curve=[(0, 0.08), (0.4, 0.42), (0.8, 0.82), (1, 0.95)],
        r_curve=[(0, 0), (0.5, 0.53), (1, 1)],
        b_curve=[(0, 0), (0.5, 0.47), (1, 1)],
        saturation_mult=0.85, lightness_add=0.04,
    ),
    FilterDefinition(
        id="warm-3", name="Peach Sweet", family="Warm",
        description="High key lightness with soft peach-pink tones",
        rgb_curve=[(0, 0.05), (0.5, 0.55), (1, 1)],
        r_curve=[(0, 0), (0.5, 0.56), (1, 1)],
        g_curve=[(0, 0), (0.5, 0.48), (1, 1)],
        b_curve=[(0, 0), (0.5, 0.50), (1, 1)],
        saturation_mult=0.80, lightness_add=0.05,
    ),
    FilterDefinition(
        id="warm-4", name="Sunset Glow", family="Warm",
        description="Intense amber highlights, deep warm shadows",
        rgb_curve=[(0, 0), (0.25, 0.20), (0.75, 0.80), (1, 1)],
        r_curve=[(0, 0), (0.5, 0.60), (1, 1)],
        b_curve=[(0, 0), (0.5, 0.40), (1, 1)],
        saturation_mult=1.20,
    ),
])

# ── 8. Moody Family — Dark obsidian, teal/orange, and forest hues ─────────────
_register_family([
    FilterDefinition(
        id="moody-1", name="Dark Obsidian", family="Moody",
        description="Crushed blacks, heavy contrast, desaturated tones",
        rgb_curve=[(0, 0), (0.3, 0.15), (0.7, 0.65), (1, 0.9)],
        saturation_mult=0.60, lightness_add=-0.05,
    ),
    FilterDefinition(
        id="moody-2", name="Moody Forest", family="Moody",
        description="Lush dark emerald shadows, desaturated yellows",
        rgb_curve=[(0, 0.02), (0.3, 0.22), (0.7, 0.70), (1, 1)],
        r_curve=[(0, 0), (0.5, 0.46), (1, 1)],
        g_curve=[(0, 0), (0.5, 0.54), (1, 1)],
        b_curve=[(0, 0), (0.5, 0.48), (1, 1)],
        saturation_mult=0.75,
    ),
    FilterDefinition(
        id="moody-3", name="Cyberpunk", family="Moody",
        description="Deep indigo shadows, hot magenta highlights",
        rgb_curve=[(0, 0), (0.2, 0.16), (0.5, 0.5), (0.8, 0.84), (1, 1)],
        r_curve=[(0, 0), (0.5, 0.54), (1, 1)],
        g_curve=[(0, 0), (0.5, 0.45), (1, 1)],
        b_curve=[(0, 0), (0.5, 0.60), (1, 1)],
        saturation_mult=1.15,
    ),
    FilterDefinition(
        id="moody-4", name="Emerald Shadow", family="Moody",
        description="Deep green shadow tinting with cold highlights",
        rgb_curve=[(0, 0), (0.3, 0.20), (0.7, 0.75), (1, 0.95)],
        shadow_tint=(0.05, 0.12, 0.05), highlight_tint=(0.04, 0.05, 0.08),
        tint_strength=0.12, saturation_mult=0.85, lightness_add=-0.02,
    ),
])

# ── 9. Artistic Family — Creative duotones, retro-teals, and pop art ──────────
_register_family([
    FilterDefinition(
        id="artistic-1", name="Retro Teal", family="Artistic",
        description="Faded vintage teal cast with yellow highlights",
        rgb_curve=[(0, 0.05), (0.5, 0.48), (1, 0.92)],
        r_curve=[(0, 0), (0.5, 0.48), (1, 1)],
        g_curve=[(0, 0), (0.5, 0.52), (1, 1)],
        b_curve=[(0, 0), (0.5, 0.55), (1, 1)],
        saturation_mult=0.78, lightness_add=0.02,
    ),
    FilterDefinition(
        id="artistic-2", name="Duotone Cool", family="Artistic",
        description="Extreme indigo shadow and icey highlight split tint",
        rgb_curve=[(0, 0.05), (0.5, 0.5), (1, 0.95)],
        shadow_tint=(0.18, 0.10, 0.05), highlight_tint=(0.08, 0.12, 0.15),
        tint_strength=0.20, saturation_mult=0.40,
    ),
    FilterDefinition(
        id="artistic-3", name="Pop Art", family="Artistic",
        description="High vibrancy with shifted color spectrum",
        rgb_curve=[(0, 0), (0.2, 0.18), (0.8, 0.85), (1, 1)],
        hue_shift=15.0, saturation_mult=1.50,
    ),
    FilterDefinition(
        id="artistic-4", name="Solarized", family="Artistic",
        description="Creative inverted midtone curves",
        rgb_curve=[(0, 0.1), (0.2, 0.5), (0.5, 0.1), (0.8, 0.8), (1, 0.9)],
        saturation_mult=1.20,
    ),
])

# ── 10. Vintage Family — Sepia-toned faded old-school looks ──────────────────
_register_family([
    FilterDefinition(
        id="vintage-1", name="Classic Sepia", family="Vintage",
        description="Warm sepia tone, lifted blacks",
        rgb_curve=[(0, 0.06), (0.3, 0.30), (0.7, 0.72), (1, 0.94)],
        r_curve=[(0, 0.05), (0.5, 0.56), (1, 0.98)],
        g_curve=[(0, 0.03), (0.5, 0.50), (1, 0.90)],
        b_curve=[(0, 0.00), (0.5, 0.42), (1, 0.80)],
        saturation_mult=0.30, lightness_add=0.03,
    ),
    FilterDefinition(
        id="vintage-2", name="Faded Kodak", family="Vintage",
        description="Faded Kodak paper print — warm midtones, pale highlights",
        rgb_curve=[(0, 0.08), (0.4, 0.38), (0.8, 0.78), (1, 0.92)],
        r_curve=[(0, 0.05), (0.5, 0.53), (1, 0.95)],
        b_curve=[(0, 0.02), (0.5, 0.44), (1, 0.86)],
        saturation_mult=0.55, lightness_add=0.04,
        shadow_tint=(0.10, 0.08, 0.04), tint_strength=0.07,
    ),
    FilterDefinition(
        id="vintage-3", name="Retro Chrome", family="Vintage",
        description="1970s chrome-print look — shifted colors, high contrast",
        rgb_curve=[(0, 0.02), (0.2, 0.16), (0.5, 0.52), (0.8, 0.85), (1, 0.97)],
        r_curve=[(0, 0.04), (0.5, 0.55), (1, 0.99)],
        g_curve=[(0, 0.02), (0.5, 0.49), (1, 0.92)],
        b_curve=[(0, 0.00), (0.5, 0.44), (1, 0.85)],
        saturation_mult=0.70, lightness_add=0.01,
        shadow_tint=(0.08, 0.06, 0.02), tint_strength=0.06,
    ),
    FilterDefinition(
        id="vintage-4", name="Old Polaroid", family="Vintage",
        description="Polaroid SX-70 — faded yellow-green cast, soft contrast",
        rgb_curve=[(0, 0.09), (0.5, 0.50), (1, 0.90)],
        r_curve=[(0, 0.06), (0.5, 0.52), (1, 0.93)],
        g_curve=[(0, 0.05), (0.5, 0.52), (1, 0.91)],
        b_curve=[(0, 0.02), (0.5, 0.44), (1, 0.84)],
        saturation_mult=0.50, lightness_add=0.05,
        shadow_tint=(0.06, 0.07, 0.02), highlight_tint=(0.05, 0.05, 0.02), tint_strength=0.06,
    ),
])

# ── 11. B&W Family — Monochrome tone curve variants ───────────────────────────
_register_family([
    FilterDefinition(
        id="bw-1", name="B&W Standard", family="B&W",
        description="Neutral black & white — classic luminance desaturation",
        rgb_curve=[(0, 0), (0.25, 0.24), (0.5, 0.50), (0.75, 0.76), (1, 1)],
        saturation_mult=0.0, lightness_add=0.0,
    ),
    FilterDefinition(
        id="bw-2", name="B&W High Contrast", family="B&W",
        description="Punchy B&W — strong S-curve, deep blacks, bright highlights",
        rgb_curve=[(0, 0), (0.15, 0.08), (0.5, 0.52), (0.85, 0.93), (1, 1)],
        saturation_mult=0.0, lightness_add=0.0,
    ),
    FilterDefinition(
        id="bw-3", name="B&W Soft Matte", family="B&W",
        description="Soft B&W — lifted blacks, gentle midtones, matte finish",
        rgb_curve=[(0, 0.07), (0.35, 0.34), (0.70, 0.70), (1, 0.93)],
        saturation_mult=0.0, lightness_add=0.03,
    ),
    FilterDefinition(
        id="bw-4", name="Selenium Tone", family="B&W",
        description="Selenium-toned silver print — cool purple-gray cast",
        rgb_curve=[(0, 0.01), (0.3, 0.27), (0.7, 0.73), (1, 0.99)],
        r_curve=[(0, 0), (0.5, 0.49), (1, 0.97)],
        b_curve=[(0, 0.01), (0.5, 0.52), (1, 1.0)],
        saturation_mult=0.10, lightness_add=0.0,
        shadow_tint=(0.06, 0.04, 0.08), tint_strength=0.06,
    ),
])

# ── 12. Cool Tones Family — Arctic, ice-blue, and cold atmospheric casts ───────
_register_family([
    FilterDefinition(
        id="cool-1", name="Arctic Light", family="Cool Tones",
        description="Clean cold light — boosted blues, reduced warmth",
        rgb_curve=[(0, 0), (0.25, 0.23), (0.75, 0.78), (1, 1)],
        r_curve=[(0, 0), (0.5, 0.47), (1, 0.95)],
        b_curve=[(0, 0.01), (0.5, 0.54), (1, 1.0)],
        saturation_mult=0.92, lightness_add=0.01,
    ),
    FilterDefinition(
        id="cool-2", name="Ice Blue", family="Cool Tones",
        description="Icy blue cast — vivid cold highlights, steel shadows",
        rgb_curve=[(0, 0), (0.2, 0.17), (0.5, 0.51), (0.8, 0.84), (1, 1)],
        r_curve=[(0, 0), (0.5, 0.45), (1, 0.92)],
        b_curve=[(0, 0.02), (0.5, 0.56), (1, 1.0)],
        saturation_mult=1.05,
        shadow_tint=(0.08, 0.08, 0.14), highlight_tint=(0.02, 0.05, 0.12), tint_strength=0.10,
    ),
    FilterDefinition(
        id="cool-3", name="Overcast", family="Cool Tones",
        description="Flat cool overcast day — muted, even, desaturated blues",
        rgb_curve=[(0, 0.04), (0.4, 0.38), (0.8, 0.79), (1, 0.95)],
        saturation_mult=0.72,
        shadow_tint=(0.06, 0.07, 0.11), tint_strength=0.06,
    ),
    FilterDefinition(
        id="cool-4", name="Moonlight", family="Cool Tones",
        description="Moonlit scene — deep indigo shadows, pale silver highlights",
        rgb_curve=[(0, 0), (0.15, 0.10), (0.5, 0.52), (0.85, 0.89), (1, 1)],
        r_curve=[(0, 0), (0.5, 0.45), (1, 0.93)],
        b_curve=[(0, 0.02), (0.5, 0.57), (1, 1.0)],
        saturation_mult=0.78, lightness_add=-0.02,
        shadow_tint=(0.10, 0.08, 0.16), highlight_tint=(0.03, 0.04, 0.10), tint_strength=0.12,
    ),
])


# ── LUT Cache ─────────────────────────────────────────────────────────────────

_lut_cache: Dict[str, np.ndarray] = {}


def _interpolate_curve(control_points: list, size: int = 256) -> np.ndarray:
    """Build a 1D lookup table from control points via cubic interpolation."""
    if not control_points:
        return np.linspace(0, 1, size, dtype=np.float32)

    xs = np.array([p[0] for p in control_points], dtype=np.float32)
    ys = np.array([p[1] for p in control_points], dtype=np.float32)
    x_range = np.linspace(0, 1, size, dtype=np.float32)

    # Use numpy interp (linear) — sufficient for smooth curves
    curve = np.interp(x_range, xs, ys).astype(np.float32)
    return np.clip(curve, 0.0, 1.0)


def _build_lut(fdef: FilterDefinition) -> np.ndarray:
    """
    Build a 32x32x32x3 3D LUT from a FilterDefinition.
    Returns array of shape [LUT_SIZE, LUT_SIZE, LUT_SIZE, 3] in BGR order.
    """
    n = LUT_SIZE

    # Create identity LUT
    # r, g, b indices map to actual float values
    b_vals = np.linspace(0, 1, n, dtype=np.float32)
    g_vals = np.linspace(0, 1, n, dtype=np.float32)
    r_vals = np.linspace(0, 1, n, dtype=np.float32)

    # Shape: [n, n, n, 3] in BGR
    bb, gg, rr = np.meshgrid(b_vals, g_vals, r_vals, indexing='ij')
    lut = np.stack([bb, gg, rr], axis=-1)  # [n, n, n, 3] BGR

    # ── Apply RGB tone curve ──────────────────────────────────────────────────
    if fdef.rgb_curve:
        curve = _interpolate_curve(fdef.rgb_curve)
        for c in range(3):
            lut[:, :, :, c] = np.interp(lut[:, :, :, c], np.linspace(0, 1, 256), curve)

    # ── Apply per-channel curves ──────────────────────────────────────────────
    curves_bgr = [fdef.b_curve, fdef.g_curve, fdef.r_curve]  # BGR order
    for c, ch_curve in enumerate(curves_bgr):
        if ch_curve:
            curve = _interpolate_curve(ch_curve)
            lut[:, :, :, c] = np.interp(lut[:, :, :, c], np.linspace(0, 1, 256), curve)

    # ── Apply HSL adjustments ──────────────────────────────────────────────────
    if fdef.saturation_mult != 1.0 or fdef.lightness_add != 0.0 or fdef.hue_shift != 0.0:
        lut_u8 = (np.clip(lut, 0, 1) * 255).astype(np.uint8)
        lut_reshaped = lut_u8.reshape(-1, 1, 3)
        hsv = cv2.cvtColor(lut_reshaped, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue shift
        if fdef.hue_shift != 0.0:
            hsv[:, :, 0] = (hsv[:, :, 0] + fdef.hue_shift / 2.0) % 180.0

        # Saturation multiplier
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * fdef.saturation_mult, 0, 255)

        # Lightness add (as V channel adjustment)
        if fdef.lightness_add != 0.0:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + fdef.lightness_add * 255.0, 0, 255)

        lut_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        lut = lut_bgr.astype(np.float32) / 255.0
        lut = lut.reshape(n, n, n, 3)

    # ── Apply shadow/highlight tinting ────────────────────────────────────────
    if fdef.shadow_tint or fdef.highlight_tint:
        luminance = 0.299 * lut[:, :, :, 2] + 0.587 * lut[:, :, :, 1] + 0.114 * lut[:, :, :, 0]

        if fdef.shadow_tint:
            sh_mask = np.clip((0.5 - luminance) * 2.0, 0.0, 1.0)[:, :, :, np.newaxis]
            tint = np.array(fdef.shadow_tint, dtype=np.float32)
            lut = lut + sh_mask * tint * fdef.tint_strength

        if fdef.highlight_tint:
            hi_mask = np.clip((luminance - 0.5) * 2.0, 0.0, 1.0)[:, :, :, np.newaxis]
            tint = np.array(fdef.highlight_tint, dtype=np.float32)
            lut = lut + hi_mask * tint * fdef.tint_strength

    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def _get_lut(filter_id: str) -> Optional[np.ndarray]:
    """Get cached LUT, building it on first access."""
    if filter_id not in _lut_cache:
        fdef = FILTER_DEFINITIONS.get(filter_id)
        if fdef is None:
            return None
        _lut_cache[filter_id] = _build_lut(fdef)
    return _lut_cache[filter_id]


def _apply_3d_lut_trilinear(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply a 3D LUT to an image using trilinear interpolation.
    img: float32 BGR [0,1], shape [H, W, 3]
    lut: float32 [LUT_SIZE, LUT_SIZE, LUT_SIZE, 3] in BGR
    """
    n = lut.shape[0]
    scale = (n - 1)

    # Scale pixel values to LUT index space
    b = img[:, :, 0] * scale
    g = img[:, :, 1] * scale
    r = img[:, :, 2] * scale

    # Floor and ceil indices
    b0 = np.clip(np.floor(b).astype(int), 0, n - 2)
    g0 = np.clip(np.floor(g).astype(int), 0, n - 2)
    r0 = np.clip(np.floor(r).astype(int), 0, n - 2)
    b1, g1, r1 = b0 + 1, g0 + 1, r0 + 1

    # Fractional offsets
    fb = (b - b0)[:, :, np.newaxis]
    fg = (g - g0)[:, :, np.newaxis]
    fr = (r - r0)[:, :, np.newaxis]

    # Trilinear interpolation (8 corners of the LUT cube)
    c000 = lut[b0, g0, r0]
    c001 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]
    c011 = lut[b0, g1, r1]
    c100 = lut[b1, g0, r0]
    c101 = lut[b1, g0, r1]
    c110 = lut[b1, g1, r0]
    c111 = lut[b1, g1, r1]

    result = (
        c000 * (1 - fb) * (1 - fg) * (1 - fr) +
        c001 * (1 - fb) * (1 - fg) * fr +
        c010 * (1 - fb) * fg * (1 - fr) +
        c011 * (1 - fb) * fg * fr +
        c100 * fb * (1 - fg) * (1 - fr) +
        c101 * fb * (1 - fg) * fr +
        c110 * fb * fg * (1 - fr) +
        c111 * fb * fg * fr
    )

    return np.clip(result, 0.0, 1.0)


def apply_filter(img: np.ndarray, filter_id: str, intensity: float = 1.0) -> np.ndarray:
    """
    Apply a named filter to a float32 BGR image [0,1].
    intensity: 0.0 (no effect) to 1.0 (full effect).
    This is the function injected into process_frame() as filter_lut_fn.
    """
    if filter_id == "none" or intensity <= 0.0:
        return img

    lut = _get_lut(filter_id)
    if lut is None:
        return img

    filtered = _apply_3d_lut_trilinear(img, lut)

    # Blend between original and filtered based on intensity
    if intensity >= 1.0:
        return filtered
    return img * (1.0 - intensity) + filtered * intensity


def get_filter_families() -> Dict[str, list]:
    """Returns filters organized by family for the UI."""
    families: Dict[str, list] = {}
    for fdef in FILTER_DEFINITIONS.values():
        families.setdefault(fdef.family, []).append(fdef)
    return families


def preload_all_luts() -> None:
    """Pre-build all LUTs at startup to avoid hitching during editing."""
    for filter_id in FILTER_DEFINITIONS:
        _get_lut(filter_id)


def get_filter_thumbnail(filter_id: str, source_img: np.ndarray, target_w: int = 72, target_h: int = 58) -> np.ndarray:
    """
    Generate a small thumbnail showing what a filter looks like on the source image,
    preserving the real aspect ratio, and filling the gaps with a blurred background.
    source_img: uint8 BGR full-size image
    Returns: uint8 BGR image of size (target_h, target_w, 3)
    """
    h, w = source_img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 1. Generate the blurred background (fill/cover target_w x target_h)
    scale_w = target_w / w
    scale_h = target_h / h
    scale = max(scale_w, scale_h)

    bg_w = int(round(w * scale))
    bg_h = int(round(h * scale))

    bg = cv2.resize(source_img, (bg_w, bg_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    # Center crop background to target_w x target_h
    start_x = max(0, (bg_w - target_w) // 2)
    start_y = max(0, (bg_h - target_h) // 2)
    bg_cropped = bg[start_y:start_y+target_h, start_x:start_x+target_w]

    # Handle tiny rounding discrepancies to ensure exact target_w x target_h size
    if bg_cropped.shape[1] != target_w or bg_cropped.shape[0] != target_h:
        bg_cropped = cv2.resize(bg_cropped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Apply heavy blur to the background cropped image
    blurred_bg = cv2.GaussianBlur(bg_cropped, (15, 15), 0)
    # Dim the blurred background slightly so the foreground stands out
    blurred_bg = (blurred_bg * 0.6).astype(np.uint8)

    # 2. Generate the foreground image (fit/contain inside target_w x target_h)
    scale_fit = min(scale_w, scale_h)
    fg_w = max(1, int(round(w * scale_fit)))
    fg_h = max(1, int(round(h * scale_fit)))

    fg = cv2.resize(source_img, (fg_w, fg_h), interpolation=cv2.INTER_AREA if scale_fit < 1.0 else cv2.INTER_LINEAR)

    # 3. Paste the foreground onto the center of the blurred background
    dx = max(0, (target_w - fg_w) // 2)
    dy = max(0, (target_h - fg_h) // 2)

    result = blurred_bg.copy()
    # Crop fg if it exceeds bounds (due to rounding)
    fg_crop_h = min(fg_h, target_h - dy)
    fg_crop_w = min(fg_w, target_w - dx)
    result[dy:dy+fg_crop_h, dx:dx+fg_crop_w] = fg[:fg_crop_h, :fg_crop_w]

    # 4. Apply the filter to the thumbnail
    thumb_f = result.astype(np.float32) / 255.0
    result_f = apply_filter(thumb_f, filter_id, 1.0)
    return (result_f * 255.0).astype(np.uint8)
