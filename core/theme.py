"""
core/theme.py
Design tokens and dynamic stylesheet compilation for Light/Dark themes.
"""

from pathlib import Path
from core.win_effects import is_windows_11

THEME_TOKENS = {
    "dark": {
        "translucent": {
            "WINDOW_BG": "rgba(28, 28, 28, 0.4)",
            "PANEL_BG": "rgba(32, 32, 32, 0.65)",
            "PREVIEW_BG": "rgba(20, 20, 20, 0.6)",
            "BORDER_COLOR": "rgba(255, 255, 255, 0.08)",
            "TEXT_COLOR": "#E8E8F0",
            "TEXT_COLOR_DIM": "#9090A8",
            "CONTROL_BG": "rgba(255, 255, 255, 0.06)",
            "CONTROL_BG_HOVER": "rgba(255, 255, 255, 0.1)",
            "CONTROL_BG_PRESSED": "rgba(255, 255, 255, 0.04)",
            "CONTROL_BORDER": "rgba(255, 255, 255, 0.08)",
            "ACCENT_COLOR": "#0078D4",
            "ACCENT_HOVER": "#2B88D8",
            "ACCENT_PRESSED": "#005A9E",
            "ACCENT_LIGHT": "rgba(0, 120, 212, 0.15)",
            "LIST_ITEM_HOVER": "rgba(255, 255, 255, 0.04)",
            "LIST_ITEM_SELECTED": "rgba(255, 255, 255, 0.12)",
            "SCROLL_HANDLE": "rgba(255, 255, 255, 0.25)",
            "SLIDER_GROOVE": "#333333",
            "DANGER_BG": "rgba(139, 51, 51, 0.2)",
            "DANGER_BORDER": "rgba(139, 51, 51, 0.4)",
            "DANGER_TEXT": "#FF8080",
            "DANGER_BG_HOVER": "rgba(139, 51, 51, 0.3)",
            "DANGER_BORDER_HOVER": "rgba(192, 80, 80, 0.6)",
        },
        "solid": {
            "WINDOW_BG": "#1C1C1C",
            "PANEL_BG": "#202020",
            "PREVIEW_BG": "#141414",
            "BORDER_COLOR": "#2D2D2D",
            "TEXT_COLOR": "#E8E8F0",
            "TEXT_COLOR_DIM": "#9090A8",
            "CONTROL_BG": "#2D2D2D",
            "CONTROL_BG_HOVER": "#323232",
            "CONTROL_BG_PRESSED": "#252525",
            "CONTROL_BORDER": "#353535",
            "ACCENT_COLOR": "#0078D4",
            "ACCENT_HOVER": "#2B88D8",
            "ACCENT_PRESSED": "#005A9E",
            "ACCENT_LIGHT": "rgba(0, 120, 212, 0.06)",
            "LIST_ITEM_HOVER": "rgba(255, 255, 255, 0.04)",
            "LIST_ITEM_SELECTED": "#35354A",
            "SCROLL_HANDLE": "#424242",
            "SLIDER_GROOVE": "#333333",
            "DANGER_BG": "#2D1F1F",
            "DANGER_BORDER": "#8B3333",
            "DANGER_TEXT": "#FF8080",
            "DANGER_BG_HOVER": "#3C2525",
            "DANGER_BORDER_HOVER": "#C05050",
        }
    },
    "light": {
        "translucent": {
            "WINDOW_BG": "rgba(248, 249, 250, 0.4)",
            "PANEL_BG": "rgba(255, 255, 255, 0.65)",
            "PREVIEW_BG": "rgba(243, 244, 246, 0.6)",
            "BORDER_COLOR": "rgba(229, 231, 235, 0.8)",
            "TEXT_COLOR": "#111827",
            "TEXT_COLOR_DIM": "#4B5563",
            "CONTROL_BG": "rgba(243, 244, 246, 0.8)",
            "CONTROL_BG_HOVER": "rgba(229, 231, 235, 0.8)",
            "CONTROL_BG_PRESSED": "rgba(209, 213, 219, 0.8)",
            "CONTROL_BORDER": "rgba(209, 213, 219, 0.8)",
            "ACCENT_COLOR": "#2563EB",
            "ACCENT_HOVER": "#1D4ED8",
            "ACCENT_PRESSED": "#1E40AF",
            "ACCENT_LIGHT": "rgba(37, 99, 235, 0.1)",
            "LIST_ITEM_HOVER": "rgba(243, 244, 246, 0.8)",
            "LIST_ITEM_SELECTED": "rgba(37, 99, 235, 0.15)",
            "SCROLL_HANDLE": "rgba(156, 163, 175, 0.4)",
            "SLIDER_GROOVE": "#D1D5DB",
            "DANGER_BG": "rgba(254, 226, 226, 0.6)",
            "DANGER_BORDER": "rgba(252, 165, 165, 0.8)",
            "DANGER_TEXT": "#991B1B",
            "DANGER_BG_HOVER": "rgba(254, 202, 202, 0.8)",
            "DANGER_BORDER_HOVER": "rgba(248, 113, 113, 0.8)",
        },
        "solid": {
            "WINDOW_BG": "#F8F9FA",
            "PANEL_BG": "#FFFFFF",
            "PREVIEW_BG": "#F3F4F6",
            "BORDER_COLOR": "#E5E7EB",
            "TEXT_COLOR": "#111827",
            "TEXT_COLOR_DIM": "#4B5563",
            "CONTROL_BG": "#F3F4F6",
            "CONTROL_BG_HOVER": "#E5E7EB",
            "CONTROL_BG_PRESSED": "#D1D5DB",
            "CONTROL_BORDER": "#D1D5DB",
            "ACCENT_COLOR": "#2563EB",
            "ACCENT_HOVER": "#1D4ED8",
            "ACCENT_PRESSED": "#1E40AF",
            "ACCENT_LIGHT": "rgba(37, 99, 235, 0.08)",
            "LIST_ITEM_HOVER": "#F3F4F6",
            "LIST_ITEM_SELECTED": "#EFF6FF",
            "SCROLL_HANDLE": "#9CA3AF",
            "SLIDER_GROOVE": "#D1D5DB",
            "DANGER_BG": "#FEE2E2",
            "DANGER_BORDER": "#FCA5A5",
            "DANGER_TEXT": "#991B1B",
            "DANGER_BG_HOVER": "#FECACA",
            "DANGER_BORDER_HOVER": "#F87171",
        }
    }
}

def compile_theme(theme_name: str) -> str:
    """
    Read theme_template.qss and replace color tokens based on platform compatibility.
    """
    project_root = Path(__file__).parent.parent
    template_path = project_root / "assets" / "styles" / "theme_template.qss"
    
    if not template_path.exists():
        return ""

    qss_text = template_path.read_text(encoding="utf-8")
    
    # Force light mode to always use solid tokens for legibility and visual elegance,
    # preventing muddy blend effects on Windows translucent backdrops.
    if theme_name == "light":
        mode = "solid"
    else:
        mode = "translucent" if is_windows_11() else "solid"
        
    tokens = THEME_TOKENS.get(theme_name, THEME_TOKENS["dark"])[mode]

    # Process replacements
    for token, val in tokens.items():
        qss_text = qss_text.replace(f"@{token}@", val)

    return qss_text
