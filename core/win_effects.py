"""
core/win_effects.py
Windows 11 platform-specific window backdrop effects wrapper.
"""

import sys
import ctypes

def is_windows_11() -> bool:
    """Check if the operating system is Windows 11 (build >= 22000)."""
    if sys.platform == "win32":
        try:
            return sys.getwindowsversion().build >= 22000
        except Exception:
            pass
    return False

def apply_win11_theme_effects(hwnd: int, dark_mode: bool):
    """
    Apply Windows 11 Fluent theme backdrop (Mica Alt or Mica) and titlebar styling.
    """
    if sys.platform != "win32":
        return

    try:
        dwmapi = ctypes.windll.dwmapi

        # 1. Force titlebar immersive dark mode
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        dark_val = ctypes.c_int(1 if dark_mode else 0)
        dwmapi.DwmSetWindowAttribute(
            hwnd,
            20,
            ctypes.byref(dark_val),
            ctypes.sizeof(dark_val)
        )

        # Retrieve windows version
        win_ver = sys.getwindowsversion()

        # 2. System backdrop selection
        # DWMWA_SYSTEMBACKDROP_TYPE = 38 (introduced in build 22621)
        # DWMSBT_MAINWINDOW = 2 (Mica), DWMSBT_TRANSIENTWINDOW = 3 (Acrylic), DWMSBT_TABBEDWINDOW = 4 (Mica Alt)
        if win_ver.build >= 22621:
            backdrop_val = ctypes.c_int(4 if dark_mode else 2)  # Mica Alt for dark, Mica for light
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                38,
                ctypes.byref(backdrop_val),
                ctypes.sizeof(backdrop_val)
            )
        elif win_ver.build >= 22000:
            # Fallback for original Windows 11 build 22000: DWMWA_MICA_EFFECT = 1029
            mica_val = ctypes.c_int(1 if dark_mode else 0)
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                1029,
                ctypes.byref(mica_val),
                ctypes.sizeof(mica_val)
            )
    except Exception as e:
        print(f"[PixClip] Win11 backdrop effect application failed: {e}")
