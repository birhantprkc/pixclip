"""
main.py
PixClip — High-Performance Desktop Image & Video Editing Application
Entry point: sets up the Qt application, applies the dark theme, and launches MainWindow.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path (for `python main.py` invocation)
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication, QTimer
from PySide6.QtGui import QFont, QIcon, QPalette, QColor

from ui.main_window import MainWindow


def load_stylesheet(app: QApplication, theme_name: str = "dark") -> None:
    from core.theme import compile_theme
    stylesheet = compile_theme(theme_name)
    app.setStyleSheet(stylesheet)


def configure_palette(app: QApplication, theme_name: str = "dark") -> None:
    """Configure QPalette at the Qt level for native widgets."""
    palette = QPalette()
    if theme_name == "dark":
        dark_color = QColor("#1C1C1C")
        mid_dark = QColor("#202020")
        accent = QColor("#0078D4")
        text = QColor("#E8E8F0")
        dim_text = QColor("#9090A8")

        palette.setColor(QPalette.ColorRole.Window, dark_color)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Base, mid_dark)
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2D2D2D"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, mid_dark)
        palette.setColor(QPalette.ColorRole.ToolTipText, text)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.Button, QColor("#252535"))
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.Highlight, accent)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.PlaceholderText, dim_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#404050"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#404050"))
    else:
        # Light mode palette - aligned with solid light design tokens
        light_color = QColor("#F8F9FA")
        light_base = QColor("#FFFFFF")
        accent = QColor("#2563EB")
        text = QColor("#111827")
        dim_text = QColor("#4B5563")

        palette.setColor(QPalette.ColorRole.Window, light_color)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Base, light_base)
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#E5E7EB"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, light_base)
        palette.setColor(QPalette.ColorRole.ToolTipText, text)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.Button, QColor("#F3F4F6"))
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#000000"))
        palette.setColor(QPalette.ColorRole.Highlight, accent)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.PlaceholderText, dim_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#9CA3AF"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#9CA3AF"))

    app.setPalette(palette)


def main() -> int:
    from core.version import __version__
    app = QApplication(sys.argv)
    app.setApplicationName("PixClip")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("PixClip")

    # Set application window icon
    logo_path = Path(__file__).parent / "assets" / "styles" / "logo.png"
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))

    # Load settings to get preferred theme
    from PySide6.QtCore import QSettings
    settings = QSettings("PixClip", "PixClip")
    theme_name = settings.value("theme", "dark")

    # Apply theme
    configure_palette(app, theme_name)
    load_stylesheet(app, theme_name)

    # Global font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Launch main window
    window = MainWindow()
    window.show()

    # If files were passed as command-line arguments, open them after window is shown
    if len(sys.argv) > 1:
        from pathlib import Path as _Path
        paths = [_Path(a) for a in sys.argv[1:] if _Path(a).exists()]
        if paths:
            QTimer.singleShot(100, lambda: window._assets._add_files(paths))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
