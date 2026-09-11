"""
ui/main_window.py
PixClip MainWindow — the central controller.

Layout:
  [Asset Panel | Preview + Filter Strip | Adjustment Panel]

Responsibilities:
- Connects all UI components
- Manages the QThread-based preview render worker
- Routes slider changes through scope-aware project state
- Handles keyboard shortcuts, file I/O, batch export
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt, Signal, QThread, QObject, QTimer, Slot, QSize, QSettings,
)
from PySide6.QtGui import (
    QAction, QKeySequence, QIcon, QPixmap, QColor, QShortcut, QImage,
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QStatusBar, QProgressBar, QLabel, QPushButton, QFileDialog,
    QMenuBar, QMenu, QDialog, QMessageBox, QFrame, QApplication,
    QComboBox, QTabWidget,
)

from core.params import AdjustmentParams, ImageRecord, ProjectState, VideoRecord
from core.pipeline import process_frame, process_frame_preview
from core.filters import apply_filter, preload_all_luts
from core.batch import BatchProcessor, ExportOptions, ExportFormat

from ui.preview_widget import PreviewWidget
from ui.asset_panel import AssetPanel
from ui.adjustment_panel import AdjustmentPanel
from ui.filter_strip import FilterStrip
from ui.filmstrip import FilmstripWidget
from ui.scope_selector import ScopeSelector
from ui.preset_dialog import PresetDialog

from ui.video_asset_panel import VideoAssetPanel
from ui.video_preview_widget import VideoPreviewWidget
from core.video_processor import VideoExportWorker
from ui.quick_video_editor import QuickVideoEditor


# ── Background Render Worker ──────────────────────────────────────────────────

class RenderWorker(QObject):
    """
    Processes a frame in a background thread and emits the result.
    Uses half-resolution for instant preview feedback.
    """
    result_ready = Signal(str, np.ndarray)

    def __init__(self):
        super().__init__()
        self._pending: Optional[tuple] = None
        self._running = True

    @Slot(str, np.ndarray, AdjustmentParams, float)
    def render(self, image_id: str, frame: np.ndarray, params: AdjustmentParams, scale: float) -> None:
        result = process_frame_preview(frame, params, scale=scale,
                                       filter_lut_fn=apply_filter)
        
        # Save to disk cache in background thread
        from core.cache import save_cached_preview
        save_cached_preview(image_id, params, result)
        
        self.result_ready.emit(image_id, result)

    @Slot(str, np.ndarray, AdjustmentParams)
    def render_full(self, image_id: str, frame: np.ndarray, params: AdjustmentParams) -> None:
        result = process_frame(frame, params, filter_lut_fn=apply_filter)
        
        # Save high resolution edited output to cache
        from core.cache import save_cached_edited
        save_cached_edited(image_id, params, result)
        
        self.result_ready.emit(image_id, result)


# ── Background Export Workers & Dialogs ────────────────────────────────────────

class SingleExportWorker(QThread):
    """Background worker to process and save a single image asynchronously."""
    finished = Signal(bool, str)  # (success, message)

    def __init__(self, img_record: ImageRecord, params: AdjustmentParams, out_path: str):
        super().__init__()
        self.record = img_record
        self.params = params
        self.out_path = out_path

    def run(self):
        try:
            # 1. Load original image if not in memory
            if self.record.original is None:
                img = cv2.imread(str(self.record.path), cv2.IMREAD_COLOR)
                if img is None:
                    self.finished.emit(False, f"Failed to read source image: {self.record.name}")
                    return
            else:
                img = self.record.original.copy()

            # 2. Process frame using full resolution pipeline
            result = process_frame(img, self.params, filter_lut_fn=apply_filter)

            # 3. Save to disk
            ext = Path(self.out_path).suffix.lower()
            if ext in (".jpg", ".jpeg"):
                cv2.imwrite(self.out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == ".png":
                cv2.imwrite(self.out_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            elif ext == ".webp":
                cv2.imwrite(self.out_path, result, [cv2.IMWRITE_WEBP_QUALITY, 90])
            else:
                cv2.imwrite(self.out_path, result)

            self.finished.emit(True, f"Successfully exported: {Path(self.out_path).name}")
        except Exception as e:
            self.finished.emit(False, f"Export failed: {str(e)}")


class BatchExportWorker(QThread):
    """Background worker for batch processing and exporting images."""
    progress = Signal(int, int, object)  # (done, total, BatchResult)
    finished = Signal(list)  # List[BatchResult]

    def __init__(self, processor: BatchProcessor, image_ids: Optional[list[str]] = None):
        super().__init__()
        self.processor = processor
        self.image_ids = image_ids

    def run(self):
        def progress_cb(done, total, result):
            self.progress.emit(done, total, result)

        results = self.processor.run(
            image_ids=self.image_ids,
            progress_callback=progress_cb
        )
        self.finished.emit(results)


def _ndarray_to_pixmap(arr: np.ndarray, size: int) -> QPixmap:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    px = QPixmap.fromImage(qi)
    return px.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)


class SingleExportLoadingDialog(QDialog):
    """Small modal dialog showing single image export activity with original and processed previews."""
    def __init__(self, record: ImageRecord, params: AdjustmentParams, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporting Image")
        self.setModal(True)
        self.setFixedSize(480, 120)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self.record = record
        self.params = params
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Left: Original thumbnail
        self.og_thumb_label = QLabel()
        self.og_thumb_label.setObjectName("AssetThumbnail")
        self.og_thumb_label.setFixedSize(70, 70)
        self.og_thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.record.thumbnail is not None:
            px = _ndarray_to_pixmap(self.record.thumbnail, 70)
            self.og_thumb_label.setPixmap(px)
        else:
            self.og_thumb_label.setText("No OG")

        # Transition arrow
        arrow_label = QLabel("➔")
        arrow_label.setObjectName("DialogArrow")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Right: Processed thumbnail
        self.proc_thumb_label = QLabel()
        self.proc_thumb_label.setObjectName("AssetThumbnail")
        self.proc_thumb_label.setFixedSize(70, 70)
        self.proc_thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.record.thumbnail is not None:
            try:
                processed_thumb = process_frame(self.record.thumbnail, self.params, filter_lut_fn=apply_filter)
                px = _ndarray_to_pixmap(processed_thumb, 70)
                self.proc_thumb_label.setPixmap(px)
            except Exception:
                self.proc_thumb_label.setText("Error")
        else:
            self.proc_thumb_label.setText("No Preview")

        # Far Right: Info & progress
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.title_label = QLabel("Exporting and applying edits...")
        self.title_label.setObjectName("DialogTitle")
        
        self.file_label = QLabel(self.record.name)
        self.file_label.setObjectName("DialogSub")
        self.file_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)

        right_layout.addWidget(self.title_label)
        right_layout.addWidget(self.file_label)
        right_layout.addWidget(self.progress_bar)

        layout.addWidget(self.og_thumb_label)
        layout.addWidget(arrow_label)
        layout.addWidget(self.proc_thumb_label)
        layout.addLayout(right_layout, stretch=1)


class ExportProgressDialog(QDialog):
    """Dialog showing batch export progress with original to processed thumbnail previews."""
    cancelled = Signal()

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("Exporting Images")
        self.setModal(True)
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # Left: Thumbnails side-by-side comparison
        thumbs_layout = QHBoxLayout()
        thumbs_layout.setSpacing(8)

        self.og_thumb_label = QLabel()
        self.og_thumb_label.setObjectName("AssetThumbnail")
        self.og_thumb_label.setFixedSize(70, 70)
        self.og_thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.og_thumb_label.setText("Preparing")

        self.arrow_label = QLabel("➔")
        self.arrow_label.setObjectName("DialogArrow")
        self.arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.proc_thumb_label = QLabel()
        self.proc_thumb_label.setObjectName("AssetThumbnail")
        self.proc_thumb_label.setFixedSize(70, 70)
        self.proc_thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.proc_thumb_label.setText("Preparing")

        thumbs_layout.addWidget(self.og_thumb_label)
        thumbs_layout.addWidget(self.arrow_label)
        thumbs_layout.addWidget(self.proc_thumb_label)

        # Right: Info & stats
        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)

        self.title_label = QLabel("Exporting Images...")
        self.title_label.setObjectName("DialogTitle")

        self.file_label = QLabel("Starting batch export...")
        self.file_label.setObjectName("DialogSub")
        self.file_label.setWordWrap(True)

        self.stats_label = QLabel("0 / 0 files")
        self.stats_label.setObjectName("DialogStats")

        info_layout.addWidget(self.title_label)
        info_layout.addWidget(self.file_label)
        info_layout.addWidget(self.stats_label)

        content_layout.addLayout(thumbs_layout)
        content_layout.addLayout(info_layout, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("DangerButton")
        self.cancel_button.clicked.connect(self._on_cancel)

        layout.addLayout(content_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.cancel_button, alignment=Qt.AlignmentFlag.AlignRight)

    def set_progress(self, filename: str, current: int, total: int, image_id: Optional[str] = None):
        self.file_label.setText(f"Exporting: {filename}")
        self.stats_label.setText(f"{current} / {total} files")
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))

        # Update thumbnails dynamically
        if image_id:
            img = self._state.get_image(image_id)
            if img and img.thumbnail is not None:
                # 1. Original (OG) thumbnail on the left
                px_og = _ndarray_to_pixmap(img.thumbnail, 70)
                self.og_thumb_label.setPixmap(px_og)
                self.og_thumb_label.setText("")

                # 2. Processed thumbnail on the right
                params = self._state.resolved_params(image_id)
                try:
                    processed_thumb = process_frame(img.thumbnail, params, filter_lut_fn=apply_filter)
                    px_proc = _ndarray_to_pixmap(processed_thumb, 70)
                    self.proc_thumb_label.setPixmap(px_proc)
                    self.proc_thumb_label.setText("")
                except Exception:
                    self.proc_thumb_label.setText("Error")

    def _on_cancel(self):
        self.cancelled.emit()
        self.reject()

    def closeEvent(self, event):
        self.cancelled.emit()
        super().closeEvent(event)


# ── Video Export Dialog ───────────────────────────────────────────────────────

class VideoExportDialog(QDialog):
    """Modal progress dialog shown during video export, with live frame preview."""
    cancelled = Signal()

    def __init__(self, record, params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporting Video")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setFixedHeight(280)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self.record = record
        self.params = params
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel(f"Exporting: {self.record.name}")
        title.setObjectName("DialogTitle")

        # Live frame preview
        self._preview_label = QLabel()
        self._preview_label.setObjectName("AssetThumbnail")
        self._preview_label.setFixedSize(160, 100)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setText("Processing...")

        # Status label
        self._status_label = QLabel("Initializing...")
        self._status_label.setObjectName("DialogSub")
        self._status_label.setWordWrap(True)

        # Stats (frames)
        self._stats_label = QLabel("0 / 0 frames")
        self._stats_label.setObjectName("DialogStats")
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        h_info = QHBoxLayout()
        h_info.addWidget(self._status_label)
        h_info.addWidget(self._stats_label)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)

        # Cancel
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("DangerButton")
        cancel_btn.clicked.connect(self._on_cancel)

        layout.addWidget(title)
        layout.addWidget(self._preview_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addLayout(h_info)
        layout.addWidget(self._progress)
        layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def set_progress(self, status: str, current: int, total: int, preview_frame=None):
        self._status_label.setText(status)
        self._stats_label.setText(f"{current} / {total} frames")
        if total > 0:
            self._progress.setValue(int((current / total) * 100))
        if preview_frame is not None:
            import cv2
            from PySide6.QtGui import QImage, QPixmap
            rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qi = QImage(rgb.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
            px = QPixmap.fromImage(qi).scaled(
                160, 100,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._preview_label.setPixmap(px)

    def _on_cancel(self):
        self.cancelled.emit()
        self.reject()

    def closeEvent(self, event):
        self.cancelled.emit()
        super().closeEvent(event)


# ── Main Window ───────────────────────────────────────────────────────────────

PREVIEW_QUALITY_LIMITS = {
    "Keep Original": None,
    "360p": 360,
    "480p": 480,
    "720p": 720,
    "960p": 960,
    "1080p": 1080,
    "2K": 1440,
}


class MainWindow(QMainWindow):
    # Internal signals for thread communication
    _render_requested = Signal(str, np.ndarray, AdjustmentParams, float)

    def __init__(self):
        super().__init__()
        self._state = ProjectState()
        self._render_in_flight = False
        self._pending_render: Optional[tuple] = None

        # Load settings
        self._settings = QSettings("PixClip", "PixClip")
        self._preview_quality = self._settings.value("preview_quality", "720p")

        # Set window icon
        logo_path = Path(__file__).parent.parent / "assets" / "styles" / "logo.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        # Initialize disk cache
        from core.cache import init_cache
        init_cache()

        self._setup_render_thread()
        self._build_ui()
        self._build_menus()
        self._connect_signals()

        # Apply Windows 11 backdrop effects on startup
        from core.win_effects import apply_win11_theme_effects, is_windows_11
        if is_windows_11():
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            theme_name = self._settings.value("theme", "dark")
            apply_win11_theme_effects(int(self.winId()), dark_mode=(theme_name == "dark"))

        # Pre-load all LUTs in background
        QTimer.singleShot(500, preload_all_luts)

        self.setWindowTitle("PixClip — Image & Video Editor")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 640)

    # ── Thread Setup ──────────────────────────────────────────────────────────

    def _setup_render_thread(self):
        self._render_thread = QThread()
        self._worker = RenderWorker()
        self._worker.moveToThread(self._render_thread)
        self._render_requested.connect(self._worker.render)
        self._worker.result_ready.connect(self._on_render_complete)
        self._render_thread.start()

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Workspace Tabs ────────────────────────────────────────────────────
        self._workspace_tabs = QTabWidget()
        self._workspace_tabs.setObjectName("WorkspaceTabs")
        self._workspace_tabs.setDocumentMode(True)
        root.addWidget(self._workspace_tabs, stretch=1)

        # ── Tab 1: Image Editor ───────────────────────────────────────────────
        image_workspace = QWidget()
        iw_layout = QVBoxLayout(image_workspace)
        iw_layout.setContentsMargins(0, 0, 0, 0)
        iw_layout.setSpacing(0)

        # Scope Selector
        self._scope = ScopeSelector()
        iw_layout.addWidget(self._scope)

        # Image splitter: Asset | Center | Adjustments
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)

        # Left panel
        self._assets = AssetPanel(self._state)
        splitter.addWidget(self._assets)

        # Center area (Preview & FilterStrip)
        center = QWidget()
        center.setObjectName("PreviewContainer")
        center.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._preview = PreviewWidget()
        self._filter_strip = FilterStrip()

        center_layout.addWidget(self._preview, stretch=1)
        center_layout.addWidget(self._filter_strip)
        splitter.addWidget(center)

        # Right panel (Adjustments)
        right = QWidget()
        right.setObjectName("AdjustmentPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Preset/reset bar
        preset_bar = QWidget()
        preset_bar.setFixedHeight(38)
        preset_bar.setObjectName("PresetBar")
        pb_layout = QHBoxLayout(preset_bar)
        pb_layout.setContentsMargins(10, 4, 10, 4)
        pb_layout.setSpacing(6)

        presets_lbl = QLabel("ADJUSTMENTS")
        presets_lbl.setObjectName("SectionHeader")

        self._btn_presets = QPushButton("Presets")
        self._btn_presets.setFixedHeight(28)
        self._btn_presets.clicked.connect(self._open_presets)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setFixedHeight(28)
        self._btn_reset.setObjectName("DangerButton")
        self._btn_reset.clicked.connect(self._reset_all)
        self._btn_reset.setToolTip("Reset all adjustments to default")

        pb_layout.addWidget(presets_lbl)
        pb_layout.addStretch()
        pb_layout.addWidget(self._btn_presets)
        pb_layout.addWidget(self._btn_reset)

        self._adj_panel = AdjustmentPanel()
        right_layout.addWidget(preset_bar)
        right_layout.addWidget(self._adj_panel, stretch=1)
        splitter.addWidget(right)

        splitter.setSizes([160, 900, 280])
        iw_layout.addWidget(splitter, stretch=1)

        # Bottom filmstrip
        self._filmstrip = FilmstripWidget()
        iw_layout.addWidget(self._filmstrip)

        self._workspace_tabs.addTab(image_workspace, "Image Editor")

        # ── Tab 2: Video Editor ───────────────────────────────────────────────
        video_workspace = QWidget()
        vw_layout = QVBoxLayout(video_workspace)
        vw_layout.setContentsMargins(0, 0, 0, 0)
        vw_layout.setSpacing(0)



        # Video splitter: Asset | Center | Adjustments
        video_splitter = QSplitter(Qt.Orientation.Horizontal)
        video_splitter.setHandleWidth(2)
        video_splitter.setChildrenCollapsible(False)

        # Left panel (Video assets)
        self._video_assets = VideoAssetPanel(self._state)
        video_splitter.addWidget(self._video_assets)

        # Center area (Video preview & FilterStrip)
        video_center = QWidget()
        video_center.setObjectName("PreviewContainer")
        video_center.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        vc_layout = QVBoxLayout(video_center)
        vc_layout.setContentsMargins(0, 0, 0, 0)
        vc_layout.setSpacing(0)

        self._video_preview = VideoPreviewWidget()
        self._video_filter_strip = FilterStrip()

        vc_layout.addWidget(self._video_preview, stretch=1)
        vc_layout.addWidget(self._video_filter_strip)
        video_splitter.addWidget(video_center)

        # Right panel (Video adjustments)
        video_right = QWidget()
        video_right.setObjectName("AdjustmentPanel")
        vr_layout = QVBoxLayout(video_right)
        vr_layout.setContentsMargins(0, 0, 0, 0)
        vr_layout.setSpacing(0)

        # Video preset/reset bar
        video_preset_bar = QWidget()
        video_preset_bar.setFixedHeight(38)
        video_preset_bar.setObjectName("PresetBar")
        vpb_layout = QHBoxLayout(video_preset_bar)
        vpb_layout.setContentsMargins(10, 4, 10, 4)
        vpb_layout.setSpacing(6)

        video_presets_lbl = QLabel("ADJUSTMENTS")
        video_presets_lbl.setObjectName("SectionHeader")

        self._btn_video_presets = QPushButton("Presets")
        self._btn_video_presets.setFixedHeight(28)
        self._btn_video_presets.clicked.connect(self._open_video_presets)

        self._btn_video_reset = QPushButton("Reset")
        self._btn_video_reset.setFixedHeight(28)
        self._btn_video_reset.setObjectName("DangerButton")
        self._btn_video_reset.clicked.connect(self._reset_video_all)
        self._btn_video_reset.setToolTip("Reset all video adjustments to default")

        vpb_layout.addWidget(video_presets_lbl)
        vpb_layout.addStretch()
        vpb_layout.addWidget(self._btn_video_presets)
        vpb_layout.addWidget(self._btn_video_reset)

        self._video_adj_panel = AdjustmentPanel()
        vr_layout.addWidget(video_preset_bar)
        vr_layout.addWidget(self._video_adj_panel, stretch=1)
        video_splitter.addWidget(video_right)

        video_splitter.setSizes([160, 900, 280])
        vw_layout.addWidget(video_splitter, stretch=1)

        self._workspace_tabs.addTab(video_workspace, "Video Editor")

        # ── Tab 3: Quick Video Editor ──────────────────────────────────────
        self._quick_editor = QuickVideoEditor(self._state)
        self._workspace_tabs.addTab(self._quick_editor, "⚡ Quick Export")

        # ── Status bar ────────────────────────────────────────────────────────
        self._status = QStatusBar()
        self._status.setSizeGripEnabled(False)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(200)
        self._progress.setVisible(False)
        self._status_lbl = QLabel("Ready — Drop images/videos to begin")
        self._status.addWidget(self._status_lbl)
        self._status.addPermanentWidget(self._progress)

        # Preview Quality Select combo box
        quality_container = QWidget()
        q_layout = QHBoxLayout(quality_container)
        q_layout.setContentsMargins(0, 0, 5, 0)
        q_layout.setSpacing(5)

        q_label = QLabel("Preview Quality:")
        q_label.setObjectName("InfoSub")

        self._quality_combo = QComboBox()
        self._quality_combo.addItems([
            "Keep Original",
            "360p",
            "480p",
            "720p",
            "960p",
            "1080p",
            "2K"
        ])
        self._quality_combo.setCurrentText(self._preview_quality)
        self._quality_combo.setToolTip("Set the maximum resolution of the editing preview to speed up adjustments.")
        self._quality_combo.currentTextChanged.connect(self._change_preview_quality)

        q_layout.addWidget(q_label)
        q_layout.addWidget(self._quality_combo)
        self._status.addPermanentWidget(quality_container)

        self.setStatusBar(self._status)

    # ── Menu Build ────────────────────────────────────────────────────────────

    def _build_menus(self):
        # We will keep the file, edit, view menus as class variables
        self._menu_file = self.menuBar().addMenu("&File")
        self._menu_edit = self.menuBar().addMenu("&Edit")
        self._menu_view = self.menuBar().addMenu("&View")

        # ── Image Actions ─────────────────────────────────────────────────────
        self._act_open_images = QAction("&Open Images…", self)
        self._act_open_images.triggered.connect(self._open_images)

        self._act_export_image = QAction("&Export Current Image…", self)
        self._act_export_image.triggered.connect(self._export_current)

        self._act_batch_export = QAction("Batch &Export All Images…", self)
        self._act_batch_export.triggered.connect(self._batch_export)

        # ── Video Actions ─────────────────────────────────────────────────────
        self._act_open_videos = QAction("&Open Videos…", self)
        self._act_open_videos.triggered.connect(self._open_videos)

        self._act_export_video = QAction("Export Current &Video…", self)
        self._act_export_video.triggered.connect(self._export_current_video)

        # ── Quick Video Actions ────────────────────────────────────────────────
        self._act_export_quick = QAction("⚡ &Quick Export Video…", self)
        self._act_export_quick.triggered.connect(self._export_quick_video)

        # ── General Actions ───────────────────────────────────────────────────
        self._act_quit = QAction("&Quit", self)
        self._act_quit.triggered.connect(self.close)

        # ── Edit Actions ──────────────────────────────────────────────────────
        self._act_reset_all = QAction("Reset Adjustments", self)
        self._act_reset_all.triggered.connect(self._reset_all)

        self._act_reset_video_all = QAction("Reset Video Adjustments", self)
        self._act_reset_video_all.triggered.connect(self._reset_video_all)

        self._act_reset_quick_all = QAction("Reset Adjustments", self)
        self._act_reset_quick_all.triggered.connect(self._reset_quick_all)

        self._act_presets = QAction("Manage Presets…", self)
        self._act_presets.triggered.connect(self._open_presets)

        self._act_video_presets = QAction("Manage Presets…", self)
        self._act_video_presets.triggered.connect(self._open_video_presets)

        # ── View Actions ──────────────────────────────────────────────────────
        self._act_fit = QAction("Fit to Window", self)
        self._act_fit.triggered.connect(self._preview_fit_active)

        self._act_100 = QAction("Actual Size (1:1)", self)
        self._act_100.triggered.connect(self._preview_100_active)

        # ── Theme Actions ─────────────────────────────────────────────────────
        self._act_dark_theme = QAction("Dark Theme", self)
        self._act_dark_theme.setCheckable(True)
        self._act_dark_theme.triggered.connect(lambda: self._set_theme("dark"))

        self._act_light_theme = QAction("Light Theme", self)
        self._act_light_theme.setCheckable(True)
        self._act_light_theme.triggered.connect(lambda: self._set_theme("light"))

        from PySide6.QtGui import QActionGroup
        self._theme_group = QActionGroup(self)
        self._theme_group.addAction(self._act_dark_theme)
        self._theme_group.addAction(self._act_light_theme)
        self._theme_group.setExclusive(True)

        # Set checked state based on settings
        theme_name = self._settings.value("theme", "dark")
        if theme_name == "light":
            self._act_light_theme.setChecked(True)
        else:
            self._act_dark_theme.setChecked(True)

        # Build initial menus for the current tab (default Tab 0: Image Editor)
        self._update_menus_for_tab(0)

        # Build permanent help menu
        self._menu_help = self.menuBar().addMenu("&Help")
        self._act_about = QAction("&About PixClip…", self)
        self._act_about.triggered.connect(self._show_about_dialog)
        self._menu_help.addAction(self._act_about)

    def _show_about_dialog(self) -> None:
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        from PySide6.QtGui import QPixmap
        from PySide6.QtCore import Qt
        from pathlib import Path
        from core.version import __version__
        
        dialog = QDialog(self)
        dialog.setWindowTitle("About PixClip")
        dialog.setModal(True)
        dialog.setFixedSize(380, 260)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # Logo
        logo_lbl = QLabel()
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = Path(__file__).parent.parent / "assets" / "styles" / "logo.png"
        if logo_path.exists():
            px = QPixmap(str(logo_path)).scaled(
                64, 64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            logo_lbl.setPixmap(px)
        layout.addWidget(logo_lbl)
        
        # Title
        title_lbl = QLabel("PixClip")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #2563EB;")
        layout.addWidget(title_lbl)
        
        # Description
        desc_lbl = QLabel(f"Creative Image & Video Editor\nVersion {__version__}\n\nHigh-Performance Desktop Media Processor\nPowered by PySide6, OpenCV, and FFmpeg.")
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_lbl.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(desc_lbl)
        
        layout.addStretch()
        
        # Close button
        btn_close = QPushButton("Close")
        btn_close.setFixedWidth(100)
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)
        
        dialog.exec()

    def _open_videos(self) -> None:
        idx = self._workspace_tabs.currentIndex()
        if idx == 1:
            self._video_assets._open_file_dialog()
        elif idx == 2:
            self._quick_editor._video_list._open_file_dialog()

    def _export_quick_video(self) -> None:
        self._quick_editor._start_export()

    def _reset_quick_all(self) -> None:
        self._quick_editor._reset_adjustments()

    def _preview_fit_active(self) -> None:
        idx = self._workspace_tabs.currentIndex()
        if idx == 0:
            self._preview._view_fit()
        elif idx == 1:
            self._video_preview.fit_in_views()

    def _preview_100_active(self) -> None:
        idx = self._workspace_tabs.currentIndex()
        if idx == 0:
            self._preview._active_view().zoom_to_100()
        elif idx == 1:
            self._video_preview._active_view().zoom_to_100()

    def _update_menus_for_tab(self, index: int) -> None:
        # Clear existing actions from menus
        self._menu_file.clear()
        self._menu_edit.clear()
        self._menu_view.clear()

        # Invalidate shortcuts on all actions first to prevent conflicts
        for act in [
            self._act_open_images, self._act_export_image, self._act_batch_export,
            self._act_open_videos, self._act_export_video, self._act_export_quick,
            self._act_reset_all, self._act_reset_video_all, self._act_reset_quick_all,
            self._act_presets, self._act_video_presets, self._act_fit, self._act_100,
            self._act_quit
        ]:
            act.setShortcut(QKeySequence(""))

        # Map shortcuts and populate menus based on active tab
        if index == 0:  # Image Editor
            self._act_open_images.setShortcut(QKeySequence("Ctrl+O"))
            self._act_export_image.setShortcut(QKeySequence("Ctrl+S"))
            self._act_batch_export.setShortcut(QKeySequence("Ctrl+Shift+S"))
            self._act_quit.setShortcut(QKeySequence("Ctrl+Q"))
            self._act_reset_all.setShortcut(QKeySequence("Ctrl+R"))
            self._act_fit.setShortcut(QKeySequence("F"))
            self._act_100.setShortcut(QKeySequence("1"))

            self._menu_file.addAction(self._act_open_images)
            self._menu_file.addAction(self._act_export_image)
            self._menu_file.addAction(self._act_batch_export)
            self._menu_file.addSeparator()
            self._menu_file.addAction(self._act_quit)

            self._menu_edit.addAction(self._act_reset_all)
            self._menu_edit.addAction(self._act_presets)

            self._menu_view.addAction(self._act_fit)
            self._menu_view.addAction(self._act_100)
            self._menu_view.addSeparator()
            theme_menu = self._menu_view.addMenu("Theme")
            theme_menu.addAction(self._act_dark_theme)
            theme_menu.addAction(self._act_light_theme)

        elif index == 1:  # Video Editor
            self._act_open_videos.setShortcut(QKeySequence("Ctrl+O"))
            self._act_export_video.setShortcut(QKeySequence("Ctrl+S"))
            self._act_quit.setShortcut(QKeySequence("Ctrl+Q"))
            self._act_reset_video_all.setShortcut(QKeySequence("Ctrl+R"))
            self._act_fit.setShortcut(QKeySequence("F"))
            self._act_100.setShortcut(QKeySequence("1"))

            self._menu_file.addAction(self._act_open_videos)
            self._menu_file.addAction(self._act_export_video)
            self._menu_file.addSeparator()
            self._menu_file.addAction(self._act_quit)

            self._menu_edit.addAction(self._act_reset_video_all)
            self._menu_edit.addAction(self._act_video_presets)

            self._menu_view.addAction(self._act_fit)
            self._menu_view.addAction(self._act_100)
            self._menu_view.addSeparator()
            theme_menu = self._menu_view.addMenu("Theme")
            theme_menu.addAction(self._act_dark_theme)
            theme_menu.addAction(self._act_light_theme)

        elif index == 2:  # Quick Export
            self._act_open_videos.setShortcut(QKeySequence("Ctrl+O"))
            self._act_export_quick.setShortcut(QKeySequence("Ctrl+S"))
            self._act_quit.setShortcut(QKeySequence("Ctrl+Q"))
            self._act_reset_quick_all.setShortcut(QKeySequence("Ctrl+R"))

            self._menu_file.addAction(self._act_open_videos)
            self._menu_file.addAction(self._act_export_quick)
            self._menu_file.addSeparator()
            self._menu_file.addAction(self._act_quit)

            self._menu_edit.addAction(self._act_reset_quick_all)
            self._menu_edit.addAction(self._act_video_presets)

            theme_menu = self._menu_view.addMenu("Theme")
            theme_menu.addAction(self._act_dark_theme)
            theme_menu.addAction(self._act_light_theme)

    def _set_theme(self, theme_name: str) -> None:
        self._settings.setValue("theme", theme_name)
        
        self._act_dark_theme.setChecked(theme_name == "dark")
        self._act_light_theme.setChecked(theme_name == "light")

        from main import configure_palette
        from core.theme import compile_theme
        
        app = QApplication.instance()
        if app:
            configure_palette(app, theme_name)
            qss = compile_theme(theme_name)
            app.setStyleSheet(qss)

        # Apply Windows 11 backdrop effects if supported
        from core.win_effects import apply_win11_theme_effects, is_windows_11
        if is_windows_11():
            apply_win11_theme_effects(int(self.winId()), dark_mode=(theme_name == "dark"))

        # Repolish self to apply new QSS rules
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    # ── Signal Connections ────────────────────────────────────────────────────

    def _connect_signals(self):
        # ── Image Editor signals ──────────────────────────────────────────────
        # Asset panel
        self._assets.image_selected.connect(self._on_image_selected)
        self._assets.images_added.connect(self._on_images_added)
        self._assets.image_removed.connect(self._on_image_removed)
        self._assets.context_menu_requested.connect(self._show_image_context_menu)

        # Filmstrip
        self._filmstrip.image_selected.connect(self._assets.select_image)
        self._filmstrip.context_menu_requested.connect(self._show_image_context_menu)

        # Adjustment panel
        self._adj_panel.param_changed.connect(self._on_param_changed)

        # Filter strip (image editor)
        self._filter_strip.filter_selected.connect(self._on_filter_selected)
        self._filter_strip.intensity_changed.connect(self._on_filter_intensity_changed)

        # Scope
        self._scope.scope_changed.connect(self._on_scope_changed)
        self._scope.group_created.connect(self._on_group_created)

        # ── Video Editor signals ──────────────────────────────────────────────
        # Video asset panel
        self._video_assets.video_selected.connect(self._on_video_selected)
        self._video_assets.context_menu_requested.connect(self._show_video_context_menu)

        # Video adjustment panel
        self._video_adj_panel.param_changed.connect(self._on_video_param_changed)

        # Video filter strip
        self._video_filter_strip.filter_selected.connect(self._on_video_filter_selected)
        self._video_filter_strip.intensity_changed.connect(self._on_video_filter_intensity_changed)

        # Quick Editor → Video Editor real-time sync
        self._quick_editor.params_updated.connect(self._on_quick_editor_params_updated)

        # Workspace tab switching
        self._workspace_tabs.currentChanged.connect(self._on_workspace_tab_changed)

    # ── Image Selection ───────────────────────────────────────────────────────

    @Slot(str)
    def _on_image_selected(self, image_id: str) -> None:
        self._state.active_id = image_id
        self._filmstrip.select(image_id)

        img = self._state.get_image(image_id)
        if img is None:
            return

        # Load current params into sliders
        params = self._state.resolved_params(image_id)
        self._adj_panel.load_params(params, silent=True)
        self._filter_strip.set_active_filter(params.filter_id, silent=True)

        # Make sure original is loaded (we need it for thumbnail/preview resolutions)
        if img.original is None:
            from core.batch import load_image_record
            load_image_record(img)

        # Update filter strip source using the small thumbnail for maximum speed
        if img.thumbnail is not None:
            self._filter_strip.set_source_image(img.thumbnail)

        # Check preview cache first!
        from core.cache import get_cached_preview
        cached_preview = get_cached_preview(image_id, params)
        preview_frame = self._get_preview_frame(img)

        if cached_preview is not None:
            # We have a cached preview! Display it instantly.
            self._preview.set_image(cached_preview, original=preview_frame)
            h, w = cached_preview.shape[:2]
            scale = 80 / max(h, w)
            tw = max(1, int(w * scale))
            th = max(1, int(h * scale))
            small = cv2.resize(cached_preview, (tw, th), interpolation=cv2.INTER_AREA)
            self._filmstrip.update_thumbnail(img.id, small)
        else:
            if preview_frame is not None:
                self._preview.set_image(preview_frame, original=preview_frame)
                self._schedule_preview(img.id, preview_frame, params)

        if img.original is not None:
            h, w = img.original.shape[:2]
            self._set_status(f"{img.name}  —  {w}×{h}px")
        else:
            self._set_status(f"{img.name}  —  Loading original...")

        # Schedule lazy loading in 50ms (in case it wasn't loaded or to refresh compare views)
        QTimer.singleShot(50, lambda: self._lazy_load_original(img))

        # Memory cleanup: keep only active original in memory to save RAM
        for other_img in self._state.images:
            if other_img.id != image_id and other_img.original is not None:
                other_img.original = None  # Free RAM!

    def _lazy_load_original(self, img: ImageRecord) -> None:
        if img.original is None:
            from core.batch import load_image_record
            load_image_record(img)
            
        if img.original is not None:
            # Update filter strip source using the small thumbnail
            if self._state.active_id == img.id:
                if img.thumbnail is not None:
                    self._filter_strip.set_source_image(img.thumbnail)
                
                # Retrieve current resolved params and cached preview
                params = self._state.resolved_params(img.id)
                from core.cache import get_cached_preview
                cached = get_cached_preview(img.id, params)
                preview_frame = self._get_preview_frame(img)
                self._preview.set_image(cached if cached is not None else preview_frame, original=preview_frame)

    @Slot(list)
    def _on_images_added(self, records: list) -> None:
        for rec in records:
            self._filmstrip.add_record(rec)
        self._save_state()

    @Slot(str)
    def _on_image_removed(self, image_id: str) -> None:
        self._filmstrip.remove_record(image_id)
        self._save_state()

    # ── Param Change Handler ──────────────────────────────────────────────────

    @Slot(str, float)
    def _on_param_changed(self, param: str, value: float) -> None:
        scope = self._scope.current_scope
        img = self._state.active_image
        if img is None:
            return

        if scope == "all":
            self._state.set_param_global(param, value)
        elif scope == "group" and self._scope.current_group:
            self._state.set_param_group(self._scope.current_group, param, value)
        else:
            self._state.set_param_individual(img.id, param, value)

        # Trigger preview
        if img.original is not None:
            params = self._state.resolved_params(img.id)
            preview_frame = self._get_preview_frame(img)
            self._schedule_preview(img.id, preview_frame, params)
        self._save_state()

    @Slot(str)
    def _on_filter_selected(self, filter_id: str) -> None:
        img = self._state.active_image
        if img is None:
            return
        self._on_param_changed("filter_id", 0)  # trigger scope path
        # Set filter_id string directly
        scope = self._scope.current_scope
        if scope == "all":
            self._state.global_params.filter_id = filter_id
            self._state.invalidate_all_caches()
        elif scope == "group" and self._scope.current_group:
            gp = self._state.group_params.setdefault(
                self._scope.current_group, AdjustmentParams()
            )
            gp.filter_id = filter_id
        else:
            img.params.filter_id = filter_id
            img.invalidate_cache()

        if img.original is not None:
            params = self._state.resolved_params(img.id)
            preview_frame = self._get_preview_frame(img)
            self._schedule_preview(img.id, preview_frame, params)
            # Update filter thumbnails
            if img.thumbnail is not None:
                self._filter_strip.update_visible_thumbnails()
        self._save_state()

    @Slot(str, object)
    def _on_scope_changed(self, scope: str, group_id) -> None:
        img = self._state.active_image
        if img and img.original is not None:
            params = self._state.resolved_params(img.id)
            self._adj_panel.load_params(params, silent=True)
            preview_frame = self._get_preview_frame(img)
            self._schedule_preview(img.id, preview_frame, params)

    @Slot(str)
    def _on_group_created(self, name: str) -> None:
        # Create group in project state
        self._state.create_group(name)
        # Assign currently active image to it
        img = self._state.active_image
        if img:
            self._state.assign_to_group(img.id, name)
            self._refresh_image_group_display(img.id)

    def _refresh_image_group_display(self, image_id: str) -> None:
        img = self._state.get_image(image_id)
        if not img:
            return
        # Refresh Asset Card
        card = self._assets._cards.get(image_id)
        if card:
            card.update_group_info()
        # Refresh Filmstrip
        thumb = self._filmstrip._thumbs.get(image_id)
        if thumb:
            thumb.update_group_tooltip(img)

    def _show_image_context_menu(self, image_id: str, pos) -> None:
        img = self._state.get_image(image_id)
        if not img:
            return

        menu = QMenu(self)
        
        # Submenu: Assign to Group
        group_menu = menu.addMenu("Assign to Group")
        
        # No Group action
        act_no_group = group_menu.addAction("No Group")
        act_no_group.setCheckable(True)
        act_no_group.setChecked(img.group_id is None)
        
        # Existing groups
        groups = self._state.get_groups()
        group_actions = {}
        if groups:
            group_menu.addSeparator()
            for g in groups:
                act = group_menu.addAction(g)
                act.setCheckable(True)
                act.setChecked(img.group_id == g)
                group_actions[act] = g
                
        group_menu.addSeparator()
        act_new_group = group_menu.addAction("+ New Group...")
        
        # Remove action
        action_remove = menu.addAction("Remove from Project")
        
        action = menu.exec(pos)
        if not action:
            return
            
        if action == action_remove:
            self._assets._remove_image(image_id)
        elif action == act_no_group:
            self._state.assign_to_group(image_id, None)
            self._refresh_image_group_display(image_id)
            if self._scope.current_scope == "group":
                self._on_scope_changed("group", self._scope.current_group)
        elif action in group_actions:
            target_group = group_actions[action]
            self._state.assign_to_group(image_id, target_group)
            self._refresh_image_group_display(image_id)
            if self._scope.current_scope == "group":
                self._on_scope_changed("group", self._scope.current_group)
        elif action == act_new_group:
            from PySide6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(
                self, "Create Group", "Enter new group name:"
            )
            name = name.strip()
            if ok and name:
                self._state.create_group(name)
                self._state.assign_to_group(image_id, name)
                # Refresh the scope dropdown
                self._scope.set_groups(self._state.get_groups())
                # Update visual labels
                self._refresh_image_group_display(image_id)
                # Also select the group and set scope to "Group" in the scope selector bar
                self._scope._group_combo.blockSignals(True)
                self._scope._group_combo.setCurrentText(name)
                self._scope._group_combo.blockSignals(False)
                self._scope._btn_group_scope.setChecked(True)
                self._scope._current_scope = "group"
                self._scope._group_combo.setEnabled(True)
                self._scope._current_group = name
                # Notify
                self._on_scope_changed("group", name)

    # ── Workspace Tab Change ──────────────────────────────────────────────────

    @Slot(int)
    def _on_workspace_tab_changed(self, index: int) -> None:
        """Handle tab switching with two-way sync between Video Editor and Quick Export."""
        prev_index = getattr(self, "_prev_tab_index", 0)

        # Pause video playback when leaving Video Editor
        if index != 1:
            self._video_preview.pause()

        # Leaving Quick Export → sync any changes back to Video Editor
        if prev_index == 2 and index != 2:
            active_vid = self._state.active_video
            if active_vid is not None and self._state.active_video_id:
                self._video_adj_panel.load_params(active_vid.params, silent=True)
                self._video_filter_strip.set_active_filter(active_vid.params.filter_id, silent=True)
                self._video_filter_strip.set_filter_intensity(active_vid.params.filter_intensity, silent=True)

        # Entering Video Editor → refresh asset list and reload active video
        if index == 1:
            self._video_assets.refresh()
            active_id = self._state.active_video_id
            if active_id:
                if not self._video_preview._record or self._video_preview._record.id != active_id:
                    self._on_video_selected(active_id)

        # Leaving Video Editor → stop quick editor
        if index != 2:
            self._quick_editor.on_tab_deactivated()
        else:
            # Entering Quick Export → sync Video Editor state into Quick Editor
            self._quick_editor.on_tab_activated()
            active_id = self._state.active_video_id
            if active_id:
                vid = self._state.get_video(active_id)
                if vid is not None:
                    self._quick_editor.sync_from_record(vid)

        self._prev_tab_index = index
        self._update_menus_for_tab(index)

    # ── Video Selection ───────────────────────────────────────────────────────

    @Slot(str)
    def _on_video_selected(self, video_id: str) -> None:
        self._state.active_video_id = video_id
        vid = self._state.get_video(video_id)
        if vid is None:
            return

        # Load current params into sliders
        params = vid.params
        self._video_adj_panel.load_params(params, silent=True)
        self._video_filter_strip.set_active_filter(params.filter_id, silent=True)
        self._video_filter_strip.set_filter_intensity(params.filter_intensity, silent=True)

        # Set filter strip source thumbnail for previews
        if vid.thumbnail is not None:
            self._video_filter_strip.set_source_image(vid.thumbnail)

        # Load video into the preview widget
        self._video_preview.set_params(params)
        self._video_preview.load_video(vid)

        self._set_status(f"{vid.name}  —  {vid.width}×{vid.height}  |  {vid.fps:.2f} fps  |  {vid.duration:.1f}s")

    # ── Video Param Change ────────────────────────────────────────────────────

    @Slot(str, float)
    def _on_video_param_changed(self, param: str, value: float) -> None:
        vid = self._state.active_video
        if vid is None:
            return
        setattr(vid.params, param, value)
        self._video_preview.set_params(vid.params)

    @Slot(str)
    def _on_video_filter_selected(self, filter_id: str) -> None:
        vid = self._state.active_video
        if vid is None:
            return
        vid.params.filter_id = filter_id
        self._video_preview.set_params(vid.params)
        if vid.thumbnail is not None:
            self._video_filter_strip.update_visible_thumbnails()

    @Slot(float)
    def _on_video_filter_intensity_changed(self, intensity: float) -> None:
        vid = self._state.active_video
        if vid is None:
            return
        vid.params.filter_intensity = intensity
        self._video_preview.set_params(vid.params)

    @Slot(float)
    def _on_filter_intensity_changed(self, intensity: float) -> None:
        img = self._state.active_image
        if img is None:
            return
        scope = self._scope.current_scope
        if scope == "all":
            self._state.global_params.filter_intensity = intensity
        elif scope == "group" and self._scope.current_group:
            gp = self._state.group_params.setdefault(self._scope.current_group, AdjustmentParams())
            gp.filter_intensity = intensity
        else:
            img.params.filter_intensity = intensity
        if img.original is not None:
            params = self._state.resolved_params(img.id)
            preview_frame = self._get_preview_frame(img)
            self._schedule_preview(img.id, preview_frame, params)

    @Slot(str)
    def _on_quick_editor_params_updated(self, video_id: str) -> None:
        """Quick Editor changed something — refresh Video Editor if it's showing the same video."""
        vid = self._state.get_video(video_id)
        if vid is None:
            return
        # Only update the Video Editor UI if it currently has the same video active
        if (self._state.active_video_id == video_id and
                self._workspace_tabs.currentIndex() != 2):
            self._video_adj_panel.load_params(vid.params, silent=True)
            self._video_filter_strip.set_active_filter(vid.params.filter_id, silent=True)
            self._video_filter_strip.set_filter_intensity(vid.params.filter_intensity, silent=True)



    # ── Video Context Menu ────────────────────────────────────────────────────

    def _show_video_context_menu(self, video_id: str, pos) -> None:
        vid = self._state.get_video(video_id)
        if not vid:
            return

        menu = QMenu(self)
        act_export = menu.addAction("Export Video…")
        act_remove = menu.addAction("Remove from Project")

        action = menu.exec(pos)
        if action == act_export:
            self._export_video(video_id)
        elif action == act_remove:
            self._video_assets._remove_video(video_id)

    # ── Video Export ──────────────────────────────────────────────────────────

    def _export_current_video(self) -> None:
        vid = self._state.active_video
        if vid is None:
            QMessageBox.information(self, "No Video", "Please select a video first.")
            return
        self._export_video(vid.id)

    def _export_video(self, video_id: str) -> None:
        vid = self._state.get_video(video_id)
        if vid is None:
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", vid.path.stem + "_pixclip.mp4",
            "MP4 Video (*.mp4)"
        )
        if not out_path:
            return

        params = vid.params

        dialog = VideoExportDialog(vid, params, self)
        worker = VideoExportWorker(vid, params, out_path)

        def on_progress(status, current, total, preview_frame):
            dialog.set_progress(status, current, total, preview_frame)
            self._set_status(f"Exporting video: {current}/{total} frames")

        def on_finished(success, message):
            dialog.accept()
            if success:
                self._set_status(f"Video export complete: {Path(out_path).name}")
                QMessageBox.information(self, "Export Complete", message)
            else:
                QMessageBox.critical(self, "Export Failed", message)
                self._set_status("Video export failed")
            worker.wait()
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        dialog.cancelled.connect(worker.cancel)

        worker.start()
        dialog.exec()

    # ── Video Presets & Reset ─────────────────────────────────────────────────

    def _open_video_presets(self) -> None:
        vid = self._state.active_video
        current = vid.params if vid else AdjustmentParams()
        dialog = PresetDialog(current, self)
        dialog.preset_loaded.connect(self._apply_video_preset)
        dialog.exec()

    @Slot(AdjustmentParams)
    def _apply_video_preset(self, params: AdjustmentParams) -> None:
        vid = self._state.active_video
        if vid is None:
            return
        vid.params = params.copy()
        self._video_adj_panel.load_params(params, silent=True)
        self._video_filter_strip.set_active_filter(params.filter_id, silent=True)
        self._video_preview.set_params(params)

    def _reset_video_all(self) -> None:
        vid = self._state.active_video
        if vid is None:
            return
        default = AdjustmentParams()
        vid.params = default
        self._video_adj_panel.load_params(default, silent=True)
        self._video_filter_strip.set_active_filter("none", silent=True)
        self._video_preview.set_params(default)

    # ── Preview Rendering ─────────────────────────────────────────────────────

    # ── Preview Quality & Resolution ──────────────────────────────────────────

    def _get_preview_frame(self, img: ImageRecord) -> Optional[np.ndarray]:
        if img.original is None:
            return None
        
        quality = self._preview_quality
        limit = PREVIEW_QUALITY_LIMITS.get(quality)
        
        h, w = img.original.shape[:2]
        if limit is None or h <= limit:
            return img.original
            
        if getattr(img, "_preview_source_resolution", None) == quality and getattr(img, "_preview_source", None) is not None:
            return img._preview_source
            
        scale = limit / h
        tw, th = max(1, int(w * scale)), limit
        img._preview_source = cv2.resize(img.original, (tw, th), interpolation=cv2.INTER_AREA)
        img._preview_source_resolution = quality
        return img._preview_source

    @Slot(str)
    def _change_preview_quality(self, quality: str) -> None:
        self._preview_quality = quality
        self._settings.setValue("preview_quality", quality)
        
        # Invalidate existing previews in disk cache
        from core.cache import clear_all_cache
        clear_all_cache()
        
        # Trigger rerender for active image
        img = self._state.active_image
        if img is not None:
            if img.original is None:
                from core.batch import load_image_record
                load_image_record(img)
                
            if img.original is not None:
                params = self._state.resolved_params(img.id)
                preview_frame = self._get_preview_frame(img)
                self._preview.set_image(preview_frame, original=preview_frame)
                self._schedule_preview(img.id, preview_frame, params)

    # ── Preview Rendering ─────────────────────────────────────────────────────

    def _schedule_preview(self, image_id: str, frame: np.ndarray, params: AdjustmentParams) -> None:
        """
        Request a new preview render. If a render is already in flight,
        the new request will be queued (overwriting any previous queued request).
        """
        # Check cache first
        from core.cache import get_cached_preview
        cached = get_cached_preview(image_id, params)
        if cached is not None:
            self._preview.set_image(cached, original=frame)
            h, w = cached.shape[:2]
            scale = 80 / max(h, w)
            tw = max(1, int(w * scale))
            th = max(1, int(h * scale))
            small = cv2.resize(cached, (tw, th), interpolation=cv2.INTER_AREA)
            self._filmstrip.update_thumbnail(image_id, small)
            return

        scale = 1.0
        self._pending_render = (image_id, frame, params, scale)
        if not self._render_in_flight:
            self._fire_next_render()

    def _fire_next_render(self) -> None:
        if self._pending_render:
            image_id, frame, params, scale = self._pending_render
            self._pending_render = None
            self._render_in_flight = True
            self._preview.set_loading(True)
            self._render_requested.emit(image_id, frame.copy(), params, scale)

    @Slot(str, np.ndarray)
    def _on_render_complete(self, image_id: str, result: np.ndarray) -> None:
        self._render_in_flight = False
        self._preview.set_loading(False)
        img = self._state.active_image
        if img is not None and img.id == image_id:
            preview_frame = self._get_preview_frame(img)
            self._preview.set_image(result, original=preview_frame)
            # Update filmstrip thumbnail with processed version
            h, w = result.shape[:2]
            scale = 80 / max(h, w)
            tw = max(1, int(w * scale))
            th = max(1, int(h * scale))
            small = cv2.resize(result, (tw, th), interpolation=cv2.INTER_AREA)
            self._filmstrip.update_thumbnail(img.id, small)
        self._fire_next_render()

    # ── File Operations ───────────────────────────────────────────────────────

    def _open_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Images", "",
            "Images (*.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp)"
        )
        if files:
            paths = [Path(f) for f in files]
            self._assets._add_files(paths)

    def _export_current(self) -> None:
        img = self._state.active_image
        if img is None:
            QMessageBox.information(self, "No Image", "Please select an image first.")
            return

        out_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Image", img.path.stem + "_edited.jpg",
            "JPEG (*.jpg);;PNG (*.png);;WebP (*.webp);;TIFF (*.tif)"
        )
        if not out_path:
            return

        params = self._state.resolved_params(img.id)

        # Show non-blocking modal loading dialog
        dialog = SingleExportLoadingDialog(img, params, self)
        worker = SingleExportWorker(img, params, out_path)

        def on_worker_finished(success, message):
            dialog.accept()
            if success:
                self._set_status(message)
            else:
                QMessageBox.critical(self, "Export Error", message)
                self._set_status("Export failed")
            worker.wait()
            worker.deleteLater()

        worker.finished.connect(on_worker_finished)
        worker.start()
        dialog.exec()

    def _batch_export(self) -> None:
        if not self._state.images:
            QMessageBox.information(self, "No Images", "Import images first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not out_dir:
            return

        options = ExportOptions(
            output_dir=Path(out_dir),
            format=ExportFormat.JPEG,
            jpeg_quality=95,
            suffix="_pixclip",
        )
        processor = BatchProcessor(self._state, options)

        # Show progress dialog
        dialog = ExportProgressDialog(self._state, self)
        worker = BatchExportWorker(processor)

        # Connect signals
        dialog.cancelled.connect(processor.cancel)

        # Update stats inside callback
        def on_worker_progress(done, total, result):
            dialog.set_progress(result.source_path.name, done, total, result.image_id)
            self._set_status(
                f"Exporting {done}/{total}: {result.source_path.name}"
                + (" ✓" if result.success else " ✗")
            )
            
        worker.progress.connect(on_worker_progress)

        def on_worker_finished(results):
            dialog.accept()
            success_count = sum(1 for r in results if r.success)
            total = len(results)
            if processor._cancelled:
                self._set_status(f"Batch export cancelled: {success_count}/{total} exported")
            else:
                self._set_status(f"Batch export complete: {success_count}/{total} succeeded")
            worker.wait()
            worker.deleteLater()

        worker.finished.connect(on_worker_finished)
        worker.start()
        dialog.exec()

    # ── Presets ───────────────────────────────────────────────────────────────

    def _open_presets(self) -> None:
        img = self._state.active_image
        current = self._state.resolved_params(img.id) if img else AdjustmentParams()
        dialog = PresetDialog(current, self)
        dialog.preset_loaded.connect(self._apply_preset)
        dialog.exec()

    @Slot(AdjustmentParams)
    def _apply_preset(self, params: AdjustmentParams) -> None:
        img = self._state.active_image
        if img is None:
            return
        scope = self._scope.current_scope
        if scope == "all":
            self._state.global_params = params.copy()
            self._state.invalidate_all_caches()
        elif scope == "group" and self._scope.current_group:
            self._state.group_params[self._scope.current_group] = params.copy()
        else:
            img.params = params.copy()
            img.invalidate_cache()

        self._adj_panel.load_params(params, silent=True)
        self._filter_strip.set_active_filter(params.filter_id, silent=True)

        if img.original is not None:
            resolved = self._state.resolved_params(img.id)
            self._schedule_preview(img.id, img.original, resolved)
        self._save_state()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_all(self) -> None:
        img = self._state.active_image
        if img is None:
            return
        scope = self._scope.current_scope
        default = AdjustmentParams()
        if scope == "all":
            self._state.global_params = default
            self._state.invalidate_all_caches()
        elif scope == "group" and self._scope.current_group:
            self._state.group_params[self._scope.current_group] = default
        else:
            img.params = default
            img.invalidate_cache()

        self._adj_panel.load_params(default, silent=True)
        self._filter_strip.set_active_filter("none", silent=True)

        if img.original is not None:
            self._schedule_preview(img.id, img.original, default)
        self._save_state()

    # ── Keyboard Shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_F:
            self._preview._view_fit()
        elif key == Qt.Key.Key_1:
            self._preview._active_view().zoom_to_100()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        super().keyReleaseEvent(event)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_lbl.setText(msg)

    def _save_state(self) -> None:
        pass

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if hasattr(self, "_video_preview") and self._video_preview is not None:
            try:
                self._video_preview.close()
            except Exception:
                pass
        if hasattr(self, "_quick_editor") and self._quick_editor is not None:
            try:
                self._quick_editor.close()
            except Exception:
                pass

        self._render_thread.quit()
        self._render_thread.wait(2000)
        try:
            from core.cache import clean_old_caches
            clean_old_caches()
        except Exception:
            pass
        super().closeEvent(event)
