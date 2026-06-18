"""
ui/quick_video_editor.py
Quick Video Editor — Tab 3 of PixClip.

Strategy: Use FFmpeg's native filtergraph for export (10-50x faster than Python
frame-loop). For preview, play the original video with OpenCV frame stepping and
show a single processed preview frame on demand.

Architecture:
  Left:   QuickVideoList — drag-drop video import (reuses VideoAssetPanel)
  Center: QuickPreviewArea — CV2 frame viewer + playback controls
  Right:  QuickAdjustPanel — compact sliders + filter strip
  Bottom: Export bar — quality preset, CRF, Export button
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, Signal, QTimer, QThread, Slot, QSize, QObject,
)
from PySide6.QtGui import (
    QImage, QPixmap, QColor, QPainter, QFont,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QFrame, QScrollArea, QSlider, QComboBox,
    QSizePolicy, QProgressBar, QFileDialog, QMessageBox,
    QDialog,
)

from core.params import AdjustmentParams, VideoRecord, ProjectState
from core.pipeline import process_frame
from core.filters import apply_filter
from core.ffmpeg_filters import (
    build_ffmpeg_vf, run_ffmpeg_export, get_video_duration_frames,
    export_filter_as_cube,
)
from ui.video_asset_panel import VideoAssetPanel
from ui.filter_strip import FilterStrip
from ui.adjustment_panel import AdjustmentPanel, ParamSlider
from ui.storyboard_strip import VideoStoryboardStrip
from ui.preview_widget import ShimmerOverlay


# ── FFmpeg Export Worker ───────────────────────────────────────────────────────

class QuickExportWorker(QThread):
    """Runs FFmpeg export in a background thread with progress updates."""
    progress = Signal(int, int, float)   # (current_frame, total_frames, fps)
    finished = Signal(bool, str)         # (success, message)

    def __init__(self, input_path: str, output_path: str,
                 params: AdjustmentParams, preset: str, crf: int):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.params = params
        self.preset = preset
        self.crf = crf
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        def progress_cb(current, total, fps):
            self.progress.emit(current, total, fps)

        success, message = run_ffmpeg_export(
            self.input_path,
            self.output_path,
            self.params,
            preset=self.preset,
            crf=self.crf,
            progress_callback=progress_cb,
            cancelled_check=lambda: self._cancelled,
        )
        self.finished.emit(success, message)


# ── Quick Export Dialog ───────────────────────────────────────────────────────

class QuickExportDialog(QDialog):
    """Sleek modal progress dialog for the quick export."""
    cancelled = Signal()

    def __init__(self, record: VideoRecord, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick Export")
        self.setModal(True)
        self.setFixedSize(440, 240)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self.record = record
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Header
        header = QHBoxLayout()
        icon_lbl = QLabel("⚡")
        icon_lbl.setObjectName("InfoFlash")
        title = QLabel(f"Quick Exporting")
        title.setObjectName("DialogTitle")
        header.addWidget(icon_lbl)
        header.addWidget(title)
        header.addStretch()

        # File label
        self._file_label = QLabel(self.record.name)
        self._file_label.setObjectName("DialogSub")
        self._file_label.setWordWrap(True)

        # Speed indicator
        self._speed_label = QLabel("Initializing FFmpeg...")
        self._speed_label.setObjectName("DialogHighlight")

        # Frame progress
        self._frame_label = QLabel("0 / 0 frames")
        self._frame_label.setObjectName("DialogStats")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        info_row = QHBoxLayout()
        info_row.addWidget(self._speed_label)
        info_row.addStretch()
        info_row.addWidget(self._frame_label)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setFixedHeight(10)
        self._progress.setTextVisible(False)

        # ETA label
        self._eta_label = QLabel("")
        self._eta_label.setObjectName("DialogSub")

        # Cancel
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("DangerButton")
        cancel_btn.setFixedWidth(90)
        cancel_btn.clicked.connect(self._on_cancel)

        layout.addLayout(header)
        layout.addWidget(self._file_label)
        layout.addLayout(info_row)
        layout.addWidget(self._progress)
        layout.addWidget(self._eta_label)
        layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self._start_time = None

    def set_progress(self, current: int, total: int, fps: float):
        import time
        if self._start_time is None:
            self._start_time = time.time()

        if total > 0:
            self._progress.setValue(int(current / total * 100))
            self._frame_label.setText(f"{current} / {total} frames")

        if fps > 0:
            self._speed_label.setText(f"⚡ {fps:.0f} fps  (FFmpeg native)")
        
        # ETA
        if self._start_time and fps > 0 and total > 0 and current > 0:
            elapsed = time.time() - self._start_time
            remaining = (total - current) / fps if fps > 0 else 0
            self._eta_label.setText(
                f"Elapsed: {elapsed:.0f}s  ·  ETA: {remaining:.0f}s"
            )

    def _on_cancel(self):
        self.cancelled.emit()
        self.reject()

    def closeEvent(self, event):
        self.cancelled.emit()
        super().closeEvent(event)


# ── Single-frame Preview Widget ───────────────────────────────────────────────

def _bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


class QuickFrameView(QLabel):
    """Displays a single video frame with fit-to-size scaling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setObjectName("PreviewContainer")
        self._pixmap: Optional[QPixmap] = None
        self._show_placeholder()

    def _show_placeholder(self):
        self.setText("No video selected\nDrop a video onto the left panel")
        self.setObjectName("PlaceholderText")

    def set_frame(self, frame: np.ndarray):
        self._pixmap = _bgr_to_pixmap(frame)
        self.setText("")
        self._scale_pixmap()

    def clear_frame(self):
        self._pixmap = None
        self._show_placeholder()

    def _scale_pixmap(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_pixmap()


# ── Playback Controller ───────────────────────────────────────────────────────

class PlaybackController(QWidget):
    """
    Minimal playback bar: play/pause + seek slider + time label.
    Uses a QTimer to advance frames from an open cv2.VideoCapture.
    """
    frame_changed = Signal(int)   # (frame_index)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.setObjectName("VideoControlsBar")

        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames = 0
        self._fps = 30.0
        self._current = 0
        self._playing = False
        self._was_playing = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        self._btn_play = QPushButton("▶")
        self._btn_play.setObjectName("IconButton")
        self._btn_play.setFixedSize(32, 32)
        self._btn_play.clicked.connect(self.toggle_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(0)
        self._slider.sliderPressed.connect(self._on_slider_press)
        self._slider.sliderReleased.connect(self._on_slider_release)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        self._lbl_time = QLabel("0:00 / 0:00")
        self._lbl_time.setObjectName("TimeLabel")
        self._lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._btn_play)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._lbl_time)

    def load(self, cap: cv2.VideoCapture, total_frames: int, fps: float):
        self.stop()
        self._cap = cap
        self._total_frames = max(1, total_frames)
        self._fps = max(1.0, fps)
        self._current = 0
        self._slider.setRange(0, self._total_frames - 1)
        self._slider.setValue(0)
        self._update_time_label()

    def release(self):
        self.stop()
        if self._cap:
            self._cap.release()
            self._cap = None

    def toggle_play(self):
        if self._playing:
            self.stop()
        else:
            self.play()

    def play(self):
        if not self._cap:
            return
        if self._current >= self._total_frames - 1:
            self.seek(0)
        self._playing = True
        self._btn_play.setText("⏸")
        self._timer.start(int(1000 / self._fps))

    def stop(self):
        self._playing = False
        self._btn_play.setText("▶")
        self._timer.stop()

    def seek(self, frame_idx: int):
        if not self._cap:
            return
        frame_idx = max(0, min(self._total_frames - 1, frame_idx))
        self._current = frame_idx
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._slider.blockSignals(True)
        self._slider.setValue(frame_idx)
        self._slider.blockSignals(False)
        self._update_time_label()
        self.frame_changed.emit(frame_idx)

    def current_frame_index(self) -> int:
        return self._current

    def _tick(self):
        if not self._cap or not self._playing:
            return
        ret, _ = self._cap.read()
        if not ret:
            self.stop()
            return
        self._current += 1
        self._slider.blockSignals(True)
        self._slider.setValue(self._current)
        self._slider.blockSignals(False)
        self._update_time_label()
        self.frame_changed.emit(self._current)
        if self._current >= self._total_frames - 1:
            self.stop()

    def _on_slider_press(self):
        self._was_playing = self._playing
        self.stop()

    def _on_slider_release(self):
        if self._was_playing:
            self.play()

    def _on_slider_moved(self, val: int):
        self.seek(val)

    def _update_time_label(self):
        def fmt(s):
            return f"{int(s)//60}:{int(s)%60:02d}"
        cur = self._current / self._fps
        total = self._total_frames / self._fps
        self._lbl_time.setText(f"{fmt(cur)} / {fmt(total)}")


# ── Quick Video Editor (main widget) ─────────────────────────────────────────

class QuickVideoRenderWorker(QObject):
    """
    Processes quick video editor frames asynchronously in a background thread.
    """
    result_ready = Signal(int, np.ndarray, np.ndarray)  # (frame_idx, raw_frame, processed_frame)
    error_occurred = Signal(str)

    def __init__(self, video_path: str = ""):
        super().__init__()
        self.video_path = video_path
        self._cap = None
        self._last_cap_path = ""

    def set_video_path(self, path: str):
        self.video_path = path

    def _ensure_cap(self):
        if not self.video_path:
            return
        if self._cap is None or self._last_cap_path != self.video_path:
            if self._cap is not None:
                self._cap.release()
            self._cap = cv2.VideoCapture(self.video_path)
            self._last_cap_path = self.video_path

    @Slot(int, object)
    def process_frame_async(self, frame_idx: int, params: AdjustmentParams):
        try:
            self._ensure_cap()
            if self._cap is None or not self._cap.isOpened():
                self.error_occurred.emit("VideoCapture not opened")
                return

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self.error_occurred.emit(f"Failed to read frame {frame_idx}")
                return

            processed = process_frame(frame, params, filter_lut_fn=apply_filter)
            self.result_ready.emit(frame_idx, frame, processed)
        except Exception as e:
            self.error_occurred.emit(str(e))

    @Slot(int, np.ndarray, object)
    def process_existing_frame_async(self, frame_idx: int, raw_frame: np.ndarray, params: AdjustmentParams):
        try:
            if raw_frame is None:
                return
            processed = process_frame(raw_frame, params, filter_lut_fn=apply_filter)
            self.result_ready.emit(frame_idx, raw_frame, processed)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._last_cap_path = ""


class QuickVideoEditor(QWidget):
    """
    The full Quick Video Editor tab.
    Layout: [VideoList | PreviewArea | AdjustPanel]
    """
    sig_render_requested = Signal(int, AdjustmentParams)
    sig_render_existing_requested = Signal(int, np.ndarray, AdjustmentParams)

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self._state = state
        self._active_record: Optional[VideoRecord] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_raw_frame: Optional[np.ndarray] = None
        self._show_processed = False
        self._export_worker: Optional[QuickExportWorker] = None

        # Setup background render thread for static preview updates
        self._render_thread = QThread()
        self._worker = QuickVideoRenderWorker()
        self._worker.moveToThread(self._render_thread)
        self._worker.result_ready.connect(self._on_render_complete)
        self._worker.error_occurred.connect(self._on_render_error)
        self.sig_render_requested.connect(self._worker.process_frame_async)
        self.sig_render_existing_requested.connect(self._worker.process_existing_frame_async)
        self._render_thread.start()

        self._render_in_flight = False
        self._pending_render = None  # Format: (frame_idx, raw_frame_or_none, params)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top: Info bar ────────────────────────────────────────────────────
        info_bar = QWidget()
        info_bar.setObjectName("InfoBar")
        info_bar.setFixedHeight(36)
        ib_layout = QHBoxLayout(info_bar)
        ib_layout.setContentsMargins(12, 0, 12, 0)
        ib_layout.setSpacing(8)

        flash = QLabel("⚡")
        flash.setObjectName("InfoFlash")

        title_lbl = QLabel("Quick Video Editor")
        title_lbl.setObjectName("InfoTitle")

        badge = QLabel("FFmpeg Native · Up to 50× faster export")
        badge.setObjectName("InfoBadge")

        self._info_lbl = QLabel("")
        self._info_lbl.setObjectName("InfoSub")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        ib_layout.addWidget(flash)
        ib_layout.addWidget(title_lbl)
        ib_layout.addWidget(badge)
        ib_layout.addStretch()
        ib_layout.addWidget(self._info_lbl)

        root.addWidget(info_bar)

        # ── Main splitter ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)

        # ── Left: Video asset list ───────────────────────────────────────────
        self._video_list = VideoAssetPanel(self._state)
        splitter.addWidget(self._video_list)

        # ── Center: Preview area ─────────────────────────────────────────────
        center = QWidget()
        center.setObjectName("PreviewContainer")
        center.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # Preview toolbar
        preview_toolbar = QWidget()
        preview_toolbar.setFixedHeight(40)
        preview_toolbar.setObjectName("ZoomToolbar")
        pt_layout = QHBoxLayout(preview_toolbar)
        pt_layout.setContentsMargins(10, 4, 10, 4)
        pt_layout.setSpacing(6)

        # Original / Processed toggle
        self._btn_original = QPushButton("Original")
        self._btn_original.setObjectName("TabButton")
        self._btn_original.setCheckable(True)
        self._btn_original.setChecked(True)
        self._btn_original.clicked.connect(lambda: self._set_preview_mode(False))

        self._btn_processed = QPushButton("Preview Filter")
        self._btn_processed.setObjectName("TabButton")
        self._btn_processed.setCheckable(True)
        self._btn_processed.setToolTip(
            "Process current frame with your adjustments (pauses playback)"
        )
        self._btn_processed.clicked.connect(lambda: self._set_preview_mode(True))

        tab_container = QWidget()
        tab_container.setObjectName("TabContainer")
        tc_layout = QHBoxLayout(tab_container)
        tc_layout.setContentsMargins(2, 2, 2, 2)
        tc_layout.setSpacing(0)
        tc_layout.addWidget(self._btn_original)
        tc_layout.addWidget(self._btn_processed)

        pt_layout.addStretch()
        pt_layout.addWidget(tab_container)
        pt_layout.addStretch()

        self._res_label = QLabel("")
        self._res_label.setObjectName("InfoSub")
        pt_layout.addWidget(self._res_label)

        # Frame viewer
        self._frame_view = QuickFrameView()

        # Playback controls
        self._playback = PlaybackController()
        self._playback.frame_changed.connect(self._on_frame_changed)

        self._storyboard = VideoStoryboardStrip()
        self._storyboard.frame_selected.connect(self._playback.seek)

        # Filter strip (moved to center bottom for consistency)
        self._filter_strip = FilterStrip()

        center_layout.addWidget(preview_toolbar)
        center_layout.addWidget(self._frame_view, stretch=1)
        center_layout.addWidget(self._storyboard)
        center_layout.addWidget(self._playback)
        center_layout.addWidget(self._filter_strip)

        splitter.addWidget(center)

        # ── Right: Adjustment panel ──────────────────────────────────────────
        right = QWidget()
        right.setObjectName("AdjustmentPanel")
        right.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Preset bar (header)
        preset_bar = QWidget()
        preset_bar.setFixedHeight(38)
        preset_bar.setObjectName("PresetBar")
        pb_layout = QHBoxLayout(preset_bar)
        pb_layout.setContentsMargins(10, 4, 10, 4)
        pb_layout.setSpacing(6)

        adj_lbl = QLabel("ADJUSTMENTS")
        adj_lbl.setObjectName("SectionHeader")

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setObjectName("DangerButton")
        self._btn_reset.setFixedHeight(26)
        self._btn_reset.clicked.connect(self._reset_adjustments)

        pb_layout.addWidget(adj_lbl)
        pb_layout.addStretch()
        pb_layout.addWidget(self._btn_reset)

        self._adj_panel = AdjustmentPanel()

        right_layout.addWidget(preset_bar)
        right_layout.addWidget(self._adj_panel, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([160, 900, 280])

        root.addWidget(splitter, stretch=1)

        # ── Bottom: Export bar ───────────────────────────────────────────────
        export_bar = self._build_export_bar()
        root.addWidget(export_bar)

        self._loading_overlay = ShimmerOverlay(self._frame_view)

    def _build_export_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("ExportBar")
        bar.setFixedHeight(52)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(10)

        # Speed preset
        speed_lbl = QLabel("Speed:")
        speed_lbl.setObjectName("InfoSub")

        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            "Ultrafast (lowest quality)",
            "Superfast",
            "Very Fast",
            "Fast (recommended)",
            "Medium (best quality)",
        ])
        self._preset_combo.setCurrentIndex(3)
        self._preset_combo.setFixedWidth(200)
        self._preset_combo.setToolTip(
            "Encoding speed preset — faster = larger file, slower = smaller/better"
        )

        # Quality (CRF)
        quality_lbl = QLabel("Quality:")
        quality_lbl.setObjectName("InfoSub")

        self._crf_combo = QComboBox()
        self._crf_combo.addItems([
            "High (CRF 18)",
            "Good (CRF 22)",
            "Medium (CRF 26)",
            "Low (CRF 30)",
        ])
        self._crf_combo.setCurrentIndex(0)
        self._crf_combo.setFixedWidth(150)

        # Export button
        self._btn_export = QPushButton("⚡  Export Video")
        self._btn_export.setObjectName("PrimaryButton")
        self._btn_export.setFixedHeight(36)
        self._btn_export.setFixedWidth(160)
        self._btn_export.clicked.connect(self._start_export)

        layout.addWidget(speed_lbl)
        layout.addWidget(self._preset_combo)
        layout.addSpacing(6)
        layout.addWidget(quality_lbl)
        layout.addWidget(self._crf_combo)
        layout.addStretch()
        layout.addWidget(self._btn_export)

        return bar

    def _connect_signals(self):
        # Video list
        self._video_list.video_selected.connect(self._on_video_selected)

        # Adjustment panel (debounced - but for quick editor we just store)
        self._adj_panel.param_changed.connect(self._on_param_changed)

        # Filter strip
        self._filter_strip.filter_selected.connect(self._on_filter_selected)

    # ── Video Selection ───────────────────────────────────────────────────────

    @Slot(str)
    def _on_video_selected(self, video_id: str):
        vid = self._state.get_video(video_id)
        if vid is None:
            return

        self._active_record = vid
        self.set_loading(True)
        self._worker.set_video_path(str(vid.path.resolve()))
        QTimer.singleShot(100, lambda: self._do_load_video(vid))

    def _do_load_video(self, vid: VideoRecord):
        # Release old capture
        if self._cap:
            self._playback.release()
            self._cap = None

        # Open new capture
        self._cap = cv2.VideoCapture(str(vid.path.resolve()))
        if not self._cap.isOpened():
            self._frame_view.clear_frame()
            self.set_loading(False)
            return

        # Load playback controller
        self._playback.load(self._cap, vid.frame_count, vid.fps)

        # Load existing params for this video
        self._adj_panel.load_params(vid.params, silent=True)
        self._filter_strip.set_active_filter(vid.params.filter_id, silent=True)
        if vid.thumbnail is not None:
            self._filter_strip.set_source_image(vid.thumbnail)

        # Show first frame
        self._seek_and_display(0)

        self._info_lbl.setText(
            f"{vid.name}  ·  {vid.width}×{vid.height}  ·  {vid.fps:.1f}fps  ·  {vid.duration:.1f}s"
        )
        self._res_label.setText(f"{vid.width}×{vid.height}")
        self._storyboard.load_video(str(vid.path.resolve()), vid.frame_count, vid.fps, vid.width, vid.height)

    def _seek_and_display(self, frame_idx: int):
        """Seek capture and display frame in current preview mode asynchronously."""
        if not self._active_record:
            return
        self.request_render(frame_idx, None, self._active_record.params)

    def _display_frame(self, frame: np.ndarray):
        """Show raw or processed frame depending on mode."""
        if self._show_processed and self._active_record:
            params = self._active_record.params
            processed = process_frame(frame, params, filter_lut_fn=apply_filter)
            self._frame_view.set_frame(processed)
        else:
            self._frame_view.set_frame(frame)

    @Slot(int)
    def _on_frame_changed(self, frame_idx: int):
        """Called by playback controller when frame advances."""
        if not self._cap or not self._cap.isOpened():
            return
            
        if self._show_processed and self._active_record:
            # Static paused seek in processed mode -> render asynchronously!
            self.request_render(frame_idx, None, self._active_record.params)
        else:
            # Playback or original mode static seek -> read and display synchronously!
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._current_raw_frame = frame
                self._frame_view.set_frame(frame)
            self._storyboard.highlight_frame(frame_idx)

    # ── Preview Mode Toggle ───────────────────────────────────────────────────

    def _set_preview_mode(self, processed: bool):
        self._show_processed = processed
        self._btn_original.setChecked(not processed)
        self._btn_processed.setChecked(processed)

        if processed and self._current_raw_frame is not None:
            # Pause playback while showing processed
            self._playback.stop()
            self.request_render(self._playback.current_frame_index(), self._current_raw_frame, self._active_record.params)
        elif self._current_raw_frame is not None:
            self._frame_view.set_frame(self._current_raw_frame)

    # ── Param Changes ─────────────────────────────────────────────────────────

    @Slot(str, float)
    def _on_param_changed(self, param: str, value: float):
        if not self._active_record:
            return
        setattr(self._active_record.params, param, value)

        # If in processed mode, refresh current frame display
        if self._show_processed and self._current_raw_frame is not None:
            self.request_render(self._playback.current_frame_index(), self._current_raw_frame, self._active_record.params)

    @Slot(str)
    def _on_filter_selected(self, filter_id: str):
        if not self._active_record:
            return
        self._active_record.params.filter_id = filter_id

        # Pre-export the cube file in background to avoid delay at export time
        if filter_id != "none":
            QTimer.singleShot(200, lambda: export_filter_as_cube(filter_id))

        if self._show_processed and self._current_raw_frame is not None:
            self.request_render(self._playback.current_frame_index(), self._current_raw_frame, self._active_record.params)

        if self._active_record.thumbnail is not None:
            self._filter_strip.update_visible_thumbnails()

    def _reset_adjustments(self):
        if not self._active_record:
            return
        default = AdjustmentParams()
        self._active_record.params = default
        self._adj_panel.load_params(default, silent=True)
        self._filter_strip.set_active_filter("none", silent=True)

        if self._show_processed and self._current_raw_frame is not None:
            self.request_render(self._playback.current_frame_index(), self._current_raw_frame, self._active_record.params)

    def update_frame_display(self):
        """Called when preview quality changes or similar global updates occur."""
        if self._active_record and self._current_raw_frame is not None:
            if self._show_processed:
                self.request_render(self._playback.current_frame_index(), self._current_raw_frame, self._active_record.params)

    def closeEvent(self, event):
        self._playback.stop()
        if hasattr(self, "_render_thread") and self._render_thread.isRunning():
            self._render_thread.quit()
            self._render_thread.wait(2000)
        super().closeEvent(event)

    # ── Asynchronous Rendering Thread Communication ──────────────────────────

    def request_render(self, frame_idx: int, raw_frame: Optional[np.ndarray], params: AdjustmentParams):
        self._pending_render = (frame_idx, raw_frame, params)
        if not self._render_in_flight:
            self._fire_next_render()

    def _fire_next_render(self):
        if self._pending_render:
            frame_idx, raw_frame, params = self._pending_render
            self._pending_render = None
            self._render_in_flight = True
            
            self.set_loading(True)
            
            if raw_frame is not None:
                self.sig_render_existing_requested.emit(frame_idx, raw_frame, params)
            else:
                self.sig_render_requested.emit(frame_idx, params)

    @Slot(int, np.ndarray, np.ndarray)
    def _on_render_complete(self, frame_idx: int, raw_frame: np.ndarray, processed_frame: np.ndarray):
        self._render_in_flight = False
        self.set_loading(False)
        
        self._current_raw_frame = raw_frame
        
        if self._show_processed:
            self._frame_view.set_frame(processed_frame)
        else:
            self._frame_view.set_frame(raw_frame)
            
        self._storyboard.highlight_frame(frame_idx)
        self._fire_next_render()

    @Slot(str)
    def _on_render_error(self, err_msg: str):
        self._render_in_flight = False
        self.set_loading(False)
        print(f"[QuickVideoEditor] Render error: {err_msg}")
        self._fire_next_render()

    # ── Export ────────────────────────────────────────────────────────────────

    def _get_preset_str(self) -> str:
        idx = self._preset_combo.currentIndex()
        presets = ["ultrafast", "superfast", "veryfast", "fast", "medium"]
        return presets[idx]

    def _get_crf(self) -> int:
        idx = self._crf_combo.currentIndex()
        crfs = [18, 22, 26, 30]
        return crfs[idx]

    def _start_export(self):
        if not self._active_record:
            QMessageBox.information(
                self, "No Video", "Please select a video first."
            )
            return

        vid = self._active_record
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Video",
            vid.path.stem + "_quick.mp4",
            "MP4 Video (*.mp4)",
        )
        if not out_path:
            return

        params = vid.params.copy()
        preset = self._get_preset_str()
        crf = self._get_crf()

        # Pause playback during export
        self._playback.stop()

        dialog = QuickExportDialog(vid, self.window())
        self._export_worker = QuickExportWorker(
            str(vid.path.resolve()), out_path, params, preset, crf
        )

        def on_progress(current, total, fps):
            dialog.set_progress(current, total, fps)

        def on_finished(success, message):
            dialog.accept()
            if self._export_worker:
                self._export_worker.wait()
            self._export_worker = None
            if success:
                QMessageBox.information(
                    self,
                    "Export Complete ⚡",
                    f"Video exported successfully!\n\n{Path(out_path).name}",
                )
            else:
                QMessageBox.critical(self, "Export Failed", message)

        self._export_worker.progress.connect(on_progress)
        self._export_worker.finished.connect(on_finished)
        dialog.cancelled.connect(self._export_worker.cancel)

        self._export_worker.start()
        dialog.exec()

    def on_tab_activated(self):
        """Called when user switches to this tab."""
        self._video_list.refresh()
        # If there is an active video in the state and it's not loaded yet, select it
        active_id = self._state.active_video_id
        if active_id:
            if not self._active_record or self._active_record.id != active_id:
                self._on_video_selected(active_id)

    def on_tab_deactivated(self):
        """Called when user switches away from this tab."""
        self._playback.stop()

    def set_loading(self, loading: bool) -> None:
        if loading:
            self._loading_overlay.show_shimmer()
        else:
            self._loading_overlay.hide_shimmer()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._loading_overlay.setGeometry(0, 0, self._frame_view.width(), self._frame_view.height())

    def closeEvent(self, event):
        self._playback.stop()
        self._storyboard.clear()
        
        if hasattr(self, "_render_thread") and self._render_thread.isRunning():
            self._worker.release()
            self._render_thread.quit()
            self._render_thread.wait()
            
        if self._cap:
            self._cap.release()
            self._cap = None
        super().closeEvent(event)
