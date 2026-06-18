"""
ui/video_preview_widget.py
Full video preview panel: supports zoom/pan, split comparison, play/pause,
and interactive timeline seeking. Processes frames dynamically.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional

from PySide6.QtCore import (
    Qt, QRectF, Signal, QPoint, QPointF, QTimer, QPropertyAnimation, Slot,
    QThread, QObject,
)
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,
    QColor, QResizeEvent, QPen, QLinearGradient, QFont,
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsObject,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QFrame, QSplitter, QButtonGroup, QStackedWidget, QGraphicsOpacityEffect,
)

from core.params import VideoRecord, AdjustmentParams
from core.pipeline import process_frame
from core.filters import apply_filter
from ui.preview_widget import ImageGraphicsView, ndarray_to_qimage, SplitImageItem, ShimmerOverlay
from ui.storyboard_strip import VideoStoryboardStrip


def format_duration(seconds: float) -> str:
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}:{secs:02d}"


class VideoRenderWorker(QObject):
    """
    Processes video frames asynchronously in a background thread.
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

    @Slot(int, object, str)
    def process_frame_async(self, frame_idx: int, params: AdjustmentParams, quality: str):
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

            processed = self._process_matrix(frame, params, quality)
            self.result_ready.emit(frame_idx, frame, processed)
        except Exception as e:
            self.error_occurred.emit(str(e))

    @Slot(int, np.ndarray, object, str)
    def process_existing_frame_async(self, frame_idx: int, raw_frame: np.ndarray, params: AdjustmentParams, quality: str):
        try:
            if raw_frame is None:
                return
            processed = self._process_matrix(raw_frame, params, quality)
            self.result_ready.emit(frame_idx, raw_frame, processed)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _process_matrix(self, frame: np.ndarray, params: AdjustmentParams, quality: str) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = 1.0
        if quality != "Keep Original":
            targets = {"360p": 360, "480p": 480, "720p": 720, "960p": 960, "1080p": 1080, "2K": 1440}
            target_h = targets.get(quality, 720)
            if h > target_h:
                scale = target_h / h

        if scale < 1.0:
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame.copy()

        processed = process_frame(small, params, filter_lut_fn=apply_filter)
        return processed

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._last_cap_path = ""


class VideoPreviewWidget(QWidget):
    """
    Video playback viewport widget with full zoom/pan support, comparative splits,
    and interactive timeline controls.
    """
    play_state_changed = Signal(bool)  # (is_playing)
    frame_changed = Signal(int)        # (frame_idx)
    sig_render_requested = Signal(int, AdjustmentParams, str)
    sig_render_existing_requested = Signal(int, np.ndarray, AdjustmentParams, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewContainer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._record: Optional[VideoRecord] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_frame_idx = 0
        self._is_playing = False
        
        # State params for rendering
        self._active_params = AdjustmentParams()
        self._preview_quality = "720p"
        
        # Playback timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        
        # Syncing variables for side-by-side zoom/pan
        self._syncing = False
        
        # Cached original/edited frame matrices for rendering comparison modes
        self._raw_frame_cache: Optional[np.ndarray] = None
        self._proc_frame_cache: Optional[np.ndarray] = None

        # Setup background render thread for static preview updates
        self._render_thread = QThread()
        self._worker = VideoRenderWorker()
        self._worker.moveToThread(self._render_thread)
        self._worker.result_ready.connect(self._on_render_complete)
        self._worker.error_occurred.connect(self._on_render_error)
        self.sig_render_requested.connect(self._worker.process_frame_async)
        self.sig_render_existing_requested.connect(self._worker.process_existing_frame_async)
        self._render_thread.start()

        self._render_in_flight = False
        self._pending_render = None  # Format: (frame_idx, raw_frame_or_none, params, quality)

        self._build_ui()
        self._connect_sync()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setObjectName("ZoomToolbar")
        toolbar.setFixedHeight(40)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 4, 10, 4)
        tb_layout.setSpacing(6)

        # Zoom buttons
        self._zoom_label = QLabel("100%")
        self._zoom_label.setObjectName("ZoomLabel")
        self._zoom_label.setFixedWidth(54)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_zoom_out = QPushButton("−")
        btn_zoom_out.setObjectName("IconButton")
        btn_zoom_out.setFixedSize(24, 24)
        btn_zoom_out.clicked.connect(lambda: self._active_view().set_zoom(self._active_view().zoom_level / 1.3))

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setObjectName("IconButton")
        btn_zoom_in.setFixedSize(24, 24)
        btn_zoom_in.clicked.connect(lambda: self._active_view().set_zoom(self._active_view().zoom_level * 1.3))

        btn_fit = QPushButton("Fit")
        btn_fit.setObjectName("IconButton")
        btn_fit.setFixedSize(30, 24)
        btn_fit.clicked.connect(self.fit_in_views)

        btn_100 = QPushButton("1:1")
        btn_100.setObjectName("IconButton")
        btn_100.setFixedSize(30, 24)
        btn_100.clicked.connect(lambda: self._active_view().zoom_to_100())

        tb_layout.addWidget(btn_zoom_out)
        tb_layout.addWidget(self._zoom_label)
        tb_layout.addWidget(btn_zoom_in)
        tb_layout.addWidget(btn_fit)
        tb_layout.addWidget(btn_100)
        
        tb_layout.addStretch()

        # View Mode Tabs (Original | Edited | Compare | Split)
        self._btn_group = QButtonGroup(self)
        
        self._tab_orig = QPushButton("Original")
        self._tab_orig.setObjectName("TabButton")
        self._tab_orig.setCheckable(True)
        
        self._tab_edit = QPushButton("Edited")
        self._tab_edit.setObjectName("TabButton")
        self._tab_edit.setCheckable(True)
        self._tab_edit.setChecked(True)
        
        self._tab_comp = QPushButton("Compare")
        self._tab_comp.setObjectName("TabButton")
        self._tab_comp.setCheckable(True)

        self._tab_split = QPushButton("Split")
        self._tab_split.setObjectName("TabButton")
        self._tab_split.setCheckable(True)

        self._btn_group.addButton(self._tab_orig, 0)
        self._btn_group.addButton(self._tab_edit, 1)
        self._btn_group.addButton(self._tab_comp, 2)
        self._btn_group.addButton(self._tab_split, 3)
        self._btn_group.idClicked.connect(self._on_tab_changed)

        tab_container = QWidget()
        tab_container.setObjectName("TabContainer")
        tc_layout = QHBoxLayout(tab_container)
        tc_layout.setContentsMargins(2, 2, 2, 2)
        tc_layout.setSpacing(0)
        tc_layout.addWidget(self._tab_orig)
        tc_layout.addWidget(self._tab_edit)
        tc_layout.addWidget(self._tab_comp)
        tc_layout.addWidget(self._tab_split)

        tb_layout.addWidget(tab_container)
        tb_layout.addStretch()

        # Resolution quality indicator (read-only info label)
        self._info_label = QLabel("")
        self._info_label.setObjectName("InfoSub")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tb_layout.addWidget(self._info_label)

        # ── Views Stack ───────────────────────────────────────────────────────
        self._stack = QStackedWidget()
        
        self._view_left = ImageGraphicsView()   # Original frame
        self._view_right = ImageGraphicsView()  # Processed frame

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(2)
        self._splitter.addWidget(self._view_left)
        self._splitter.addWidget(self._view_right)

        self._view_split = ImageGraphicsView()  # Split slider view
        self._view_split._scene.removeItem(self._view_split._item)
        self._split_item = SplitImageItem()
        self._view_split._scene.addItem(self._split_item)

        self._stack.addWidget(self._splitter)
        self._stack.addWidget(self._view_split)

        # Hide left view initially (Edited is default)
        self._view_left.hide()
        self._stack.setCurrentIndex(0)

        # Loading / shimmer overlay
        self._loading_overlay = ShimmerOverlay(self._stack)

        # ── Video Controls Bar ────────────────────────────────────────────────
        controls_bar = QWidget()
        controls_bar.setFixedHeight(44)
        controls_bar.setObjectName("VideoControlsBar")
        
        cb_layout = QHBoxLayout(controls_bar)
        cb_layout.setContentsMargins(12, 0, 12, 0)
        cb_layout.setSpacing(10)

        # Play / Pause button
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("IconButton")
        self.btn_play.setFixedSize(30, 30)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setCursor(Qt.CursorShape.PointingHandCursor)

        # Timeline seek slider
        self.slider_timeline = QSlider(Qt.Orientation.Horizontal)
        self.slider_timeline.setRange(0, 100)
        self.slider_timeline.setValue(0)
        self.slider_timeline.sliderMoved.connect(self._on_slider_moved)
        self.slider_timeline.sliderPressed.connect(self._on_slider_pressed)
        self.slider_timeline.sliderReleased.connect(self._on_slider_released)

        # Timeline duration label
        self.lbl_time = QLabel("0:00 / 0:00")
        self.lbl_time.setObjectName("TimeLabel")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._storyboard = VideoStoryboardStrip()
        self._storyboard.frame_selected.connect(self.seek_to_frame)

        cb_layout.addWidget(self.btn_play)
        cb_layout.addWidget(self.slider_timeline, stretch=1)
        cb_layout.addWidget(self.lbl_time)

        # Assemble Preview
        layout.addWidget(toolbar)
        layout.addWidget(self._stack, stretch=1)
        layout.addWidget(self._storyboard)
        layout.addWidget(controls_bar)

    def _connect_sync(self):
        """Synchronize pan and zoom between all views."""
        views = [self._view_left, self._view_right, self._view_split]

        def sync_zoom(zoom, source):
            if not self._syncing:
                self._syncing = True
                for v in views:
                    if v != source:
                        v.set_zoom(zoom, emit=False)
                self._on_zoom_changed(zoom)
                self._syncing = False

        self._view_left.zoom_changed.connect(lambda z: sync_zoom(z, self._view_left))
        self._view_right.zoom_changed.connect(lambda z: sync_zoom(z, self._view_right))
        self._view_split.zoom_changed.connect(lambda z: sync_zoom(z, self._view_split))

        def sync_h(val, source):
            if not self._syncing:
                self._syncing = True
                for v in views:
                    if v != source:
                        v.horizontalScrollBar().setValue(val)
                self._syncing = False

        self._view_left.horizontalScrollBar().valueChanged.connect(lambda v: sync_h(v, self._view_left))
        self._view_right.horizontalScrollBar().valueChanged.connect(lambda v: sync_h(v, self._view_right))
        self._view_split.horizontalScrollBar().valueChanged.connect(lambda v: sync_h(v, self._view_split))

        def sync_v(val, source):
            if not self._syncing:
                self._syncing = True
                for v in views:
                    if v != source:
                        v.verticalScrollBar().setValue(val)
                self._syncing = False

        self._view_left.verticalScrollBar().valueChanged.connect(lambda v: sync_v(v, self._view_left))
        self._view_right.verticalScrollBar().valueChanged.connect(lambda v: sync_v(v, self._view_right))
        self._view_split.verticalScrollBar().valueChanged.connect(lambda v: sync_v(v, self._view_split))

    def _active_view(self) -> ImageGraphicsView:
        idx = self._btn_group.checkedId()
        if idx == 0: return self._view_left
        elif idx == 1: return self._view_right
        elif idx == 2: return self._view_right  # Either is fine, they are synced
        else: return self._view_split

    def fit_in_views(self):
        self._view_left._fit_on_next = True
        self._view_right._fit_on_next = True
        self._view_split._fit_on_next = True
        self._view_left.fit_in_view()
        self._view_right.fit_in_view()
        self._view_split.fit_in_view()

    def _on_zoom_changed(self, zoom: float):
        self._zoom_label.setText(f"{zoom * 100:.0f}%")

    def _on_tab_changed(self, idx: int):
        """0: Original, 1: Edited, 2: Compare, 3: Split"""
        if idx == 3:
            self._stack.setCurrentIndex(1)
        else:
            self._stack.setCurrentIndex(0)
            if idx == 0:
                self._view_left.show()
                self._view_right.hide()
            elif idx == 1:
                self._view_left.hide()
                self._view_right.show()
            elif idx == 2:
                self._view_left.show()
                self._view_right.show()
                w = self._splitter.width()
                self._splitter.setSizes([w // 2, w // 2])
        if idx != 0 and self._raw_frame_cache is not None and self._proc_frame_cache is None:
            self.request_render(self._current_frame_idx, self._raw_frame_cache, self._active_params, self._preview_quality)
        else:
            self.update_frame_display()

    # ── Playback Controls & Frame Operations ──────────────────────────────────

    def load_video(self, record: VideoRecord) -> None:
        """Loads a new video and resets playing timeline states."""
        self.pause()
        self.set_loading(True)
        self._worker.set_video_path(str(record.path.resolve()))
        QTimer.singleShot(100, lambda: self._do_load_video(record))

    def _do_load_video(self, record: VideoRecord) -> None:
        self._record = record
        if self._cap is not None:
            self._cap.release()
            
        self._cap = cv2.VideoCapture(str(record.path.resolve()))
        self._current_frame_idx = 0
        
        # Configure UI slider
        self.slider_timeline.setRange(0, record.frame_count - 1)
        self.slider_timeline.setValue(0)
        self._update_time_label()

        self._info_label.setText(f"{record.width}x{record.height} | {record.fps:.2f} fps")

        # Invalidate caches
        self._raw_frame_cache = None
        self._proc_frame_cache = None
        
        # Reset zoom anchor fits
        self._view_left._fit_on_next = True
        self._view_right._fit_on_next = True
        self._view_split._fit_on_next = True
        
        self._storyboard.load_video(str(record.path.resolve()), record.frame_count, record.fps, record.width, record.height)
        self.render_current_frame()

    def set_params(self, params: AdjustmentParams):
        """Update active parameters and refresh current frame render."""
        self._active_params = params.copy()
        if not self._is_playing:
            self.render_current_frame()

    def set_preview_quality(self, quality_str: str):
        """Change downscaling quality setting and re-render frame."""
        self._preview_quality = quality_str
        if not self._is_playing:
            self.render_current_frame()

    def toggle_play(self):
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if not self._record or not self._cap:
            return
        if self._current_frame_idx >= self._record.frame_count - 1:
            # Loop back to start
            self.seek_to_frame(0)
            
        self._is_playing = True
        self.btn_play.setText("⏸")
        self.play_state_changed.emit(True)
        
        # Set timer interval matched to video FPS
        interval_ms = int(1000.0 / self._record.fps)
        self._timer.start(interval_ms)

    def pause(self):
        self._is_playing = False
        self.btn_play.setText("▶")
        self._timer.stop()
        self.play_state_changed.emit(False)

    def seek_to_frame(self, frame_idx: int):
        if not self._record or not self._cap:
            return
        
        frame_idx = max(0, min(self._record.frame_count - 1, frame_idx))
        self._current_frame_idx = frame_idx
        
        # Update slider UI silently without triggering signals
        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(frame_idx)
        self.slider_timeline.blockSignals(False)
        
        self._update_time_label()
        
        self.request_render(frame_idx, None, self._active_params, self._preview_quality)

    def render_current_frame(self):
        """Re-reads and processes the current frame frame."""
        if not self._record or not self._cap:
            return
        if self._raw_frame_cache is not None:
            self.request_render(self._current_frame_idx, self._raw_frame_cache, self._active_params, self._preview_quality)
        else:
            self.request_render(self._current_frame_idx, None, self._active_params, self._preview_quality)

    def _process_frame_matrix(self, frame: np.ndarray) -> np.ndarray:
        """Stateless helper to downscale and process frame."""
        if frame is None:
            return frame
        
        # 1. Calculate downscaling preview ratio
        h, w = frame.shape[:2]
        scale = 1.0
        
        if self._preview_quality != "Keep Original":
            targets = {"360p": 360, "480p": 480, "720p": 720, "960p": 960, "1080p": 1080, "2K": 1440}
            target_h = targets.get(self._preview_quality, 720)
            if h > target_h:
                scale = target_h / h

        # 2. Downscale frame
        if scale < 1.0:
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame.copy()

        # 3. Apply stateless parameters pipeline
        processed = process_frame(small, self._active_params, filter_lut_fn=apply_filter)
        return processed

    def update_frame_display(self):
        """Displays cached original and edited matrices into active stacked views."""
        if self._raw_frame_cache is None:
            return

        mode_id = self._btn_group.checkedId()
        if mode_id != 0 and self._proc_frame_cache is None:
            return
        
        # Scale original frame size to match edited downscale size for zoom parity
        raw_scaled = self._raw_frame_cache
        if mode_id != 0 and raw_scaled.shape[:2] != self._proc_frame_cache.shape[:2]:
            raw_scaled = cv2.resize(raw_scaled, 
                                    (self._proc_frame_cache.shape[1], self._proc_frame_cache.shape[0]), 
                                    interpolation=cv2.INTER_AREA)

        # 1. Update views depending on active tab
        if mode_id == 0:  # Original
            self._view_left.set_image(raw_scaled)
        elif mode_id == 1:  # Edited
            self._view_right.set_image(self._proc_frame_cache)
        elif mode_id == 2:  # Compare
            self._view_left.set_image(raw_scaled)
            self._view_right.set_image(self._proc_frame_cache)
        elif mode_id == 3:  # Split Slider
            self._split_item.orig_pixmap = QPixmap.fromImage(ndarray_to_qimage(raw_scaled))
            self._split_item.edit_pixmap = QPixmap.fromImage(ndarray_to_qimage(self._proc_frame_cache))
            self._view_split._scene.setSceneRect(QRectF(self._split_item.orig_pixmap.rect()))
            
            if self._view_split._fit_on_next:
                self._view_split.fit_in_view()
                self._view_split._fit_on_next = False
            self._split_item.update()

    def _on_timer_tick(self):
        """Advances video frame by frame during playing state."""
        if not self._record or not self._cap or not self._is_playing:
            return
        
        ret, frame = self._cap.read()
        if not ret or frame is None:
            # End of video reached
            self.pause()
            return
        
        self._current_frame_idx += 1
        
        # Update slider timeline silently
        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(self._current_frame_idx)
        self.slider_timeline.blockSignals(False)
        
        self._update_time_label()

        # Process and render
        self._raw_frame_cache = frame
        mode_id = self._btn_group.checkedId()
        if mode_id != 0:
            self._proc_frame_cache = self._process_frame_matrix(frame)
        else:
            self._proc_frame_cache = None
        self.update_frame_display()
        
        self.frame_changed.emit(self._current_frame_idx)
        self._storyboard.highlight_frame(self._current_frame_idx)

        # Handle wrap-around end limits
        if self._current_frame_idx >= self._record.frame_count - 1:
            self.pause()

    def _update_time_label(self):
        if not self._record:
            self.lbl_time.setText("0:00 / 0:00")
            return
        
        cur_time = self._current_frame_idx / self._record.fps
        self.lbl_time.setText(f"{format_duration(cur_time)} / {format_duration(self._record.duration)}")

    # ── Slider Interaction events ─────────────────────────────────────────────

    def _on_slider_pressed(self):
        self._was_playing_before_seek = self._is_playing
        self.pause()

    def _on_slider_released(self):
        if getattr(self, "_was_playing_before_seek", False):
            self.play()

    def _on_slider_moved(self, val: int):
        self.seek_to_frame(val)

    # ── Asynchronous Rendering Thread Communication ──────────────────────────

    def request_render(self, frame_idx: int, raw_frame: Optional[np.ndarray], params: AdjustmentParams, quality: str):
        if self._is_playing:
            return  # During playback, we skip async queue rendering
            
        self._pending_render = (frame_idx, raw_frame, params, quality)
        if not self._render_in_flight:
            self._fire_next_render()

    def _fire_next_render(self):
        if self._pending_render and not self._is_playing:
            frame_idx, raw_frame, params, quality = self._pending_render
            self._pending_render = None
            self._render_in_flight = True
            
            self.set_loading(True)
            
            if raw_frame is not None:
                self.sig_render_existing_requested.emit(frame_idx, raw_frame, params, quality)
            else:
                self.sig_render_requested.emit(frame_idx, params, quality)

    @Slot(int, np.ndarray, np.ndarray)
    def _on_render_complete(self, frame_idx: int, raw_frame: np.ndarray, processed_frame: np.ndarray):
        self._render_in_flight = False
        self.set_loading(False)
        
        if self._is_playing:
            self._fire_next_render()
            return
            
        self._current_frame_idx = frame_idx
        self._raw_frame_cache = raw_frame
        
        mode_id = self._btn_group.checkedId()
        if mode_id != 0:
            self._proc_frame_cache = processed_frame
        else:
            self._proc_frame_cache = None
            
        self.update_frame_display()
        self._storyboard.highlight_frame(frame_idx)
        
        self._fire_next_render()

    @Slot(str)
    def _on_render_error(self, err_msg: str):
        self._render_in_flight = False
        self.set_loading(False)
        print(f"[VideoPreviewWidget] Render error: {err_msg}")
        self._fire_next_render()

    # ── Overlay & general events ──────────────────────────────────────────────

    def set_loading(self, loading: bool) -> None:
        if loading:
            self._loading_overlay.show_shimmer()
        else:
            self._loading_overlay.hide_shimmer()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._loading_overlay.setGeometry(0, 0, self._stack.width(), self._stack.height())

    def closeEvent(self, event):
        self.pause()
        self._storyboard.clear()
        
        if hasattr(self, "_render_thread") and self._render_thread.isRunning():
            self._worker.release()
            self._render_thread.quit()
            self._render_thread.wait()
            
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().closeEvent(event)
