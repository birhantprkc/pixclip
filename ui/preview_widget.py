"""
ui/preview_widget.py
Hardware-accelerated preview panel using QGraphicsView.
Supports pan (middle-mouse drag), zoom (scroll wheel), and fit-to-window.
Features a tabbed interface (Original | Edited | Compare | Split) for reliable visual feedback.
"""

from __future__ import annotations
import cv2
import numpy as np
from PySide6.QtCore import (
    Qt, QRectF, Signal, QPoint, QPointF, QTimer, QPropertyAnimation, QEvent, QSettings,
)
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,
    QColor, QResizeEvent, QPen, QLinearGradient, QFont,
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsObject,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSplitter, QButtonGroup, QStackedWidget, QGraphicsOpacityEffect,
)


def ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a uint8 BGR OpenCV array to a QImage (RGB)."""
    if arr is None:
        return QImage()
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)


class SplitImageItem(QGraphicsObject):
    """A custom QGraphicsObject that draws two images side-by-side with a draggable split line."""
    def __init__(self):
        super().__init__()
        self.orig_pixmap = QPixmap()
        self.edit_pixmap = QPixmap()
        self.ratio = 0.5
        self._dragging = False
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def boundingRect(self) -> QRectF:
        if not self.orig_pixmap.isNull():
            return QRectF(self.orig_pixmap.rect())
        elif not self.edit_pixmap.isNull():
            return QRectF(self.edit_pixmap.rect())
        return QRectF()

    def paint(self, painter: QPainter, option, widget):
        if self.orig_pixmap.isNull():
            return

        rect = self.boundingRect()
        split_x = rect.width() * self.ratio

        # Draw left half (original)
        painter.setClipRect(0, 0, split_x, rect.height())
        painter.drawPixmap(0, 0, self.orig_pixmap)

        # Draw right half (edited)
        painter.setClipRect(split_x, 0, rect.width() - split_x, rect.height())
        if not self.edit_pixmap.isNull():
            painter.drawPixmap(rect.toRect(), self.edit_pixmap, self.edit_pixmap.rect())

        painter.setClipping(False)

        # Draw split line
        pen = QPen(QColor(255, 255, 255, 200), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(split_x, 0), QPointF(split_x, rect.height()))

        # Draw a small handle
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        handle_rect = QRectF(split_x - 4, rect.height() / 2 - 20, 8, 40)
        painter.drawRoundedRect(handle_rect, 4, 4)

    def _get_tolerance(self) -> float:
        views = self.scene().views()
        if views:
            scale = views[0].transform().m11()
            return 20.0 / scale if scale > 0 else 20.0
        return 20.0

    def _is_on_handle(self, pos: QPointF) -> bool:
        split_x = self.boundingRect().width() * self.ratio
        return abs(pos.x() - split_x) < self._get_tolerance()

    def hoverMoveEvent(self, event):
        if self._is_on_handle(event.pos()):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if self._is_on_handle(event.pos()):
            self._dragging = True
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self._dragging:
            w = self.boundingRect().width()
            if w > 0:
                self.ratio = max(0.0, min(1.0, event.pos().x() / w))
                self.update()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            event.accept()
        else:
            event.ignore()


class ImageGraphicsView(QGraphicsView):
    """
    Custom QGraphicsView with smooth pan/zoom for a single image.
    """
    zoom_changed = Signal(float)  # emits zoom level 0.0 to N

    ZOOM_MIN = 0.05
    ZOOM_MAX = 16.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)

        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPoint()
        self._fit_on_next = True

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_image(self, arr: np.ndarray) -> None:
        """Display a new image."""
        if arr is None:
            return
        qi = ndarray_to_qimage(arr)
        px = QPixmap.fromImage(qi)
        self._item.setPixmap(px)
        self._scene.setSceneRect(QRectF(px.rect()))
        if self._fit_on_next:
            self.fit_in_view()
            self._fit_on_next = False

    def clear_image(self) -> None:
        self._item.setPixmap(QPixmap())

    def fit_in_view(self) -> None:
        """Scale image to fit the viewport while preserving aspect ratio."""
        items = self._scene.items()
        if not items:
            return
        rect = items[0].boundingRect()
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.zoom_changed.emit(self._zoom)

    def set_zoom(self, zoom: float, emit: bool = True) -> None:
        zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, zoom))
        factor = zoom / self._zoom
        self.scale(factor, factor)
        self._zoom = zoom
        if emit:
            self.zoom_changed.emit(self._zoom)

    def zoom_to_100(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        self.zoom_changed.emit(1.0)

    @property
    def zoom_level(self) -> float:
        return self._zoom

    # ── Events ────────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else (1 / 1.12)
        new_zoom = self._zoom * factor
        if self.ZOOM_MIN <= new_zoom <= self.ZOOM_MAX:
            self.scale(factor, factor)
            self._zoom = new_zoom
            self.zoom_changed.emit(self._zoom)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit_on_next:
            self.fit_in_view()


class ShimmerOverlay(QWidget):
    """
    Sleek overlay showing a semi-transparent dark look
    and a diagonal linear gradient sweeping shimmer.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.offset = -1.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_shimmer)
        self.timer.start(16)  # ~60 FPS
        
        # Opacity effect for smooth transitions
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(180)
        self.hide()

        if parent:
            parent.installEventFilter(self)
            self.setGeometry(0, 0, parent.width(), parent.height())

    def eventFilter(self, watched, event):
        if watched == self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(0, 0, watched.width(), watched.height())
        return super().eventFilter(watched, event)

    def show_shimmer(self):
        if self.parent():
            self.setGeometry(0, 0, self.parent().width(), self.parent().height())
        self.show()
        self.raise_()
        self.fade_animation.stop()
        self.fade_animation.setStartValue(self.opacity_effect.opacity())
        self.fade_animation.setEndValue(0.85)  # max opacity
        self.fade_animation.start()

    def hide_shimmer(self):
        self.fade_animation.stop()
        self.fade_animation.setStartValue(self.opacity_effect.opacity())
        self.fade_animation.setEndValue(0.0)
        
        # Connect to hide after fade finishes
        def on_fade_done():
            if self.opacity_effect.opacity() == 0.0:
                self.hide()
            try:
                self.fade_animation.finished.disconnect()
            except Exception:
                pass
                
        self.fade_animation.finished.connect(on_fade_done)
        self.fade_animation.start()

    def _update_shimmer(self):
        self.offset += 0.02
        if self.offset > 1.2:
            self.offset = -0.6
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        settings = QSettings("PixClip", "PixClip")
        is_light = settings.value("theme", "dark") == "light"

        # 1. Semi-transparent base overlay
        if is_light:
            painter.fillRect(self.rect(), QColor(248, 249, 250, 160))
        else:
            painter.fillRect(self.rect(), QColor(10, 10, 15, 120))

        # 2. Draw sweeping diagonal gradient shimmer
        w, h = self.width(), self.height()
        if w > 0 and h > 0:
            gradient = QLinearGradient(
                self.offset * w, 0,
                (self.offset + 0.5) * w, h
            )
            shimmer_color = QColor(37, 99, 235) if is_light else QColor(124, 106, 247)
            gradient.setColorAt(0.0, QColor(shimmer_color.red(), shimmer_color.green(), shimmer_color.blue(), 0))
            gradient.setColorAt(0.5, QColor(shimmer_color.red(), shimmer_color.green(), shimmer_color.blue(), 30 if is_light else 40))
            gradient.setColorAt(1.0, QColor(shimmer_color.red(), shimmer_color.green(), shimmer_color.blue(), 0))
            
            painter.fillRect(self.rect(), gradient)
            
        # 3. Draw a spinning mini circular indicator in the center
        arc_color = QColor(37, 99, 235, 220) if is_light else QColor(124, 106, 247, 200)
        painter.setPen(QPen(arc_color, 3, Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        center_x = w // 2
        center_y = h // 2
        r = 18
        start_angle = int(self.offset * 720) % 360
        painter.drawArc(center_x - r, center_y - r, r * 2, r * 2, start_angle * 16, 270 * 16)
        
        # Draw text
        text_color = QColor(17, 24, 39, 220) if is_light else QColor(232, 232, 240, 220)
        painter.setPen(text_color)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "\n\nRendering...")


class PreviewWidget(QWidget):
    """
    Full preview panel with 4-tab viewing mode (Original, Edited, Compare, Split).
    Automatically scales the 'Original' rendering to match the 'Edited' rendering 
    size so that side-by-side zoom scales match perfectly.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewContainer")
        self._before_arr: np.ndarray | None = None
        self._current_arr: np.ndarray | None = None
        self._syncing = False
        self._build_ui()
        self._connect_sync()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setObjectName("ZoomToolbar")
        toolbar.setObjectName("ZoomToolbar")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 4, 10, 4)
        tb_layout.setSpacing(6)

        # Left: Zoom controls
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
        btn_fit.clicked.connect(self._view_fit)

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

        # Center: View Mode Tabs
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

        # Right: Info label
        self._info_label = QLabel("")
        self._info_label.setObjectName("PreviewInfoLabel")
        tb_layout.addWidget(self._info_label)

        # ── Views ─────────────────────────────────────────────────────────────
        self._stack = QStackedWidget()
        
        self._view_left = ImageGraphicsView()   # Original view
        self._view_right = ImageGraphicsView()  # Edited view

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

        # Hide left view initially (since Edited is default)
        self._view_left.hide()
        self._stack.setCurrentIndex(0)

        # Loading overlay
        self._loading_overlay = ShimmerOverlay(self._stack)

        layout.addWidget(toolbar)
        layout.addWidget(self._stack, stretch=1)

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

    def _view_fit(self):
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

    def set_image(self, arr: np.ndarray, original: np.ndarray = None) -> None:
        """
        Update the preview. 'arr' is Edited, 'original' is Original.
        Scales 'original' to exactly match 'arr' (which is a 0.5 scale preview image)
        so that pixel sizes align during side-by-side or split comparisons.
        """
        self._current_arr = arr
        self._view_right.set_image(arr)

        if original is not None:
            self._before_arr = original
            
        if self._before_arr is not None and arr is not None:
            if self._before_arr.shape[:2] != arr.shape[:2]:
                scaled_before = cv2.resize(self._before_arr, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                scaled_before = self._before_arr

            self._view_left.set_image(scaled_before)
            
            self._split_item.orig_pixmap = QPixmap.fromImage(ndarray_to_qimage(scaled_before))
            self._split_item.edit_pixmap = QPixmap.fromImage(ndarray_to_qimage(arr))
            self._view_split._scene.setSceneRect(QRectF(self._split_item.orig_pixmap.rect()))
            
            if self._view_split._fit_on_next:
                self._view_split.fit_in_view()
                self._view_split._fit_on_next = False
                
            self._split_item.update()

    def set_info(self, text: str) -> None:
        self._info_label.setText(text)

    def set_loading(self, loading: bool) -> None:
        if loading:
            self._loading_overlay.show_shimmer()
        else:
            self._loading_overlay.hide_shimmer()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._loading_overlay.setGeometry(0, 0, self._stack.width(), self._stack.height())
