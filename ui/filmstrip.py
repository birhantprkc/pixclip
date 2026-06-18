"""
ui/filmstrip.py
Bottom horizontal filmstrip showing all imported images as selectable thumbnails.
Syncs with AssetPanel selection.
"""

from __future__ import annotations
import cv2
import numpy as np
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QScrollArea, QLabel, QFrame, QSizePolicy,
)

from core.params import ImageRecord

STRIP_THUMB = 72


def make_blurred_bg_thumbnail(img: np.ndarray, target_size: int = 72) -> np.ndarray:
    """
    Creates a square target_size x target_size composite with a blurred background
    and centered foreground preserving the aspect ratio.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    target_w = target_size
    target_h = target_size

    # 1. Generate the blurred background (fill/cover target_w x target_h)
    scale_w = target_w / w
    scale_h = target_h / h
    scale = max(scale_w, scale_h)

    bg_w = int(round(w * scale))
    bg_h = int(round(h * scale))

    bg = cv2.resize(img, (bg_w, bg_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    # Center crop background to target_w x target_h
    start_x = max(0, (bg_w - target_w) // 2)
    start_y = max(0, (bg_h - target_h) // 2)
    bg_cropped = bg[start_y:start_y+target_h, start_x:start_x+target_w]

    if bg_cropped.shape[1] != target_w or bg_cropped.shape[0] != target_h:
        bg_cropped = cv2.resize(bg_cropped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Apply heavy blur to the background
    blurred_bg = cv2.GaussianBlur(bg_cropped, (15, 15), 0)
    # Dim the blurred background
    blurred_bg = (blurred_bg * 0.6).astype(np.uint8)

    # 2. Generate the foreground image (fit/contain inside target_w x target_h)
    scale_fit = min(scale_w, scale_h)
    fg_w = max(1, int(round(w * scale_fit)))
    fg_h = max(1, int(round(h * scale_fit)))

    fg = cv2.resize(img, (fg_w, fg_h), interpolation=cv2.INTER_AREA if scale_fit < 1.0 else cv2.INTER_LINEAR)

    # 3. Paste the foreground onto the center of the blurred background
    dx = max(0, (target_w - fg_w) // 2)
    dy = max(0, (target_h - fg_h) // 2)

    result = blurred_bg.copy()
    fg_crop_h = min(fg_h, target_h - dy)
    fg_crop_w = min(fg_w, target_w - dx)
    result[dy:dy+fg_crop_h, dx:dx+fg_crop_w] = fg[:fg_crop_h, :fg_crop_w]

    return result


def _ndarray_to_pixmap(arr: np.ndarray, size: int) -> QPixmap:
    # 1. Create the blurred background composite
    thumb_bgr = make_blurred_bg_thumbnail(arr, size)
    # 2. Convert to QPixmap
    rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


class FilmstripScrollArea(QScrollArea):
    """Custom scroll area that redirects vertical mouse wheel events to horizontal scrolling."""
    def wheelEvent(self, event):
        if event.angleDelta().y() != 0:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - event.angleDelta().y()
            )
            event.accept()
        else:
            super().wheelEvent(event)


class FilmstripThumb(QFrame):
    """Single thumbnail in the filmstrip with hover and active select states."""
    clicked = Signal(str)
    context_menu = Signal(str, object)

    def __init__(self, record: ImageRecord, parent=None):
        super().__init__(parent)
        self.setObjectName("FilmstripThumb")
        self._id = record.id
        self._active = False
        self.setFixedSize(STRIP_THUMB + 6, STRIP_THUMB + 6)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda pos: self.context_menu.emit(self._id, self.mapToGlobal(pos))
        )
        self._build_ui()
        self.update_group_tooltip(record)

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        self._img = QLabel()
        self._img.setFixedSize(STRIP_THUMB, STRIP_THUMB)
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setObjectName("ThumbImage")
        layout.addWidget(self._img)
        self._update_border()

    def update_group_tooltip(self, record: ImageRecord) -> None:
        self.setToolTip(f"Group: {record.group_id}" if record.group_id else "No Group")

    def set_thumbnail(self, arr: np.ndarray) -> None:
        px = _ndarray_to_pixmap(arr, STRIP_THUMB)
        self._img.setPixmap(px)

    def set_active(self, active: bool) -> None:
        self._active = active
        self._update_border()

    def _update_border(self):
        self.setProperty("active", "true" if self._active else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._id)
        super().mousePressEvent(event)


class FilmstripWidget(QWidget):
    """Horizontal filmstrip that mirrors the asset panel selection."""
    image_selected = Signal(str)
    context_menu_requested = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FilmstripContainer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(STRIP_THUMB + 20)
        self._thumbs: dict[str, FilmstripThumb] = {}
        self._selected_ids: List[str] = [] # future-ready multi-select tracking
        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = FilmstripScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        self._row_widget = QWidget()
        self._row_widget.setStyleSheet("background: transparent;")
        self._row_layout = QHBoxLayout(self._row_widget)
        self._row_layout.setContentsMargins(8, 6, 8, 6)
        self._row_layout.setSpacing(8)
        self._row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        scroll.setWidget(self._row_widget)
        outer.addWidget(scroll)

    def add_record(self, record: ImageRecord) -> None:
        # Avoid duplicate cards
        if record.id in self._thumbs:
            return
        thumb = FilmstripThumb(record)
        thumb.clicked.connect(self._on_thumb_clicked)
        thumb.context_menu.connect(self.context_menu_requested.emit)
        if record.thumbnail is not None:
            thumb.set_thumbnail(record.thumbnail)
        self._thumbs[record.id] = thumb
        self._row_layout.addWidget(thumb)

    def update_thumbnail(self, image_id: str, arr: np.ndarray) -> None:
        thumb = self._thumbs.get(image_id)
        if thumb:
            thumb.set_thumbnail(arr)

    def remove_record(self, image_id: str) -> None:
        thumb = self._thumbs.pop(image_id, None)
        if thumb:
            self._row_layout.removeWidget(thumb)
            thumb.deleteLater()
        if image_id in self._selected_ids:
            self._selected_ids.remove(image_id)

    def select(self, image_id: str) -> None:
        self._selected_ids = [image_id]
        for tid, t in self._thumbs.items():
            t.set_active(tid == image_id)

    def _on_thumb_clicked(self, image_id: str) -> None:
        self.select(image_id)
        self.image_selected.emit(image_id)
