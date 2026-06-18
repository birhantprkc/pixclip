"""
ui/asset_panel.py
Left-side asset panel: grid view of imported images with drag-and-drop support.
Displays thumbnails with selection state and group badges.
"""

from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt, Signal, QSize, QMimeData, QTimer, QThread, QObject, Slot,
)
from PySide6.QtGui import (
    QImage, QPixmap, QDragEnterEvent, QDropEvent, QColor,
    QPainter, QBrush, QFont, QPen,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QGridLayout, QSizePolicy, QToolButton,
    QMenu, QDialog, QProgressBar,
)

from core.params import ImageRecord, ProjectState
from core.batch import load_image_record
from ui.flow_layout import FlowLayout
from core.cache import save_cached_thumbnail, get_cached_thumbnail


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
THUMB_SIZE = 100


def make_blurred_bg_thumbnail(img: np.ndarray, target_size: int = 100) -> np.ndarray:
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


class ThumbnailCard(QFrame):
    """Individual thumbnail card in the asset grid with edit status indicator."""
    clicked = Signal(str)         # image_id
    context_menu = Signal(str, object)  # image_id, QPoint

    def __init__(self, record: ImageRecord, parent=None):
        super().__init__(parent)
        self.record = record
        self._active = False
        self.setObjectName("AssetThumbnail")
        self.setFixedSize(THUMB_SIZE + 10, THUMB_SIZE + 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda pos: self.context_menu.emit(record.id, self.mapToGlobal(pos))
        )
        self._build_ui()
        self.update_group_info()
        self.update_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._thumb_label.setObjectName("ThumbImage")

        name_layout = QHBoxLayout()
        name_layout.setContentsMargins(2, 0, 2, 0)
        name_layout.setSpacing(4)

        self._status_indicator = QLabel()
        self._status_indicator.setFixedSize(8, 8)
        self._status_indicator.setObjectName("StatusIndicator")
        
        self._name_label = QLabel(
            self.record.name[:12] + "…" if len(self.record.name) > 12 else self.record.name
        )
        self._name_label.setObjectName("ThumbName")

        name_layout.addWidget(self._status_indicator)
        name_layout.addWidget(self._name_label, stretch=1)

        layout.addWidget(self._thumb_label)
        layout.addLayout(name_layout)

        self._update_border()

    def update_group_info(self) -> None:
        base_name = self.record.name
        disp_name = base_name[:12] + "…" if len(base_name) > 12 else base_name
        if self.record.group_id:
            self._name_label.setText(f"[{self.record.group_id}] {disp_name}")
            self._name_label.setProperty("hasGroup", "true")
            self.setToolTip(f"Name: {base_name}\nGroup: {self.record.group_id}")
        else:
            self._name_label.setText(disp_name)
            self._name_label.setProperty("hasGroup", "false")
            self.setToolTip(f"Name: {base_name}\nNo Group")
        self._name_label.style().unpolish(self._name_label)
        self._name_label.style().polish(self._name_label)

    def set_thumbnail(self, arr: np.ndarray) -> None:
        px = _ndarray_to_pixmap(arr, THUMB_SIZE)
        self._thumb_label.setPixmap(px)

    def set_active(self, active: bool) -> None:
        self._active = active
        self._update_border()

    def update_status(self) -> None:
        """Update the status indicator color depending on edits."""
        if self.record.params and not self.record.params.is_default():
            self._status_indicator.setProperty("edited", "true")
            self._status_indicator.setToolTip("Edited")
        else:
            self._status_indicator.setProperty("edited", "false")
            self._status_indicator.setToolTip("Original")
        self._status_indicator.style().unpolish(self._status_indicator)
        self._status_indicator.style().polish(self._status_indicator)

    def _update_border(self):
        self.setProperty("active", "true" if self._active else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.record.id)
        super().mousePressEvent(event)


class ImportWorker(QThread):
    """Background worker to import images and generate thumbnails asynchronously."""
    progress = Signal(str, int, int, object)  # (filename, current, total, ImageRecord)
    finished = Signal(bool)  # (is_cancelled)

    def __init__(self, paths: List[Path]):
        super().__init__()
        self.paths = paths
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import uuid
        total = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._cancelled:
                self.finished.emit(True)
                return

            filename = path.name
            try:
                # Deterministic ID based on path hash
                record_id = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()
                
                # Check if thumbnail already exists in disk cache
                cached_thumb = get_cached_thumbnail(record_id)
                
                if cached_thumb is not None:
                    record = ImageRecord(path=path, id=record_id)
                    record.thumbnail = cached_thumb
                    self.progress.emit(filename, i + 1, total, record)
                    continue

                # Read image data to generate thumbnail
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    # Build thumbnail (160x160 max, aspect-preserving)
                    h, w = img.shape[:2]
                    scale = min(160 / w, 160 / h)
                    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
                    thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

                    # Save to cache
                    save_cached_thumbnail(record_id, thumb)

                    record = ImageRecord(path=path, id=record_id)
                    record.thumbnail = thumb
                    self.progress.emit(filename, i + 1, total, record)
                else:
                    self.progress.emit(f"Failed loading: {filename}", i + 1, total, None)
            except Exception as e:
                self.progress.emit(f"Error: {filename}", i + 1, total, None)

        self.finished.emit(self._cancelled)


class ImportProgressDialog(QDialog):
    """Sleek modal dialog showing import progress with a cancel button."""
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Importing Images")
        self.setModal(True)
        self.setMinimumWidth(380)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.title_label = QLabel("Importing Images...")
        self.title_label.setObjectName("DialogTitle")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)

        self.file_label = QLabel("Initializing...")
        self.file_label.setObjectName("DialogSub")
        self.file_label.setWordWrap(True)

        self.stats_label = QLabel("0 / 0 files")
        self.stats_label.setObjectName("DialogStats")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("DangerButton")
        self.cancel_button.clicked.connect(self._on_cancel)

        h_info = QHBoxLayout()
        h_info.addWidget(self.file_label)
        h_info.addWidget(self.stats_label)

        layout.addWidget(self.title_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(h_info)
        layout.addWidget(self.cancel_button, alignment=Qt.AlignmentFlag.AlignRight)

    def set_progress(self, filename: str, current: int, total: int):
        self.file_label.setText(f"Processing: {filename}")
        self.stats_label.setText(f"{current} / {total} files")
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))

    def _on_cancel(self):
        self.cancelled.emit()
        self.reject()

    def closeEvent(self, event):
        self.cancelled.emit()
        super().closeEvent(event)


class AssetPanel(QWidget):
    """
    Left asset panel with drag-and-drop import, thumbnail grid,
    and selection management.
    """
    image_selected = Signal(str)         # image_id
    images_added = Signal(list)          # List[ImageRecord]
    image_removed = Signal(str)          # image_id
    context_menu_requested = Signal(str, object)  # image_id, pos

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.setObjectName("AssetPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._state = state
        self._cards: dict[str, ThumbnailCard] = {}
        self._import_worker: Optional[ImportWorker] = None
        self._import_dialog: Optional[ImportProgressDialog] = None
        self.setAcceptDrops(True)
        self.setMinimumWidth(130)
        self.setMaximumWidth(450)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("AssetHeader")
        header.setFixedHeight(40)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 0, 8, 0)

        h_label = QLabel("PHOTOS")
        h_label.setObjectName("SectionHeader")

        self._count_label = QLabel("0")
        self._count_label.setObjectName("CountLabel")

        btn_add = QPushButton("+")
        btn_add.setObjectName("IconButton")
        btn_add.setFixedSize(26, 26)
        btn_add.setToolTip("Add Images")
        btn_add.clicked.connect(self._open_file_dialog)

        h_layout.addWidget(h_label)
        h_layout.addWidget(self._count_label)
        h_layout.addStretch()
        h_layout.addWidget(btn_add)

        # ── Scroll grid ───────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._grid_widget = QWidget()
        self._grid = FlowLayout(self._grid_widget, margin=6, spacing=6)

        # Drop hint
        self._hint = QLabel("Drop photos\nhere")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setObjectName("DropHint")
        self._grid.addWidget(self._hint)

        scroll.setWidget(self._grid_widget)

        layout.addWidget(header)
        layout.addWidget(scroll, stretch=1)

    def _open_file_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Images", "",
            "Images (*.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp)"
        )
        if files:
            self._add_files([Path(f) for f in files])

    def _add_files(self, paths: List[Path]) -> None:
        # Filter supported extensions and duplicates
        to_import = []
        for path in paths:
            if path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            # Avoid duplicates
            existing = [img.path for img in self._state.images]
            if path in existing:
                continue
            to_import.append(path)

        if not to_import:
            return

        # Disable main window interaction optionally or show dialog
        self._import_worker = ImportWorker(to_import)
        self._import_dialog = ImportProgressDialog(self.window())

        self._import_worker.progress.connect(self._on_import_progress)
        self._import_worker.finished.connect(self._on_import_finished)
        self._import_dialog.cancelled.connect(self._import_worker.cancel)

        self._import_worker.start()
        self._import_dialog.exec()

    def _on_import_progress(self, filename: str, current: int, total: int, record: Optional[ImageRecord]) -> None:
        if self._import_dialog:
            self._import_dialog.set_progress(filename, current, total)

        if record is not None:
            # Add to project state
            self._state.add_image(record)
            # Add card
            self._add_card(record)
            # Update thumbnail card dynamically
            if record.thumbnail is not None:
                self._cards[record.id].set_thumbnail(record.thumbnail)
            
            # Emit added signal for live filmstrip updates
            self.images_added.emit([record])

    def _on_import_finished(self, is_cancelled: bool) -> None:
        if self._import_dialog:
            self._import_dialog.accept()
            self._import_dialog = None

        self._count_label.setText(str(len(self._state.images)))
        self._hint.setVisible(len(self._state.images) == 0)

        # Auto-select active image if none is set
        if not self._state.active_id and self._state.images:
            self.select_image(self._state.images[0].id)

        # Cleanup worker
        if self._import_worker:
            self._import_worker.wait()
            self._import_worker.deleteLater()
            self._import_worker = None

    def _add_card(self, record: ImageRecord) -> None:
        card = ThumbnailCard(record)
        card.clicked.connect(self.select_image)
        card.context_menu.connect(self.context_menu_requested.emit)
        self._cards[record.id] = card
        self._grid.addWidget(card)

    def select_image(self, image_id: str) -> None:
        # Deselect all
        for card in self._cards.values():
            card.set_active(False)
        # Activate selected
        card = self._cards.get(image_id)
        if card:
            card.set_active(True)
        self._state.active_id = image_id
        self.image_selected.emit(image_id)

    # Deleted _show_context_menu as context menus are now delegated to MainWindow

    def _remove_image(self, image_id: str) -> None:
        card = self._cards.pop(image_id, None)
        if card:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._state.remove_image(image_id)
        self._count_label.setText(str(len(self._state.images)))
        if not self._cards:
            self._hint.setVisible(True)
        self.image_removed.emit(image_id)

    # ── Drag & Drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("#AssetPanel { border: 1px solid #0078D4; }")

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent) -> None:
        self.setStyleSheet("")
        paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                paths.extend(
                    p for p in path.iterdir()
                    if p.suffix.lower() in SUPPORTED_EXTS
                )
            elif path.suffix.lower() in SUPPORTED_EXTS:
                paths.append(path)
        if paths:
            self._add_files(paths)
        event.acceptProposedAction()

    def closeEvent(self, event) -> None:
        if hasattr(self, "_import_worker") and self._import_worker is not None:
            if self._import_worker.isRunning():
                self._import_worker.cancel()
                self._import_worker.wait()
        super().closeEvent(event)
