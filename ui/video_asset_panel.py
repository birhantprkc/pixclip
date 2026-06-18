"""
ui/video_asset_panel.py
Left-side video asset panel: grid view of imported videos with drag-and-drop support.
Displays video thumbnails, duration badges, selection states, and groups.
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

from core.params import VideoRecord, ProjectState
from core.video_processor import get_video_info
from ui.flow_layout import FlowLayout

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
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


def format_duration(seconds: float) -> str:
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}:{secs:02d}"


class VideoThumbnailCard(QFrame):
    """Individual thumbnail card in the video grid with duration and selection indicator."""
    clicked = Signal(str)         # video_id
    context_menu = Signal(str, object)  # video_id, QPoint

    def __init__(self, record: VideoRecord, parent=None):
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

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setObjectName("AssetThumbImage")

        name_layout = QHBoxLayout()
        name_layout.setContentsMargins(2, 0, 2, 0)
        name_layout.setSpacing(4)

        # Video symbol/icon
        self._status_indicator = QLabel("▶")
        self._status_indicator.setObjectName("AssetStatus")
        
        self._name_label = QLabel(
            self.record.name[:12] + "…" if len(self.record.name) > 12 else self.record.name
        )
        self._name_label.setObjectName("AssetName")

        self._duration_lbl = QLabel(format_duration(self.record.duration))
        self._duration_lbl.setObjectName("AssetDuration")

        name_layout.addWidget(self._status_indicator)
        name_layout.addWidget(self._name_label, stretch=1)
        name_layout.addWidget(self._duration_lbl)

        layout.addWidget(self._thumb_label)
        layout.addLayout(name_layout)

        self._update_border()

    def update_group_info(self) -> None:
        base_name = self.record.name
        disp_name = base_name[:12] + "…" if len(base_name) > 12 else base_name
        if self.record.group_id:
            self._name_label.setText(f"[{self.record.group_id}] {disp_name}")
            self._name_label.setProperty("hasGroup", True)
            self.setToolTip(f"Name: {base_name}\nGroup: {self.record.group_id}")
        else:
            self._name_label.setText(disp_name)
            self._name_label.setProperty("hasGroup", False)
            self.setToolTip(f"Name: {base_name}\nNo Group")
        self._name_label.style().unpolish(self._name_label)
        self._name_label.style().polish(self._name_label)

    def set_thumbnail(self, arr: np.ndarray) -> None:
        px = _ndarray_to_pixmap(arr, THUMB_SIZE)
        self._thumb_label.setPixmap(px)

    def set_active(self, active: bool) -> None:
        self._active = active
        self._update_border()

    def _update_border(self):
        self.setProperty("active", "true" if self._active else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.record.id)
        super().mousePressEvent(event)


class VideoImportWorker(QThread):
    """Background worker to import videos and extract thumbnails asynchronously."""
    progress = Signal(str, int, int, object)  # (filename, current, total, VideoRecord)
    finished = Signal(bool)  # (is_cancelled)

    def __init__(self, paths: List[Path]):
        super().__init__()
        self.paths = paths
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._cancelled:
                self.finished.emit(True)
                return

            filename = path.name
            try:
                record_id = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()
                
                # Fetch metadata and thumbnail
                meta, thumb = get_video_info(path)
                
                if meta["fps"] > 0:
                    record = VideoRecord(
                        path=path,
                        id=record_id,
                        duration=meta["duration"],
                        fps=meta["fps"],
                        frame_count=meta["frame_count"],
                        width=meta["width"],
                        height=meta["height"]
                    )
                    record.thumbnail = thumb
                    self.progress.emit(filename, i + 1, total, record)
                else:
                    self.progress.emit(f"Failed loading: {filename}", i + 1, total, None)
            except Exception as e:
                self.progress.emit(f"Error: {filename}", i + 1, total, None)

        self.finished.emit(self._cancelled)


class VideoImportProgressDialog(QDialog):
    """Sleek modal dialog showing video import progress."""
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Importing Videos")
        self.setModal(True)
        self.setMinimumWidth(380)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.title_label = QLabel("Importing Videos...")
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


class VideoAssetPanel(QFrame):
    """
    Left video asset panel with drag-and-drop import, thumbnail grid,
    and selection management.
    """
    video_selected = Signal(str)         # video_id
    videos_added = Signal(list)          # List[VideoRecord]
    video_removed = Signal(str)          # video_id
    context_menu_requested = Signal(str, object)  # video_id, pos

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.setObjectName("AssetPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._state = state
        self._cards: dict[str, VideoThumbnailCard] = {}
        self._import_worker: Optional[VideoImportWorker] = None
        self._import_dialog: Optional[VideoImportProgressDialog] = None
        self.setAcceptDrops(True)
        self.setMinimumWidth(130)
        self.setMaximumWidth(450)
        self._build_ui()
        self.refresh()

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

        h_label = QLabel("VIDEOS")
        h_label.setObjectName("SectionHeader")

        self._count_label = QLabel("0")
        self._count_label.setObjectName("InfoSub")

        btn_add = QPushButton("+")
        btn_add.setObjectName("IconButton")
        btn_add.setFixedSize(26, 26)
        btn_add.setToolTip("Add Videos")
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
        self._hint = QLabel("Drop videos\nhere")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setObjectName("DropHint")
        self._grid.addWidget(self._hint)

        scroll.setWidget(self._grid_widget)

        layout.addWidget(header)
        layout.addWidget(scroll, stretch=1)

    def _open_file_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Videos", "",
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if files:
            self._add_files([Path(f) for f in files])

    def _add_files(self, paths: List[Path]) -> None:
        to_import = []
        for path in paths:
            if path.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
                continue
            existing = [vid.path for vid in self._state.videos]
            if path in existing:
                continue
            to_import.append(path)

        if not to_import:
            return

        self._import_worker = VideoImportWorker(to_import)
        self._import_dialog = VideoImportProgressDialog(self.window())

        self._import_worker.progress.connect(self._on_import_progress)
        self._import_worker.finished.connect(self._on_import_finished)
        self._import_dialog.cancelled.connect(self._import_worker.cancel)

        self._import_worker.start()
        self._import_dialog.exec()

    def _on_import_progress(self, filename: str, current: int, total: int, record: Optional[VideoRecord]) -> None:
        if self._import_dialog:
            self._import_dialog.set_progress(filename, current, total)

        if record is not None:
            self._state.add_video(record)
            self._add_card(record)
            if record.thumbnail is not None:
                self._cards[record.id].set_thumbnail(record.thumbnail)
            self.videos_added.emit([record])

    def _on_import_finished(self, is_cancelled: bool) -> None:
        if self._import_dialog:
            self._import_dialog.accept()
            self._import_dialog = None

        self._count_label.setText(str(len(self._state.videos)))
        self._hint.setVisible(len(self._state.videos) == 0)

        # Auto-select active video if none is set
        if not self._state.active_video_id and self._state.videos:
            self.select_video(self._state.videos[0].id)

        if self._import_worker:
            self._import_worker.wait()
            self._import_worker.deleteLater()
            self._import_worker = None

    def _add_card(self, record: VideoRecord) -> None:
        card = VideoThumbnailCard(record)
        card.clicked.connect(self.select_video)
        card.context_menu.connect(self.context_menu_requested.emit)
        self._cards[record.id] = card
        self._grid.addWidget(card)

    def select_video(self, video_id: str) -> None:
        for card in self._cards.values():
            card.set_active(False)
        card = self._cards.get(video_id)
        if card:
            card.set_active(True)
        self._state.active_video_id = video_id
        self.video_selected.emit(video_id)

    def _remove_video(self, video_id: str) -> None:
        card = self._cards.pop(video_id, None)
        if card:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._state.remove_video(video_id)
        self._count_label.setText(str(len(self._state.videos)))
        if not self._cards:
            self._hint.setVisible(True)
        self.video_removed.emit(video_id)

    def refresh(self) -> None:
        """Refresh the grid from the project state."""
        # Remove all existing cards from grid layout
        for card in list(self._cards.values()):
            self._grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        
        # Populate from state
        for video in self._state.videos:
            self._add_card(video)
            if video.thumbnail is not None:
                self._cards[video.id].set_thumbnail(video.thumbnail)
                
        self._count_label.setText(str(len(self._state.videos)))
        self._hint.setVisible(len(self._state.videos) == 0)
        
        # Restore selection state
        if self._state.active_video_id:
            for card_id, card in self._cards.items():
                card.set_active(card_id == self._state.active_video_id)

    # ── Drag & Drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragActive", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                paths.extend(
                    p for p in path.iterdir()
                    if p.suffix.lower() in SUPPORTED_VIDEO_EXTS
                )
            elif path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
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
