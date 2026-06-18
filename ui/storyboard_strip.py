"""
ui/storyboard_strip.py
Video storyboard strip (frame picker): extracts evenly spaced thumbnails
from a video in a background thread and presents them in a horizontally
scrolling row. Clicking a thumbnail seeks the player to that frame.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal, QThread, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame,
)

def make_blurred_bg_thumbnail(frame: np.ndarray, target_w: int = 96, target_h: int = 54) -> np.ndarray:
    """
    Creates a target_w x target_h composite with a blurred background and centered foreground.
    """
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 1. Generate the blurred background (fill/cover target_w x target_h)
    scale_w = target_w / w
    scale_h = target_h / h
    scale = max(scale_w, scale_h)

    bg_w = int(round(w * scale))
    bg_h = int(round(h * scale))

    bg = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

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

    fg = cv2.resize(frame, (fg_w, fg_h), interpolation=cv2.INTER_AREA if scale_fit < 1.0 else cv2.INTER_LINEAR)

    # 3. Paste the foreground onto the center of the blurred background
    dx = max(0, (target_w - fg_w) // 2)
    dy = max(0, (target_h - fg_h) // 2)

    result = blurred_bg.copy()
    fg_crop_h = min(fg_h, target_h - dy)
    fg_crop_w = min(fg_w, target_w - dx)
    result[dy:dy+fg_crop_h, dx:dx+fg_crop_w] = fg[:fg_crop_h, :fg_crop_w]

    return result

class StoryboardWorker(QThread):
    """Asynchronously extracts 8 thumbnails from a video file."""
    frame_extracted = Signal(int, QImage, str)  # (frame_idx, image, timestamp_str)
    finished = Signal()

    def __init__(self, video_path: str, frame_count: int, fps: float):
        super().__init__()
        self.video_path = video_path
        self.frame_count = frame_count
        self.fps = fps
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if self.frame_count <= 0:
            self.finished.emit()
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return

        # We will extract 8 frames evenly spaced
        num_frames = 8
        indices = []
        if self.frame_count < num_frames:
            indices = list(range(self.frame_count))
        else:
            step = self.frame_count / num_frames
            indices = [int(i * step) for i in range(num_frames)]
            indices[-1] = self.frame_count - 1

        for idx in indices:
            if self._cancelled:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Scale and composite with blurred background (96 x 54)
            tw, th = 96, 54
            thumb_bgr = make_blurred_bg_thumbnail(frame, tw, th)

            # Convert BGR to RGB
            rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data.tobytes(), tw, th, 3 * tw, QImage.Format.Format_RGB888)
            qimg_copy = qimg.copy()

            # Format timestamp
            secs = idx / self.fps if self.fps > 0 else 0.0
            mins = int(secs) // 60
            remaining_secs = int(secs) % 60
            timestamp_str = f"{mins}:{remaining_secs:02d}"

            self.frame_extracted.emit(idx, qimg_copy, timestamp_str)

        cap.release()
        self.finished.emit()


class StoryboardThumbnail(QFrame):
    """A single frame thumbnail button in the storyboard strip."""
    clicked = Signal(int)

    def __init__(self, frame_idx: int, image: QImage, timestamp: str, parent=None):
        super().__init__(parent)
        self.frame_idx = frame_idx
        self._active = False
        
        lbl_w = 96
        lbl_h = 54
        
        self.setFixedSize(lbl_w + 4, lbl_h + 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("FilmstripThumb")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        self._lbl_img = QLabel()
        self._lbl_img.setFixedSize(lbl_w, lbl_h)
        self._lbl_img.setScaledContents(True)
        self._lbl_img.setPixmap(QPixmap.fromImage(image))
        self._lbl_img.setObjectName("StoryboardThumbImage")
        
        self._lbl_time = QLabel(timestamp)
        self._lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_time.setObjectName("AssetDuration")
        
        layout.addWidget(self._lbl_img)
        layout.addWidget(self._lbl_time)
        
        self._update_border()

    def update_image(self, image: QImage):
        self._lbl_img.setPixmap(QPixmap.fromImage(image))

    def set_active(self, active: bool):
        if self._active != active:
            self._active = active
            self._update_border()

    def _update_border(self):
        self.setProperty("active", "true" if self._active else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.frame_idx)
        super().mousePressEvent(event)


class VideoStoryboardStrip(QWidget):
    """Horizontal strip showing video storyboard thumbnails for seeking."""
    frame_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self.setObjectName("FilmstripContainer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(1)
        
        # Header
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 4, 0)
        lbl = QLabel("STORYBOARD / FRAME PICKER")
        lbl.setObjectName("SectionHeader")
        
        self._lbl_status = QLabel("Ready")
        self._lbl_status.setObjectName("InfoSub")
        
        header.addWidget(lbl)
        header.addStretch()
        header.addWidget(self._lbl_status)
        layout.addLayout(header)

        # Scroll Area for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self._content = QWidget()
        self._content_layout = QHBoxLayout(self._content)
        self._content_layout.setContentsMargins(4, 2, 4, 2)
        self._content_layout.setSpacing(8)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        scroll.setWidget(self._content)
        layout.addWidget(scroll)

        self._thumbnails: dict[int, StoryboardThumbnail] = {}
        self._worker: Optional[StoryboardWorker] = None
        self._current_frame_idx = -1

    def load_video(self, video_path: str, frame_count: int, fps: float, width: int = 0, height: int = 0):
        self.clear()
        
        if frame_count <= 0:
            return
            
        self._lbl_status.setText("Extracting frames...")
        
        # Calculate storyboard frame indices immediately
        num_frames = 8
        indices = []
        if frame_count < num_frames:
            indices = list(range(frame_count))
        else:
            step = frame_count / num_frames
            indices = [int(i * step) for i in range(num_frames)]
            indices[-1] = frame_count - 1
            
        lbl_h = 54
        lbl_w = 96
            
        # Create a skeleton placeholder QImage
        from PySide6.QtGui import QColor, QPainter, QFont
        placeholder = QImage(lbl_w, lbl_h, QImage.Format.Format_RGB32)
        placeholder.fill(QColor(32, 32, 32))
        painter = QPainter(placeholder)
        painter.setPen(QColor(70, 70, 85))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(placeholder.rect(), Qt.AlignmentFlag.AlignCenter, "Loading...")
        painter.end()

        # Generate skeleton cards
        for idx in indices:
            secs = idx / fps if fps > 0 else 0.0
            mins = int(secs) // 60
            remaining_secs = int(secs) % 60
            timestamp_str = f"{mins}:{remaining_secs:02d}"
            
            thumb = StoryboardThumbnail(idx, placeholder, timestamp_str, self)
            thumb.clicked.connect(self.frame_selected.emit)
            self._thumbnails[idx] = thumb
            self._content_layout.addWidget(thumb)
            
        if self._current_frame_idx == -1:
            self.highlight_frame(0)
        
        self._worker = StoryboardWorker(video_path, frame_count, fps)
        self._worker.frame_extracted.connect(self._on_frame_extracted)
        self._worker.finished.connect(self._on_extraction_finished)
        self._worker.start()

    def clear(self):
        if self._worker:
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
            
        for child in list(self._thumbnails.values()):
            self._content_layout.removeWidget(child)
            child.deleteLater()
        self._thumbnails.clear()
        self._current_frame_idx = -1
        self._lbl_status.setText("Ready")

    def highlight_frame(self, frame_idx: int):
        if not self._thumbnails:
            return
            
        # Find closest storyboard frame index
        closest_idx = min(self._thumbnails.keys(), key=lambda k: abs(k - frame_idx))
        
        if closest_idx != self._current_frame_idx:
            if self._current_frame_idx in self._thumbnails:
                self._thumbnails[self._current_frame_idx].set_active(False)
            self._current_frame_idx = closest_idx
            self._thumbnails[closest_idx].set_active(True)

    def _on_frame_extracted(self, frame_idx: int, image: QImage, timestamp: str):
        if frame_idx in self._thumbnails:
            self._thumbnails[frame_idx].update_image(image)

    def _on_extraction_finished(self):
        self._lbl_status.setText("Storyboard loaded")
        if self._worker:
            self._worker.wait()
            self._worker.deleteLater()
            self._worker = None
