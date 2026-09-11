"""
ui/filter_strip.py
Horizontal scrollable filter selector strip.
Shows filter family tabs + thumbnail previews with active state highlighting.
Includes a filter intensity slider shown when a filter is active.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional

from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QFont
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QLabel,
    QPushButton, QFrame, QSizePolicy, QSlider,
)

from core.filters import get_filter_families, get_filter_thumbnail, FILTER_DEFINITIONS


THUMB_W, THUMB_H = 72, 58


def _arr_to_pixmap(arr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


class FilterButton(QWidget):
    """Individual filter button with thumbnail preview."""
    clicked = Signal(str)  # filter_id

    def __init__(self, filter_id: str, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("FilterButton")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._filter_id = filter_id
        self._active = False
        self.setFixedSize(THUMB_W + 6, THUMB_H + 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_ui(name)

    def _build_ui(self, name: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        self._thumb = QLabel()
        self._thumb.setFixedSize(THUMB_W, THUMB_H)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setObjectName("FilterThumbImage")

        self._name_lbl = QLabel(name)
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_lbl.setObjectName("FilterName")

        layout.addWidget(self._thumb)
        layout.addWidget(self._name_lbl)
        self._update_style()

    def set_thumbnail(self, arr: np.ndarray) -> None:
        px = _arr_to_pixmap(arr)
        self._thumb.setPixmap(px)

    def set_active(self, active: bool) -> None:
        self._active = active
        self._update_style()

    def _update_style(self):
        self.setProperty("active", "true" if self._active else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self._name_lbl.setProperty("active", "true" if self._active else "false")
        self._name_lbl.style().unpolish(self._name_lbl)
        self._name_lbl.style().polish(self._name_lbl)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._filter_id)
        super().mousePressEvent(event)


class FilterStrip(QWidget):
    """
    Horizontal filter strip with family tabs and thumbnail previews.
    Includes an intensity slider shown when a filter is active.
    """
    filter_selected = Signal(str)    # filter_id (or "none")
    intensity_changed = Signal(float)  # 0.0 – 1.0

    # Height when intensity bar is visible vs hidden
    _HEIGHT_BASE = THUMB_H + 70
    _HEIGHT_WITH_INTENSITY = THUMB_H + 98

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FilterStrip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(self._HEIGHT_BASE)
        self._source_img: Optional[np.ndarray] = None
        self._buttons: dict[str, FilterButton] = {}
        self._active_filter: str = "none"
        self._active_family: str = ""
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setObjectName("DividerLine")
        div.setFixedHeight(1)
        outer.addWidget(div)

        # Inner content
        inner = QWidget()
        inner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        inner.setObjectName("FilterInner")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 6, 8, 4)
        inner_layout.setSpacing(4)

        # ── Family tabs ───────────────────────────────────────────────────────
        families_layout = QHBoxLayout()
        families_layout.setSpacing(4)

        families = get_filter_families()
        self._family_tabs: dict[str, QPushButton] = {}

        # "None" tab
        none_btn = QPushButton("None")
        none_btn.setCheckable(True)
        none_btn.setChecked(True)
        none_btn.setFixedHeight(22)
        none_btn.setObjectName("FilterFamilyTab")
        none_btn.clicked.connect(lambda: self._on_family_tab("none"))
        self._family_tabs["none"] = none_btn
        families_layout.addWidget(none_btn)

        for family_name in families:
            btn = QPushButton(family_name)
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setObjectName("FilterFamilyTab")
            btn.clicked.connect(lambda checked, fn=family_name: self._on_family_tab(fn))
            self._family_tabs[family_name] = btn
            families_layout.addWidget(btn)

        families_layout.addStretch()
        inner_layout.addLayout(families_layout)

        # ── Filter scroll area ────────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setFixedHeight(THUMB_H + 28)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._filter_row = QWidget()
        self._filter_layout = QHBoxLayout(self._filter_row)
        self._filter_layout.setContentsMargins(0, 0, 0, 0)
        self._filter_layout.setSpacing(6)
        self._filter_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Populate ALL buttons (hidden initially)
        for fdef in FILTER_DEFINITIONS.values():
            btn = FilterButton(fdef.id, fdef.name)
            btn.clicked.connect(self._on_filter_clicked)
            self._buttons[fdef.id] = btn
            self._filter_layout.addWidget(btn)
            btn.setVisible(False)

        self._scroll.setWidget(self._filter_row)
        inner_layout.addWidget(self._scroll)

        # ── Intensity row (hidden until a filter is active) ───────────────────
        self._intensity_row = QWidget()
        self._intensity_row.setVisible(False)
        intensity_layout = QHBoxLayout(self._intensity_row)
        intensity_layout.setContentsMargins(0, 2, 0, 2)
        intensity_layout.setSpacing(8)

        intensity_lbl = QLabel("Intensity")
        intensity_lbl.setObjectName("ParamLabel")
        intensity_lbl.setFixedWidth(56)

        self._intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self._intensity_slider.setRange(0, 100)
        self._intensity_slider.setValue(100)
        self._intensity_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._intensity_slider.valueChanged.connect(self._on_intensity_changed)

        self._intensity_val_lbl = QLabel("100%")
        self._intensity_val_lbl.setObjectName("ValueLabel")
        self._intensity_val_lbl.setFixedWidth(38)
        self._intensity_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        intensity_layout.addWidget(intensity_lbl)
        intensity_layout.addWidget(self._intensity_slider)
        intensity_layout.addWidget(self._intensity_val_lbl)

        inner_layout.addWidget(self._intensity_row)

        outer.addWidget(inner)

        # Show "None" family initially (empty)
        self._show_family_buttons("none")

    def _on_intensity_changed(self, val: int) -> None:
        self._intensity_val_lbl.setText(f"{val}%")
        self.intensity_changed.emit(val / 100.0)

    def _show_intensity_row(self, visible: bool) -> None:
        self._intensity_row.setVisible(visible)
        new_h = self._HEIGHT_WITH_INTENSITY if visible else self._HEIGHT_BASE
        self.setFixedHeight(new_h)

    def _on_family_tab(self, family: str) -> None:
        # Update tab checked states
        for fname, btn in self._family_tabs.items():
            btn.setChecked(fname == family)
        self._active_family = family
        self._show_family_buttons(family)

        if family == "none":
            self._deactivate_all()
            self.filter_selected.emit("none")

    def _show_family_buttons(self, family: str) -> None:
        for fid, btn in self._buttons.items():
            if family == "none":
                btn.setVisible(False)
            else:
                fdef = FILTER_DEFINITIONS[fid]
                btn.setVisible(fdef.family == family)

    def _on_filter_clicked(self, filter_id: str) -> None:
        if self._active_filter == filter_id:
            # Toggle off
            self._deactivate_all()
            self._show_intensity_row(False)
            self.filter_selected.emit("none")
        else:
            self._deactivate_all()
            self._active_filter = filter_id
            if filter_id in self._buttons:
                self._buttons[filter_id].set_active(True)
            self._show_intensity_row(True)
            self.filter_selected.emit(filter_id)

    def _deactivate_all(self) -> None:
        self._active_filter = "none"
        for btn in self._buttons.values():
            btn.set_active(False)

    def set_source_image(self, arr: np.ndarray) -> None:
        """Update filter thumbnails to reflect the current image."""
        self._source_img = arr
        if arr is None:
            return
        for fid, btn in self._buttons.items():
            if btn.isVisible():
                thumb = get_filter_thumbnail(fid, arr, target_w=THUMB_W, target_h=THUMB_H)
                btn.set_thumbnail(thumb)

    def update_visible_thumbnails(self) -> None:
        """Regenerate thumbnails for visible buttons (called on family tab switch)."""
        if self._source_img is None:
            return
        for fid, btn in self._buttons.items():
            if btn.isVisible():
                thumb = get_filter_thumbnail(fid, self._source_img, target_w=THUMB_W, target_h=THUMB_H)
                btn.set_thumbnail(thumb)

    def set_active_filter(self, filter_id: str, silent: bool = True) -> None:
        """Set the active filter and update the intensity row visibility."""
        self._deactivate_all()
        if filter_id != "none":
            self._active_filter = filter_id
            if filter_id in self._buttons:
                self._buttons[filter_id].set_active(True)
            # Switch to the correct family tab
            fdef = FILTER_DEFINITIONS.get(filter_id)
            if fdef:
                for fname, btn in self._family_tabs.items():
                    btn.setChecked(fname == fdef.family)
                self._active_family = fdef.family
                self._show_family_buttons(fdef.family)
            self._show_intensity_row(True)
        else:
            # Reset to "None" tab
            for fname, btn in self._family_tabs.items():
                btn.setChecked(fname == "none")
            self._active_family = "none"
            self._show_family_buttons("none")
            self._show_intensity_row(False)

    def set_filter_intensity(self, intensity: float, silent: bool = True) -> None:
        """Set the intensity slider value (0.0–1.0). Used for two-way sync."""
        val = max(0, min(100, int(round(intensity * 100))))
        if silent:
            self._intensity_slider.blockSignals(True)
        self._intensity_slider.setValue(val)
        self._intensity_val_lbl.setText(f"{val}%")
        if silent:
            self._intensity_slider.blockSignals(False)

    def get_filter_intensity(self) -> float:
        """Return current intensity as 0.0–1.0."""
        return self._intensity_slider.value() / 100.0
