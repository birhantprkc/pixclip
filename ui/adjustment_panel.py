"""
ui/adjustment_panel.py
The right-hand slider panel with debounced live preview updates.
Two sections: Clarity & Sharpness (top priority), and Light & Exposure.
"""

from __future__ import annotations
from PySide6.QtCore import Signal, QTimer, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QScrollArea, QFrame, QSizePolicy, QPushButton,
)
from PySide6.QtGui import QFont

from core.params import AdjustmentParams


# Parameter definitions: (param_name, display_label, min, max, default)
CLARITY_PARAMS = [
    ("clarity",   "Clarity",   0,    100,  0),
    ("sharpness", "Sharpness", 0,    100,  0),
]

LIGHT_PARAMS = [
    ("exposure",     "Exposure",    -100, 100, 0),
    ("brightness",   "Brightness",  -100, 100, 0),
    ("contrast",     "Contrast",    -100, 100, 0),
    ("lightness",    "Lightness",   -100, 100, 0),
    ("highlights",   "Highlights",  -100, 100, 0),
    ("shadows",      "Shadows",     -100, 100, 0),
    ("light_range",  "Light Range", -100, 100, 0),
    ("dark_range",   "Dark Range",  -100, 100, 0),
]


class ParamSlider(QWidget):
    """A single labeled slider row with live value display."""
    value_changed = Signal(str, float)  # (param_name, value)

    def __init__(self, param: str, label: str, min_val: int, max_val: int,
                 default: int = 0, parent=None):
        super().__init__(parent)
        self._param = param
        self._default = default
        self._build_ui(label, min_val, max_val, default)

    def _build_ui(self, label: str, min_val: int, max_val: int, default: int):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)

        # Top row: label + value
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setObjectName("ParamLabel")

        self._value_label = QLabel(f"{default:+d}" if default != 0 else "0")
        self._value_label.setObjectName("ValueLabel")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value_label.setFixedWidth(38)

        top.addWidget(lbl)
        top.addStretch()
        top.addWidget(self._value_label)

        # Slider Row
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(6)

        btn_minus = QPushButton("−")
        btn_minus.setObjectName("IconButton")
        btn_minus.setFixedSize(20, 20)
        btn_minus.clicked.connect(lambda: self._slider.setValue(self._slider.value() - 1))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(min_val)
        self._slider.setMaximum(max_val)
        self._slider.setValue(default)
        self._slider.setTickPosition(QSlider.TickPosition.NoTicks)

        if min_val < 0:
            self._slider.setStyleSheet("")  # use global style

        self._slider.valueChanged.connect(self._on_slider_changed)

        btn_plus = QPushButton("+")
        btn_plus.setObjectName("IconButton")
        btn_plus.setFixedSize(20, 20)
        btn_plus.clicked.connect(lambda: self._slider.setValue(self._slider.value() + 1))

        slider_layout.addWidget(btn_minus)
        slider_layout.addWidget(self._slider)
        slider_layout.addWidget(btn_plus)

        layout.addLayout(top)
        layout.addLayout(slider_layout)

    def _on_slider_changed(self, val: int):
        text = f"{val:+d}" if val != 0 else "0"
        self._value_label.setText(text)
        self.value_changed.emit(self._param, float(val))

    def set_value(self, val: float, silent: bool = False) -> None:
        if silent:
            self._slider.blockSignals(True)
        self._slider.setValue(int(val))
        text = f"{int(val):+d}" if int(val) != 0 else "0"
        self._value_label.setText(text)
        if silent:
            self._slider.blockSignals(False)

    def get_value(self) -> float:
        return float(self._slider.value())

    def reset(self, silent: bool = False) -> None:
        self.set_value(self._default, silent=silent)

    def mouseDoubleClickEvent(self, event):
        """Double-click anywhere on the row to reset to default."""
        self.reset()


class SectionDivider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SectionDivider")
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)


class SectionHeader(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("SectionHeader")


class AdjustmentPanel(QWidget):
    """
    Scrollable panel of adjustment sliders.
    Emits param_changed(param_name, value) with 50ms debounce.
    """
    param_changed = Signal(str, float)   # debounced emit to trigger preview
    reset_all = Signal()

    DEBOUNCE_MS = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AdjustmentPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)

        self._sliders: dict[str, ParamSlider] = {}
        self._pending: dict[str, float] = {}
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(self.DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._flush_pending)

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 10, 14, 20)
        layout.setSpacing(4)

        # ── Clarity & Sharpness ───────────────────────────────────────────────
        layout.addWidget(SectionHeader("CLARITY & SHARPNESS"))
        for param, label, mn, mx, default in CLARITY_PARAMS:
            s = ParamSlider(param, label, mn, mx, default)
            s.value_changed.connect(self._on_value_changed)
            self._sliders[param] = s
            layout.addWidget(s)

        layout.addSpacing(8)
        layout.addWidget(SectionDivider())
        layout.addSpacing(8)

        # ── Light & Exposure ──────────────────────────────────────────────────
        layout.addWidget(SectionHeader("LIGHT & EXPOSURE"))
        for param, label, mn, mx, default in LIGHT_PARAMS:
            s = ParamSlider(param, label, mn, mx, default)
            s.value_changed.connect(self._on_value_changed)
            self._sliders[param] = s
            layout.addWidget(s)

        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _on_value_changed(self, param: str, value: float) -> None:
        self._pending[param] = value
        self._debounce_timer.start()

    def _flush_pending(self) -> None:
        for param, value in self._pending.items():
            self.param_changed.emit(param, value)
        self._pending.clear()

    def load_params(self, params: AdjustmentParams, silent: bool = True) -> None:
        """Populate all sliders from an AdjustmentParams instance."""
        for param, slider in self._sliders.items():
            val = getattr(params, param, slider._default)
            slider.set_value(val, silent=silent)

    def get_params(self) -> AdjustmentParams:
        """Read current slider values into an AdjustmentParams."""
        p = AdjustmentParams()
        for param, slider in self._sliders.items():
            setattr(p, param, slider.get_value())
        return p

    def reset_all_sliders(self) -> None:
        for slider in self._sliders.values():
            slider.reset(silent=True)
        # Flush a full reset emit
        p = AdjustmentParams()
        for param in self._sliders:
            self.param_changed.emit(param, getattr(p, param))
