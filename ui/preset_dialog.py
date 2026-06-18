"""
ui/preset_dialog.py
Preset management dialog: save current settings, load, and delete presets.
Also supports import/export to external .json files.
"""

from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QFrame, QWidget,
)

from core.params import AdjustmentParams
import core.presets as preset_io


class PresetDialog(QDialog):
    """
    Full preset management dialog.
    preset_loaded signal emits when user applies a preset.
    """
    preset_loaded = Signal(AdjustmentParams)

    def __init__(self, current_params: AdjustmentParams, parent=None):
        super().__init__(parent)
        self._current_params = current_params
        self.setWindowTitle("Presets")
        self.setModal(True)
        self.setMinimumSize(380, 440)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ── Save current ──────────────────────────────────────────────────────
        save_frame = QFrame()
        save_frame.setObjectName("PresetSaveFrame")
        sf_layout = QVBoxLayout(save_frame)
        sf_layout.setContentsMargins(12, 10, 12, 10)
        sf_layout.setSpacing(6)

        sf_label = QLabel("Save Current Settings as Preset")
        sf_label.setObjectName("InfoSub")

        name_row = QHBoxLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Preset name…")
        self._name_edit.returnPressed.connect(self._save_preset)

        save_btn = QPushButton("Save")
        save_btn.setObjectName("PrimaryButton")
        save_btn.setFixedWidth(60)
        save_btn.clicked.connect(self._save_preset)

        name_row.addWidget(self._name_edit)
        name_row.addWidget(save_btn)

        sf_layout.addWidget(sf_label)
        sf_layout.addLayout(name_row)
        layout.addWidget(save_frame)

        # ── Preset list ───────────────────────────────────────────────────────
        list_label = QLabel("SAVED PRESETS")
        list_label.setObjectName("SectionHeader")
        layout.addWidget(list_label)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemDoubleClicked.connect(self._apply_selected)
        layout.addWidget(self._list, stretch=1)

        # ── Action buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("PrimaryButton")
        apply_btn.clicked.connect(self._apply_selected)

        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("DangerButton")
        delete_btn.clicked.connect(self._delete_selected)

        import_btn = QPushButton("Import…")
        import_btn.clicked.connect(self._import_preset)

        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self._export_preset)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)

        btn_row.addWidget(apply_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(import_btn)
        btn_row.addWidget(export_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _refresh_list(self):
        self._list.clear()
        for name, _ in preset_io.list_presets():
            item = QListWidgetItem(name)
            self._list.addItem(item)

    def _save_preset(self):
        name = self._name_edit.text().strip()
        if not name:
            return
        preset_io.save_preset(name, self._current_params)
        self._name_edit.clear()
        self._refresh_list()

    def _apply_selected(self):
        item = self._list.currentItem()
        if not item:
            return
        params = preset_io.load_preset(item.text())
        if params:
            self.preset_loaded.emit(params)
            self.accept()

    def _delete_selected(self):
        item = self._list.currentItem()
        if not item:
            return
        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Delete preset '{item.text()}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            preset_io.delete_preset(item.text())
            self._refresh_list()

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Preset", "", "PixClip Preset (*.json)"
        )
        if path:
            result = preset_io.import_preset(Path(path))
            if result:
                self._refresh_list()
            else:
                QMessageBox.warning(self, "Import Failed", "Could not read preset file.")

    def _export_preset(self):
        item = self._list.currentItem()
        if not item:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Preset", f"{item.text()}.json", "PixClip Preset (*.json)"
        )
        if path:
            preset_io.export_preset(item.text(), Path(path))
