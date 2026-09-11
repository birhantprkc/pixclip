"""
ui/profile_library_dialog.py
Profile Library dialog — save, load, and delete named EditProfiles backed
by auto_enhance.ProfileLibrary (JSON file on disk).

Reuses the same visual language as PresetDialog but wires to ProfileLibrary
instead of core.presets so profiles survive across app restarts.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QPushButton, QMessageBox, QFrame,
)

from core.params import AdjustmentParams
from auto_enhance import EditProfile, ProfileLibrary

# Persistent store lives at ~/.pixclip/profiles.json
_PROFILES_PATH = Path.home() / ".pixclip" / "profiles.json"


def _get_library() -> ProfileLibrary:
    """Return the singleton-ish ProfileLibrary for the app's profile store."""
    return ProfileLibrary(_PROFILES_PATH)


def profile_to_params(profile: EditProfile) -> AdjustmentParams:
    """
    Convert an EditProfile → AdjustmentParams so it can be pushed into the
    app's existing slider / preview pipeline without any separate copy.

    Range mapping (documented per field):
      clarity:            profile.clarity * 10
          (profile: 0–100  →  slider: 0–1000, displayed /10)
      sharpness:          profile.sharpness * 10
          (same as clarity)
      contrast:           profile.contrast
          (both: -100 → +100,  1:1)
      exposure:           profile.exposure
          (both: -100 → +100,  1:1)
      shadows:            profile.shadow_recovery
          (profile: 0–100 → slider: 0 → +100, lifts dark regions)
      highlights:         -profile.highlight_recovery
          (profile: 0–100 → slider: -100 → 0, pulls down blown highlights)
      brightness:         0.0 (neutral; exposure handles EV)
      lightness:          0.0 (neutral)
      filter_id:          profile.filter_style (or subtle vivid boost if vibrance > 10)
      filter_intensity:   profile.filter_intensity
    """
    p = AdjustmentParams()
    p.clarity    = round(profile.clarity * 10.0, 1)          # 0–100 → 0–1000
    p.sharpness  = round(profile.sharpness * 10.0, 1)        # 0–100 → 0–1000
    p.contrast   = round(profile.contrast, 1)                # -100→+100, 1:1
    p.exposure   = round(profile.exposure, 1)                # -100→+100, 1:1
    p.shadows    = round(max(0.0, min(100.0, profile.shadow_recovery)), 1)
    p.highlights = round(-max(0.0, min(100.0, profile.highlight_recovery)), 1)
    p.brightness = 0.0
    p.lightness  = 0.0

    p.filter_id        = profile.filter_style
    p.filter_intensity = profile.filter_intensity

    # If no filter style was chosen but vibrance is high, use subtle vivid preset
    if p.filter_id == "none" and profile.vibrance > 10.0:
        p.filter_id = "vivid-1"
        p.filter_intensity = round(min(0.45, profile.vibrance / 60.0), 2)

    return p


def params_to_profile(params: AdjustmentParams) -> EditProfile:
    """
    Reverse mapping: AdjustmentParams → EditProfile so the current slider
    state can be saved into the ProfileLibrary.

    Inverse of profile_to_params (same fields, inverted formulae).
    """
    return EditProfile(
        clarity            = round(params.clarity / 10.0, 1),
        sharpness          = round(params.sharpness / 10.0, 1),
        contrast           = round(params.contrast, 1),
        vibrance           = 25.0 if params.filter_id.startswith("vivid") else 0.0,
        exposure           = round(params.exposure, 1),
        shadow_recovery    = round(max(0.0, params.shadows), 1),
        highlight_recovery = round(max(0.0, -params.highlights), 1),
        denoise            = 0.0,     # no slider source
        warmth             = 0.0,     # no slider source
        tint               = 0.0,     # no slider source
        filter_style       = params.filter_id,
        filter_intensity   = params.filter_intensity,
    )


class ProfileLibraryDialog(QDialog):
    """
    Modal dialog for managing the persistent ProfileLibrary.

    Signals
    -------
    profile_loaded(AdjustmentParams)
        Emitted when the user clicks Apply.  The caller should push these
        params through the app's existing slider state exactly as it would
        for any manual edit.
    """
    profile_loaded = Signal(AdjustmentParams)

    def __init__(self, current_params: AdjustmentParams, parent=None):
        super().__init__(parent)
        self._current_params = current_params
        self._library = _get_library()
        self.setWindowTitle("Profile Library")
        self.setModal(True)
        self.setMinimumSize(400, 460)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ── Save current settings ─────────────────────────────────────────
        save_frame = QFrame()
        save_frame.setObjectName("PresetSaveFrame")
        sf_layout = QVBoxLayout(save_frame)
        sf_layout.setContentsMargins(12, 10, 12, 10)
        sf_layout.setSpacing(6)

        sf_label = QLabel("Save Current Settings as Profile")
        sf_label.setObjectName("InfoSub")

        name_row = QHBoxLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Profile name…")
        self._name_edit.returnPressed.connect(self._save_profile)

        save_btn = QPushButton("Save")
        save_btn.setObjectName("PrimaryButton")
        save_btn.setFixedWidth(60)
        save_btn.clicked.connect(self._save_profile)

        name_row.addWidget(self._name_edit)
        name_row.addWidget(save_btn)

        sf_layout.addWidget(sf_label)
        sf_layout.addLayout(name_row)
        layout.addWidget(save_frame)

        # ── Profile list ──────────────────────────────────────────────────
        list_label = QLabel("SAVED PROFILES")
        list_label.setObjectName("SectionHeader")
        layout.addWidget(list_label)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemDoubleClicked.connect(self._apply_selected)
        layout.addWidget(self._list, stretch=1)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("PrimaryButton")
        apply_btn.clicked.connect(self._apply_selected)

        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("DangerButton")
        delete_btn.clicked.connect(self._delete_selected)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)

        btn_row.addWidget(apply_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _refresh_list(self):
        self._list.clear()
        for name in self._library.list_names():
            self._list.addItem(QListWidgetItem(name))

    def _save_profile(self):
        name = self._name_edit.text().strip()
        if not name:
            return
        profile = params_to_profile(self._current_params)
        self._library.add(name, profile)
        self._name_edit.clear()
        self._refresh_list()

    def _apply_selected(self):
        item = self._list.currentItem()
        if not item:
            return
        profile = self._library.get(item.text())
        if profile is None:
            QMessageBox.warning(self, "Profile Error", "Could not load profile.")
            return
        params = profile_to_params(profile)
        self.profile_loaded.emit(params)
        self.accept()

    def _delete_selected(self):
        item = self._list.currentItem()
        if not item:
            return
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Delete profile '{item.text()}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._library.delete(item.text())
            self._refresh_list()
