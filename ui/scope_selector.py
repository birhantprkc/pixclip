"""
ui/scope_selector.py
Scope control: determines whether slider changes apply to All / Group / Selected image.
"""

from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QButtonGroup,
    QComboBox, QInputDialog,
)


class ScopeSelector(QWidget):
    """
    Sleek, Segmented Three-mode scope selector: This Image | Group | All
    Emits scope_changed(scope_str, group_id_or_none) and group_created(group_name).
    """
    scope_changed = Signal(str, object)  # ("all"|"group"|"one", group_id|None)
    group_created = Signal(str)          # (group_name)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ScopeBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(36)
        self._current_scope = "one"  # Start with single image scope as default/user-friendly baseline
        self._current_group: str | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        scope_lbl = QLabel("APPLY FILTER TO")
        scope_lbl.setObjectName("SectionHeader")

        # Segmented Control Container
        self._segmented_container = QWidget()
        self._segmented_container.setObjectName("SegmentedContainer")
        
        seg_layout = QHBoxLayout(self._segmented_container)
        seg_layout.setContentsMargins(2, 2, 2, 2)
        seg_layout.setSpacing(2)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        self._btn_one = QPushButton("This Image")
        self._btn_one.setCheckable(True)
        self._btn_one.setChecked(True)

        self._btn_group_scope = QPushButton("Group")
        self._btn_group_scope.setCheckable(True)

        self._btn_all = QPushButton("All")
        self._btn_all.setCheckable(True)

        # Map to original button indices: All=0, Group=1, This Image=2
        self._btn_group.addButton(self._btn_all, 0)
        self._btn_group.addButton(self._btn_group_scope, 1)
        self._btn_group.addButton(self._btn_one, 2)

        seg_layout.addWidget(self._btn_one)
        seg_layout.addWidget(self._btn_group_scope)
        seg_layout.addWidget(self._btn_all)

        self._group_combo = QComboBox()
        self._group_combo.setObjectName("GroupCombo")
        self._group_combo.setFixedWidth(150) # give it more width since it now has placeholders and actions
        self._group_combo.setEnabled(False)  # start disabled as "This Image" is selected by default

        self._btn_group.idClicked.connect(self._on_scope_changed)
        self._group_combo.currentTextChanged.connect(self._on_group_changed)

        layout.addWidget(scope_lbl)
        layout.addWidget(self._segmented_container)
        layout.addWidget(self._group_combo)
        layout.addStretch()
        
        # Populate initial list
        self.set_groups([])

    def _on_scope_changed(self, btn_id: int) -> None:
        scopes = ["all", "group", "one"]
        self._current_scope = scopes[btn_id]
        self._group_combo.setEnabled(btn_id == 1)
        self.scope_changed.emit(self._current_scope, self._current_group)

    def _on_group_changed(self, text: str) -> None:
        if text == "+ Create New Group...":
            from PySide6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(
                self.window(), "Create Group", "Enter new group name:"
            )
            name = name.strip()
            if ok and name:
                self._group_combo.blockSignals(True)
                # Remove placeholder item if it exists
                if self._group_combo.itemText(0).startswith("No Groups"):
                    self._group_combo.removeItem(0)
                
                # Insert the new group name before the "+ Create New Group..." item
                count = self._group_combo.count()
                self._group_combo.insertItem(count - 1, name)
                self._group_combo.setCurrentText(name)
                self._group_combo.blockSignals(False)
                
                self._current_group = name
                self.group_created.emit(name)
                self._btn_group_scope.setChecked(True) # select "Group" mode
                self._current_scope = "group"
                self._group_combo.setEnabled(True)
                self.scope_changed.emit(self._current_scope, self._current_group)
            else:
                # Restore previous group selection
                self._group_combo.blockSignals(True)
                if self._current_group:
                    self._group_combo.setCurrentText(self._current_group)
                else:
                    self._group_combo.setCurrentIndex(0 if self._group_combo.count() > 0 else -1)
                self._group_combo.blockSignals(False)
            return

        if text.startswith("No Groups"):
            self._current_group = None
        else:
            self._current_group = text if text else None
        self.scope_changed.emit(self._current_scope, self._current_group)

    def set_groups(self, groups: list[str]) -> None:
        """Refresh the group combo box."""
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        
        if not groups:
            self._group_combo.addItem("No Groups (Create one below)")
        else:
            for g in groups:
                if g: # Avoid empty group strings
                    self._group_combo.addItem(g)
                
        self._group_combo.addItem("+ Create New Group...")
        
        # Restore selection
        if self._current_group and self._current_group in groups:
            self._group_combo.setCurrentText(self._current_group)
        else:
            if groups:
                self._current_group = groups[0]
                self._group_combo.setCurrentText(groups[0])
            else:
                self._current_group = None
                self._group_combo.setCurrentIndex(0)
                
        self._group_combo.blockSignals(False)

    @property
    def current_scope(self) -> str:
        return self._current_scope

    @property
    def current_group(self) -> str | None:
        return self._current_group
