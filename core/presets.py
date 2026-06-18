"""
core/presets.py
Preset save/load system for PixClip.
Presets are stored as JSON files in the user's presets directory.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, List

from core.params import AdjustmentParams

PRESETS_DIR = Path.home() / ".pixclip" / "presets"


def _ensure_dir() -> Path:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return PRESETS_DIR


def save_preset(name: str, params: AdjustmentParams) -> Path:
    """Save a preset to disk as JSON. Returns the saved file path."""
    dir_ = _ensure_dir()
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    path = dir_ / f"{safe_name}.json"
    data = {"name": name, "params": params.to_dict()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_preset(name: str) -> Optional[AdjustmentParams]:
    """Load a preset by name from disk."""
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    path = PRESETS_DIR / f"{safe_name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AdjustmentParams.from_dict(data["params"])
    except Exception:
        return None


def load_preset_from_path(path: Path) -> Optional[tuple[str, AdjustmentParams]]:
    """Load a preset from an explicit file path. Returns (name, params) or None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("name", path.stem)
        params = AdjustmentParams.from_dict(data["params"])
        return name, params
    except Exception:
        return None


def delete_preset(name: str) -> bool:
    """Delete a preset file. Returns True if successful."""
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    path = PRESETS_DIR / f"{safe_name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_presets() -> List[tuple[str, AdjustmentParams]]:
    """Return all saved presets as (name, params) tuples, sorted by name."""
    dir_ = _ensure_dir()
    results = []
    for path in sorted(dir_.glob("*.json")):
        result = load_preset_from_path(path)
        if result:
            results.append(result)
    return results


def export_preset(name: str, export_path: Path) -> bool:
    """Copy a preset to an arbitrary export path for sharing."""
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    src = PRESETS_DIR / f"{safe_name}.json"
    if not src.exists():
        return False
    export_path.write_bytes(src.read_bytes())
    return True


def import_preset(import_path: Path) -> Optional[tuple[str, AdjustmentParams]]:
    """Import a preset from an external file. Returns (name, params) if valid."""
    result = load_preset_from_path(import_path)
    if result:
        name, params = result
        save_preset(name, params)
    return result
