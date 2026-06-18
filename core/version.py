"""
core/version.py
Version manager: reads version metadata from the root version.json.
"""

import json
from pathlib import Path

def get_version_info() -> dict:
    root_dir = Path(__file__).parent.parent.resolve()
    version_json = root_dir / "version.json"
    if version_json.exists():
        try:
            return json.loads(version_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "version": "1.0.0",
        "build": 100,
        "release_date": "2026-06-18",
        "description": "Creative Image & Video Editor"
    }

VERSION_INFO = get_version_info()
__version__ = VERSION_INFO.get("version", "1.0.0")
