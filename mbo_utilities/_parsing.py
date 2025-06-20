import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _make_json_serializable(obj):
    """Convert metadata to JSON serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _load_existing(save_path: Path) -> list[dict[str, Any]]:
    if not save_path.exists():
        return []
    try:
        return json.loads(save_path.read_text())
    except Exception:
        return []


def _increment_label(existing: list[dict[str, Any]], base_label: str) -> str:
    count = 1
    labels = {e["label"] for e in existing if "label" in e}
    if base_label not in labels:
        return base_label
    while f"{base_label} [{count+1}]" in labels:
        count += 1
    return f"{base_label} [{count+1}]"
