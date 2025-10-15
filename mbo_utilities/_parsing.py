import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def parse_scanimage_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Parse metadata from a ScanImage dictionary to a more JSON-friendly format.

    Parameters
    ----------
    metadata : dict[str, Any]
        The original metadata dictionary from ScanImage.

    Returns
    -------
    dict[str, Any]
        A JSON-serializable version of the metadata.
    """
    return _make_json_serializable(metadata)


def _make_json_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
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
    while f"{base_label} [{count + 1}]" in labels:
        count += 1
    return f"{base_label} [{count + 1}]"