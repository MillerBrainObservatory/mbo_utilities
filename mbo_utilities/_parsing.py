import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _is_disabled_si_module(value) -> bool:
    """Check if a scanimage module dict has enable=false."""
    if not isinstance(value, dict):
        return False
    enable_val = value.get("enable")
    if enable_val is False:
        return True
    return bool(isinstance(enable_val, str) and enable_val.lower() in ("false", "0"))


def _filter_disabled_modules(metadata: dict, recursive: bool = True) -> dict:
    """Filter out disabled scanimage modules (hXxx with enable=false) from metadata."""
    if not isinstance(metadata, dict):
        return metadata
    result = {}
    for k, v in metadata.items():
        # skip disabled h-modules
        if k.startswith("h") and _is_disabled_si_module(v):
            continue
        # recursively filter nested dicts (like SI dict containing hModules)
        if recursive and isinstance(v, dict):
            v = _filter_disabled_modules(v, recursive=True)
        result[k] = v
    return result


def _make_json_serializable(obj, filter_disabled: bool = True):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        # filter out disabled SI modules recursively
        if filter_disabled:
            obj = _filter_disabled_modules(obj, recursive=True)
        return {k: _make_json_serializable(v, filter_disabled=False) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v, filter_disabled=False) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def _convert_paths_to_strings(obj, filter_disabled: bool = True):
    """
    Recursively convert all pathlib.Path objects to strings in a nested structure.

    This is essential before saving dicts with np.save() since pickled Path objects
    (PosixPath/WindowsPath) cannot be loaded cross-platform.

    Parameters
    ----------
    obj : any
        The object to convert. Can be a dict, list, tuple, Path, or any other type.
    filter_disabled : bool
        If True (default), filter out disabled SI modules at top level.

    Returns
    -------
    any
        The same structure with all Path objects converted to strings.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        # filter out disabled SI modules recursively
        if filter_disabled:
            obj = _filter_disabled_modules(obj, recursive=True)
        return {k: _convert_paths_to_strings(v, filter_disabled=False) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert_paths_to_strings(v, filter_disabled=False) for v in obj)
    if isinstance(obj, np.ndarray):
        # Don't recurse into numpy arrays - they shouldn't contain Paths
        return obj
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
