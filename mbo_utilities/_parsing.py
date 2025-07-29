import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mbo_utilities.file_io import MBO_PIPELINE_TAGS

def _parse_si_to_dict_nested(metadata: dict[str, Any]) -> dict[str, Any]:
    pass


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
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
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


def _normalize_file_url(path):
    """
    Derive a folder tag from a filename based on “planeN”, “roiN”, or "tagN" patterns.

    Parameters
    ----------
    path : str or pathlib.Path
        File path or name whose stem will be parsed.

    Returns
    -------
    str
        If the stem starts with “plane”, “roi”, or “res” followed by an integer,
        returns that tag plus the integer (e.g. “plane3”, “roi7”, “res2”).
        Otherwise returns the original stem unchanged.

    Examples
    --------
    >>> _normalize_file_url("plane_01.tif")
    'plane1'
    >>> _normalize_file_url("plane2.bin")
    'plane2'
    >>> _normalize_file_url("roi5.raw")
    'roi5'
    >>> _normalize_file_url("ROI_10.dat")
    'roi10'
    >>> _normalize_file_url("res-3.h5")
    'res3'
    >>> _normalize_file_url("assembled_data_1.tiff")
    'assembled_data_1'
    >>> _normalize_file_url("file_12.tif")
    'file_12'
    """
    name = Path(path).stem
    for tag in MBO_PIPELINE_TAGS:
        low = name.lower()
        if low.startswith(tag):
            suffix = name[len(tag) :]
            if suffix and (suffix[0] in ("_", "-")):
                suffix = suffix[1:]
            if suffix.isdigit():
                return f"{tag}{int(suffix)}"
    return name
