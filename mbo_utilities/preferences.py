"""
User preferences management for MBO Utilities.

This module provides centralized storage and retrieval of user preferences,
including recent files, default directories, and GUI settings.

All preferences are stored in ~/mbo/settings/ as JSON files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mbo_utilities import log
import contextlib

logger = log.get("preferences")

# Maximum number of recent files/folders to track
MAX_RECENT_FILES = 20


def _get_settings_dir() -> Path:
    """Get the settings directory, creating it if needed."""
    settings_dir = Path.home() / ".mbo" / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir


def _get_preferences_path() -> Path:
    """Get path to the main preferences file."""
    return _get_settings_dir() / "preferences.json"


def _load_preferences() -> dict:
    """Load preferences from disk."""
    path = _get_preferences_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load preferences: {e}")
    return {}


def _save_preferences(prefs: dict) -> None:
    """Save preferences to disk."""
    path = _get_preferences_path()
    try:
        path.write_text(json.dumps(prefs, indent=2, default=str))
    except OSError as e:
        logger.warning(f"Failed to save preferences: {e}")


def get_recent_files() -> list[dict]:
    """
    Get list of recently opened files/folders.

    Returns
    -------
    list[dict]
        List of recent file entries, each containing:
        - path: str - absolute path to file/folder
        - timestamp: str - ISO format datetime when last opened
        - type: str - 'file' or 'folder'

    Examples
    --------
    >>> recent = get_recent_files()
    >>> for entry in recent[:5]:
    ...     print(entry['path'])
    """
    prefs = _load_preferences()
    return prefs.get("recent_files", [])


def add_recent_file(path: str | Path, file_type: str = "auto") -> None:
    """
    Add a file or folder to the recent files list.

    Parameters
    ----------
    path : str or Path
        Path to the file or folder.
    file_type : str, optional
        Type of entry: 'file', 'folder', or 'auto' (detect automatically).
        Default is 'auto'.

    Examples
    --------
    >>> add_recent_file("/path/to/data.tiff")
    >>> add_recent_file("/path/to/folder", file_type="folder")
    """
    path = Path(path).resolve()
    if not path.exists():
        logger.debug(f"Path does not exist, not adding to recent: {path}")
        return

    if file_type == "auto":
        file_type = "folder" if path.is_dir() else "file"

    prefs = _load_preferences()
    recent = prefs.get("recent_files", [])

    # Remove any existing entry for this path
    recent = [r for r in recent if r.get("path") != str(path)]

    # Add new entry at the beginning
    entry = {
        "path": str(path),
        "timestamp": datetime.now().isoformat(),
        "type": file_type,
    }
    recent.insert(0, entry)

    # Trim to max size
    recent = recent[:MAX_RECENT_FILES]

    prefs["recent_files"] = recent
    _save_preferences(prefs)



# keys for different dialog contexts - each dialog type has its own cached path
_DIR_KEYS = {
    # General file/folder opening
    "open_file": "last_open_file_dir",      # File > Open File
    "open_folder": "last_open_folder_dir",  # File > Open Folder
    # Save dialogs
    "save_as": "last_save_as_dir",          # Save As dialog
    # Suite2p pipeline
    "suite2p_output": "last_suite2p_output_dir",  # Run tab > Browse for output
    "suite2p_chan2": "last_suite2p_chan2_dir",    # Registration > Channel 2 file
    "suite2p_stat": "last_suite2p_stat_dir",      # Load stat.npy for diagnostics
    "suite2p_ops": "last_suite2p_ops_dir",        # Load ops.npy for results viewer
    "suite2p_diagnostics": "last_suite2p_diagnostics_dir",  # Load plane folder for diagnostics
    # Grid search
    "grid_search": "last_grid_search_dir",  # Grid search results folder
}


def get_last_dir(context: str) -> Path | None:
    """
    Get the last directory used for a specific dialog context.

    Parameters
    ----------
    context : str
        One of: 'open_file', 'open_folder', 'save_as', 'suite2p_output',
        'suite2p_chan2', 'suite2p_stat', 'suite2p_ops', 'suite2p_diagnostics'

    Returns
    -------
    Path or None
        The last directory for this context if it exists, otherwise None.
    """
    key = _DIR_KEYS.get(context)
    if not key:
        logger.warning(f"Unknown directory context: {context}")
        return None

    prefs = _load_preferences()
    path_str = prefs.get(key)
    if path_str:
        path = Path(path_str)
        if path.exists():
            return path
    return None


def set_last_dir(context: str, path: str | Path) -> None:
    """
    Set the last directory used for a specific dialog context.

    Parameters
    ----------
    context : str
        One of: 'open_file', 'open_folder', 'save_as', 'suite2p_output',
        'suite2p_chan2', 'suite2p_stat', 'suite2p_ops', 'suite2p_diagnostics'
    path : str or Path
        Path to directory (if file is given, parent directory is used).
    """
    key = _DIR_KEYS.get(context)
    if not key:
        logger.warning(f"Unknown directory context: {context}")
        return

    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    if path.is_dir():
        prefs = _load_preferences()
        prefs[key] = str(path)
        _save_preferences(prefs)


def get_last_open_dir() -> Path | None:
    """
    Get the last directory used to open files.

    Returns
    -------
    Path or None
        The last open directory if it exists, otherwise None.

    Notes
    -----
    This is a legacy function. Prefer using get_last_dir('open_file') or
    get_last_dir('open_folder') for context-specific directories.
    """
    # Try the new context-specific keys first
    for context in ("open_file", "open_folder"):
        result = get_last_dir(context)
        if result:
            return result

    # Fall back to legacy key
    prefs = _load_preferences()
    path_str = prefs.get("last_open_dir")
    if path_str:
        path = Path(path_str)
        if path.exists():
            return path
    return None


def set_last_open_dir(path: str | Path) -> None:
    """
    Set the last directory used to open files.

    Parameters
    ----------
    path : str or Path
        Path to directory (if file is given, parent directory is used).

    Notes
    -----
    This is a legacy function that updates both open_file and legacy key.
    Prefer using set_last_dir('open_file', path) for new code.
    """
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    if path.is_dir():
        prefs = _load_preferences()
        prefs["last_open_dir"] = str(path)
        # Also update new context-specific key
        prefs[_DIR_KEYS["open_file"]] = str(path)
        _save_preferences(prefs)


def get_last_save_dir() -> Path | None:
    """
    Get the last directory used to save files.

    Returns
    -------
    Path or None
        The last save directory if it exists, otherwise None.

    Notes
    -----
    This is a legacy function. Prefer using get_last_dir('save_as') for
    the Save As dialog specifically.
    """
    # Try the new context-specific key first
    result = get_last_dir("save_as")
    if result:
        return result

    # Fall back to legacy key
    prefs = _load_preferences()
    path_str = prefs.get("last_save_dir")
    if path_str:
        path = Path(path_str)
        if path.exists():
            return path
    # Fall back to legacy location
    legacy_path = _get_settings_dir() / "last_savedir.json"
    if legacy_path.exists():
        try:
            data = json.loads(legacy_path.read_text())
            path = Path(data.get("last_savedir", ""))
            if path.exists():
                return path
        except (json.JSONDecodeError, OSError):
            pass
    return None


def set_last_save_dir(path: str | Path) -> None:
    """
    Set the last directory used to save files.

    Parameters
    ----------
    path : str or Path
        Path to directory (if file is given, parent directory is used).

    Notes
    -----
    This is a legacy function. Prefer using set_last_dir('save_as', path)
    for new code.
    """
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    if path.is_dir():
        prefs = _load_preferences()
        prefs["last_save_dir"] = str(path)
        # Also update new context-specific key
        prefs[_DIR_KEYS["save_as"]] = str(path)
        _save_preferences(prefs)
        # Also update legacy location for backwards compatibility
        legacy_path = _get_settings_dir() / "last_savedir.json"
        with contextlib.suppress(OSError):
            legacy_path.write_text(json.dumps({"last_savedir": str(path)}))


def get_default_open_dir() -> Path:
    """
    Get the default directory to use when opening files.

    Checks in order:
    1. Last open directory (if exists)
    2. Most recent file's parent directory (if exists)
    3. Last save directory (if exists)
    4. User's home directory

    Returns
    -------
    Path
        A valid directory path.
    """
    # Try last open dir
    last_open = get_last_open_dir()
    if last_open:
        return last_open

    # Try most recent file's directory
    recent = get_recent_files()
    for entry in recent:
        path = Path(entry.get("path", ""))
        if path.exists():
            parent = path.parent if path.is_file() else path
            if parent.is_dir():
                return parent

    # Try last save dir
    last_save = get_last_save_dir()
    if last_save:
        return last_save

    # Fall back to home
    return Path.home()
