"""
Reusable GUI widgets for mbo_utilities.

This module requires imgui extras. Install with:
    pip install mbo_utilities[imgui]

Examples
--------
>>> from mbo_utilities.widgets import select_folder, select_files
>>> from pathlib import Path
>>>
>>> # Select a folder (note: native folder dialogs don't show files)
>>> folder = select_folder(
...     title="Select Suite2p output",
...     start_path=Path.home()
... )
>>>
>>> # Select one or more files
>>> paths = select_files(
...     title="Select TIFF files",
...     filters=["TIFF Files", "*.tif *.tiff"]
... )
"""

from mbo_utilities.graphics.simple_selector import (
    SimpleSelector,
    select_folder,
    select_files,
)

__all__ = [
    "SimpleSelector",
    "select_folder",
    "select_files",
]
