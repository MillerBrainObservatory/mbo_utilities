"""
Reusable GUI widgets for mbo_utilities.

This module requires imgui extras. Install with:
    pip install mbo_utilities[imgui]

Examples
--------
>>> from mbo_utilities.widgets import select_file, select_folder, select_files
>>> from pathlib import Path
>>>
>>> # Select a single file
>>> path = select_file(
...     title="Select ops.npy",
...     filters=["Numpy Files", "*.npy"],
...     start_path=Path.home()
... )
>>>
>>> # Select a folder
>>> folder = select_folder(
...     title="Select Suite2p output",
...     start_path=Path.home()
... )
>>>
>>> # Select multiple files
>>> paths = select_files(
...     title="Select TIFF files",
...     filters=["TIFF Files", "*.tif *.tiff"]
... )
"""

from mbo_utilities.graphics.simple_selector import (
    SimpleSelector,
    select_file,
    select_folder,
    select_files,
)

__all__ = [
    "SimpleSelector",
    "select_file",
    "select_folder",
    "select_files",
]
