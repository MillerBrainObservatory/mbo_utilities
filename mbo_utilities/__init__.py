from pathlib import Path

from .file_io import (
    get_files,
    zstack_from_files,
    npy_to_dask,
    read_scan,
    save_png,
    save_mp4,
    zarr_to_dask,
    expand_paths,
)
from .assembly import save_as
from .metadata import is_raw_scanimage, get_metadata, params_from_metadata
from .image import fix_scan_phase, return_scan_offset
from .util import norm_minmax, smooth_data, is_running_jupyter

from .graphics import run_gui

__version__ = (Path(__file__).parent / "VERSION").read_text().strip()

__all__ = [
    "run_gui",
    # image processing
    "fix_scan_phase",
    "return_scan_offset",
    # file_io
    "scanreader",
    "npy_to_dask",
    "zarr_to_dask",
    "get_files",
    "zstack_from_files",
    "read_scan",
    "save_png",
    "save_mp4",
    "expand_paths",
    # metadata
    "is_raw_scanimage",
    "get_metadata",
    "params_from_metadata",
    # util
    "norm_minmax",
    "smooth_data",
    "is_running_jupyter",
    # assembly
    "save_as",
]
