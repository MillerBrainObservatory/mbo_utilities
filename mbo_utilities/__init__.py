"""
mbo_utilities - Miller Brain Observatory data processing utilities.

This package uses lazy imports to minimize startup time. Heavy dependencies
like numpy, dask, and tifffile are only loaded when actually needed.

For fastest CLI startup (e.g., `mbo --download-notebook`), avoid importing
from this module directly - use `from mbo_utilities.graphics.run_gui import _cli_entry`.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mbo_utilities")
except PackageNotFoundError:
    # fallback for editable installs
    __version__ = "0.0.0"


# Define what's available for lazy loading
__all__ = [
    # Core I/O
    "imread",
    "imwrite",
    "SUPPORTED_FTYPES",
    # File utilities
    "get_mbo_dirs",
    "files_to_dask",
    "get_files",
    "expand_paths",
    "load_ops",
    "write_ops",
    "get_plane_from_filename",
    "merge_zarr_zplanes",
    # Metadata
    "is_raw_scanimage",
    "get_metadata",
    # Preferences
    "get_recent_files",
    "add_recent_file",
    "get_last_open_dir",
    "set_last_open_dir",
    "get_last_save_dir",
    "set_last_save_dir",
    # Utilities
    "norm_minmax",
    "smooth_data",
    "is_running_jupyter",
    "is_imgui_installed",
    "subsample_array",
    # Visualization
    "save_mp4",
    "save_png",
]


def __getattr__(name):
    """Lazy import attributes to avoid loading heavy dependencies at startup."""
    # Core I/O (lazy_array -> array_types -> numpy, dask, tifffile)
    if name in ("imread", "imwrite", "SUPPORTED_FTYPES"):
        from . import lazy_array
        return getattr(lazy_array, name)

    # File utilities (file_io -> dask, tifffile, zarr)
    if name in (
        "get_mbo_dirs",
        "files_to_dask",
        "get_files",
        "expand_paths",
        "load_ops",
        "write_ops",
        "get_plane_from_filename",
        "merge_zarr_zplanes",
    ):
        from . import file_io
        return getattr(file_io, name)

    # Metadata (metadata -> tifffile)
    if name in ("is_raw_scanimage", "get_metadata"):
        from . import metadata
        return getattr(metadata, name)

    # Preferences (lightweight, no heavy deps)
    if name in (
        "get_recent_files",
        "add_recent_file",
        "get_last_open_dir",
        "set_last_open_dir",
        "get_last_save_dir",
        "set_last_save_dir",
    ):
        from . import preferences
        return getattr(preferences, name)

    # Utilities (util -> potentially torch, pandas)
    if name in (
        "norm_minmax",
        "smooth_data",
        "is_running_jupyter",
        "is_imgui_installed",
        "subsample_array",
    ):
        from . import util
        return getattr(util, name)

    # Visualization (plot_util -> matplotlib, imageio)
    if name in ("save_mp4", "save_png"):
        from . import plot_util
        return getattr(plot_util, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
