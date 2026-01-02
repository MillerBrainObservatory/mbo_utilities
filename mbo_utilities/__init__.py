"""
mbo_utilities - Miller Brain Observatory data processing utilities.

This package uses lazy imports to minimize startup time. Heavy dependencies
like numpy, dask, and tifffile are only loaded when actually needed.

For fastest CLI startup (e.g., `mbo --download-notebook`), avoid importing
from this module directly - use `from mbo_utilities.gui.run_gui import _cli_entry`.
"""

from importlib.metadata import version, PackageNotFoundError
import warnings

# Suppress annoying CuPy warning about CUDA path (usually harmless if CUDA works)
warnings.filterwarnings("ignore", category=UserWarning, message="CUDA path could not be detected")

try:
    __version__ = version("mbo_utilities")
except PackageNotFoundError:
    # fallback for editable installs
    __version__ = "0.0.0"


# Define what's available for lazy loading
__all__ = [
    "MBO_SUPPORTED_FTYPES",
    "VoxelSize",
    "add_recent_file",
    # CLI utilities
    "download_file",
    "download_notebook",
    "expand_paths",
    "files_to_dask",
    "get_all_input_patterns",
    "get_all_marker_files",
    "get_all_output_patterns",
    # Pipeline registry
    "get_all_pipelines",
    "get_files",
    "get_last_open_dir",
    "get_last_save_dir",
    # File utilities
    "get_mbo_dirs",
    "get_metadata",
    "get_pipeline_info",
    # Preferences
    "get_recent_files",
    "get_voxel_size",
    # Core I/O
    "imread",
    "imwrite",
    "is_imgui_installed",
    # Metadata
    "is_raw_scanimage",
    "is_running_jupyter",
    "load_ops",
    "merge_zarr_zplanes",
    # Utilities
    "norm_minmax",
    "normalize_resolution",
    "select_files",
    # File/folder selection (GUI)
    "select_folder",
    "set_last_open_dir",
    "set_last_save_dir",
    "smooth_data",
    "subsample_array",
    # Visualization
    "to_video",
    "write_ops",
]


def __getattr__(name):
    """Lazy import attributes to avoid loading heavy dependencies at startup."""
    # Core I/O
    if name == "imread":
        from .reader import imread
        return imread
    if name == "imwrite":
        from .writer import imwrite
        return imwrite
    if name == "MBO_SUPPORTED_FTYPES":
        from .reader import MBO_SUPPORTED_FTYPES
        return MBO_SUPPORTED_FTYPES

    # File utilities (file_io -> dask, tifffile, zarr)
    if name in (
        "get_mbo_dirs",
        "files_to_dask",
        "get_files",
        "expand_paths",
        "merge_zarr_zplanes",
    ):
        from . import file_io
        return getattr(file_io, name)

    # Suite2p ops utilities
    if name == "load_ops":
        from .arrays.suite2p import load_ops
        return load_ops
    if name == "write_ops":
        from ._writers import write_ops
        return write_ops

    # Metadata (metadata -> tifffile)
    if name in ("is_raw_scanimage", "get_metadata", "get_voxel_size", "normalize_resolution", "VoxelSize"):
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

    # Video export (_writers -> imageio)
    if name == "to_video":
        from ._writers import to_video
        return to_video

    # CLI utilities (cli -> click, urllib only)
    if name in ("download_file", "download_notebook"):
        from . import cli
        return getattr(cli, name)

    # File/folder selection (widgets -> imgui, wgpu)
    if name in ("select_folder", "select_files"):
        from .gui.simple_selector import select_folder, select_files
        return select_folder if name == "select_folder" else select_files

    # Pipeline registry (triggers array module imports to register pipelines)
    if name in (
        "get_all_pipelines",
        "get_pipeline_info",
        "get_all_input_patterns",
        "get_all_output_patterns",
        "get_all_marker_files",
    ):
        # first register all pipelines
        from .arrays import register_all_pipelines
        register_all_pipelines()
        # then return the requested function
        from . import pipeline_registry
        return getattr(pipeline_registry, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
