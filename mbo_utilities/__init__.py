"""
mbo_utilities - Miller Brain Observatory data processing utilities.

This package uses lazy imports to minimize startup time. Heavy dependencies
like numpy, dask, and tifffile are only loaded when actually needed.
"""

import warnings

# Suppress annoying CuPy warning about CUDA path (usually harmless if CUDA works)
warnings.filterwarnings("ignore", category=UserWarning, message="CUDA path could not be detected")


# Define what's available for lazy loading
__all__ = [
    "AxialShiftView",
    "MBO_SUPPORTED_FTYPES",
    "VoxelSize",
    "add_recent_file",
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
    # Pluggable array API
    "LazyArray",
    "register_array_class",
    # Metadata
    "is_raw_scanimage",
    "load_npy",
    "load_ops",
    "merge_zarr_zplanes",
    "normalize_resolution",
    "select_files",
    # File/folder selection (GUI)
    "select_folder",
    "set_last_open_dir",
    "set_last_save_dir",
    # Visualization
    "to_video",
    "with_axial_shifts",
    "with_phasecorr",
    "write_ops",
]


def __getattr__(name):
    """Lazy import attributes to avoid loading heavy dependencies at startup."""
    # Version (importlib.metadata pulls email + zipfile; defer to keep CLI startup fast)
    if name == "__version__":
        try:
            from importlib.metadata import version
            return version("mbo_utilities")
        except Exception:
            return "0.0.0"  # editable install / metadata unavailable

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

    # Pluggable array API (dependency-light; safe to import early)
    if name in ("LazyArray", "register_array_class"):
        from .lazy_array import LazyArray, register_array_class
        return LazyArray if name == "LazyArray" else register_array_class

    if name == "get_mbo_dirs":
        from .preferences import get_mbo_dirs
        return get_mbo_dirs

    if name == "files_to_dask":
        from .arrays import files_to_dask
        return files_to_dask

    # Axial plane-shift apply/remove (read-time, non-destructive)
    if name in ("with_axial_shifts", "AxialShiftView"):
        from .arrays import with_axial_shifts, AxialShiftView
        return with_axial_shifts if name == "with_axial_shifts" else AxialShiftView

    # Read-time bidirectional phase correction (non-destructive)
    if name == "with_phasecorr":
        from .arrays import with_phasecorr
        return with_phasecorr

    # File utilities (file_io -> tifffile, zarr)
    if name in (
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

    if name == "load_npy":
        from .file_io import load_npy
        return load_npy

    # Video export (_writers -> imageio)
    if name == "to_video":
        from ._writers import to_video
        return to_video

    # File/folder selection (widgets -> imgui, wgpu)
    if name in ("select_folder", "select_files"):
        from .gui.widgets.simple_selector import select_folder, select_files
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
