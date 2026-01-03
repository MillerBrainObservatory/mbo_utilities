"""
Array types for mbo_utilities.

This package provides lazy array readers for various imaging data formats:
- Suite2pArray: Suite2p binary files (.bin + ops.npy)
- H5Array: HDF5 datasets
- TiffArray: Generic TIFF files
- MBOTiffArray: Dask-backed MBO processed TIFFs
- ScanImageArray: Base class for raw ScanImage TIFFs with phase correction
- LBMArray: LBM (Light Beads Microscopy) stacks
- PiezoArray: Piezo z-stacks with optional frame averaging
- CalibrationArray: Pollen/bead calibration stacks (LBM + piezo)
- SinglePlaneArray: Single-plane time series
- open_scanimage: Factory function for auto-detecting ScanImage stack type
- NumpyArray: NumPy arrays and .npy files
- NWBArray: NWB (Neurodata Without Borders) files
- ZarrArray: Zarr v3 stores (including OME-Zarr)
- BinArray: Raw binary files without ops.npy
- IsoviewArray: Isoview lightsheet microscopy data

Legacy aliases:
- MboRawArray: Alias for ScanImageArray (backwards compatibility)

Also provides:
- Registration utilities (validate_s3d_registration, register_zplanes_s3d)
- RoiFeatureMixin: Mixin for multi-ROI support (use hasattr(arr, 'roi_mode') for detection)
- Common helpers (normalize_roi, etc.)
- Features (DimLabels, DimLabelsMixin for dimension labeling)

Array classes are lazy-loaded on first access to improve startup time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# base module is lightweight, import eagerly
from mbo_utilities.arrays._base import (
    CHUNKS_3D,
    CHUNKS_4D,
    _axes_or_guess,
    _build_output_path,
    _imwrite_base,
    _normalize_planes,
    _safe_get_metadata,
    _sanitize_suffix,
    _to_tzyx,
    iter_rois,
    normalize_roi,
    supports_roi,
)

if TYPE_CHECKING:
    from mbo_utilities.arrays._registration import (
        register_zplanes_s3d as register_zplanes_s3d,
        validate_s3d_registration as validate_s3d_registration,
    )
    from mbo_utilities.arrays.bin import BinArray as BinArray
    from mbo_utilities.arrays.h5 import H5Array as H5Array
    from mbo_utilities.arrays.isoview import IsoviewArray as IsoviewArray
    from mbo_utilities.arrays.numpy import NumpyArray as NumpyArray
    from mbo_utilities.arrays.nwb import NWBArray as NWBArray
    from mbo_utilities.arrays.suite2p import (
        Suite2pArray as Suite2pArray,
        find_suite2p_plane_dirs as find_suite2p_plane_dirs,
    )
    from mbo_utilities.arrays.tiff import (
        CalibrationArray as CalibrationArray,
        LBMArray as LBMArray,
        MBOTiffArray as MBOTiffArray,
        MboRawArray as MboRawArray,
        PiezoArray as PiezoArray,
        ScanImageArray as ScanImageArray,
        SinglePlaneArray as SinglePlaneArray,
        TiffArray as TiffArray,
        find_tiff_plane_files as find_tiff_plane_files,
        open_scanimage as open_scanimage,
    )
    from mbo_utilities.arrays.zarr import ZarrArray as ZarrArray

# lazy loading map: name -> (module, attr)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # array classes
    "Suite2pArray": (".suite2p", "Suite2pArray"),
    "find_suite2p_plane_dirs": (".suite2p", "find_suite2p_plane_dirs"),
    "H5Array": (".h5", "H5Array"),
    "TiffArray": (".tiff", "TiffArray"),
    "MBOTiffArray": (".tiff", "MBOTiffArray"),
    "ScanImageArray": (".tiff", "ScanImageArray"),
    "MboRawArray": (".tiff", "ScanImageArray"),  # backwards compat alias
    "LBMArray": (".tiff", "LBMArray"),
    "PiezoArray": (".tiff", "PiezoArray"),
    "CalibrationArray": (".tiff", "CalibrationArray"),
    "SinglePlaneArray": (".tiff", "SinglePlaneArray"),
    "open_scanimage": (".tiff", "open_scanimage"),
    "find_tiff_plane_files": (".tiff", "find_tiff_plane_files"),
    "NumpyArray": (".numpy", "NumpyArray"),
    "NWBArray": (".nwb", "NWBArray"),
    "ZarrArray": (".zarr", "ZarrArray"),
    "BinArray": (".bin", "BinArray"),
    "IsoviewArray": (".isoview", "IsoviewArray"),
    "_extract_tiff_plane_number": (".tiff", "_extract_tiff_plane_number"),
    # registration
    "validate_s3d_registration": ("._registration", "validate_s3d_registration"),
    "register_zplanes_s3d": ("._registration", "register_zplanes_s3d"),
    # features subpackage
    "features": (".features", None),
    # ROI mixin
    "RoiFeatureMixin": (".features._roi", "RoiFeatureMixin"),
}

# cache loaded modules
_loaded: dict[str, object] = {}


def __getattr__(name: str) -> object:
    if name in _loaded:
        return _loaded[name]

    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module

        module = import_module(module_name, package="mbo_utilities.arrays")
        # if attr_name is None, return the module itself (for subpackages)
        obj = module if attr_name is None else getattr(module, attr_name)
        _loaded[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)


def register_all_pipelines() -> None:
    """
    Import all array modules to trigger pipeline registration.

    Call this before using pipeline_registry if you want all
    array types registered.
    """
    from importlib import import_module

    # import all array modules to trigger their pipeline registrations
    for module_name, _ in set(_LAZY_IMPORTS.values()):
        import_module(module_name, package="mbo_utilities.arrays")


__all__ = [
    "CHUNKS_3D",
    "CHUNKS_4D",
    "BinArray",
    "CalibrationArray",
    "H5Array",
    "IsoviewArray",
    "LBMArray",
    "MBOTiffArray",
    "MboRawArray",  # backwards compat alias
    "NWBArray",
    "NumpyArray",
    "PiezoArray",
    # ROI mixin
    "RoiFeatureMixin",
    "ScanImageArray",
    "SinglePlaneArray",
    # Array classes
    "Suite2pArray",
    "TiffArray",
    "ZarrArray",
    "_axes_or_guess",
    "_build_output_path",
    "_extract_tiff_plane_number",
    "_imwrite_base",
    "_normalize_planes",
    "_safe_get_metadata",
    "_sanitize_suffix",
    "_to_tzyx",
    # Features subpackage
    "features",
    # Suite2p helpers
    "find_suite2p_plane_dirs",
    # TIFF helpers
    "find_tiff_plane_files",
    "iter_rois",
    "normalize_roi",
    "open_scanimage",
    # Pipeline registration
    "register_all_pipelines",
    "register_zplanes_s3d",
    # Helpers
    "supports_roi",
    # Registration
    "validate_s3d_registration",
]
