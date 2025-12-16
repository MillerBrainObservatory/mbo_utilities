"""
Array types for mbo_utilities.

This package provides lazy array readers for various imaging data formats:
- Suite2pArray: Suite2p binary files (.bin + ops.npy)
- H5Array: HDF5 datasets
- TiffArray: Generic TIFF files
- MBOTiffArray: Dask-backed MBO processed TIFFs
- MboRawArray: Raw ScanImage TIFFs with phase correction
- NumpyArray: NumPy arrays and .npy files
- NWBArray: NWB (Neurodata Without Borders) files
- ZarrArray: Zarr v3 stores (including OME-Zarr)
- BinArray: Raw binary files without ops.npy
- IsoviewArray: Isoview lightsheet microscopy data

Also provides:
- Registration utilities (validate_s3d_registration, register_zplanes_s3d)
- Common helpers (supports_roi, normalize_roi, iter_rois, etc.)

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
        MBOTiffArray as MBOTiffArray,
        MboRawArray as MboRawArray,
        TiffArray as TiffArray,
        find_tiff_plane_files as find_tiff_plane_files,
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
    "MboRawArray": (".tiff", "MboRawArray"),
    "find_tiff_plane_files": (".tiff", "find_tiff_plane_files"),
    "NumpyArray": (".numpy", "NumpyArray"),
    "NWBArray": (".nwb", "NWBArray"),
    "ZarrArray": (".zarr", "ZarrArray"),
    "BinArray": (".bin", "BinArray"),
    "IsoviewArray": (".isoview", "IsoviewArray"),
    # registration
    "validate_s3d_registration": ("._registration", "validate_s3d_registration"),
    "register_zplanes_s3d": ("._registration", "register_zplanes_s3d"),
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
        obj = getattr(module, attr_name)
        _loaded[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)


__all__ = [
    # Array classes
    "Suite2pArray",
    "H5Array",
    "TiffArray",
    "MBOTiffArray",
    "MboRawArray",
    "NumpyArray",
    "NWBArray",
    "ZarrArray",
    "BinArray",
    "IsoviewArray",
    # Suite2p helpers
    "find_suite2p_plane_dirs",
    # TIFF helpers
    "find_tiff_plane_files",
    # Registration
    "validate_s3d_registration",
    "register_zplanes_s3d",
    # Helpers
    "supports_roi",
    "normalize_roi",
    "iter_rois",
    "_normalize_planes",
    "_build_output_path",
    "_imwrite_base",
    "_to_tzyx",
    "_axes_or_guess",
    "_safe_get_metadata",
    "_sanitize_suffix",
    "CHUNKS_3D",
    "CHUNKS_4D",
]
