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
"""

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
from mbo_utilities.arrays._registration import (
    register_zplanes_s3d,
    validate_s3d_registration,
)
from mbo_utilities.arrays.bin import BinArray
from mbo_utilities.arrays.h5 import H5Array
from mbo_utilities.arrays.isoview import IsoviewArray
from mbo_utilities.arrays.numpy import NumpyArray
from mbo_utilities.arrays.nwb import NWBArray
from mbo_utilities.arrays.suite2p import (
    Suite2pArray,
    Suite2pVolumeArray,
    find_suite2p_plane_dirs,
)
from mbo_utilities.arrays.tiff import (
    MBOTiffArray,
    MboRawArray,
    TiffArray,
    TiffVolumeArray,
    find_tiff_plane_files,
)
from mbo_utilities.arrays.zarr import ZarrArray

__all__ = [
    # Array classes
    "Suite2pArray",
    "Suite2pVolumeArray",
    "H5Array",
    "TiffArray",
    "TiffVolumeArray",
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
