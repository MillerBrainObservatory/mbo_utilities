"""
Backwards compatibility shim for array_types module.

This module re-exports all array types from their new locations in the
arrays package for backwards compatibility with existing code.

New code should import directly from mbo_utilities.arrays:
    from mbo_utilities.arrays import Suite2pArray, ZarrArray, ...
"""

# Re-export everything from the arrays package
from mbo_utilities.arrays import (
    # Array classes
    BinArray,
    H5Array,
    IsoviewArray,
    MBOTiffArray,
    MboRawArray,
    NumpyArray,
    NWBArray,
    Suite2pArray,
    TiffArray,
    ZarrArray,
    # Registration functions
    register_zplanes_s3d,
    validate_s3d_registration,
    # Helper functions
    CHUNKS_3D,
    CHUNKS_4D,
    _axes_or_guess,
    _build_output_path,
    _imwrite_base,
    _normalize_planes,
    _safe_get_metadata,
    _to_tzyx,
    iter_rois,
    normalize_roi,
    supports_roi,
)

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
    "CHUNKS_3D",
    "CHUNKS_4D",
]
