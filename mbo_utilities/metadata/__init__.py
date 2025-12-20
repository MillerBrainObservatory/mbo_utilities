"""
mbo_utilities.metadata - metadata handling for calcium imaging data.

this package provides:
- standardized parameter definitions and aliases
- scanimage-specific metadata parsing
- stack type detection (lbm, piezo, single plane)
- voxel size extraction and normalization
- file I/O for extracting metadata from TIFF files
"""
from .base import (
    MetadataParameter,
    VoxelSize,
    METADATA_PARAMS,
    ALIAS_MAP,
    get_canonical_name,
)

from .params import (
    get_param,
    get_voxel_size,
    normalize_resolution,
    normalize_metadata,
)

from .scanimage import (
    StackType,
    detect_stack_type,
    is_lbm_stack,
    is_piezo_stack,
    get_lbm_ai_sources,
    get_num_color_channels,
    get_num_zplanes,
    get_frames_per_slice,
    get_log_average_factor,
    get_z_step_size,
    compute_num_timepoints,
    get_stack_info,
)

# file I/O functions
from .io import (
    has_mbo_metadata,
    is_raw_scanimage,
    get_metadata,
    get_metadata_single,
    get_metadata_batch,
    query_tiff_pages,
    clean_scanimage_metadata,
    default_ops,
    _build_ome_metadata,
)

__all__ = [
    # base types
    "MetadataParameter",
    "VoxelSize",
    "METADATA_PARAMS",
    "ALIAS_MAP",
    "get_canonical_name",
    # parameter access
    "get_param",
    "get_voxel_size",
    "normalize_resolution",
    "normalize_metadata",
    # scanimage detection
    "StackType",
    "detect_stack_type",
    "is_lbm_stack",
    "is_piezo_stack",
    "get_lbm_ai_sources",
    "get_num_color_channels",
    "get_num_zplanes",
    "get_frames_per_slice",
    "get_log_average_factor",
    "get_z_step_size",
    "compute_num_timepoints",
    "get_stack_info",
    # file I/O
    "has_mbo_metadata",
    "is_raw_scanimage",
    "get_metadata",
    "get_metadata_single",
    "get_metadata_batch",
    "query_tiff_pages",
    "clean_scanimage_metadata",
    "default_ops",
    "_build_ome_metadata",
]
