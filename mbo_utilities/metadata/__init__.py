"""
mbo_utilities.metadata - metadata handling for calcium imaging data.

this package provides:
- standardized parameter definitions and aliases
- scanimage-specific metadata parsing
- stack type detection (lbm, piezo, pollen, single plane)
- voxel size extraction and normalization
- file I/O for extracting metadata from TIFF files
"""
from .base import (
    MetadataParameter,
    VoxelSize,
    RoiMode,
    METADATA_PARAMS,
    ALIAS_MAP,
    get_canonical_name,
    IMAGING_METADATA_KEYS,
    get_imaging_metadata_info,
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
    is_pollen_stack,
    get_lbm_ai_sources,
    get_num_color_channels,
    get_num_zplanes,
    get_frames_per_slice,
    get_log_average_factor,
    get_z_step_size,
    get_num_volumes,
    get_num_slices,
    get_frames_per_volume,
    compute_num_timepoints,
    get_roi_info,
    get_frame_rate,
    get_stack_info,
    extract_roi_slices,
)

# file I/O functions
from .io import (
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
    "ALIAS_MAP",
    # imaging metadata (core params for display/editing)
    "IMAGING_METADATA_KEYS",
    "METADATA_PARAMS",
    # base types
    "MetadataParameter",
    "RoiMode",
    # scanimage detection
    "StackType",
    "VoxelSize",
    "_build_ome_metadata",
    "clean_scanimage_metadata",
    "compute_num_timepoints",
    "default_ops",
    "detect_stack_type",
    "extract_roi_slices",
    "get_canonical_name",
    "get_frame_rate",
    "get_frames_per_slice",
    "get_imaging_metadata_info",
    "get_lbm_ai_sources",
    "get_log_average_factor",
    "get_metadata",
    "get_metadata_batch",
    "get_metadata_single",
    "get_num_color_channels",
    "get_num_slices",
    "get_num_volumes",
    "get_num_zplanes",
    "get_frames_per_volume",
    # parameter access
    "get_param",
    "get_roi_info",
    "get_stack_info",
    "get_voxel_size",
    "get_z_step_size",
    # file I/O
    "is_lbm_stack",
    "is_piezo_stack",
    "is_pollen_stack",
    "is_raw_scanimage",
    "normalize_metadata",
    "normalize_resolution",
    "query_tiff_pages",
]
