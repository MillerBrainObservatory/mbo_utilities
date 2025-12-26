"""
Array features for mbo_utilities.

Features are composable properties that can be attached to array classes.
Following the fastplotlib pattern, each feature is a self-contained class
that manages its own state and events.

Available features:
- DimLabels: dimension labeling system (T, Z, Y, X, etc.)
- VoxelSizeFeature: physical pixel/voxel dimensions
- FrameRateFeature: temporal sampling frequency
- DisplayRangeFeature: min/max for display scaling
- ROIFeature: multi-ROI handling
- DataTypeFeature: dtype with lazy conversion
- CompressionFeature: compression settings
- ChunkSizeFeature: chunking configuration
- ZStatsFeature: z-plane statistics
- PhaseCorrectionFeature: bidirectional scan correction

Mixins:
- DimLabelsMixin: adds dims property and related methods to array classes
"""

from __future__ import annotations

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent
from mbo_utilities.arrays.features._chunks import (
    CHUNKS_2D,
    CHUNKS_3D,
    CHUNKS_4D,
    ChunkSizeFeature,
    estimate_chunk_memory,
    normalize_chunks,
)
from mbo_utilities.arrays.features._compression import (
    Codec,
    CompressionFeature,
    CompressionSettings,
)
from mbo_utilities.arrays.features._dim_labels import (
    DEFAULT_DIMS,
    DIM_DESCRIPTIONS,
    DimLabels,
    KNOWN_ORDERINGS,
    get_dim_index,
    get_dims,
    get_num_planes,
    get_slider_dims,
    infer_dims,
    parse_dims,
)
from mbo_utilities.arrays.features._display_range import (
    DisplayRange,
    DisplayRangeFeature,
)
from mbo_utilities.arrays.features._dtype import DataTypeFeature
from mbo_utilities.arrays.features._frame_rate import FrameRateFeature
from mbo_utilities.arrays.features._mixin import DimLabelsMixin
from mbo_utilities.arrays.features._phase_correction import (
    PhaseCorrectionFeature,
    PhaseCorrMethod,
)
from mbo_utilities.arrays.features._roi import ROIFeature
from mbo_utilities.arrays.features._voxel_size import (
    VoxelSizeFeature,
    get_voxel_size_from_metadata,
)
from mbo_utilities.arrays.features._zstats import PlaneStats, ZStatsFeature

__all__ = [
    # base
    "ArrayFeature",
    "ArrayFeatureEvent",
    # dim labels
    "DimLabels",
    "DimLabelsMixin",
    "DEFAULT_DIMS",
    "DIM_DESCRIPTIONS",
    "KNOWN_ORDERINGS",
    "parse_dims",
    "get_dims",
    "get_num_planes",
    "get_slider_dims",
    "get_dim_index",
    "infer_dims",
    # voxel size
    "VoxelSizeFeature",
    "get_voxel_size_from_metadata",
    # frame rate
    "FrameRateFeature",
    # display range
    "DisplayRange",
    "DisplayRangeFeature",
    # roi
    "ROIFeature",
    # dtype
    "DataTypeFeature",
    # compression
    "Codec",
    "CompressionFeature",
    "CompressionSettings",
    # chunks
    "ChunkSizeFeature",
    "CHUNKS_2D",
    "CHUNKS_3D",
    "CHUNKS_4D",
    "normalize_chunks",
    "estimate_chunk_memory",
    # zstats
    "ZStatsFeature",
    "PlaneStats",
    # phase correction
    "PhaseCorrectionFeature",
    "PhaseCorrMethod",
]
