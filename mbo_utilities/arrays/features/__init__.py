"""
Array features for mbo_utilities.

Features are composable properties that can be attached to array classes.
Following the fastplotlib pattern, each feature is a self-contained class
that manages its own state and events.

Available features:
- DimLabels: dimension labeling system (T, Z, Y, X, etc.)

Mixins:
- DimLabelsMixin: adds dims property and related methods to array classes
"""

from __future__ import annotations

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent
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
from mbo_utilities.arrays.features._mixin import DimLabelsMixin

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
]
