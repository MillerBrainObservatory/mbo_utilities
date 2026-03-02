"""
Mixin for adding feature support to array classes.

Provides a standardized way to integrate features like DimLabels
into array classes.
"""

from __future__ import annotations


from mbo_utilities.arrays.features._dim_labels import DimLabels
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class DimLabelsMixin:
    """
    Mixin that adds dimension labeling support to array classes.

    Add this mixin to array classes to provide:
    - _dim_labels: DimLabels feature instance
    - dims: property for getting/setting dimension labels
    - dim_index(): method to get index of a dimension

    Usage
    -----
    class MyArray(DimLabelsMixin, ReductionMixin):
        def __init__(self, path, dims=None):
            self._data = np.load(path)
            # initialize dim labels feature
            self._init_dim_labels(dims)

        @property
        def ndim(self):
            return self._data.ndim

        @property
        def shape(self):
            return self._data.shape

    # now you can use:
    arr = MyArray("data.npy", dims="TZYX")
    arr.dims  # ('T', 'Z', 'Y', 'X')
    arr.dims = "ZTYX"  # change labels
    arr.dim_index("Z")  # 0
    """

    _dim_labels: DimLabels | None = None

    def _init_dim_labels(self, dims: str | Sequence[str] | None = None) -> None:
        """
        Initialize the DimLabels feature.

        Call this in __init__ after setting up the array data.

        Parameters
        ----------
        dims : str | Sequence[str] | None
            initial dimension labels. if None, inferred from ndim.
        """
        self._dim_labels = DimLabels(dims=dims, ndim=self.ndim)

    @property
    def dims(self) -> tuple[str, ...] | None:
        """
        Dimension labels for this array.

        Returns
        -------
        tuple[str, ...] | None
            dimension labels like ('T', 'Z', 'Y', 'X'), or None if not set
        """
        if self._dim_labels is None:
            return None
        return self._dim_labels.value

    @dims.setter
    def dims(self, value: str | Sequence[str] | None) -> None:
        """
        Set dimension labels.

        Parameters
        ----------
        value : str | Sequence[str] | None
            new dimension labels (e.g., "TZYX" or ("T", "Z", "Y", "X"))
        """
        if self._dim_labels is None:
            self._init_dim_labels(value)
        else:
            self._dim_labels.set_value(self, value)

    @property
    def slider_dims(self) -> tuple[str, ...]:
        """
        Dimensions that should have sliders in viewers.

        These are all non-spatial dimensions (everything except Y, X).

        Returns
        -------
        tuple[str, ...]
            slider dimension labels
        """
        if self._dim_labels is None:
            return ()
        return self._dim_labels.slider_dims

    def dim_index(self, label: str) -> int | None:
        """
        Get the index of a dimension by label.

        Parameters
        ----------
        label : str
            dimension label (case-insensitive)

        Returns
        -------
        int | None
            index of the dimension, or None if not found
        """
        if self._dim_labels is None:
            return None
        return self._dim_labels.index(label)

    def has_dim(self, label: str) -> bool:
        """
        Check if array has a specific dimension.

        Parameters
        ----------
        label : str
            dimension label to check (case-insensitive)

        Returns
        -------
        bool
            True if dimension exists
        """
        if self._dim_labels is None:
            return False
        return self._dim_labels.has(label)
