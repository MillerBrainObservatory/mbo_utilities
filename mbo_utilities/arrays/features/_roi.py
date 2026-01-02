"""
ROI (Region of Interest) feature for arrays.

Provides multi-ROI handling for ScanImage data.

Classes
-------
ROIFeature
    Standalone feature object for managing ROI state.
RoiFeatureMixin
    Mixin class that adds ROI properties to array classes.
    Presence of `roi_mode` attribute indicates ROI support (duck typing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent
from mbo_utilities.metadata.base import RoiMode

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class ROIFeature(ArrayFeature):
    """
    ROI feature for arrays.

    Manages multi-ROI state for ScanImage data.

    Parameters
    ----------
    current : int | None
        current ROI index (1-based), None for stitched view
    num_rois : int
        total number of ROIs
    mode : RoiMode | str
        ROI handling mode (concat_y or separate)

    Examples
    --------
    >>> roi = ROIFeature(current=1, num_rois=3)
    >>> roi.current
    1
    >>> roi.num_rois
    3
    >>> list(roi.iter_all())
    [1, 2, 3]
    """

    def __init__(
        self,
        current: int | None = None,
        num_rois: int = 1,
        mode: RoiMode | str = RoiMode.concat_y,
        property_name: str = "roi",
    ):
        super().__init__(property_name=property_name)
        self._current = current
        self._num_rois = max(1, num_rois)

        if isinstance(mode, str):
            self._mode = RoiMode.from_string(mode)
        else:
            self._mode = mode

    @property
    def value(self) -> int | None:
        """Current ROI index (1-based), None for stitched."""
        return self._current

    @property
    def current(self) -> int | None:
        """Current ROI index (1-based), None for stitched view."""
        return self._current

    @property
    def num_rois(self) -> int:
        """Total number of ROIs."""
        return self._num_rois

    @property
    def mode(self) -> RoiMode:
        """ROI handling mode."""
        return self._mode

    @property
    def is_multi_roi(self) -> bool:
        """True if there are multiple ROIs."""
        return self._num_rois > 1

    @property
    def is_stitched(self) -> bool:
        """True if viewing stitched (all ROIs concatenated)."""
        return self._current is None

    def set_value(self, array, value: int | None) -> None:
        """
        Set current ROI.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : int | None
            ROI index (1-based), None for stitched view
        """
        if value is not None and (value < 1 or value > self._num_rois):
            raise ValueError(
                f"ROI index {value} out of range [1, {self._num_rois}]"
            )

        old_value = self._current
        self._current = value

        if old_value != self._current:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={
                    "value": self._current,
                    "old_value": old_value,
                    "is_stitched": self.is_stitched,
                },
            )
            self._call_event_handlers(event)

    def set_mode(self, mode: RoiMode | str) -> None:
        """
        Set ROI handling mode.

        Parameters
        ----------
        mode : RoiMode | str
            new mode
        """
        if isinstance(mode, str):
            self._mode = RoiMode.from_string(mode)
        else:
            self._mode = mode

    def iter_all(self) -> Iterator[int]:
        """
        Iterate over all ROI indices.

        Yields
        ------
        int
            ROI indices from 1 to num_rois
        """
        yield from range(1, self._num_rois + 1)

    def select_next(self) -> int | None:
        """
        Select next ROI (wraps around).

        Returns
        -------
        int | None
            new current ROI index
        """
        if self._current is None:
            self._current = 1
        else:
            self._current = (self._current % self._num_rois) + 1
        return self._current

    def select_previous(self) -> int | None:
        """
        Select previous ROI (wraps around).

        Returns
        -------
        int | None
            new current ROI index
        """
        if self._current is None:
            self._current = self._num_rois
        else:
            self._current = ((self._current - 2) % self._num_rois) + 1
        return self._current

    def select_stitched(self) -> None:
        """Select stitched (all ROIs) view."""
        self._current = None

    def __repr__(self) -> str:
        if self._current is None:
            return f"ROIFeature(stitched, {self._num_rois} rois, {self._mode.value})"
        return f"ROIFeature(roi={self._current}/{self._num_rois}, {self._mode.value})"


class RoiFeatureMixin:
    """
    Mixin class that adds multi-ROI support to array classes.

    This mixin provides the ROI interface for ScanImage multi-ROI data.
    Feature detection uses duck typing: presence of `roi_mode` attribute
    indicates the array supports ROI operations.

    Usage
    -----
    Check for ROI support::

        if hasattr(arr, 'roi_mode'):
            # array supports ROI operations
            for roi_idx in arr.iter_rois():
                ...

    Required attributes (set by implementing class):
        _metadata : dict
            Metadata dict containing 'roi_groups' if multi-ROI
        _roi : int | None
            Current ROI selection (internal state)
        _rois : list[dict]
            Extracted ROI slice info (set via _extract_roi_info)

    Properties provided:
        roi_groups : list[dict]
            Raw ROI group definitions from ScanImage metadata
        roi_slices : list[dict]
            Computed slice information for each ROI
        num_rois : int
            Number of ROIs (1 if single ROI)
        roi : int | None
            Current ROI selection (1-based index, None for stitched)
        roi_mode : RoiMode
            Current ROI handling mode (concat_y or separate)
        is_multi_roi : bool
            True if data has multiple ROIs

    Methods provided:
        iter_rois() : Iterator[int | None]
            Iterate over ROI indices based on current selection
    """

    # ROI mode: controlled by the class or set externally
    _roi_mode: RoiMode = RoiMode.concat_y

    @property
    def roi_groups(self) -> list:
        """
        Raw ROI group definitions from ScanImage metadata.

        Returns empty list if no multi-ROI data.
        """
        md = getattr(self, "_metadata", None) or {}
        groups = md.get("roi_groups", [])
        if isinstance(groups, dict):
            return [groups]
        return groups if groups else []

    @property
    def roi_slices(self) -> list:
        """
        Computed slice information for each ROI.

        Each dict contains:
        - y_start: starting y pixel (inclusive)
        - y_end: ending y pixel (exclusive)
        - width: ROI width in pixels
        - height: ROI height in pixels
        - x: x offset (always 0 for strip ROIs)
        - slice: slice object for y-axis indexing

        Returns empty list if no ROI info available.
        """
        return getattr(self, "_rois", []) or []

    @property
    def num_rois(self) -> int:
        """Number of ROIs. Returns 1 if no multi-ROI data."""
        rois = self.roi_slices
        return len(rois) if rois else 1

    @property
    def roi(self) -> int | None:
        """
        Current ROI selection.

        Values:
        - None: stitched view (all ROIs concatenated)
        - 0: split all ROIs (for iteration)
        - 1..num_rois: specific ROI (1-based index)
        """
        return getattr(self, "_roi", None)

    @roi.setter
    def roi(self, value: int | Sequence[int] | None):
        """
        Set current ROI selection.

        Parameters
        ----------
        value : int | Sequence[int] | None
            - None: stitched view
            - 0: split all ROIs
            - 1..num_rois: specific ROI
            - list/tuple: multiple specific ROIs
        """
        if value is not None and value != 0:
            num = self.num_rois
            if isinstance(value, int):
                if value < 1 or value > num:
                    raise ValueError(
                        f"ROI index {value} out of bounds. "
                        f"Valid range: 1 to {num} (1-indexed). "
                        f"Use roi=0 to split all ROIs, or roi=None to stitch."
                    )
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if v < 1 or v > num:
                        raise ValueError(
                            f"ROI index {v} in {value} out of bounds. "
                            f"Valid range: 1 to {num} (1-indexed)."
                        )
        self._roi = value

    @property
    def roi_mode(self) -> RoiMode:
        """
        ROI handling mode.

        Values:
        - RoiMode.concat_y: concatenate ROIs horizontally (stitched view)
        - RoiMode.separate: write each ROI to separate file
        """
        return getattr(self, "_roi_mode", RoiMode.concat_y)

    @roi_mode.setter
    def roi_mode(self, value: RoiMode | str):
        """Set ROI handling mode."""
        if isinstance(value, str):
            self._roi_mode = RoiMode.from_string(value)
        else:
            self._roi_mode = value

    @property
    def is_multi_roi(self) -> bool:
        """True if data has multiple ROIs."""
        return self.num_rois > 1

    def iter_rois(self) -> Iterator[int | None]:
        """
        Iterate over ROI indices based on current selection.

        Yields ROI indices according to MBO semantics:
        - roi=None: yields None (stitched full-FOV image)
        - roi=0: yields each ROI index from 1..num_rois (split all)
        - roi=int > 0: yields that ROI only
        - roi=list/tuple: yields each element

        Yields
        ------
        int | None
            ROI index (1-based) or None for stitched view
        """
        roi = self.roi
        num = self.num_rois

        if roi is None:
            yield None
        elif roi == 0:
            yield from range(1, num + 1)
        elif isinstance(roi, int):
            yield roi
        elif isinstance(roi, (list, tuple)):
            for r in roi:
                if r == 0:
                    yield from range(1, num + 1)
                else:
                    yield r
        else:
            yield None
