"""
ROI (Region of Interest) feature for arrays.

Provides multi-ROI handling for ScanImage data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent
from mbo_utilities.metadata.base import RoiMode

if TYPE_CHECKING:
    from typing import Iterator


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
        """current ROI index (1-based), None for stitched"""
        return self._current

    @property
    def current(self) -> int | None:
        """current ROI index (1-based), None for stitched view"""
        return self._current

    @property
    def num_rois(self) -> int:
        """total number of ROIs"""
        return self._num_rois

    @property
    def mode(self) -> RoiMode:
        """ROI handling mode"""
        return self._mode

    @property
    def is_multi_roi(self) -> bool:
        """True if there are multiple ROIs"""
        return self._num_rois > 1

    @property
    def is_stitched(self) -> bool:
        """True if viewing stitched (all ROIs concatenated)"""
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
        if value is not None:
            if value < 1 or value > self._num_rois:
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
