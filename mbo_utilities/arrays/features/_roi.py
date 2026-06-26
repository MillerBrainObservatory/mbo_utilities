"""
ROI (Region of Interest) feature for arrays.

Provides multi-ROI handling for ScanImage data.

Classes
-------
RoiFeatureMixin
    Mixin class that adds ROI properties to array classes.
    Presence of `roi_mode` attribute indicates ROI support (duck typing).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING


class RoiMode(str, Enum):
    """Mode for handling multi-ROI (mROI) data from ScanImage.

    Attributes
    ----------
    concat_y : str
        Concatenate ROIs along Y axis into a single FOV (default).
    separate : str
        Write each ROI to separate files.
    """

    concat_y = "concat_y"
    separate = "separate"

    @classmethod
    def from_string(cls, value: str) -> RoiMode:
        """Case-insensitive lookup of RoiMode from string."""
        value_lower = value.lower().strip()
        for member in cls:
            if member.value.lower() == value_lower:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown RoiMode: {value!r}. Valid modes: {valid}")

    @property
    def description(self) -> str:
        """Human-readable description of the mode."""
        descriptions = {
            RoiMode.concat_y: "horizontally concatenate ROIs",
            RoiMode.separate: "separate ROI files",
        }
        return descriptions.get(self, self.value)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


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
