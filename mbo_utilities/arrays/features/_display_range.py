"""
Display range feature for arrays.

Provides min/max values for display scaling (vmin/vmax).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent

if TYPE_CHECKING:
    pass


class DisplayRange(NamedTuple):
    """Display range as (vmin, vmax) tuple."""

    vmin: float
    vmax: float

    @property
    def span(self) -> float:
        """range span (vmax - vmin)"""
        return self.vmax - self.vmin

    @property
    def center(self) -> float:
        """center value"""
        return (self.vmin + self.vmax) / 2


class DisplayRangeFeature(ArrayFeature):
    """
    Display range feature for arrays.

    Manages min/max values for display scaling with lazy computation.

    Parameters
    ----------
    vmin : float | None
        minimum display value
    vmax : float | None
        maximum display value

    Examples
    --------
    >>> dr = DisplayRangeFeature(0, 4095)
    >>> dr.value
    DisplayRange(vmin=0, vmax=4095)
    >>> dr.vmin
    0
    >>> dr.span
    4095
    """

    def __init__(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        property_name: str = "display_range",
    ):
        super().__init__(property_name=property_name)
        self._vmin = vmin
        self._vmax = vmax
        self._auto_computed = False

    @property
    def value(self) -> DisplayRange | None:
        """display range as DisplayRange namedtuple"""
        if self._vmin is None or self._vmax is None:
            return None
        return DisplayRange(self._vmin, self._vmax)

    @property
    def vmin(self) -> float | None:
        """minimum display value"""
        return self._vmin

    @property
    def vmax(self) -> float | None:
        """maximum display value"""
        return self._vmax

    @property
    def span(self) -> float | None:
        """range span (vmax - vmin)"""
        if self._vmin is None or self._vmax is None:
            return None
        return self._vmax - self._vmin

    @property
    def center(self) -> float | None:
        """center value"""
        if self._vmin is None or self._vmax is None:
            return None
        return (self._vmin + self._vmax) / 2

    @property
    def is_auto_computed(self) -> bool:
        """True if range was auto-computed from data"""
        return self._auto_computed

    def set_value(self, array, value) -> None:
        """
        Set display range.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : DisplayRange | tuple | None
            new display range as DisplayRange, (vmin, vmax) tuple, or None
        """
        old_value = self.value

        if value is None:
            self._vmin = None
            self._vmax = None
            self._auto_computed = False
        elif isinstance(value, DisplayRange):
            self._vmin = value.vmin
            self._vmax = value.vmax
            self._auto_computed = False
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            self._vmin = float(value[0])
            self._vmax = float(value[1])
            self._auto_computed = False
        else:
            raise TypeError(f"expected DisplayRange, tuple, or None, got {type(value)}")

        new_value = self.value
        if old_value != new_value:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"value": new_value, "old_value": old_value},
            )
            self._call_event_handlers(event)

    def compute_from_frame(self, array, frame_idx: int = 0, plane_idx: int = 0) -> None:
        """
        Compute display range from a single frame.

        Parameters
        ----------
        array : array-like
            the array to compute range from
        frame_idx : int
            frame index to sample
        plane_idx : int
            plane index to sample (for 4D data)
        """
        if array.ndim == 4:
            frame = np.asarray(array[frame_idx, plane_idx])
        elif array.ndim == 3:
            frame = np.asarray(array[frame_idx])
        else:
            frame = np.asarray(array)

        self._vmin = float(frame.min())
        self._vmax = float(frame.max())
        self._auto_computed = True

    def compute_from_percentile(
        self,
        array,
        low: float = 1.0,
        high: float = 99.0,
        sample_frames: int = 10,
    ) -> None:
        """
        Compute display range from percentiles.

        Parameters
        ----------
        array : array-like
            the array to compute range from
        low : float
            low percentile (0-100)
        high : float
            high percentile (0-100)
        sample_frames : int
            number of frames to sample for percentile calculation
        """
        # sample frames evenly
        n_frames = len(array)
        if n_frames <= sample_frames:
            indices = list(range(n_frames))
        else:
            indices = np.linspace(0, n_frames - 1, sample_frames, dtype=int).tolist()

        samples = []
        for i in indices:
            if array.ndim == 4:
                frame = np.asarray(array[i, 0])
            elif array.ndim == 3:
                frame = np.asarray(array[i])
            else:
                frame = np.asarray(array)
            samples.append(frame.ravel())

        data = np.concatenate(samples)
        self._vmin = float(np.percentile(data, low))
        self._vmax = float(np.percentile(data, high))
        self._auto_computed = True

    def clip(self, data: np.ndarray) -> np.ndarray:
        """
        Clip data to display range.

        Parameters
        ----------
        data : np.ndarray
            data to clip

        Returns
        -------
        np.ndarray
            clipped data
        """
        if self._vmin is None or self._vmax is None:
            return data
        return np.clip(data, self._vmin, self._vmax)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] based on display range.

        Parameters
        ----------
        data : np.ndarray
            data to normalize

        Returns
        -------
        np.ndarray
            normalized data in [0, 1]
        """
        if self._vmin is None or self._vmax is None:
            raise ValueError("display range not set")
        span = self._vmax - self._vmin
        if span == 0:
            return np.zeros_like(data, dtype=np.float32)
        return (data.astype(np.float32) - self._vmin) / span

    def __repr__(self) -> str:
        if self._vmin is None or self._vmax is None:
            return "DisplayRangeFeature(None)"
        auto = " [auto]" if self._auto_computed else ""
        return f"DisplayRangeFeature({self._vmin:.1f}, {self._vmax:.1f}){auto}"
