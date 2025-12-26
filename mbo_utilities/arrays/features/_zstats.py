"""
Z-plane statistics feature for arrays.

Provides per-plane statistics (mean, std, SNR) for quality assessment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent

if TYPE_CHECKING:
    pass


class PlaneStats(NamedTuple):
    """Statistics for a single z-plane."""

    mean: float
    std: float
    snr: float
    min: float = 0.0
    max: float = 0.0


class ZStatsFeature(ArrayFeature):
    """
    Z-plane statistics feature for arrays.

    Manages per-plane statistics for quality assessment.

    Parameters
    ----------
    num_planes : int
        number of z-planes

    Examples
    --------
    >>> zs = ZStatsFeature(num_planes=10)
    >>> zs.compute(arr)  # compute from array
    >>> zs.mean  # mean per plane
    [100.5, 102.3, ...]
    >>> zs.snr[0]  # SNR for first plane
    15.2
    """

    def __init__(
        self,
        num_planes: int = 1,
        property_name: str = "zstats",
    ):
        super().__init__(property_name=property_name)
        self._num_planes = num_planes
        self._mean: list[float] | None = None
        self._std: list[float] | None = None
        self._snr: list[float] | None = None
        self._min: list[float] | None = None
        self._max: list[float] | None = None
        self._computed = False

    @property
    def value(self) -> dict | None:
        """statistics as dict with mean, std, snr lists"""
        if not self._computed:
            return None
        return {
            "mean": self._mean,
            "std": self._std,
            "snr": self._snr,
            "min": self._min,
            "max": self._max,
        }

    @property
    def mean(self) -> list[float] | None:
        """mean intensity per plane"""
        return self._mean

    @property
    def std(self) -> list[float] | None:
        """standard deviation per plane"""
        return self._std

    @property
    def snr(self) -> list[float] | None:
        """signal-to-noise ratio per plane (mean/std)"""
        return self._snr

    @property
    def is_computed(self) -> bool:
        """True if statistics have been computed"""
        return self._computed

    @property
    def num_planes(self) -> int:
        """number of z-planes"""
        return self._num_planes

    def get_plane_stats(self, plane_idx: int) -> PlaneStats | None:
        """
        Get statistics for a specific plane.

        Parameters
        ----------
        plane_idx : int
            plane index (0-based)

        Returns
        -------
        PlaneStats | None
            statistics for the plane, or None if not computed
        """
        if not self._computed:
            return None
        if plane_idx < 0 or plane_idx >= self._num_planes:
            raise IndexError(f"plane index {plane_idx} out of range")
        return PlaneStats(
            mean=self._mean[plane_idx],
            std=self._std[plane_idx],
            snr=self._snr[plane_idx],
            min=self._min[plane_idx] if self._min else 0.0,
            max=self._max[plane_idx] if self._max else 0.0,
        )

    def set_value(self, array, value: dict) -> None:
        """
        Set statistics from dict.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : dict
            statistics dict with mean, std, snr keys
        """
        if not isinstance(value, dict):
            raise TypeError(f"expected dict, got {type(value)}")

        required_keys = ("mean", "std", "snr")
        for key in required_keys:
            if key not in value:
                raise ValueError(f"missing required key: {key}")

        old_value = self.value
        self._mean = list(value["mean"])
        self._std = list(value["std"])
        self._snr = list(value["snr"])
        self._min = list(value.get("min", [0.0] * len(self._mean)))
        self._max = list(value.get("max", [0.0] * len(self._mean)))
        self._num_planes = len(self._mean)
        self._computed = True

        event = ArrayFeatureEvent(
            type=self._property_name,
            info={"value": self.value, "old_value": old_value},
        )
        self._call_event_handlers(event)

    def compute(
        self,
        array,
        sample_frames: int = 100,
        subsample_spatial: int = 4,
    ) -> None:
        """
        Compute statistics from array.

        Parameters
        ----------
        array : array-like
            4D array (T, Z, Y, X) or 3D array (T, Y, X)
        sample_frames : int
            number of frames to sample
        subsample_spatial : int
            spatial subsampling factor (1 = no subsampling)
        """
        if array.ndim == 3:
            self._num_planes = 1
            self._compute_3d(array, sample_frames, subsample_spatial)
        elif array.ndim == 4:
            self._num_planes = array.shape[1]
            self._compute_4d(array, sample_frames, subsample_spatial)
        else:
            raise ValueError(f"expected 3D or 4D array, got {array.ndim}D")

        self._computed = True

    def _compute_3d(self, array, sample_frames: int, subsample: int) -> None:
        """Compute stats for 3D (TYX) array."""
        n_frames = len(array)
        if n_frames <= sample_frames:
            indices = list(range(n_frames))
        else:
            indices = np.linspace(0, n_frames - 1, sample_frames, dtype=int).tolist()

        samples = []
        for i in indices:
            frame = np.asarray(array[i])
            if subsample > 1:
                frame = frame[::subsample, ::subsample]
            samples.append(frame.ravel())

        data = np.concatenate(samples)
        self._mean = [float(np.mean(data))]
        self._std = [float(np.std(data))]
        self._snr = [self._mean[0] / self._std[0] if self._std[0] > 0 else 0.0]
        self._min = [float(np.min(data))]
        self._max = [float(np.max(data))]

    def _compute_4d(self, array, sample_frames: int, subsample: int) -> None:
        """Compute stats for 4D (TZYX) array."""
        n_frames = array.shape[0]
        n_planes = array.shape[1]

        if n_frames <= sample_frames:
            indices = list(range(n_frames))
        else:
            indices = np.linspace(0, n_frames - 1, sample_frames, dtype=int).tolist()

        self._mean = []
        self._std = []
        self._snr = []
        self._min = []
        self._max = []

        for z in range(n_planes):
            samples = []
            for i in indices:
                frame = np.asarray(array[i, z])
                if subsample > 1:
                    frame = frame[::subsample, ::subsample]
                samples.append(frame.ravel())

            data = np.concatenate(samples)
            mean_val = float(np.mean(data))
            std_val = float(np.std(data))

            self._mean.append(mean_val)
            self._std.append(std_val)
            self._snr.append(mean_val / std_val if std_val > 0 else 0.0)
            self._min.append(float(np.min(data)))
            self._max.append(float(np.max(data)))

    def best_plane(self) -> int | None:
        """
        Find plane with highest SNR.

        Returns
        -------
        int | None
            index of best plane, or None if not computed
        """
        if not self._computed or not self._snr:
            return None
        return int(np.argmax(self._snr))

    def to_dict(self) -> dict:
        """
        Convert to serializable dict.

        Returns
        -------
        dict
            statistics dict for serialization
        """
        if not self._computed:
            return {}
        return {
            "mean": self._mean,
            "std": self._std,
            "snr": self._snr,
            "min": self._min,
            "max": self._max,
            "num_planes": self._num_planes,
        }

    def __repr__(self) -> str:
        if not self._computed:
            return f"ZStatsFeature({self._num_planes} planes, not computed)"
        avg_snr = np.mean(self._snr) if self._snr else 0
        return f"ZStatsFeature({self._num_planes} planes, avg SNR={avg_snr:.1f})"
