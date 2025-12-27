"""
statistics feature for arrays.

provides per-slice statistics (mean, std, SNR) for quality assessment.
generalizes to z-planes, cameras, rois, or any sliceable dimension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent

if TYPE_CHECKING:
    pass


class SliceStats(NamedTuple):
    """statistics for a single slice (plane, camera, roi, etc.)."""

    mean: float
    std: float
    snr: float
    min: float = 0.0
    max: float = 0.0


class StatsFeature(ArrayFeature):
    """
    statistics feature for arrays.

    manages per-slice statistics for quality assessment. generalizes to
    z-planes, cameras, rois, or any dimension you want to track.

    Parameters
    ----------
    num_slices : int
        number of slices (planes, cameras, rois, etc.)
    slice_label : str
        label for the slice dimension (e.g. "z", "camera", "roi")
    property_name : str
        name for this feature instance

    Examples
    --------
    >>> # z-plane stats (default)
    >>> stats = StatsFeature(num_slices=10, slice_label="z")
    >>> stats.compute(arr)
    >>> stats.mean  # mean per z-plane
    [100.5, 102.3, ...]

    >>> # camera stats for isoview
    >>> stats = StatsFeature(num_slices=4, slice_label="camera")
    >>> stats.mean_images  # mean image per camera
    """

    def __init__(
        self,
        num_slices: int = 1,
        slice_label: str = "z",
        property_name: str = "stats",
    ):
        super().__init__(property_name=property_name)
        self._num_slices = num_slices
        self._slice_label = slice_label
        self._mean: list[float] | None = None
        self._std: list[float] | None = None
        self._snr: list[float] | None = None
        self._min: list[float] | None = None
        self._max: list[float] | None = None
        self._mean_images: list[np.ndarray] | None = None
        self._computed = False
        self._progress = 0.0
        self._current_slice = 0

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
            "slice_label": self._slice_label,
        }

    @property
    def mean(self) -> list[float] | None:
        """mean intensity per slice"""
        return self._mean

    @property
    def std(self) -> list[float] | None:
        """standard deviation per slice"""
        return self._std

    @property
    def snr(self) -> list[float] | None:
        """signal-to-noise ratio per slice (mean/std)"""
        return self._snr

    @property
    def mean_images(self) -> list[np.ndarray] | None:
        """mean image per slice (for mean subtraction)"""
        return self._mean_images

    @property
    def is_computed(self) -> bool:
        """True if statistics have been computed"""
        return self._computed

    @property
    def num_slices(self) -> int:
        """number of slices"""
        return self._num_slices

    @property
    def slice_label(self) -> str:
        """label for slices (z, camera, roi, etc.)"""
        return self._slice_label

    @slice_label.setter
    def slice_label(self, value: str) -> None:
        """set slice label"""
        self._slice_label = str(value)

    @property
    def progress(self) -> float:
        """computation progress (0.0 to 1.0)"""
        return self._progress

    @property
    def current_slice(self) -> int:
        """current slice being computed"""
        return self._current_slice

    def get_slice_stats(self, slice_idx: int) -> SliceStats | None:
        """
        get statistics for a specific slice.

        Parameters
        ----------
        slice_idx : int
            slice index (0-based)

        Returns
        -------
        SliceStats | None
            statistics for the slice, or None if not computed
        """
        if not self._computed:
            return None
        if slice_idx < 0 or slice_idx >= self._num_slices:
            raise IndexError(f"slice index {slice_idx} out of range")
        return SliceStats(
            mean=self._mean[slice_idx],
            std=self._std[slice_idx],
            snr=self._snr[slice_idx],
            min=self._min[slice_idx] if self._min else 0.0,
            max=self._max[slice_idx] if self._max else 0.0,
        )

    def get_mean_image(self, slice_idx: int) -> np.ndarray | None:
        """
        get mean image for a specific slice.

        Parameters
        ----------
        slice_idx : int
            slice index (0-based)

        Returns
        -------
        np.ndarray | None
            mean image, or None if not computed
        """
        if self._mean_images is None:
            return None
        if slice_idx < 0 or slice_idx >= len(self._mean_images):
            raise IndexError(f"slice index {slice_idx} out of range")
        return self._mean_images[slice_idx]

    def set_value(self, array, value: dict) -> None:
        """
        set statistics from dict.

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
        self._num_slices = len(self._mean)
        if "slice_label" in value:
            self._slice_label = value["slice_label"]
        if "mean_images" in value:
            self._mean_images = value["mean_images"]
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
        compute_mean_images: bool = True,
        slice_axis: int | None = None,
    ) -> None:
        """
        compute statistics from array.

        Parameters
        ----------
        array : array-like
            array to compute stats from. typically 3D (T, Y, X) or 4D (T, Z, Y, X)
        sample_frames : int
            number of frames to sample for scalar stats
        subsample_spatial : int
            spatial subsampling factor for scalar stats (1 = no subsampling)
        compute_mean_images : bool
            whether to compute full mean images for each slice
        slice_axis : int | None
            axis to slice along. if None, auto-detect:
            - 3D arrays: single slice (whole array)
            - 4D arrays: axis 1 (z-planes)
        """
        self._progress = 0.0
        self._current_slice = 0

        if array.ndim == 3:
            self._num_slices = 1
            self._compute_3d(array, sample_frames, subsample_spatial, compute_mean_images)
        elif array.ndim == 4:
            axis = slice_axis if slice_axis is not None else 1
            self._num_slices = array.shape[axis]
            self._compute_4d(array, sample_frames, subsample_spatial, compute_mean_images, axis)
        else:
            raise ValueError(f"expected 3D or 4D array, got {array.ndim}D")

        self._computed = True
        self._progress = 1.0

    def _compute_3d(
        self,
        array,
        sample_frames: int,
        subsample: int,
        compute_mean_images: bool,
    ) -> None:
        """compute stats for 3D (TYX) array."""
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

        if compute_mean_images:
            # compute full mean image
            acc = np.zeros(array[0].shape, dtype=np.float64)
            for i in indices:
                acc += np.asarray(array[i])
            self._mean_images = [acc / len(indices)]

        self._progress = 1.0

    def _compute_4d(
        self,
        array,
        sample_frames: int,
        subsample: int,
        compute_mean_images: bool,
        slice_axis: int,
    ) -> None:
        """compute stats for 4D array along given axis."""
        n_frames = array.shape[0]
        n_slices = array.shape[slice_axis]

        if n_frames <= sample_frames:
            indices = list(range(n_frames))
        else:
            indices = np.linspace(0, n_frames - 1, sample_frames, dtype=int).tolist()

        self._mean = []
        self._std = []
        self._snr = []
        self._min = []
        self._max = []
        self._mean_images = [] if compute_mean_images else None

        for s in range(n_slices):
            self._current_slice = s
            samples = []

            # build slice for this dimension
            for i in indices:
                if slice_axis == 1:
                    frame = np.asarray(array[i, s])
                elif slice_axis == 2:
                    frame = np.asarray(array[i, :, s])
                else:
                    # generic slicing
                    idx = [i] + [slice(None)] * (array.ndim - 1)
                    idx[slice_axis] = s
                    frame = np.asarray(array[tuple(idx)])

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

            if compute_mean_images:
                # compute full mean image for this slice
                if slice_axis == 1:
                    first_frame = array[0, s]
                elif slice_axis == 2:
                    first_frame = array[0, :, s]
                else:
                    idx = [0] + [slice(None)] * (array.ndim - 1)
                    idx[slice_axis] = s
                    first_frame = array[tuple(idx)]

                acc = np.zeros(first_frame.shape, dtype=np.float64)
                for i in indices:
                    if slice_axis == 1:
                        acc += np.asarray(array[i, s])
                    elif slice_axis == 2:
                        acc += np.asarray(array[i, :, s])
                    else:
                        idx = [i] + [slice(None)] * (array.ndim - 1)
                        idx[slice_axis] = s
                        acc += np.asarray(array[tuple(idx)])
                self._mean_images.append(acc / len(indices))

            self._progress = (s + 1) / n_slices

    def best_slice(self) -> int | None:
        """
        find slice with highest SNR.

        Returns
        -------
        int | None
            index of best slice, or None if not computed
        """
        if not self._computed or not self._snr:
            return None
        return int(np.argmax(self._snr))

    def to_dict(self) -> dict:
        """
        convert to serializable dict.

        Returns
        -------
        dict
            statistics dict for serialization (excludes mean_images)
        """
        if not self._computed:
            return {}
        return {
            "mean": self._mean,
            "std": self._std,
            "snr": self._snr,
            "min": self._min,
            "max": self._max,
            "num_slices": self._num_slices,
            "slice_label": self._slice_label,
        }

    def reset(self) -> None:
        """reset to initial state."""
        self._mean = None
        self._std = None
        self._snr = None
        self._min = None
        self._max = None
        self._mean_images = None
        self._computed = False
        self._progress = 0.0
        self._current_slice = 0

    def __repr__(self) -> str:
        if not self._computed:
            return f"StatsFeature({self._num_slices} {self._slice_label}, not computed)"
        avg_snr = np.mean(self._snr) if self._snr else 0
        return f"StatsFeature({self._num_slices} {self._slice_label}, avg SNR={avg_snr:.1f})"


# backwards compatibility aliases
PlaneStats = SliceStats
ZStatsFeature = StatsFeature
