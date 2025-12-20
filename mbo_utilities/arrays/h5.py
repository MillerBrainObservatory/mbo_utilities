"""
HDF5 array reader.

This module provides H5Array for reading HDF5 datasets as lazy arrays.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin
from mbo_utilities.metadata import get_param

logger = log.get("arrays.h5")


class H5Array(ReductionMixin):
    """
    Lazy array reader for HDF5 datasets.

    Wraps an h5py.Dataset to provide array-like access with lazy loading.
    Auto-detects common dataset names ('mov', 'data', 'scan_corrections').

    Parameters
    ----------
    filenames : Path or str
        Path to HDF5 file.
    dataset : str, optional
        Dataset name to open. If None, auto-detects from common names.

    Attributes
    ----------
    shape : tuple[int, ...]
        Dataset shape.
    dtype : np.dtype
        Data type.
    ndim : int
        Number of dimensions.
    dataset_name : str
        Name of the opened dataset.

    Examples
    --------
    >>> arr = H5Array("data.h5")
    >>> arr.shape
    (10000, 512, 512)
    >>> frame = arr[0]  # Get first frame
    """

    def __init__(self, filenames: Path | str, dataset: str = None):
        self.filenames = Path(filenames)
        self._f = h5py.File(self.filenames, "r")

        # Auto-detect dataset if not specified
        if dataset is None:
            if "mov" in self._f:
                dataset = "mov"
            elif "data" in self._f:
                dataset = "data"
            elif "scan_corrections" in self._f:
                dataset = "scan_corrections"
                logger.info(f"Detected pollen calibration file: {self.filenames.name}")
            else:
                available = list(self._f.keys())
                if not available:
                    raise ValueError(f"No datasets found in {self.filenames}")
                dataset = available[0]
                logger.warning(
                    f"Using first available dataset '{dataset}' in {self.filenames.name}. "
                    f"Available: {available}"
                )

        try:
            self._d = self._f[dataset]
        except KeyError:
            available = list(self._f.keys())
            raise KeyError(
                f"Dataset '{dataset}' not found in {self.filenames}. "
                f"Available datasets: {available}"
            ) from None

        self.dataset_name = dataset
        self.shape = self._d.shape
        self._dtype = self._d.dtype
        self.ndim = self._d.ndim
        self._target_dtype = None

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame (frame 0, plane 0)."""
        if not hasattr(self, '_cached_vmin'):
            frame = self[0, 0] if self.ndim == 4 else self[0]
            frame = np.asarray(frame)
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        """Min from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        """Max from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmax

    @property
    def num_planes(self) -> int:
        """Number of Z-planes in the dataset."""
        # try to get from metadata first using canonical lookup
        nplanes = get_param(self.metadata, "nplanes")
        if nplanes is not None:
            return int(nplanes)

        # infer from shape based on data dimensionality
        if self.ndim >= 4:  # (T, Z, Y, X) - volumetric time series
            return int(self.shape[1])
        elif self.ndim == 3:  # (T, Y, X) - single plane time series
            return 1
        elif self.ndim == 1:
            # special case: pollen scan_corrections (nc,)
            if self.dataset_name == "scan_corrections":
                return int(self.shape[0])
            return 1
        elif self.ndim == 2:  # (Y, X) - single frame
            return 1

        return 1

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Expand ellipsis to match ndim
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            n_missing = self.ndim - (len(key) - 1)
            key = key[:idx] + (slice(None),) * n_missing + key[idx + 1 :]

        slices = []
        result_shape = []
        dim = 0
        for k in key:
            if k is None:
                result_shape.append(1)
            else:
                slices.append(k)
                dim += 1

        data = self._d[tuple(slices)]

        for i, k in enumerate(key):
            if k is None:
                data = np.expand_dims(data, axis=i)

        if self._target_dtype is not None:
            data = data.astype(self._target_dtype)
        return data

    def __array__(self):
        n = min(10, self.shape[0])
        return self._d[:n]

    def close(self):
        """Close the HDF5 file."""
        self._f.close()

    @property
    def metadata(self) -> dict:
        """File-level attributes as metadata dictionary."""
        return dict(self._f.attrs)

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        """Write H5Array to disk in various formats."""
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            **kwargs,
        )
