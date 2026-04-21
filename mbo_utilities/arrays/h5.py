"""
HDF5 array reader.

This module provides H5Array for reading HDF5 datasets as lazy arrays.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin, Shape5DMixin
from mbo_utilities.metadata import get_param
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.h5")

# register hdf5 pipeline info
_H5_INFO = PipelineInfo(
    name="hdf5",
    description="HDF5 datasets",
    input_patterns=[
        "**/*.h5",
        "**/*.hdf5",
        "**/*.hdf",
    ],
    output_patterns=[
        "**/*.h5",
        "**/*.hdf5",
    ],
    input_extensions=["h5", "hdf5", "hdf"],
    output_extensions=["h5", "hdf5"],
    marker_files=[],
    category="reader",
)
register_pipeline(_H5_INFO)


class H5Array(ReductionMixin, Shape5DMixin):
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

    def __init__(self, filenames: Path | str, dataset: str | None = None):
        # stored as a list for consistency with every other array class;
        # consumers (incl. lbm_suite2p_python.run_plane) assume filenames[0].
        self.filenames = [Path(filenames)]
        path = self.filenames[0]
        self._f = h5py.File(path, "r")

        # Auto-detect dataset if not specified
        if dataset is None:
            if "mov" in self._f:
                dataset = "mov"
            elif "data" in self._f:
                dataset = "data"
            elif "scan_corrections" in self._f:
                dataset = "scan_corrections"
                logger.info(f"Detected pollen calibration file: {path.name}")
            else:
                available = list(self._f.keys())
                if not available:
                    raise ValueError(f"No datasets found in {path}")
                dataset = available[0]
                logger.warning(
                    f"Using first available dataset '{dataset}' in {path.name}. "
                    f"Available: {available}"
                )

        try:
            self._d = self._f[dataset]
        except KeyError:
            available = list(self._f.keys())
            raise KeyError(
                f"Dataset '{dataset}' not found in {path}. "
                f"Available datasets: {available}"
            ) from None

        self.dataset_name = dataset
        self.shape = self._d.shape
        self._dtype = self._d.dtype
        self.ndim = self._d.ndim
        self._target_dtype = None

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        s = self.shape
        if len(s) == 5:
            return s
        if len(s) == 4:
            return (s[0], 1, s[1], s[2], s[3])
        if len(s) == 3:
            return (s[0], 1, 1, s[1], s[2])
        if len(s) == 2:
            return (1, 1, 1, s[0], s[1])
        return (1, 1, 1, 1, s[0]) if len(s) == 1 else (1, 1, 1, 1, 1)

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    # _compute_frame_vminmax / vmin / vmax inherited from ReductionMixin

    @property
    def num_planes(self) -> int:
        """Number of Z-planes in the dataset (index 2 in 5D TCZYX)."""
        # try to get from metadata first using canonical lookup
        nplanes = get_param(self.metadata, "nplanes")
        if nplanes is not None:
            return int(nplanes)

        # special case: pollen scan_corrections (1D)
        if self.dataset_name == "scan_corrections" and len(self._d.shape) == 1:
            return int(self._d.shape[0])

        return self._shape5d()[2]

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

    def __array__(self, dtype=None, copy=None):
        # return first frame for fast histogram/preview (prevents accidental full load)
        data = self._d[0]
        if self._target_dtype is not None:
            data = data.astype(self._target_dtype)
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def close(self):
        """Close the HDF5 file."""
        self._f.close()

    @property
    def metadata(self) -> dict:
        """File-level attributes as metadata dictionary. Always returns dict, never None."""
        return dict(self._f.attrs) if self._f.attrs else {}

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        # Update file-level attributes
        for k, v in value.items():
            try:
                self._f.attrs[k] = v
            except (TypeError, ValueError):
                # Skip values that can't be stored in HDF5 attrs
                pass

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
