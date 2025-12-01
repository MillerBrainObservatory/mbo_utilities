"""
Zarr array reader.

This module provides ZarrArray for reading Zarr v3 stores, including OME-Zarr.
Presents data in TZYX format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin

logger = log.get("arrays.zarr")


class ZarrArray(ReductionMixin):
    """
    Reader for Zarr stores (including OME-Zarr).

    Presents data as (T, Z, H, W) with Z=1..nz. Supports both standard
    zarr arrays and OME-Zarr groups with "0" arrays.

    Parameters
    ----------
    filenames : str, Path, or sequence
        Path(s) to zarr store(s).
    compressor : str, optional
        Compressor name (not currently used for reading).
    rois : list[int] or int, optional
        ROI filter (not currently used).

    Attributes
    ----------
    shape : tuple[int, int, int, int]
        Shape as (T, Z, H, W).
    dtype : np.dtype
        Data type.
    zs : list
        List of zarr arrays.

    Examples
    --------
    >>> arr = ZarrArray("data.zarr")
    >>> arr.shape
    (10000, 1, 512, 512)
    >>> frame = arr[0, 0]  # Get first frame of first z-plane
    """

    def __init__(
        self,
        filenames: str | Path | Sequence[str | Path],
        compressor: str | None = "default",
        rois: list[int] | int | None = None,
    ):
        try:
            import zarr
        except ImportError:
            logger.error(
                "zarr is not installed. Install with `uv pip install zarr>=3.1.3`."
            )
            raise

        if isinstance(filenames, (str, Path)):
            filenames = [filenames]

        self.filenames = [Path(p).with_suffix(".zarr") for p in filenames]
        self.rois = rois
        for p in self.filenames:
            if not p.exists():
                raise FileNotFoundError(f"No zarr store at {p}")

        # Open zarr stores - handle both standard arrays and OME-Zarr groups
        opened = [zarr.open(p, mode="r") for p in self.filenames]

        # If we opened a Group (OME-Zarr structure), get the "0" array
        self.zs = []
        self._groups = []
        for z in opened:
            if isinstance(z, zarr.Group):
                if "0" not in z:
                    raise ValueError(
                        f"OME-Zarr group missing '0' array in {z.store.path}"
                    )
                self.zs.append(z["0"])
                self._groups.append(z)
            else:
                self.zs.append(z)
                self._groups.append(None)

        shapes = [z.shape for z in self.zs]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent shapes across zarr stores: {shapes}")

        # For OME-Zarr, metadata is on the group; for standard zarr, on the array
        self._metadata = []
        for i, z in enumerate(self.zs):
            if self._groups[i] is not None:
                self._metadata.append(dict(self._groups[i].attrs))
            else:
                self._metadata.append(dict(z.attrs))
        self.compressor = compressor

    @property
    def metadata(self):
        """Return metadata as a dict."""
        if not self._metadata:
            md = {}
        else:
            md = self._metadata[0]

        # Ensure critical keys are present
        if "num_frames" not in md and "nframes" not in md:
            if self.zs:
                md["num_frames"] = int(self.zs[0].shape[0])

        return md

    @property
    def zstats(self) -> dict | None:
        """
        Return pre-computed z-statistics from metadata if available.

        Returns
        -------
        dict | None
            Dictionary with keys 'mean', 'std', 'snr' (each a list of floats),
            or None if not available.
        """
        md = self.metadata
        if "zstats" in md:
            return md["zstats"]
        return None

    @zstats.setter
    def zstats(self, value: dict):
        """Store z-statistics in metadata for persistence."""
        if not isinstance(value, dict):
            raise TypeError(f"zstats must be a dict, got {type(value)}")
        if not all(k in value for k in ("mean", "std", "snr")):
            raise ValueError("zstats must contain 'mean', 'std', and 'snr' keys")

        if not self._metadata:
            self._metadata = [{}]
        self._metadata[0]["zstats"] = value

    @metadata.setter
    def metadata(self, value: dict):
        """Set metadata. Updates the first zarr file's metadata."""
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")

        if not self._metadata:
            self._metadata = [value]
        else:
            self._metadata[0] = value

    @property
    def shape(self) -> tuple[int, int, int, int]:
        first_shape = self.zs[0].shape
        if len(first_shape) == 4:
            return first_shape
        elif len(first_shape) == 3:
            t, h, w = first_shape
            return t, len(self.zs), h, w
        else:
            raise ValueError(
                f"Unexpected zarr shape: {first_shape}. "
                f"Expected 3D (T, H, W) or 4D (T, Z, H, W)"
            )

    @property
    def dtype(self):
        return self.zs[0].dtype

    @property
    def size(self):
        return np.prod(self.shape)

    def __array__(self):
        """Materialize full array into memory: (T, Z, H, W)."""
        if len(self.zs) == 1 and len(self.zs[0].shape) == 4:
            return np.asarray(self.zs[0][:])
        arrs = [z[:] for z in self.zs]
        return np.stack(arrs, axis=1)


    @property
    def ndim(self):
        return 4

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        t_key, z_key, y_key, x_key = key

        def normalize(idx):
            if isinstance(idx, range):
                if len(idx) == 0:
                    return slice(0, 0)
                return slice(idx.start, idx.stop, idx.step)
            if isinstance(idx, list) and len(idx) > 0:
                if all(idx[i] + 1 == idx[i + 1] for i in range(len(idx) - 1)):
                    return slice(idx[0], idx[-1] + 1)
                else:
                    return np.array(idx)
            return idx

        t_key = normalize(t_key)
        y_key = normalize(y_key)
        x_key = normalize(x_key)
        z_key = normalize(z_key)

        is_single_4d = len(self.zs) == 1 and len(self.zs[0].shape) == 4

        if is_single_4d:
            return self.zs[0][t_key, z_key, y_key, x_key]

        if len(self.zs) == 1:
            if isinstance(z_key, int):
                if z_key != 0:
                    raise IndexError("Z dimension has size 1, only index 0 is valid")
                return self.zs[0][t_key, y_key, x_key]
            elif isinstance(z_key, slice):
                data = self.zs[0][t_key, y_key, x_key]
                return data[:, np.newaxis, ...]
            else:
                return self.zs[0][t_key, y_key, x_key]

        # Multi-zarr case
        if isinstance(z_key, int):
            return self.zs[z_key][t_key, y_key, x_key]

        if isinstance(z_key, slice):
            z_indices = range(len(self.zs))[z_key]
        elif isinstance(z_key, np.ndarray) or isinstance(z_key, list):
            z_indices = z_key
        else:
            z_indices = range(len(self.zs))

        arrs = [self.zs[i][t_key, y_key, x_key] for i in z_indices]
        return np.stack(arrs, axis=1)

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite: bool = False,
        target_chunk_mb: int = 50,
        ext: str = ".tiff",
        progress_callback=None,
        debug: bool = False,
        planes: list[int] | int | None = None,
        **kwargs,
    ):
        """Write ZarrArray to disk in various formats."""
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
