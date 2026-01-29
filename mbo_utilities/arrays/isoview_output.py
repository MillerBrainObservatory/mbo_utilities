"""Generic lazy array for IsoView pipeline outputs (TIF, KLB, or Zarr)."""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# patterns to exclude (masks, auxiliary files)
EXCLUDE_PATTERNS = [
    "Mask",
    "mask",
    "minIntensity",
    "coords",
]


class IsoViewOutputArray:
    """
    Generic lazy array for IsoView pipeline outputs.

    Handles TM folder structure with various file formats (.tif, .klb, .zarr)
    and view dimensions (cameras, channels, or custom groupings).

    Shape: (T, Z, Views, Y, X) or (Z, Views, Y, X) for single timepoint

    Parameters
    ----------
    path : str or Path
        Path to directory containing TM* folders or a single TM folder
    view_dim : str, optional
        Name for view dimension in dims tuple. Default auto-detects:
        - 'cm' if CM## pattern found in filenames
        - 'ch' if only CHN## pattern (no CM) found
        - 'view' otherwise

    Examples
    --------
    >>> arr = IsoViewOutputArray("path/to/output")
    >>> arr.shape
    (4, 38, 4, 1848, 768)  # (t, z, views, y, x)
    >>> arr.dims
    ('t', 'z', 'cm', 'y', 'x')
    >>> arr.views
    [0, 1, 2, 3]  # camera indices
    >>> frame = arr[0, 10, 0]  # t=0, z=10, view=0
    """

    def __init__(self, path: str | Path, view_dim: str | None = None):
        self.base_path = Path(path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        # discover structure
        self._discover_structure()

        # auto-detect or use provided view dimension name
        self._view_dim = view_dim or self._detect_view_dim()

        # cache for loaded volumes
        self._cache = {}
        self._metadata = {}

    def _discover_structure(self):
        """Find TM folders, file type, and views."""
        # detect if single TM or multi-TM
        if self.base_path.name.startswith("TM"):
            self._single_timepoint = True
            self.tm_folders = [self.base_path]
        else:
            self.tm_folders = sorted(
                [d for d in self.base_path.iterdir()
                 if d.is_dir() and d.name.startswith("TM")],
                key=lambda x: int(x.name[2:])
            )
            self._single_timepoint = len(self.tm_folders) == 1

            if not self.tm_folders:
                raise ValueError(f"No TM* folders found in {self.base_path}")

        # detect file type from first TM folder
        first_tm = self.tm_folders[0]
        self._detect_file_type(first_tm)

        # discover views
        self._discover_views(first_tm)

    def _detect_file_type(self, tm_folder: Path):
        """Detect file type (.tif, .klb, or .zarr) in TM folder."""
        # check for each type, excluding mask files
        tif_files = [f for f in tm_folder.glob("*.tif")
                     if not any(x in f.name for x in EXCLUDE_PATTERNS)]
        klb_files = [f for f in tm_folder.glob("*.klb")
                     if not any(x in f.name for x in EXCLUDE_PATTERNS)]
        zarr_files = [f for f in tm_folder.glob("*.zarr")
                      if not any(x in f.name for x in EXCLUDE_PATTERNS)]

        if tif_files:
            self._file_ext = ".tif"
            self._data_files = tif_files
        elif klb_files:
            self._file_ext = ".klb"
            self._data_files = klb_files
        elif zarr_files:
            self._file_ext = ".zarr"
            self._data_files = zarr_files
        else:
            raise ValueError(
                f"No supported data files (.tif, .klb, .zarr) in {tm_folder}"
            )

        logger.info(f"IsoViewOutputArray: detected {self._file_ext} files")

    def _discover_views(self, tm_folder: Path):
        """Parse filenames to find views and determine shape."""
        # patterns for extracting view info
        # SPM00_TM000000_CM00_CHN01.tif -> camera=0, channel=1
        # SPM00_TM000000_CHN00.tif -> channel=0 (no camera)
        pattern_cm = re.compile(
            r"SPM(\d+)_TM(\d+)_CM(\d+)_CHN(\d+)" + re.escape(self._file_ext)
        )
        pattern_chn = re.compile(
            r"SPM(\d+)_TM(\d+)_CHN(\d+)" + re.escape(self._file_ext)
        )

        self._views = []
        self._view_type = None  # 'cm' or 'ch'
        self._specimen = None
        self._file_map = {}  # view_idx -> filename pattern parts

        for f in sorted(self._data_files):
            # try CM pattern first
            match = pattern_cm.match(f.name)
            if match:
                specimen, timepoint, camera, channel = map(int, match.groups())
                self._specimen = specimen
                self._view_type = "cm"
                if camera not in self._views:
                    self._views.append(camera)
                    self._file_map[camera] = {"camera": camera, "channel": channel}
                continue

            # try CHN-only pattern
            match = pattern_chn.match(f.name)
            if match:
                specimen, timepoint, channel = map(int, match.groups())
                self._specimen = specimen
                self._view_type = "ch"
                if channel not in self._views:
                    self._views.append(channel)
                    self._file_map[channel] = {"channel": channel}
                continue

        if not self._views:
            raise ValueError(f"No valid data files found in {tm_folder}")

        # sort views
        self._views = sorted(self._views)

        # get shape from first file
        first_file = self._get_file_path(0, 0)
        self._read_shape(first_file)

        logger.info(
            f"IsoViewOutputArray: views={self._views}, shape={self._single_shape}, "
            f"view_type={self._view_type}"
        )

    def _read_shape(self, file_path: Path):
        """Read shape and dtype from first file."""
        if self._file_ext == ".tif":
            import tifffile
            with tifffile.TiffFile(str(file_path)) as tif:
                # get shape from first series
                shape = tif.series[0].shape
                self._dtype = tif.series[0].dtype
        elif self._file_ext == ".klb":
            import pyklb
            header = pyklb.readheader(str(file_path))
            dims = header['imagesize_tczyx']
            spatial_dims = [d for d in dims if d > 1]
            if len(spatial_dims) >= 3:
                shape = tuple(spatial_dims[-3:])
            else:
                shape = (1,) * (3 - len(spatial_dims)) + tuple(spatial_dims)
            self._dtype = np.dtype(header['datatype'])
        elif self._file_ext == ".zarr":
            import zarr
            z = zarr.open(file_path, mode="r")
            if isinstance(z, zarr.Group):
                arr = z["0"] if "0" in z else list(z.values())[0]
            else:
                arr = z
            shape = arr.shape
            self._dtype = arr.dtype
        else:
            raise ValueError(f"Unsupported file type: {self._file_ext}")

        # shape should be (Z, Y, X)
        if len(shape) == 3:
            self._single_shape = shape
        elif len(shape) == 2:
            self._single_shape = (1,) + shape
        else:
            # take last 3 dims
            self._single_shape = shape[-3:]

    def _get_file_path(self, t_idx: int, view_idx: int) -> Path:
        """Get file path for timepoint and view index."""
        tm_folder = self.tm_folders[t_idx]
        tp = int(tm_folder.name[2:])  # extract timepoint from folder name
        sp = self._specimen if self._specimen is not None else 0
        view = self._views[view_idx]

        if self._view_type == "cm":
            info = self._file_map[view]
            filename = f"SPM{sp:02d}_TM{tp:06d}_CM{info['camera']:02d}_CHN{info['channel']:02d}{self._file_ext}"
        else:  # ch
            filename = f"SPM{sp:02d}_TM{tp:06d}_CHN{view:02d}{self._file_ext}"

        return tm_folder / filename

    def _read_volume(self, t_idx: int, view_idx: int) -> np.ndarray:
        """Read 3D volume for timepoint and view."""
        cache_key = (t_idx, view_idx)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._get_file_path(t_idx, view_idx)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if self._file_ext == ".tif":
            import tifffile
            data = tifffile.imread(str(path))
        elif self._file_ext == ".klb":
            import pyklb
            data = pyklb.readfull(str(path))
        elif self._file_ext == ".zarr":
            import zarr
            z = zarr.open(path, mode="r")
            if isinstance(z, zarr.Group):
                arr = z["0"] if "0" in z else list(z.values())[0]
            else:
                arr = z
            data = arr[:]
        else:
            raise ValueError(f"Unsupported file type: {self._file_ext}")

        # ensure 3d
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._cache[cache_key] = data
        return data

    def _detect_view_dim(self) -> str:
        """Auto-detect view dimension name from file pattern."""
        if self._view_type == "cm":
            return "cm"
        elif self._view_type == "ch":
            return "ch"
        return "view"

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape: (T, Z, Views, Y, X) or (Z, Views, Y, X)."""
        z, y, x = self._single_shape
        if self._single_timepoint:
            return (z, len(self._views), y, x)
        return (len(self.tm_folders), z, len(self._views), y, x)

    @property
    def dtype(self):
        """Array data type."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return 4 if self._single_timepoint else 5

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(np.prod(self.shape))

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension labels for sliders.

        Spatial dims (Y, X) are uppercase per mbo_utilities convention.
        Slider dims (t, z, cm/ch) are lowercase for fastplotlib.
        """
        if self._single_timepoint:
            return ("z", self._view_dim, "Y", "X")
        return ("t", "z", self._view_dim, "Y", "X")

    @property
    def views(self) -> list[int]:
        """List of view indices (camera or channel numbers)."""
        return self._views

    @property
    def num_views(self) -> int:
        """Number of views."""
        return len(self._views)

    @property
    def num_timepoints(self) -> int:
        """Number of timepoints."""
        return len(self.tm_folders)

    @property
    def num_planes(self) -> int:
        """Number of Z-planes."""
        return self._single_shape[0]

    @property
    def metadata(self) -> dict:
        """Return metadata as dict."""
        meta = dict(self._metadata)

        meta["num_timepoints"] = self.num_timepoints
        meta["nframes"] = self.num_timepoints
        meta["num_frames"] = self.num_timepoints
        meta["Ly"] = self._single_shape[1]
        meta["Lx"] = self._single_shape[2]
        meta["nplanes"] = self._single_shape[0]
        meta["num_planes"] = self._single_shape[0]
        meta["views"] = self._views
        meta["view_dim"] = self._view_dim
        meta["shape"] = self.shape
        meta["file_type"] = self._file_ext
        meta["single_timepoint"] = self._single_timepoint
        meta["specimen"] = self._specimen

        return meta

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata.update(value)

    def __len__(self) -> int:
        """Length is first dimension (T or Z)."""
        return self.shape[0]

    def __getitem__(self, key):
        """Index the array: (T, Z, Views, Y, X) or (Z, Views, Y, X)."""
        if not isinstance(key, tuple):
            key = (key,)

        def to_indices(k, max_val):
            if isinstance(k, int):
                if k < 0:
                    k = max_val + k
                return [k]
            if isinstance(k, slice):
                return list(range(*k.indices(max_val)))
            if isinstance(k, (list, np.ndarray)):
                return list(k)
            return list(range(max_val))

        if self._single_timepoint:
            key = key + (slice(None),) * (4 - len(key))
            z_key, view_key, y_key, x_key = key
            t_indices = [0]
            t_key = 0
        else:
            key = key + (slice(None),) * (5 - len(key))
            t_key, z_key, view_key, y_key, x_key = key
            t_indices = to_indices(t_key, len(self.tm_folders))

        z_indices = to_indices(z_key, self._single_shape[0])
        view_indices = to_indices(view_key, len(self._views))

        out_shape = (
            len(t_indices),
            len(z_indices),
            len(view_indices),
            *self._single_shape[1:],
        )

        if isinstance(y_key, int):
            out_shape = (*out_shape[:3], 1, out_shape[4])
        elif isinstance(y_key, slice):
            y_size = len(range(*y_key.indices(self._single_shape[1])))
            out_shape = (*out_shape[:3], y_size, out_shape[4])

        if isinstance(x_key, int):
            out_shape = (*out_shape[:4], 1)
        elif isinstance(x_key, slice):
            x_size = len(range(*x_key.indices(self._single_shape[2])))
            out_shape = (*out_shape[:4], x_size)

        result = np.empty(out_shape, dtype=self._dtype)

        for ti, t_idx in enumerate(t_indices):
            for vi, view_idx in enumerate(view_indices):
                data = self._read_volume(t_idx, view_idx)
                sliced = data[z_key, y_key, x_key]

                if isinstance(z_key, int):
                    sliced = sliced[np.newaxis, ...]
                if isinstance(y_key, int):
                    sliced = sliced[:, np.newaxis, :]
                if isinstance(x_key, int):
                    sliced = sliced[:, :, np.newaxis]

                result[ti, :, vi, ...] = sliced

        # squeeze singleton dimensions
        if self._single_timepoint:
            result = np.squeeze(result, axis=0)
            int_indexed = [
                isinstance(z_key, int),
                isinstance(view_key, int),
                isinstance(y_key, int),
                isinstance(x_key, int),
            ]
            for ax in range(3, -1, -1):
                if int_indexed[ax] and ax < result.ndim and result.shape[ax] == 1:
                    result = np.squeeze(result, axis=ax)
        else:
            int_indexed = [
                isinstance(t_key, int),
                isinstance(z_key, int),
                isinstance(view_key, int),
                isinstance(y_key, int),
                isinstance(x_key, int),
            ]
            for ax in range(4, -1, -1):
                if int_indexed[ax] and ax < result.ndim and result.shape[ax] == 1:
                    result = np.squeeze(result, axis=ax)

        return result

    def __array__(self) -> np.ndarray:
        """Materialize full array into memory."""
        return self[:]

    @property
    def filenames(self) -> list[Path]:
        """Source file paths."""
        files = []
        for ti in range(len(self.tm_folders)):
            for vi in range(len(self._views)):
                path = self._get_file_path(ti, vi)
                if path.exists():
                    files.append(path)
        return files

    def close(self) -> None:
        """Release resources (clear cache)."""
        self._cache.clear()

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
        """Write array to disk."""
        from mbo_utilities.arrays._base import _imwrite_base

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

    def __repr__(self):
        return (
            f"IsoViewOutputArray(shape={self.shape}, dtype={self.dtype}, "
            f"views={self._views}, dims={self.dims}, file_type='{self._file_ext}')"
        )