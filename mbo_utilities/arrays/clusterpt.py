"""Lazy array loader for clusterPT (IsoView-Processing) KLB output files."""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# register clusterpt pipeline info
_CLUSTERPT_INFO = PipelineInfo(
    name="clusterpt",
    description="IsoView-Processing clusterPT corrected data (KLB)",
    input_patterns=[
        "**/TM??????/SPM??_TM??????_CM??_CHN??.klb",
        "**/SPM??/*.corrected/*/TM??????/",
    ],
    output_patterns=[],
    input_extensions=["klb"],
    output_extensions=[],
    marker_files=["*.configuration.mat"],
    category="reader",
)
register_pipeline(_CLUSTERPT_INFO)


def _check_pyklb():
    """Check if pyklb is available."""
    try:
        import pyklb
        return pyklb
    except ImportError:
        raise ImportError(
            "pyklb is required for KLB files. Install with: "
            "pip install mbo_utilities[isoview]"
        )


class ClusterPTArray:
    """
    Lazy loader for clusterPT (IsoView-Processing) KLB output files.

    Conforms to LazyArrayProtocol for compatibility with mbo_utilities imread/imwrite.

    clusterPT output structure:
        outputFolder/
            TM000000/
                SPM00_TM000000_CM00_CHN00.klb
                SPM00_TM000000_CM01_CHN00.klb
                SPM00_TM000000.configuration.mat
                SPM00_TM000000_CHN00.xml
            TM000001/
                ...

    Shape: (T, Z, Views, Y, X) for multi-timepoint, (Z, Views, Y, X) for single.
    Views are (camera, channel) combinations.

    Parameters
    ----------
    path : str or Path
        Path to directory containing TM* folders with .klb files.

    Examples
    --------
    >>> arr = ClusterPTArray("path/to/corrected")
    >>> arr.shape
    (100, 38, 2, 1848, 768)  # (t, z, views, y, x)
    >>> arr.dims
    ('t', 'z', 'cm', 'y', 'x')
    >>> arr.views
    [(0, 0), (1, 0)]  # (camera, channel) per view
    >>> frame = arr[0, 10, 0]  # t=0, z=10, view=0
    """

    def __init__(self, path: str | Path):
        self.base_path = Path(path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        # check for pyklb
        _check_pyklb()

        # detect if single TM or multi-TM
        if self.base_path.name.startswith("TM"):
            klb_files = list(self.base_path.glob("*.klb"))
            if klb_files:
                self._single_timepoint = True
                self.tm_folders = [self.base_path]
            else:
                raise ValueError(f"TM folder {self.base_path} contains no .klb files")
        else:
            self.tm_folders = sorted(
                [d for d in self.base_path.iterdir()
                 if d.is_dir() and d.name.startswith("TM")],
                key=lambda x: int(x.name[2:])
            )
            self._single_timepoint = len(self.tm_folders) == 1

            if not self.tm_folders:
                raise ValueError(f"No TM* folders found in {self.base_path}")

        # discover views from first timepoint
        self._discover_views(self.tm_folders[0])

        # cache for klb data: (t_idx, view_idx) -> ndarray
        self._cache = {}
        self._metadata = {}

    def _discover_views(self, tm_folder: Path):
        """Parse KLB files to find camera/channel combinations and shape."""
        import pyklb

        # pattern: SPM00_TM000000_CM00_CHN00.klb
        pattern = re.compile(
            r"SPM(\d+)_TM(\d+)_CM(\d+)_CHN(\d+)\.klb"
        )

        klb_files = sorted(tm_folder.glob("*.klb"))
        if not klb_files:
            raise ValueError(f"No .klb files in {tm_folder}")

        self._views = []  # [(camera, channel), ...]
        self._specimen = None

        for kf in klb_files:
            match = pattern.match(kf.name)
            if not match:
                # skip mask files, projections, etc
                continue

            specimen, timepoint, camera, channel = map(int, match.groups())
            self._specimen = specimen

            view = (camera, channel)
            if view not in self._views:
                self._views.append(view)

        if not self._views:
            raise ValueError(f"No valid KLB data files in {tm_folder}")

        # sort views by (camera, channel)
        self._views = sorted(self._views)

        # read first file to get shape/dtype
        first_view = self._views[0]
        first_file = self._get_klb_path(tm_folder, first_view)
        header = pyklb.readheader(str(first_file))

        # klb stores as (x, y, z) so we need to reverse
        # header['imagesize_tczyx'] gives size in t,c,z,y,x order
        # but for our 3d volumes it's typically just spatial dims
        dims = header['imagesize_tczyx']
        # filter out singleton dimensions at front
        spatial_dims = [d for d in dims if d > 1]
        if len(spatial_dims) >= 3:
            self._single_shape = tuple(spatial_dims[-3:])  # (z, y, x)
        else:
            # might be 2d, pad with 1
            self._single_shape = (1,) * (3 - len(spatial_dims)) + tuple(spatial_dims)

        self._dtype = np.dtype(header['datatype'])

        logger.info(
            f"ClusterPTArray: timepoints={len(self.tm_folders)}, views={len(self._views)}, "
            f"shape={self._single_shape}, dtype={self._dtype}"
        )

    def _get_klb_path(self, tm_folder: Path, view: tuple[int, int]) -> Path:
        """Get path to KLB file for a timepoint folder and view."""
        camera, channel = view
        # extract timepoint from folder name
        tp = int(tm_folder.name[2:])
        sp = self._specimen if self._specimen is not None else 0

        filename = f"SPM{sp:02d}_TM{tp:06d}_CM{camera:02d}_CHN{channel:02d}.klb"
        return tm_folder / filename

    def _read_klb(self, t_idx: int, view_idx: int) -> np.ndarray:
        """Read KLB file for timepoint and view index."""
        cache_key = (t_idx, view_idx)
        if cache_key in self._cache:
            return self._cache[cache_key]

        import pyklb

        view = self._views[view_idx]
        tm_folder = self.tm_folders[t_idx]
        klb_path = self._get_klb_path(tm_folder, view)

        if not klb_path.exists():
            raise FileNotFoundError(f"KLB file not found: {klb_path}")

        # pyklb returns array in (z, y, x) order
        data = pyklb.readfull(str(klb_path))

        # ensure 3d
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._cache[cache_key] = data
        return data

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Array shape.

        Returns
        -------
        - Single TM: (Z, Views, Y, X) - 4D
        - Multi TM: (T, Z, Views, Y, X) - 5D
        """
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
    def metadata(self) -> dict:
        """Return metadata as dict."""
        meta = dict(self._metadata)

        # standard keys
        num_t = self.num_timepoints
        meta["num_timepoints"] = num_t
        meta["nframes"] = num_t
        meta["num_frames"] = num_t
        meta["Ly"] = self._single_shape[1]
        meta["Lx"] = self._single_shape[2]
        meta["nplanes"] = self._single_shape[0]
        meta["num_planes"] = self._single_shape[0]
        meta["views"] = self._views
        meta["shape"] = self.shape
        meta["structure"] = "clusterpt"
        meta["single_timepoint"] = self._single_timepoint
        meta["specimen"] = self._specimen

        return meta

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata.update(value)

    @property
    def views(self) -> list[tuple[int, int]]:
        """List of (camera, channel) tuples for each view index."""
        return self._views

    @property
    def num_views(self) -> int:
        """Number of camera/channel views."""
        return len(self._views)

    @property
    def num_timepoints(self) -> int:
        """Number of timepoints."""
        return len(self.tm_folders)

    def __len__(self) -> int:
        """Length is first dimension (T or Z)."""
        return self.shape[0]

    def view_index(self, camera: int, channel: int) -> int:
        """Get view index for a specific camera/channel combination."""
        try:
            return self._views.index((camera, channel))
        except ValueError:
            raise ValueError(
                f"No view for camera={camera}, channel={channel}. "
                f"Available views: {self._views}"
            )

    def __getitem__(self, key):
        """
        Index the array.

        Shape:
        - Single TM: (Z, Views, Y, X) - 4D
        - Multi TM: (T, Z, Views, Y, X) - 5D
        """
        if not isinstance(key, tuple):
            key = (key,)

        def to_indices(k, max_val):
            """Convert key to list of indices."""
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
            # 4D indexing: (Z, Views, Y, X)
            key = key + (slice(None),) * (4 - len(key))
            z_key, view_key, y_key, x_key = key
            t_indices = [0]
            t_key = 0
        else:
            # 5D indexing: (T, Z, Views, Y, X)
            key = key + (slice(None),) * (5 - len(key))
            t_key, z_key, view_key, y_key, x_key = key
            t_indices = to_indices(t_key, len(self.tm_folders))

        z_indices = to_indices(z_key, self._single_shape[0])
        view_indices = to_indices(view_key, len(self._views))

        # build output array (5D internally)
        out_shape = (
            len(t_indices),
            len(z_indices),
            len(view_indices),
            *self._single_shape[1:],  # Y, X
        )

        # handle Y, X slicing
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
                data = self._read_klb(t_idx, view_idx)

                # index (Z, Y, X)
                sliced = data[z_key, y_key, x_key]

                # handle dimension reduction
                if isinstance(z_key, int):
                    sliced = sliced[np.newaxis, ...]
                if isinstance(y_key, int):
                    sliced = sliced[:, np.newaxis, :]
                if isinstance(x_key, int):
                    sliced = sliced[:, :, np.newaxis]

                result[ti, :, vi, ...] = sliced

        # squeeze singleton dimensions from integer indexing
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
        """Source file paths for LazyArrayProtocol."""
        files = []
        for tm in self.tm_folders:
            for view in self._views:
                path = self._get_klb_path(tm, view)
                if path.exists():
                    files.append(path)
        return files

    @property
    def dims(self) -> tuple[str, ...]:
        """
        Dimension labels for LazyArrayProtocol.

        Returns
        -------
        tuple[str, ...]
            ('z', 'cm', 'y', 'x') for single timepoint
            ('t', 'z', 'cm', 'y', 'x') for multi-timepoint
        """
        if self._single_timepoint:
            return ("z", "cm", "y", "x")
        return ("t", "z", "cm", "y", "x")

    @property
    def num_planes(self) -> int:
        """Number of Z-planes."""
        return self._single_shape[0]

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
        """Write ClusterPTArray to disk."""
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
            f"ClusterPTArray(shape={self.shape}, dtype={self.dtype}, "
            f"views={self._views}, timepoints={len(self.tm_folders)})"
        )
