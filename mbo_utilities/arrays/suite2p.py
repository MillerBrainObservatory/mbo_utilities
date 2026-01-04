"""
Suite2p binary array reader.

This module provides Suite2pArray for reading Suite2p binary output files
(data.bin, data_raw.bin) with their associated ops.npy metadata.

Automatically detects single-plane vs multi-plane (volume) directories.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin
from mbo_utilities.arrays.features import DimLabels

from mbo_utilities.metadata import get_param
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline
from mbo_utilities.util import load_npy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# register suite2p pipeline info
_SUITE2P_INFO = PipelineInfo(
    name="suite2p",
    description="Suite2p binary output files (registered/raw)",
    input_patterns=[
        "**/ops.npy",
        "**/data.bin",
        "**/data_raw.bin",
    ],
    output_patterns=[
        "**/ops.npy",
        "**/data.bin",
        "**/data_raw.bin",
        "**/stat.npy",
        "**/iscell.npy",
        "**/spks.npy",
        "**/F.npy",
        "**/Fneu.npy",
    ],
    input_extensions=["npy", "bin"],
    output_extensions=["npy", "bin"],
    marker_files=["ops.npy"],
    category="segmentation",
)
register_pipeline(_SUITE2P_INFO)

logger = log.get("arrays.suite2p")


def _extract_plane_number(name: str) -> int | None:
    """Extract plane number from directory name like 'plane01_stitched' or 'plane14'."""
    match = re.search(r"plane(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def find_suite2p_plane_dirs(directory: Path) -> list[Path]:
    """
    Find Suite2p plane directories in a parent directory.

    Looks for subdirectories containing ops.npy files, sorted by plane number.

    Parameters
    ----------
    directory : Path
        Parent directory to search.

    Returns
    -------
    list[Path]
        List of plane directories sorted by plane number.
    """
    plane_dirs = []
    for subdir in directory.iterdir():
        if subdir.is_dir():
            ops_file = subdir / "ops.npy"
            if ops_file.exists():
                plane_dirs.append(subdir)

    # Sort by plane number extracted from directory name
    def sort_key(p):
        num = _extract_plane_number(p.name)
        return num if num is not None else float("inf")

    return sorted(plane_dirs, key=sort_key)


class _SinglePlaneReader:
    """Internal reader for a single Suite2p plane."""

    def __init__(self, ops_path: Path, use_raw: bool = False):
        self.ops_path = ops_path
        self.metadata = load_npy(ops_path).item()

        ops_dir = ops_path.parent
        self.raw_file = ops_dir / "data_raw.bin"
        self.reg_file = ops_dir / "data.bin"

        # choose which file to use
        if use_raw and self.raw_file.exists():
            self.active_file = self.raw_file
        elif self.reg_file.exists():
            self.active_file = self.reg_file
        elif self.raw_file.exists():
            self.active_file = self.raw_file
        else:
            raise FileNotFoundError(
                f"No binary files found in {ops_dir}\n"
                f"Expected either:\n"
                f"  - {self.reg_file} (registered)\n"
                f"  - {self.raw_file} (raw)"
            )

        self.Ly = get_param(self.metadata, "Ly")
        self.Lx = get_param(self.metadata, "Lx")
        self.nframes = get_param(self.metadata, "nframes")
        self.shape = (self.nframes, self.Ly, self.Lx)
        self.dtype = np.int16

        # validate file size
        expected_bytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        actual_bytes = self.active_file.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"Binary file {self.active_file.name} is too small!\n"
                f"Expected: {expected_bytes:,} bytes for shape {self.shape}\n"
                f"Actual: {actual_bytes:,} bytes"
            )
        if actual_bytes > expected_bytes:
            warnings.warn(
                f"Binary file {self.active_file.name} is larger than expected.\n"
                f"Expected: {expected_bytes:,} bytes\n"
                f"Actual: {actual_bytes:,} bytes\nExtra data will be ignored.",
                UserWarning, stacklevel=2,
            )

        self._file = np.memmap(
            self.active_file, mode="r", dtype=self.dtype, shape=self.shape
        )

    def switch_channel(self, use_raw: bool = False):
        new_file = self.raw_file if use_raw else self.reg_file
        if not new_file.exists():
            raise FileNotFoundError(new_file)
        self._file = np.memmap(new_file, mode="r", dtype=self.dtype, shape=self.shape)
        self.active_file = new_file

    def __getitem__(self, key):
        return self._file[key]

    def close(self):
        self._file._mmap.close()


class Suite2pArray(ReductionMixin):
    """
    Lazy array reader for Suite2p binary output files.

    Auto-detects single-plane vs multi-plane (volumetric) data:
    - Single plane: reads ops.npy + data.bin from one directory → 3D (T, Y, X)
    - Volume: reads multiple plane directories → 4D (T, Z, Y, X)

    Parameters
    ----------
    filename : str or Path
        Path to ops.npy, a .bin file, or a directory containing plane subdirs.
    use_raw : bool, optional
        If True, prefer data_raw.bin over data.bin. Default False.
    dims : str | Sequence[str] | None, optional
        Dimension labels. If None, inferred from shape (TZYX or TYX).

    Attributes
    ----------
    shape : tuple
        Shape as (T, Y, X) for single plane or (T, Z, Y, X) for volume.
    dtype : np.dtype
        Data type (always np.int16 for Suite2p).
    metadata : dict
        Contents of ops.npy (first plane for volumes).
    is_volumetric : bool
        True if this represents multi-plane data.
    num_planes : int
        Number of Z-planes (1 for single plane).
    dims : tuple[str, ...]
        Dimension labels.

    Examples
    --------
    >>> # Single plane
    >>> arr = Suite2pArray("suite2p_output/plane0/ops.npy")
    >>> arr.shape
    (10000, 512, 512)

    >>> # Volume (auto-detected from directory structure)
    >>> arr = Suite2pArray("suite2p_output/")
    >>> arr.shape
    (10000, 14, 512, 512)
    >>> arr.is_volumetric
    True
    """

    def __init__(
        self,
        filename: str | Path,
        use_raw: bool = False,
        dims: str | Sequence[str] | None = None,
    ):
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)

        self._use_raw = use_raw
        self._planes: list[_SinglePlaneReader] = []
        self._is_volumetric = False

        # determine if this is a volume or single plane
        if path.is_dir():
            # check for plane subdirectories
            plane_dirs = find_suite2p_plane_dirs(path)
            if plane_dirs:
                self._init_volume(plane_dirs, use_raw)
            else:
                # maybe directory contains ops.npy directly
                ops_file = path / "ops.npy"
                if ops_file.exists():
                    self._init_single_plane(ops_file, use_raw)
                else:
                    raise ValueError(
                        f"Directory {path} has no plane subdirectories or ops.npy"
                    )
        elif path.suffix == ".npy" and path.stem == "ops":
            # check if parent has sibling plane directories (volume)
            parent = path.parent.parent
            plane_dirs = find_suite2p_plane_dirs(parent)
            if len(plane_dirs) > 1:
                self._init_volume(plane_dirs, use_raw)
            else:
                self._init_single_plane(path, use_raw)
        elif path.suffix == ".bin":
            # user explicitly selected a .bin file - load only that plane
            ops_path = path.with_name("ops.npy")
            if not ops_path.exists():
                raise FileNotFoundError(f"Missing ops.npy near {path}")
            self._init_single_plane(ops_path, use_raw)
        else:
            raise ValueError(f"Unsupported input: {path}")

        # initialize dimension labels
        self._dim_labels = DimLabels(dims, ndim=self.ndim)

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension labels."""
        return self._dim_labels.value



    def _init_single_plane(self, ops_path: Path, use_raw: bool):
        """Initialize as single-plane array."""
        self._is_volumetric = False
        reader = _SinglePlaneReader(ops_path, use_raw)
        self._planes = [reader]

        self._nframes = reader.nframes
        self._ly = reader.Ly
        self._lx = reader.Lx
        self._dtype = reader.dtype
        self._metadata = dict(reader.metadata)
        self.num_rois = self._metadata.get("num_rois", 1)
        self.filenames = [reader.active_file]

    def _init_volume(self, plane_dirs: list[Path], use_raw: bool):
        """Initialize as volumetric array."""
        self._is_volumetric = True

        for pdir in plane_dirs:
            ops_file = pdir / "ops.npy"
            reader = _SinglePlaneReader(ops_file, use_raw)
            self._planes.append(reader)

        # validate consistent shapes
        shapes = [(p.Ly, p.Lx) for p in self._planes]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent spatial shapes across planes: {shapes}")

        nframes = [p.nframes for p in self._planes]
        if len(set(nframes)) != 1:
            logger.warning(
                f"Inconsistent frame counts across planes: {nframes}. "
                f"Using minimum: {min(nframes)}"
            )

        self._nframes = min(nframes)
        self._nz = len(self._planes)
        self._ly, self._lx = shapes[0]
        self._dtype = self._planes[0].dtype

        # aggregate metadata from first plane
        self._metadata = dict(self._planes[0].metadata)
        self._metadata["nplanes"] = self._nz
        self._metadata["num_planes"] = self._nz  # alias for backward compat
        self._metadata["plane_dirs"] = [str(p.ops_path.parent) for p in self._planes]
        self.num_rois = get_param(self._metadata, "num_rois", default=1)
        self.filenames = [p.active_file for p in self._planes]

        logger.info(
            f"Loaded Suite2p volume: {self._nframes} frames, {self._nz} planes, "
            f"{self._ly}x{self._lx} px"
        )

    @property
    def is_volumetric(self) -> bool:
        """True if this array represents multi-plane data."""
        return self._is_volumetric

    @property
    def num_planes(self) -> int:
        """Number of Z-planes."""
        return len(self._planes)

    @property
    def shape(self) -> tuple:
        if self._is_volumetric:
            return (self._nframes, self._nz, self._ly, self._lx)
        return (self._nframes, self._ly, self._lx)

    @property
    def ndim(self) -> int:
        return 4 if self._is_volumetric else 3

    @property
    def dtype(self):
        from mbo_utilities.util import get_dtype
        return self._target_dtype if hasattr(self, "_target_dtype") and self._target_dtype else get_dtype(self._dtype)

    @property
    def metadata(self) -> dict:
        """Return metadata as dict. Always returns dict, never None."""
        return self._metadata if self._metadata is not None else {}

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame."""
        if not hasattr(self, "_cached_vmin"):
            if self._is_volumetric:
                frame = np.asarray(self[0, 0])
            else:
                frame = np.asarray(self[0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        """Min from first frame for display."""
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        """Max from first frame for display."""
        self._compute_frame_vminmax()
        return self._cached_vmax

    def __len__(self) -> int:
        return self._nframes

    def __getitem__(self, key):
        if self._is_volumetric:
            return self._getitem_volume(key)
        return self._getitem_single(key)

    def _getitem_single(self, key):
        """Index single-plane array."""
        out = self._planes[0][key]
        if hasattr(self, "_target_dtype") and self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def _getitem_volume(self, key):
        """Index volumetric array."""
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        t_key, z_key, y_key, x_key = key

        # normalize t_key
        if isinstance(t_key, slice):
            start, stop, step = t_key.indices(self._nframes)
            t_key = slice(start, stop, step)
        elif isinstance(t_key, int):
            if t_key < 0:
                t_key = self._nframes + t_key
            if t_key >= self._nframes:
                raise IndexError(f"Time index {t_key} out of bounds for {self._nframes} frames")

        # handle z indexing
        if isinstance(z_key, int):
            if z_key < 0:
                z_key = self._nz + z_key
            if z_key < 0 or z_key >= self._nz:
                raise IndexError(f"Z index {z_key} out of bounds for {self._nz} planes")
            out = self._planes[z_key][t_key, y_key, x_key]
        else:
            if isinstance(z_key, slice):
                z_indices = range(self._nz)[z_key]
            elif isinstance(z_key, (list, np.ndarray)):
                z_indices = z_key
            else:
                z_indices = range(self._nz)

            arrs = [self._planes[i][t_key, y_key, x_key] for i in z_indices]
            out = np.stack(arrs, axis=1)

        if hasattr(self, "_target_dtype") and self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def __array__(self) -> np.ndarray:
        """Materialize array into memory."""
        if self._is_volumetric:
            arrs = [p._file[:self._nframes] for p in self._planes]
            return np.stack(arrs, axis=1)
        n = min(10, self._nframes)
        return np.stack([self._planes[0][i] for i in range(n)], axis=0)

    def switch_channel(self, use_raw: bool = False):
        """Switch all planes between raw and registered data."""
        for plane in self._planes:
            plane.switch_channel(use_raw=use_raw)
        self.filenames = [p.active_file for p in self._planes]

    def close(self):
        """Close all memory-mapped files."""
        for plane in self._planes:
            plane.close()

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
        """Write Suite2pArray to disk in various formats."""
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

    def imshow(self, **kwargs):
        """Display array using fastplotlib ImageWidget."""
        arrays = []
        names = []

        if not self._is_volumetric:
            # single plane - try to show both raw and registered
            plane = self._planes[0]
            if plane.raw_file.exists():
                try:
                    raw_reader = _SinglePlaneReader(plane.ops_path, use_raw=True)
                    raw_arr = Suite2pArray.__new__(Suite2pArray)
                    raw_arr._planes = [raw_reader]
                    raw_arr._is_volumetric = False
                    raw_arr._nframes = raw_reader.nframes
                    raw_arr._ly = raw_reader.Ly
                    raw_arr._lx = raw_reader.Lx
                    raw_arr._dtype = raw_reader.dtype
                    raw_arr._metadata = {}
                    arrays.append(raw_arr)
                    names.append("raw")
                except Exception as e:
                    logger.warning(f"Could not open raw file: {e}")

            if plane.reg_file.exists():
                try:
                    reg_reader = _SinglePlaneReader(plane.ops_path, use_raw=False)
                    reg_arr = Suite2pArray.__new__(Suite2pArray)
                    reg_arr._planes = [reg_reader]
                    reg_arr._is_volumetric = False
                    reg_arr._nframes = reg_reader.nframes
                    reg_arr._ly = reg_reader.Ly
                    reg_arr._lx = reg_reader.Lx
                    reg_arr._dtype = reg_reader.dtype
                    reg_arr._metadata = {}
                    arrays.append(reg_arr)
                    names.append("registered")
                except Exception as e:
                    logger.warning(f"Could not open registered file: {e}")

            if not arrays:
                arrays.append(self)
                names.append(self._planes[0].active_file.name)
        else:
            arrays.append(self)
            names.append("volume")

        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        histogram_widget = kwargs.get("histogram_widget", True)
        window_funcs = kwargs.get("window_funcs")

        import fastplotlib as fpl

        return fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            figure_shape=(1, len(arrays)),
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )


def _add_suite2p_labels(
    root_group,
    suite2p_dirs: list[Path],
    T: int,
    Z: int,
    Y: int,
    X: int,
    dtype,
    compression_level: int,
):
    """
    Add Suite2p segmentation masks as OME-Zarr labels.

    Creates a 'labels' subgroup with ROI masks from Suite2p stat.npy files.
    Follows OME-NGFF v0.5 labels specification.
    """
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    logger.info("Creating labels array from Suite2p masks...")

    labels_group = root_group.create_group("labels", overwrite=True)

    label_codecs = [BytesCodec(), GzipCodec(level=compression_level)]
    masks = zarr.create(
        store=labels_group.store,
        path="labels/0",
        shape=(Z, Y, X),
        chunks=(1, Y, X),
        dtype=np.uint32,
        codecs=label_codecs,
        overwrite=True,
    )

    roi_id = 1

    for zi, s2p_dir in enumerate(suite2p_dirs):
        stat_path = s2p_dir / "stat.npy"
        iscell_path = s2p_dir / "iscell.npy"

        if not stat_path.exists():
            logger.warning(f"stat.npy not found in {s2p_dir}, skipping z={zi}")
            continue

        stat = load_npy(stat_path)

        if iscell_path.exists():
            iscell = load_npy(iscell_path)[:, 0].astype(bool)
        else:
            iscell = np.ones(len(stat), dtype=bool)

        plane_mask = np.zeros((Y, X), dtype=np.uint32)

        for _roi_idx, (roi_stat, is_cell) in enumerate(zip(stat, iscell, strict=False)):
            if not is_cell:
                continue

            ypix = roi_stat.get("ypix", [])
            xpix = roi_stat.get("xpix", [])

            if len(ypix) == 0 or len(xpix) == 0:
                continue

            ypix = np.clip(ypix, 0, Y - 1)
            xpix = np.clip(xpix, 0, X - 1)

            plane_mask[ypix, xpix] = roi_id
            roi_id += 1

        masks[zi, :, :] = plane_mask
        logger.debug(f"Added {(plane_mask > 0).sum()} labeled pixels for z-plane {zi + 1}/{Z}")

    labels_metadata = {
        "version": "0.5",
        "labels": ["0"],
    }

    label_array_meta = {
        "version": "0.5",
        "image-label": {
            "version": "0.5",
            "colors": [],
            "source": {"image": "../../0"},
        },
    }

    labels_group.attrs.update(labels_metadata)
    labels_group["0"].attrs.update(label_array_meta)

    logger.info(f"Added {roi_id - 1} total ROIs across {Z} z-planes")


def load_ops(ops_input: str | Path | list[str | Path]):
    """Simple utility to load a suite2p npy file."""
    if isinstance(ops_input, (str, Path)):
        return load_npy(ops_input).item()
    if isinstance(ops_input, dict):
        return ops_input
    logger.warning("No valid ops file provided, returning empty dict.")
    return {}


