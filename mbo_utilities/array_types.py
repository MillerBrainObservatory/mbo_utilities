from __future__ import annotations

import copy
import os
import tempfile
import threading
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, List, Sequence

import h5py
import numpy as np
import tifffile
from dask import array as da
from tifffile import TiffFile

from mbo_utilities import log
from mbo_utilities._protocols import get_dims, get_num_planes
from mbo_utilities._writers import _write_plane
from mbo_utilities.file_io import (
    _convert_range_to_slice,
    expand_paths,
    derive_tag_from_filename,
)
from mbo_utilities.metadata import get_metadata
from mbo_utilities.phasecorr import ALL_PHASECORR_METHODS, bidir_phasecorr
from mbo_utilities.util import subsample_array, listify_index

logger = log.get("array_types")

CHUNKS_4D = {0: 1, 1: "auto", 2: -1, 3: -1}
CHUNKS_3D = {0: 1, 1: -1, 2: -1}


def supports_roi(obj):
    return hasattr(obj, "roi") and hasattr(obj, "num_rois")


def normalize_roi(value):
    """Return ROI as None, int, or list[int] with consistent semantics."""
    if value in (None, (), [], False):
        return None
    if value is True:
        return 0  # “split ROIs” GUI flag
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def iter_rois(obj):
    """Yield ROI indices based on MBO semantics.

    - roi=None → yield None (stitched full-FOV image)
    - roi=0 → yield each ROI index from 1..num_rois (split all)
    - roi=int > 0 → yield that ROI only
    - roi=list/tuple → yield each element (as given)
    """
    if not supports_roi(obj):
        yield None
        return

    roi = getattr(obj, "roi", None)
    num_rois = getattr(obj, "num_rois", 1)

    if roi is None:
        yield None
    elif roi == 0:
        yield from range(1, num_rois + 1)
    elif isinstance(roi, int):
        yield roi
    elif isinstance(roi, (list, tuple)):
        for r in roi:
            if r == 0:
                yield from range(1, num_rois + 1)
            else:
                yield r


def _normalize_planes(planes, num_planes: int) -> list[int]:
    """
    Normalize planes argument to 0-indexed list.

    Parameters
    ----------
    planes : int | list | tuple | None
        Planes to write (1-based indexing from user).
    num_planes : int
        Total number of planes available.

    Returns
    -------
    list[int]
        0-indexed plane indices.
    """
    if planes is None:
        return list(range(num_planes))
    if isinstance(planes, int):
        return [planes - 1]  # 1-based to 0-based
    return [p - 1 for p in planes]


def _build_output_path(
    outpath: Path,
    plane_idx: int,
    roi: int | None,
    ext: str,
    output_name: str | None = None,
    structural: bool = False,
    has_multiple_rois: bool = False,
    **kwargs,
) -> Path:
    """
    Build output file path for a single plane.

    Parameters
    ----------
    outpath : Path
        Base output directory.
    plane_idx : int
        0-indexed plane number.
    roi : int | None
        ROI index (1-based) or None for stitched/single ROI.
    ext : str
        File extension (without dot).
    output_name : str | None
        Override output filename (for .bin files).
    structural : bool
        If True, use data_chan2.bin naming for structural channel.
    has_multiple_rois : bool
        If True and roi is None, use "_stitched" suffix.

    Returns
    -------
    Path
        Full output file path.
    """
    plane_num = plane_idx + 1  # Convert to 1-based for filenames

    # Determine suffix based on ROI
    if roi is None:
        roi_suffix = "_stitched" if has_multiple_rois else ""
    else:
        roi_suffix = f"_roi{roi}"

    if ext == "bin":
        if output_name:
            # Caller specified exact output - use it directly
            if structural:
                return outpath / "data_chan2.bin"
            return outpath / output_name

        # Build subdirectory structure
        subdir = f"plane{plane_num:02d}{roi_suffix}"
        plane_dir = outpath / subdir
        plane_dir.mkdir(parents=True, exist_ok=True)

        if structural:
            return plane_dir / "data_chan2.bin"
        return plane_dir / "data_raw.bin"
    else:
        # Non-binary formats: single file per plane
        return outpath / f"plane{plane_num:02d}{roi_suffix}.{ext}"


def _imwrite_base(
    arr,
    outpath: Path | str,
    planes: int | list | tuple | None = None,
    ext: str = ".tiff",
    overwrite: bool = False,
    target_chunk_mb: int = 50,
    progress_callback=None,
    debug: bool = False,
    roi_iterator=None,
    **kwargs,
) -> Path:
    """
    Common implementation for array _imwrite() methods.

    This function handles the common pattern of:
    1. Normalizing planes argument (1-based to 0-based)
    2. Iterating over ROIs (if applicable)
    3. Building output paths
    4. Calling _write_plane() for each plane

    Parameters
    ----------
    arr : LazyArrayProtocol
        Array to write. Must have shape, metadata, and support indexing.
    outpath : Path | str
        Output directory.
    planes : int | list | tuple | None
        Planes to write (1-based indexing). None means all planes.
    ext : str
        Output format extension (e.g., '.tiff', '.bin', '.zarr').
    overwrite : bool
        Whether to overwrite existing files.
    target_chunk_mb : int
        Target chunk size in MB for streaming writes.
    progress_callback : callable | None
        Progress callback function.
    debug : bool
        Enable debug output.
    roi_iterator : iterator | None
        Custom ROI iterator for arrays with ROI support.
        If None, uses iter_rois(arr) which yields [None] for arrays without ROIs.
    **kwargs
        Additional arguments passed to _write_plane().

    Returns
    -------
    Path
        Output directory path.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    ext_clean = ext.lower().lstrip(".")

    # Get metadata
    md = dict(arr.metadata) if arr.metadata else {}

    # Get dimensions using protocol helpers
    dims = get_dims(arr)
    num_planes = get_num_planes(arr)

    # Extract shape info
    nframes = arr.shape[0] if "T" in dims else 1
    Ly, Lx = arr.shape[-2], arr.shape[-1]

    # Update metadata
    md["Ly"] = Ly
    md["Lx"] = Lx
    md["nframes"] = nframes
    md["num_frames"] = nframes  # alias for backwards compatibility

    # Normalize planes to 0-indexed list
    planes_list = _normalize_planes(planes, num_planes)

    # Use provided ROI iterator or default
    roi_iter = roi_iterator if roi_iterator is not None else iter_rois(arr)

    # Check if array has multiple ROIs (for "_stitched" suffix)
    has_multiple_rois = getattr(arr, "num_rois", 1) > 1

    for roi in roi_iter:
        # Update array's ROI if it supports it
        if roi is not None and hasattr(arr, "roi"):
            arr.roi = roi

        for plane_idx in planes_list:
            target = _build_output_path(
                outpath, plane_idx, roi, ext_clean,
                output_name=kwargs.get("output_name"),
                structural=kwargs.get("structural", False),
                has_multiple_rois=has_multiple_rois,
            )

            if target.exists() and not overwrite:
                logger.warning(f"File {target} already exists. Skipping write.")
                continue

            # Build plane-specific metadata
            plane_md = md.copy()
            plane_md["plane"] = plane_idx + 1  # 1-based in metadata
            if roi is not None:
                plane_md["roi"] = roi
                plane_md["mroi"] = roi  # alias

            _write_plane(
                arr,
                target,
                overwrite=overwrite,
                target_chunk_mb=target_chunk_mb,
                metadata=plane_md,
                progress_callback=progress_callback,
                debug=debug,
                dshape=(nframes, Ly, Lx),
                plane_index=plane_idx,
                **kwargs,
            )

    return outpath


def _to_tzyx(a: da.Array, axes: str) -> da.Array:
    order = [ax for ax in ["T", "Z", "C", "S", "Y", "X"] if ax in axes]
    perm = [axes.index(ax) for ax in order]
    a = da.transpose(a, axes=perm)
    have_T = "T" in order
    pos = {ax: i for i, ax in enumerate(order)}
    tdim = a.shape[pos["T"]] if have_T else 1
    merge_dims = [d for d, ax in enumerate(order) if ax in ("Z", "C", "S")]
    if merge_dims:
        front = []
        if have_T:
            front.append(pos["T"])
        rest = [d for d in range(a.ndim) if d not in front]
        a = da.transpose(a, axes=front + rest)
        newshape = [
            tdim if have_T else 1,
            int(np.prod([a.shape[i] for i in rest[:-2]])),
            a.shape[-2],
            a.shape[-1],
        ]
        a = a.reshape(newshape)
    else:
        if have_T:
            if a.ndim == 3:
                a = da.expand_dims(a, 1)
        else:
            a = da.expand_dims(a, 0)
            a = da.expand_dims(a, 1)
        if order[-2:] != ["Y", "X"]:
            yx_pos = [order.index("Y"), order.index("X")]
            keep = [i for i in range(len(order)) if i not in yx_pos]
            a = da.transpose(a, axes=keep + yx_pos)
    return a


def _axes_or_guess(arr_ndim: int) -> str:
    if arr_ndim == 2:
        return "YX"
    elif arr_ndim == 3:
        return "ZYX"
    elif arr_ndim == 4:
        return "TZYX"
    else:
        return "Unknown"


def _safe_get_metadata(path: Path) -> dict:
    try:
        return get_metadata(path)
    except Exception:
        return {}


def validate_s3d_registration(s3d_job_dir: Path, num_planes: int = None) -> bool:
    """
    Validate that Suite3D registration completed successfully.

    Parameters
    ----------
    s3d_job_dir : Path
        Path to the Suite3D job directory (e.g., 's3d-preprocessed')
    num_planes : int, optional
        Expected number of planes. If provided, validates that plane_shifts has correct length.

    Returns
    -------
    bool
        True if valid registration results exist, False otherwise.
    """
    if not s3d_job_dir or not Path(s3d_job_dir).is_dir():
        return False

    s3d_job_dir = Path(s3d_job_dir)
    summary_path = s3d_job_dir / "summary" / "summary.npy"

    if not summary_path.is_file():
        logger.warning(f"Suite3D summary file not found: {summary_path}.")
        return False

    try:
        summary = np.load(summary_path, allow_pickle=True).item()

        if not isinstance(summary, dict):
            logger.warning(f"Suite3D summary is not a dict: {type(summary)}")
            return False

        if "plane_shifts" not in summary:
            logger.warning("Suite3D summary missing 'plane_shifts' key")
            return False

        plane_shifts = summary["plane_shifts"]

        if not isinstance(plane_shifts, (list, np.ndarray)):
            logger.warning(f"plane_shifts has invalid type: {type(plane_shifts)}")
            return False

        plane_shifts = np.asarray(plane_shifts)

        if plane_shifts.ndim != 2 or plane_shifts.shape[1] != 2:
            logger.warning(
                f"plane_shifts has invalid shape: {plane_shifts.shape}, expected (n_planes, 2)"
            )
            return False

        if num_planes is not None and len(plane_shifts) != num_planes:
            logger.warning(
                f"plane_shifts length {len(plane_shifts)} doesn't match expected {num_planes} planes"
            )
            return False

        logger.debug(
            f"Valid Suite3D registration found with {len(plane_shifts)} plane shifts"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to validate Suite3D registration: {e}")
        return False


def register_zplanes_s3d(
    filenames, metadata, outpath=None, progress_callback=None
) -> Path | None:
    # these are heavy imports, lazy import for now
    try:
        # https://github.com/MillerBrainObservatory/mbo_utilities/issues/35
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        from suite3d.job import Job  # noqa

        HAS_SUITE3D = True
    except ImportError:
        HAS_SUITE3D = False
        Job = None

    try:
        import cupy

        HAS_CUPY = True
    except ImportError:
        HAS_CUPY = False
        cupy = None
    if not HAS_SUITE3D:
        logger.warning(
            "Suite3D is not installed. Cannot preprocess."
            "Set register_z = False in imwrite, or install Suite3D:"
            "`pip install mbo_utilities[suite3d, cuda12] # CUDA 12.x or"
            "'pip install mbo_utilities[suite3d, cuda11] # CUDA 11.x"
        )
        return None
    if not HAS_CUPY:
        logger.warning(
            "CuPy is not installed. Cannot preprocess."
            "Set register_z = False in imwrite, or install CuPy:"
            "`pip install cupy-cuda12x` # CUDA 12.x or"
            "`pip install cupy-cuda11x` # CUDA 11.x"
        )
        return None

    if "frame_rate" not in metadata or "num_planes" not in metadata:
        logger.warning(
            "Missing required metadata for axial alignment: frame_rate / num_planes"
        )
        return None

    if outpath is not None:
        job_path = Path(outpath)
    else:
        job_path = Path(str(filenames[0].parent) + ".summary")

    job_id = metadata.get("job_id", "preprocessed")

    params = {
        "fs": metadata["frame_rate"],
        "planes": np.arange(metadata["num_planes"]),
        "n_ch_tif": metadata["num_planes"],
        "tau": metadata.get("tau", 1.3),
        "lbm": metadata.get("lbm", True),
        "fuse_strips": metadata.get("fuse_planes", False),
        "subtract_crosstalk": metadata.get("subtract_crosstalk", False),
        "init_n_frames": metadata.get("init_n_frames", 500),
        "n_init_files": metadata.get("n_init_files", 1),
        "n_proc_corr": metadata.get("n_proc_corr", 15),
        "max_rigid_shift_pix": metadata.get("max_rigid_shift_pix", 150),
        "3d_reg": metadata.get("3d_reg", True),
        "gpu_reg": metadata.get("gpu_reg", True),
        "block_size": metadata.get("block_size", [64, 64]),
    }
    if Job is None:
        logger.warning("Suite3D Job class not available.")
        return None

    job = Job(
        str(job_path),
        job_id,
        create=True,
        overwrite=True,
        verbosity=-1,
        tifs=filenames,
        params=params,
        progress_callback=progress_callback,
    )
    job._report(0.01, "Launching Suite3D job...")
    logger.debug("Running Suite3D job...")
    job.run_init_pass()
    out_dir = job_path / f"s3d-{job_id}"
    metadata["s3d-job"] = str(out_dir)
    metadata["s3d-params"] = params
    logger.info(f"Preprocessed data saved to {out_dir}")
    return out_dir


@dataclass
class Suite2pArray:
    filename: str | Path
    metadata: dict = field(init=False)
    active_file: Path = field(init=False)
    raw_file: Path = field(default=None)
    reg_file: Path = field(default=None)

    def __post_init__(self):
        path = Path(self.filename)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix == ".npy" and path.stem == "ops":
            ops_path = path
        elif path.suffix == ".bin":
            ops_path = path.with_name("ops.npy")
            if not ops_path.exists():
                raise FileNotFoundError(f"Missing ops.npy near {path}")
        else:
            raise ValueError(f"Unsupported input: {path}")

        self.metadata = np.load(ops_path, allow_pickle=True).item()
        self.num_rois = self.metadata.get("num_rois", 1)

        # resolve both possible bins - always look in the same directory as ops.npy
        # (metadata paths may be stale if data was moved)
        ops_dir = ops_path.parent
        self.raw_file = ops_dir / "data_raw.bin"
        self.reg_file = ops_dir / "data.bin"

        # choose which one to use
        if path.suffix == ".bin":
            # User clicked directly on a .bin file - use that specific file
            self.active_file = path
            if not self.active_file.exists():
                raise FileNotFoundError(
                    f"Binary file not found: {self.active_file}\n"
                    f"Available files in {ops_dir}:\n"
                    f"  - data.bin: {'exists' if self.reg_file.exists() else 'missing'}\n"
                    f"  - data_raw.bin: {'exists' if self.raw_file.exists() else 'missing'}"
                )
        else:
            # User clicked on directory/ops.npy - choose best available file
            # Prefer registered (data.bin) over raw (data_raw.bin)
            if self.reg_file.exists():
                self.active_file = self.reg_file
            elif self.raw_file.exists():
                self.active_file = self.raw_file
            else:
                raise FileNotFoundError(
                    f"No binary files found in {ops_dir}\n"
                    f"Expected either:\n"
                    f"  - {self.reg_file} (registered)\n"
                    f"  - {self.raw_file} (raw)\n"
                    f"Please check that Suite2p processing completed successfully."
                )

        self.Ly = self.metadata["Ly"]
        self.Lx = self.metadata["Lx"]
        self.nframes = self.metadata.get("nframes", self.metadata.get("n_frames"))
        self.shape = (self.nframes, self.Ly, self.Lx)
        self.dtype = np.int16

        # Validate file size matches expected shape
        expected_bytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        actual_bytes = self.active_file.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"Binary file {self.active_file.name} is too small!\n"
                f"Expected: {expected_bytes:,} bytes for shape {self.shape}\n"
                f"Actual: {actual_bytes:,} bytes\n"
                f"File may be corrupted or ops.npy metadata may be incorrect."
            )
        elif actual_bytes > expected_bytes:
            import warnings
            warnings.warn(
                f"Binary file {self.active_file.name} is larger than expected.\n"
                f"Expected: {expected_bytes:,} bytes for shape {self.shape}\n"
                f"Actual: {actual_bytes:,} bytes\n"
                f"Extra data will be ignored.",
                UserWarning
            )

        self._file = np.memmap(
            self.active_file, mode="r", dtype=self.dtype, shape=self.shape
        )
        self.filenames = [self.active_file]

    def switch_channel(self, use_raw=False):
        new_file = self.raw_file if use_raw else self.reg_file
        if not new_file.exists():
            raise FileNotFoundError(new_file)
        self._file = np.memmap(new_file, mode="r", dtype=self.dtype, shape=self.shape)
        self.active_file = new_file

    def __getitem__(self, key):
        return self._file[key]

    def __len__(self):
        return self.shape[0]

    def __array__(self):
        n = min(10, self.nframes) if self.nframes >= 10 else self.nframes
        return np.stack([self._file[i] for i in range(n)], axis=0)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def min(self):
        return float(self._file[0].min())

    @property
    def max(self):
        return float(self._file[0].max())

    def close(self):
        self._file._mmap.close()  # type: ignore

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
        arrays = []
        names = []

        # Try to load both files if they exist
        raw_loaded = False
        reg_loaded = False

        if self.raw_file.exists():
            try:
                raw = Suite2pArray(self.raw_file)
                arrays.append(raw)
                names.append("raw")
                raw_loaded = True
            except Exception as e:
                logger.warning(f"Could not open raw file {self.raw_file}: {e}")

        if self.reg_file.exists():
            try:
                reg = Suite2pArray(self.reg_file)
                arrays.append(reg)
                names.append("registered")
                reg_loaded = True
            except Exception as e:
                logger.warning(f"Could not open registered file {self.reg_file}: {e}")

        # If neither file could be loaded, show the currently active file
        if not arrays:
            arrays.append(self)
            if self.active_file == self.raw_file:
                names.append("raw")
            elif self.active_file == self.reg_file:
                names.append("registered")
            else:
                names.append(self.active_file.name)

        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        histogram_widget = kwargs.get("histogram_widget", True)
        window_funcs = kwargs.get("window_funcs", None)

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


class H5Array:
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
        self.dtype = self._d.dtype
        self.ndim = self._d.ndim

    @property
    def num_planes(self) -> int:
        # Try to get from metadata first
        metadata = self.metadata
        if "num_planes" in metadata:
            return int(metadata["num_planes"])

        # Infer from shape based on data dimensionality
        if self.ndim >= 4:  # (T, Z, Y, X) - volumetric time series
            return int(self.shape[1])
        elif self.ndim == 3:  # (T, Y, X) - single plane time series
            return 1
        elif self.ndim == 1:
            # Special case: pollen scan_corrections (nc,)
            if self.dataset_name == "scan_corrections":
                return int(self.shape[0])
            return 1
        elif self.ndim == 2:  # (Y, X) - single frame
            return 1

        # Fallback
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

        return data

    def min(self) -> float:
        return float(self._d[0].min())

    def max(self) -> float:
        return float(self._d[0].max())

    def __array__(self):
        n = min(10, self.shape[0])
        return self._d[:n]

    def close(self):
        self._f.close()

    @property
    def metadata(self) -> dict:
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


@dataclass
class MBOTiffArray:
    filenames: list[Path]
    _chunks: tuple[int, ...] | dict | None = None
    roi: int | None = None
    _metadata: dict | None = field(default=None, init=False)
    _dask_array: da.Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.filenames:
            raise ValueError("No filenames provided.")

        # allow string paths
        self.filenames = [Path(f) for f in self.filenames]

        # collect metadata from first TIFF
        self._metadata = get_metadata(self.filenames)
        self.num_rois = self.metadata.get("num_rois", 1)

        self.tags = [derive_tag_from_filename(f) for f in self.filenames]

    @property
    def metadata(self) -> dict:
        return self._metadata or {}

    @metadata.setter
    def metadata(self, metadata: dict):
        self._metadata = metadata

    @property
    def chunks(self):
        return self._chunks or CHUNKS_4D

    def _get_file_shape_dtype(self):
        """Get shape/dtype from first file only, cache it."""
        if not hasattr(self, '_cached_shape'):
            with tifffile.TiffFile(self.filenames[0]) as tf:
                self._cached_shape = tf.series[0].shape
                self._cached_dtype = tf.series[0].dtype
        return self._cached_shape, self._cached_dtype

    @property
    def dask(self) -> da.Array:
        if self._dask_array is not None:
            return self._dask_array

        # Get metadata from first file only
        shape, dtype = self._get_file_shape_dtype()

        # Use aszarr for truly lazy, memory-mapped access
        lazy_arrays = []
        for fname in self.filenames:
            arr = tifffile.imread(fname, aszarr=True)
            lazy_arrays.append(da.from_zarr(arr))

        if len(lazy_arrays) == 1:
            darr = lazy_arrays[0]
            # TZYX format: (time, z, y, x)
            if darr.ndim == 2:
                # Single 2D image -> add T and Z dimensions
                darr = darr[None, None, :, :]  # (1, 1, y, x)
            elif darr.ndim == 3:
                # 3D stack (t, y, x) -> add Z dimension
                darr = darr[:, None, :, :]  # (t, 1, y, x)
            # else: already 4D (t, z, y, x)
        else:
            # Concatenate along time axis - use native zarr chunks for fast access
            darr = da.concatenate(lazy_arrays, axis=0)
            if darr.ndim == 3:
                # After concat: (t, y, x) with shape (189879, 456, 448)
                # Insert Z dimension at axis 1 to get TZYX
                darr = darr[:, None, :, :]  # Shape: (189879, 1, 456, 448) = TZYX
            # else: 4D from files, already TZYX
            # No rechunking - native zarr chunks from tifffile are optimal for memory-mapped access

        self._dask_array = darr
        return darr

    @property
    def shape(self):
        if self._dask_array is not None:
            return tuple(self._dask_array.shape)
        # Infer shape without building full dask array
        file_shape, _ = self._get_file_shape_dtype()
        if len(self.filenames) == 1:
            if len(file_shape) == 2:
                # Single 2D image: (1, 1, y, x) = TZYX
                return (1, 1, *file_shape)
            elif len(file_shape) == 3:
                # Single 3D stack (t, y, x): add Z -> (t, 1, y, x) = TZYX
                return (file_shape[0], 1, file_shape[1], file_shape[2])
            return file_shape
        else:
            # Multiple files: each file shape (t, y, x), concat on T
            total_t = sum(1 for _ in self.filenames) * file_shape[0]
            if len(file_shape) == 3:
                # Files are (t, y, x), concat to (total_t, y, x), add Z -> (total_t, 1, y, x) = TZYX
                return (total_t, 1, file_shape[1], file_shape[2])
            return (total_t, *file_shape[1:])

    @property
    def ndim(self):
        return self.dask.ndim

    def __getitem__(self, key):
        key = tuple(
            slice(k.start, k.stop) if isinstance(k, range) else k
            for k in (key if isinstance(key, tuple) else (key,))
        )
        return self.dask[key]

    def __getattr__(self, attr):
        # Prevent recursion by never delegating internal attributes
        if attr.startswith('_') or attr in ('dask', 'filenames', 'metadata'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        # Use object.__getattribute__ to avoid recursion when accessing self.dask
        try:
            dask_arr = object.__getattribute__(self, '_dask_array')
            if dask_arr is None:
                # Force dask property to initialize
                dask_arr = object.__getattribute__(self, 'dask')
            return getattr(dask_arr, attr)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

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
        """Write MBOTiffArray to disk in various formats."""
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


class TiffArray:
    """
    Lazy TIFF array reader using TiffFile handles and asarray() on __getitem__.

    Similar pattern to MboRawArray but for generic TIFFs without ScanImage metadata.
    Opens TiffFile handles on init (no data read), extracts shape from metadata/first page,
    and reads data lazily via tf.asarray(key=frames) when indexed.
    """

    def __init__(self, files: str | Path | List[str] | List[Path]):
        import json
        from mbo_utilities.metadata import query_tiff_pages

        # Normalize to list of Paths
        if isinstance(files, (str, Path)):
            self.filenames = expand_paths(files)
        else:
            self.filenames = [Path(f) for f in files]
        self.filenames = [Path(p) for p in self.filenames]

        # Open TiffFile handles (no data read yet)
        self.tiff_files = [TiffFile(f) for f in self.filenames]
        self._tiff_lock = threading.Lock()

        # Extract info from first file's first page (no seeks)
        tf = self.tiff_files[0]
        page0 = tf.pages.first
        self._page_shape = page0.shape  # (Y, X) for 2D page
        self._dtype = page0.dtype

        # Try to get frame count from metadata without seeking
        # Fallback chain: shaped_description -> IFD estimate -> len(pages)
        self._frames_per_file = []
        self._num_frames = 0

        for i, (tfile, fpath) in enumerate(zip(self.tiff_files, self.filenames)):
            nframes = None

            # Method 1: ImageDescription JSON (tifffile shaped writes)
            if i == 0:
                desc = page0.description
            else:
                desc = tfile.pages.first.description

            if desc:
                try:
                    meta = json.loads(desc)
                    if "shape" in meta and isinstance(meta["shape"], list):
                        # First dimension is frames for 3D+ arrays
                        if len(meta["shape"]) >= 3:
                            nframes = meta["shape"][0]
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

            # Method 2: IFD offset estimation (fast for uniform pages)
            # Only trust if result > 1 (IFD estimate can fail on non-uniform TIFFs)
            if nframes is None:
                try:
                    est = query_tiff_pages(fpath)
                    if est > 1:
                        nframes = est
                except Exception:
                    pass

            # Method 3: Fallback to len(pages) - triggers seek but guaranteed correct
            if nframes is None:
                nframes = len(tfile.pages)

            self._frames_per_file.append(nframes)
            self._num_frames += nframes

        # Build metadata dict
        self._metadata = {
            "shape": self.shape,
            "dtype": str(self._dtype),
            "nframes": self._num_frames,
            "num_frames": self._num_frames,
            "frames_per_file": self._frames_per_file,
            "file_paths": [str(p) for p in self.filenames],
            "num_files": len(self.filenames),
        }

        self.num_rois = 1
        self._target_dtype = None

    @property
    def shape(self) -> tuple[int, ...]:
        # Return TZYX format: (frames, 1, Y, X)
        return (self._num_frames, 1, self._page_shape[0], self._page_shape[1])

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    @property
    def ndim(self) -> int:
        return 4

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __getitem__(self, key):
        """Read frames lazily using tf.asarray(key=frame_indices)."""
        if not isinstance(key, tuple):
            key = (key,)

        # Parse the key into frame indices
        t_key = key[0] if len(key) > 0 else slice(None)
        z_key = key[1] if len(key) > 1 else slice(None)
        y_key = key[2] if len(key) > 2 else slice(None)
        x_key = key[3] if len(key) > 3 else slice(None)

        # Convert to list of frame indices
        frames = listify_index(t_key, self._num_frames)
        if not frames:
            return np.empty((0, 1) + self._page_shape, dtype=self.dtype)

        # Read the requested frames
        out = self._read_frames(frames)

        # Apply spatial slicing if needed
        if y_key != slice(None) or x_key != slice(None):
            out = out[:, :, y_key, x_key]

        # Handle z dimension (always 1 for generic TIFFs)
        z_indices = listify_index(z_key, 1)
        if z_indices != [0]:
            out = out[:, z_indices, :, :]

        # Squeeze dimensions for integer indices
        squeeze_axes = []
        if isinstance(t_key, int):
            squeeze_axes.append(0)
        if isinstance(z_key, int):
            squeeze_axes.append(1 - len([a for a in squeeze_axes if a < 1]))
        if squeeze_axes:
            out = np.squeeze(out, axis=tuple(squeeze_axes))

        # Apply dtype conversion if astype() was called
        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)

        return out

    def _read_frames(self, frames: list[int]) -> np.ndarray:
        """Read specific frame indices across all files."""
        buf = np.empty((len(frames), 1, self._page_shape[0], self._page_shape[1]), dtype=self._dtype)

        start = 0
        frame_set = set(frames)
        frame_to_buf_idx = {f: i for i, f in enumerate(frames)}

        for tf, nframes in zip(self.tiff_files, self._frames_per_file):
            end = start + nframes
            # Find which requested frames are in this file
            file_frames = [f for f in frames if start <= f < end]
            if not file_frames:
                start = end
                continue

            # Convert global frame indices to local file indices
            local_indices = [f - start for f in file_frames]

            with self._tiff_lock:
                try:
                    chunk = tf.asarray(key=local_indices)
                except Exception as e:
                    raise IOError(
                        f"TiffArray: Failed to read frames {local_indices} from {tf.filename}\n"
                        f"File may be corrupted or incomplete.\n"
                        f": {type(e).__name__}: {e}"
                    ) from e

            # Handle single frame case where asarray returns 2D
            if chunk.ndim == 2:
                chunk = chunk[np.newaxis, ...]

            # Copy to output buffer
            for local_idx, global_frame in zip(local_indices, file_frames):
                buf_idx = frame_to_buf_idx[global_frame]
                chunk_idx = local_indices.index(local_idx)
                buf[buf_idx, 0] = chunk[chunk_idx]

            start = end

        return buf

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def __array__(self):
        """Return first 10 frames as numpy array."""
        n = min(10, self._num_frames)
        return self[:n]

    def min(self) -> float:
        return float(np.min(self[0]))

    def max(self) -> float:
        return float(np.max(self[0]))

    def imshow(self, **kwargs):
        import fastplotlib as fpl

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        window_funcs = kwargs.get("window_funcs", None)
        return fpl.ImageWidget(
            data=self,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )

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
        """Write TiffArray to disk in various formats."""
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


class MboRawArray:
    def __init__(
        self,
        files: str | Path | list,
        roi: int | Sequence[int] | None = None,
        fix_phase: bool = True,
        phasecorr_method: str = "mean",
        border: int | tuple[int, int, int, int] = 3,
        upsample: int = 5,
        max_offset: int = 4,
        use_fft: bool = False,
        fft_method: str = "2d",
    ):
        self.filenames = [files] if isinstance(files, (str, Path)) else list(files)
        self.tiff_files = [TiffFile(f) for f in self.filenames]
        self._tiff_lock = threading.Lock()  # Protect concurrent access to tiff_files

        # Initialize data attributes first (needed for roi setter validation)
        self._metadata = get_metadata(self.filenames)
        self.num_channels = self._metadata["num_planes"]
        self.num_rois = self._metadata.get("num_rois", 1)

        # Now set roi (this will call the setter which validates against num_rois)
        self._roi = roi
        self.roi = roi

        self._fix_phase = fix_phase
        self._use_fft = use_fft
        self._fft_method = fft_method
        self._phasecorr_method = phasecorr_method
        self.border = border
        self.max_offset = max_offset
        self.upsample = upsample
        self._offset = 0.0
        self._mean_subtraction = False
        self.pbar = None
        self.show_pbar = False
        self.logger = logger

        # Debug flags
        self.debug_flags = {
            "frame_idx": True,
            "roi_array_shape": False,
            "phase_offset": False,
        }
        self.num_frames = self._metadata["num_frames"]
        self._source_dtype = self._metadata["dtype"]  # Original dtype from file
        self._target_dtype = None  # Target dtype for conversion (set by astype)
        self._ndim = self._metadata["ndim"]

        self._frames_per_file = self._metadata.get("frames_per_file", None)

        self._rois = self._extract_roi_info()

    def _extract_roi_info(self):
        """
        Extract ROI positions and dimensions from metadata.
        Uses actual TIFF page dimensions, excluding flyback lines.
        """
        # Get ROI info from metadata
        roi_groups = self._metadata["roi_groups"]
        if isinstance(roi_groups, dict):
            roi_groups = [roi_groups]

        # Use actual TIFF dimensions
        actual_page_width = self._page_width
        actual_page_height = self._page_height
        num_fly_to_lines = self._metadata.get("num_fly_to_lines", 0)

        # Get heights from metadata
        heights_from_metadata = []
        for roi_data in roi_groups:
            scanfields = roi_data["scanfields"]
            if isinstance(scanfields, list):
                scanfields = scanfields[0]
            heights_from_metadata.append(scanfields["pixelResolutionXY"][1])

        # Calculate actual heights: distribute available height (excluding flyback) proportionally
        total_metadata_height = sum(heights_from_metadata)
        total_available_height = (
            actual_page_height - (len(roi_groups) - 1) * num_fly_to_lines
        )

        # Calculate actual heights for each ROI (proportionally)
        actual_heights = []
        remaining_height = total_available_height
        for i, metadata_height in enumerate(heights_from_metadata):
            if i == len(heights_from_metadata) - 1:
                # Last ROI gets remaining height to avoid rounding errors
                height = remaining_height
            else:
                height = int(
                    round(
                        metadata_height * total_available_height / total_metadata_height
                    )
                )
                remaining_height -= height
            actual_heights.append(height)

        # Build ROI info
        rois = []
        y_offset = 0

        for i, (roi_data, height) in enumerate(zip(roi_groups, actual_heights)):
            roi_info = {
                "y_start": y_offset,
                "y_end": y_offset + height,  # Exclude flyback lines
                "width": actual_page_width,
                "height": height,
                "x": 0,
                "slice": slice(y_offset, y_offset + height),  # Only the ROI data
            }
            rois.append(roi_info)

            # Move to next ROI position (skip flyback lines)
            y_offset += height + num_fly_to_lines

        # Debug info
        logger.debug(
            f"ROI structure: {[(r['y_start'], r['y_end'], r['height']) for r in rois]}"
        )
        logger.debug(
            f"Total calculated height: {y_offset - num_fly_to_lines}, actual page: {actual_page_height}"
        )

        return rois

    @property
    def ndim(self):
        # Return actual array dimensions, not TIFF page dimensions
        return len(self.shape)

    @property
    def dtype(self):
        """Return target dtype if set via astype(), otherwise source dtype."""
        return self._target_dtype if self._target_dtype is not None else self._source_dtype

    @property
    def metadata(self):
        self._metadata.update(
            {
                "fix_phase": self.fix_phase,
                "phasecorr_method": self.phasecorr_method,
                "offset": self.offset,
                "border": self.border,
                "upsample": self.upsample,
                "max_offset": self.max_offset,
                "nframes": self.num_frames,
                "num_frames": self.num_frames,  # alias for backwards compatibility
                "use_fft": self.use_fft,
                "mean_subtraction": self.mean_subtraction,
            }
        )
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata.update(value)

    @property
    def rois(self):
        """ROI's hold information about the size, position and shape of the ROIs."""
        return self._rois

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value: float | np.ndarray):
        """
        Set the phase offset for phase correction.
        If value is a scalar, it applies the same offset to all frames.
        If value is an array, it must match the number of frames.
        """
        if isinstance(value, int):
            self._offset = float(value)
        self._offset = value

    @property
    def use_fft(self):
        return self._use_fft

    @use_fft.setter
    def use_fft(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("use_fft must be a boolean value.")
        self._use_fft = value

    @property
    def phasecorr_method(self):
        """
        Get the current phase correction method.
        """
        return self._phasecorr_method

    @phasecorr_method.setter
    def phasecorr_method(self, value: str | None):
        """
        Set the phase correction method.
        """
        if value not in ALL_PHASECORR_METHODS:
            raise ValueError(
                f"Unsupported phase correction method: {value}. "
                f"Supported methods are: {ALL_PHASECORR_METHODS}"
            )
        if value is None:
            self.fix_phase = False
        self._phasecorr_method = value

    @property
    def fix_phase(self):
        """
        Get whether phase correction is applied.
        If True, phase correction is applied to the data.
        """
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value: bool):
        """
        Set whether to apply phase correction.
        If True, phase correction is applied to the data.
        """
        if not isinstance(value, bool):
            raise ValueError("do_phasecorr must be a boolean value.")
        self._fix_phase = value

    @property
    def mean_subtraction(self):
        """Get whether mean subtraction is enabled."""
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool):
        """Set whether mean subtraction is enabled."""
        if not isinstance(value, bool):
            raise ValueError("mean_subtraction must be a boolean value.")
        self._mean_subtraction = value

    @property
    def roi(self):
        """
        Get the current ROI index.
        If roi is None, returns -1 to indicate no specific ROI.
        """
        return self._roi

    @roi.setter
    def roi(self, value):
        """
        Set the current ROI index.
        If value is None, sets roi to -1 to indicate no specific ROI.
        """
        # Validate ROI bounds
        if value is not None and value != 0:  # 0 means "split all", None means "stitch"
            if isinstance(value, int):
                if value < 1 or value > self.num_rois:
                    raise ValueError(
                        f"ROI index {value} out of bounds.\n"
                        f"Valid range: 1 to {self.num_rois} (1-indexed)\n"
                        f"Use roi=0 to split all ROIs, or roi=None to stitch."
                    )
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if v < 1 or v > self.num_rois:
                        raise ValueError(
                            f"ROI index {v} in {value} out of bounds.\n"
                            f"Valid range: 1 to {self.num_rois} (1-indexed)"
                        )
        self._roi = value

    @property
    def output_xslices(self):
        x_offset = 0
        slices = []
        for roi in self._rois:
            slices.append(slice(x_offset, x_offset + roi["width"]))
            x_offset += roi["width"]
        return slices

    @property
    def output_yslices(self):
        return [slice(0, roi["height"]) for roi in self._rois]

    @property
    def yslices(self):
        return [roi["slice"] for roi in self._rois]

    @property
    def xslices(self):
        return [slice(0, roi["width"]) for roi in self._rois]

    def _read_pages(self, frames, chans, yslice=slice(None), xslice=slice(None), **_):
        pages = [f * self.num_channels + z for f in frames for z in chans]
        tiff_width_px = len(listify_index(xslice, self._page_width))
        tiff_height_px = len(listify_index(yslice, self._page_height))
        buf = np.empty((len(pages), tiff_height_px, tiff_width_px), dtype=self.dtype)

        start = 0
        # Use cached frames_per_file to avoid slow len(tf.pages) calls
        # Note: frames_per_file is per-time-frame, need to multiply by num_channels for total pages
        # If not available, fall back to len(tf.pages) which triggers seek
        tiff_iterator = (
            zip(self.tiff_files, (f * self.num_channels for f in self._frames_per_file))
            if self._frames_per_file is not None
            else ((tf, len(tf.pages)) for tf in self.tiff_files)
        )

        for tf, num_pages in tiff_iterator:
            end = start + num_pages
            idxs = [i for i, p in enumerate(pages) if start <= p < end]
            if not idxs:
                start = end
                continue

            frame_idx = [pages[i] - start for i in idxs]
            # Use lock to protect TiffFile access from concurrent threads
            with self._tiff_lock:
                try:
                    chunk = tf.asarray(key=frame_idx)
                except Exception as e:
                    # TODO: wrap this for all array types
                    raise IOError(
                        f"MboRawArray: Failed to read pages {frame_idx} from TIFF file {tf.filename}\n"
                        f"File may be corrupted or incomplete.\n"
                        f": {type(e).__name__}: {e}"
                    ) from e
            if chunk.ndim == 2:  # Single page was squeezed to 2D
                chunk = chunk[np.newaxis, ...]  # Add back the first dimension
            chunk = chunk[..., yslice, xslice]

            if self.fix_phase:
                # Apply phase correction to the current chunk
                corrected, offset = bidir_phasecorr(
                    chunk,
                    method=self.phasecorr_method,
                    upsample=self.upsample,
                    max_offset=self.max_offset,
                    border=self.border,
                    use_fft=self.use_fft,
                    fft_method=self._fft_method,
                )
                buf[idxs] = corrected
                self.offset = offset
            else:
                buf[idxs] = chunk
                self.offset = 0.0
            start = end

        return buf.reshape(len(frames), len(chans), tiff_height_px, tiff_width_px)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t_key, z_key, _, _ = tuple(_convert_range_to_slice(k) for k in key) + (
            slice(None),
        ) * (4 - len(key))
        frames = listify_index(t_key, self.num_frames)
        chans = listify_index(z_key, self.num_channels)
        if not frames or not chans:
            return np.empty(0)

        out = self.process_rois(frames, chans)

        squeeze = []
        if isinstance(t_key, int):
            squeeze.append(0)
        if isinstance(z_key, int):
            squeeze.append(1)
        if squeeze:
            if isinstance(out, tuple):
                out = tuple(np.squeeze(x, axis=tuple(squeeze)) for x in out)
            else:
                out = np.squeeze(out, axis=tuple(squeeze))

        # Convert dtype if astype() was called
        if self._target_dtype is not None:
            if isinstance(out, tuple):
                out = tuple(x.astype(self._target_dtype) for x in out)
            else:
                out = out.astype(self._target_dtype)

        return out

    def process_rois(self, frames, chans):
        """Dispatch ROI processing. Handles single ROI, multiple ROIs, or all ROIs (None)."""
        if self.roi is not None:
            if isinstance(self.roi, list):
                return tuple(
                    self.process_single_roi(r - 1, frames, chans) for r in self.roi
                )
            elif self.roi == 0:
                return tuple(
                    self.process_single_roi(r, frames, chans)
                    for r in range(self.num_rois)
                )
            elif isinstance(self.roi, int):
                return self.process_single_roi(self.roi - 1, frames, chans)

        # roi=None: Horizontally concatenate ROIs
        total_width = sum(roi["width"] for roi in self._rois)
        max_height = max(roi["height"] for roi in self._rois)
        out = np.zeros(
            (len(frames), len(chans), max_height, total_width), dtype=self.dtype
        )

        for roi_idx in range(self.num_rois):
            roi_data = self._read_pages(
                frames,
                chans,
                yslice=self._rois[roi_idx]["slice"],  # Where to extract from TIFF
                xslice=slice(None),
            )
            # Where to place in output (horizontal concatenation)
            oys = self.output_yslices[roi_idx]
            oxs = self.output_xslices[roi_idx]
            out[:, :, oys, oxs] = roi_data

        return out

    def process_single_roi(self, roi_idx, frames, chans):
        roi = self._rois[roi_idx]
        return self._read_pages(
            frames,
            chans,
            yslice=roi["slice"],
            xslice=slice(None),  # or slice(0, roi['width'])
        )

    @property
    def num_planes(self):
        """Alias for num_channels (ScanImage terminology)."""
        return self.num_channels

    def min(self):
        """
        Returns the minimum value of the first tiff page.
        """
        page = self.tiff_files[0].pages[0]
        return np.min(page.asarray())

    def max(self):
        """
        Returns the maximum value of the first tiff page.
        """
        page = self.tiff_files[0].pages[0]
        return np.max(page.asarray())

    @property
    def shape(self):
        """Shape is relative to the current ROI."""
        if self.roi is not None:
            if not isinstance(self.roi, (list, tuple)):
                if self.roi > 0:
                    roi = self._rois[self.roi - 1]
                    return (
                        self.num_frames,
                        self.num_channels,
                        roi["height"],
                        roi["width"],
                    )
        # roi = None: return horizontally concatenated shape
        total_width = sum(roi["width"] for roi in self._rois)
        max_height = max(roi["height"] for roi in self._rois)
        return (
            self.num_frames,
            self.num_channels,
            max_height,
            total_width,
        )

    def size(self):
        """Total number of elements."""
        total_width = sum(roi["width"] for roi in self._rois)
        max_height = max(roi["height"] for roi in self._rois)
        return self.num_frames * self.num_channels * max_height * total_width

    def astype(self, dtype, copy=True):
        """
        Set target dtype for lazy conversion on data access.

        This modifies the array in-place to convert dtype when data is loaded,
        preserving lazy loading behavior.

        Parameters
        ----------
        dtype : np.dtype or str
            Target dtype for conversion
        copy : bool, optional
            Kept for numpy compatibility (dtype conversion always creates a copy)

        Returns
        -------
        self : MboRawArray
            Returns self for method chaining

        Examples
        --------
        >>> arr = imread("data.tif")  # MboRawArray, dtype=uint16
        >>> arr_float = arr.astype(np.float32)  # No data loaded yet
        >>> frame = arr_float[0]  # Data loaded and converted on access
        """
        self._target_dtype = np.dtype(dtype)
        return self

    @property
    def _page_height(self):
        return self._metadata["page_height"]

    @property
    def _page_width(self):
        return self._metadata["page_width"]

    def __array__(self):
        """
        Convert the scan data to a NumPy array.
        Calculate the size of the scan and subsample to keep under memory limits.
        """
        return subsample_array(self, ignore_dims=[-1, -2, -3])

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
        """Write MboRawArray to disk in various formats."""
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            roi_iterator=iter_rois(self),
            **kwargs,
        )

    def imshow(self, **kwargs):
        arrays = []
        names = []
        # if roi is None, use a single array.roi = None
        # if roi is 0, get a list of all ROIs by deeepcopying the array and setting each roi
        for roi in iter_rois(self):
            arr = copy.copy(self)
            arr.roi = roi
            arr.fix_phase = False  # disable phase correction for initial display
            arr.use_fft = False
            arrays.append(arr)
            names.append(f"ROI {roi}" if roi else "Stitched mROIs")

        figure_shape = (1, len(arrays))

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get(
            "figure_kwargs",
            {
                "size": (600, 600),
            },
        )
        window_funcs = kwargs.get("window_funcs", None)
        import fastplotlib as fpl

        return fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,  # "canvas": canvas},
            figure_shape=figure_shape,
            graphic_kwargs={"vmin": arrays[0].min(), "vmax": arrays[0].max()},
            window_funcs=window_funcs,
        )


class NumpyArray:
    """
    Lazy array wrapper for NumPy arrays and .npy files.

    Conforms to LazyArrayProtocol for compatibility with mbo_utilities I/O
    and processing pipelines. Supports 2D (image), 3D (time series), and
    4D (volumetric) data.

    Parameters
    ----------
    array : np.ndarray, str, or Path
        Either a numpy array (will be saved to temp file for memory mapping)
        or a path to a .npy file.
    metadata : dict, optional
        Metadata dictionary. If not provided, basic metadata is inferred
        from array shape.

    Examples
    --------
    >>> # From .npy file
    >>> arr = NumpyArray("data.npy")
    >>> arr.shape
    (100, 512, 512)

    >>> # From in-memory array (creates temp file)
    >>> data = np.random.randn(100, 512, 512).astype(np.float32)
    >>> arr = NumpyArray(data)
    >>> arr[0:10]  # Lazy slicing

    >>> # 4D volumetric data
    >>> vol = NumpyArray("volume.npy")  # shape: (T, Z, Y, X)
    >>> vol.ndim
    4
    """

    def __init__(self, array: np.ndarray | str | Path, metadata: dict | None = None):
        if isinstance(array, (str, Path)):
            self.path = Path(array)
            if not self.path.exists():
                raise FileNotFoundError(f"Numpy file not found: {self.path}")
            self.data = np.load(self.path, mmap_mode="r")
            self._tempfile = None
        elif isinstance(array, np.ndarray):
            logger.info("Creating temporary .npy file for array.")
            tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
            np.save(tmp, array)  # type: ignore
            tmp.close()
            self.path = Path(tmp.name)
            self.data = np.load(self.path, mmap_mode="r")
            self._tempfile = tmp
            logger.debug(f"Temporary file created at {self.path}")
        else:
            raise TypeError(f"Expected np.ndarray or path, got {type(array)}")

        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.ndim = self.data.ndim
        self._metadata = metadata or {}
        self._min: float | None = None
        self._max: float | None = None

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        """Return length of first dimension (number of frames for 3D/4D)."""
        return self.shape[0]

    def __array__(self):
        return np.asarray(self.data)

    @property
    def filenames(self) -> list[Path]:
        return [self.path]

    @property
    def metadata(self) -> dict:
        # Ensure basic metadata is always present
        md = dict(self._metadata)
        if "nframes" not in md:
            md["nframes"] = self.shape[0] if self.ndim >= 1 else 1
        if "num_frames" not in md:
            md["num_frames"] = md["nframes"]
        if "Ly" not in md and self.ndim >= 2:
            md["Ly"] = self.shape[-2]
        if "Lx" not in md and self.ndim >= 2:
            md["Lx"] = self.shape[-1]
        return md

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError("metadata must be a dict")
        self._metadata = value

    @property
    def min(self) -> float:
        """Minimum value in array (computed from first frame, cached)."""
        if self._min is None:
            self._min = float(self.data[0].min()) if self.ndim >= 1 else float(self.data.min())
        return self._min

    @property
    def max(self) -> float:
        """Maximum value in array (computed from first frame, cached)."""
        if self._max is None:
            self._max = float(self.data[0].max()) if self.ndim >= 1 else float(self.data.max())
        return self._max

    def close(self):
        """Release resources and clean up temporary files."""
        if self._tempfile:
            try:
                Path(self._tempfile.name).unlink(missing_ok=True)
            except Exception:
                pass
            self._tempfile = None

    def __del__(self):
        self.close()

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
        """Write NumpyArray to disk in various formats."""
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
        import fastplotlib as fpl
        import numpy as np

        histogram_widget = kwargs.pop("histogram_widget", True)
        figure_kwargs = kwargs.pop("figure_kwargs", {"size": (800, 800)})
        graphic_kwargs = kwargs.pop("graphic_kwargs", {"vmin": self.min, "vmax": self.max})

        # Set up slider dimensions based on array dimensionality
        if self.ndim == 4:
            slider_dim_names = ("t", "z")
            window_funcs = kwargs.pop("window_funcs", (np.mean, None))
            window_sizes = kwargs.pop("window_sizes", (1, None))
        elif self.ndim == 3:
            slider_dim_names = ("t",)
            window_funcs = kwargs.pop("window_funcs", (np.mean,))
            window_sizes = kwargs.pop("window_sizes", (1,))
        else:
            slider_dim_names = None
            window_funcs = None
            window_sizes = None

        return fpl.ImageWidget(
            data=self.data,
            slider_dim_names=slider_dim_names,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            graphic_kwargs=graphic_kwargs,
            **kwargs,
        )


class NWBArray:
    def __init__(self, path: Path | str):
        try:
            from pynwb import read_nwb
        except ImportError:
            raise ImportError(
                "pynwb is not installed. Install with `pip install pynwb`."
            )
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No NWB file found at {self.path}")

        self.filenames = [self.path]

        nwbfile = read_nwb(path)
        self.data = nwbfile.acquisition["TwoPhotonSeries"].data
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.ndim = self.data.ndim
        self._metadata = {}

    def __getitem__(self, item):
        return self.data[item]

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError("metadata must be a dict")
        self._metadata = value

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
        """Write NWBArray to disk in various formats."""
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


class ZarrArray:
    """
    Reader for _write_zarr outputs.
    Presents data as (T, Z, H, W) with Z=1..nz.
    """

    def __init__(
        self,
        filenames: str | Path | Sequence[str | Path],
        compressor: str | None = "default",
        rois: list[int] | int | None = None,
    ):
        try:
            import zarr
            # v3.0 +
        except ImportError:
            logger.error(
                "zarr is not installed. Install with `uv pip install zarr>=3.1.3`."
            )
            zarr = None
            return

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
        self._groups = []  # Store groups separately to access their metadata
        for z in opened:
            if isinstance(z, zarr.Group):
                # OME-Zarr structure: access the "0" array
                if "0" not in z:
                    raise ValueError(
                        f"OME-Zarr group missing '0' array in {z.store.path}"
                    )
                self.zs.append(z["0"])
                self._groups.append(z)  # Keep reference to group for metadata
            else:
                # Standard zarr array
                self.zs.append(z)
                self._groups.append(None)

        shapes = [z.shape for z in self.zs]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent shapes across zarr stores: {shapes}")

        # For OME-Zarr, metadata is on the group; for standard zarr, it's on the array
        self._metadata = []
        for i, z in enumerate(self.zs):
            if self._groups[i] is not None:
                # OME-Zarr: metadata on group
                self._metadata.append(dict(self._groups[i].attrs))
            else:
                # Standard zarr: metadata on array
                self._metadata.append(dict(z.attrs))
        self.compressor = compressor

    @property
    def metadata(self):
        """
        Return metadata as a dict.
        - If single zarr file: return its metadata dict
        - If multiple zarr files: return the first one's metadata

        Note: _metadata is internally a list of dicts (one per zarr file)
        """
        if not self._metadata:
            md = {}
        else:
            md = self._metadata[0]

        # Ensure critical keys are present - extract from shape if missing
        # This provides backward compatibility with old zarr files
        if "num_frames" not in md and "nframes" not in md:
            # Extract from shape: (T, H, W)
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
        """
        Store z-statistics in metadata for persistence.

        Parameters
        ----------
        value : dict
            Dictionary with keys 'mean', 'std', 'snr' (each a list of floats).
        """
        if not isinstance(value, dict):
            raise TypeError(f"zstats must be a dict, got {type(value)}")
        if not all(k in value for k in ("mean", "std", "snr")):
            raise ValueError("zstats must contain 'mean', 'std', and 'snr' keys")

        # Update internal metadata
        if not self._metadata:
            self._metadata = [{}]
        self._metadata[0]["zstats"] = value

        # Also persist to zarr attrs if we have write access
        # (This will be saved when the zarr is written via _write_zarr)

    @metadata.setter
    def metadata(self, value: dict):
        """
        Set metadata. Updates the first zarr file's metadata.

        Args:
            value: dict of metadata to set
        """
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")

        if not self._metadata:
            self._metadata = [value]
        else:
            # Update first metadata dict
            self._metadata[0] = value

    @property
    def shape(self) -> tuple[int, int, int, int]:
        first_shape = self.zs[0].shape
        if len(first_shape) == 4:
            # Single merged 4D zarr: (T, Z, H, W)
            return first_shape
        elif len(first_shape) == 3:
            # Multiple 3D zarrs: stack them as (T, Z, H, W)
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
        # Check if single 4D merged array
        if len(self.zs) == 1 and len(self.zs[0].shape) == 4:
            # Already 4D, just return it
            return np.asarray(self.zs[0][:])

        # Multiple 3D arrays: stack them along Z axis
        arrs = [z[:] for z in self.zs]
        stacked = np.stack(arrs, axis=1)  # (T, Z, H, W)
        return stacked

    @property
    def min(self):
        """Minimum of first zarr store."""
        return float(self.zs[0][:].min())

    @property
    def max(self):
        """Maximum of first zarr store."""
        return float(self.zs[0][:].max())

    @property
    def ndim(self):
        # this will always be 4D, since we add a Z dimension if needed
        return 4  # (T, Z, H, W)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        t_key, z_key, y_key, x_key = key

        def normalize(idx):
            # convert range objects to slices (zarr doesn't support range objects)
            if isinstance(idx, range):
                # Convert range to slice for zarr compatibility
                if len(idx) == 0:
                    return slice(0, 0)
                return slice(idx.start, idx.stop, idx.step)
            # convert contiguous lists to slices for zarr
            if isinstance(idx, list) and len(idx) > 0:
                if all(idx[i] + 1 == idx[i + 1] for i in range(len(idx) - 1)):
                    return slice(idx[0], idx[-1] + 1)
                else:
                    return np.array(idx)  # will require looping later
            return idx

        t_key = normalize(t_key)
        y_key = normalize(y_key)
        x_key = normalize(x_key)
        z_key = normalize(z_key)  # Also normalize z_key

        # Check if we have a single 4D merged zarr or multiple 3D zarrs
        is_single_4d = len(self.zs) == 1 and len(self.zs[0].shape) == 4

        if is_single_4d:
            # Single merged 4D zarr: directly index with all 4 dimensions
            return self.zs[0][t_key, z_key, y_key, x_key]

        # Multiple 3D zarrs: stack them
        if len(self.zs) == 1:
            # Single 3D zarr: z_key must be 0 or slice(None)
            if isinstance(z_key, int):
                if z_key != 0:
                    raise IndexError("Z dimension has size 1, only index 0 is valid")
                return self.zs[0][t_key, y_key, x_key]
            elif isinstance(z_key, slice):
                # Return with Z dimension added
                data = self.zs[0][t_key, y_key, x_key]
                return data[:, np.newaxis, ...]  # Add Z dimension
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
            # Fallback: assume all z
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


@dataclass
class BinArray:
    """
    Read/write raw binary files (Suite2p format) without requiring ops.npy.

    This class provides a lightweight interface for working with raw binary
    files (.bin) directly, without needing the full Suite2p context that
    Suite2pArray provides. Useful for workflows that manipulate individual
    binary files (e.g., data_raw.bin vs data.bin).

    Parameters
    ----------
    filename : str or Path
        Path to the binary file
    shape : tuple, optional
        Shape of the data as (nframes, Ly, Lx). If None and file exists,
        will try to infer from adjacent ops.npy file.
    dtype : np.dtype, default=np.int16
        Data type of the binary file
    metadata : dict, optional
        Additional metadata to store with the array

    Examples
    --------
    >>> # Read existing binary with known shape
    >>> arr = BinArray("data_raw.bin", shape=(1000, 512, 512))
    >>> frame = arr[0]

    >>> # Create new binary file
    >>> arr = BinArray("output.bin", shape=(100, 256, 256))
    >>> arr[0] = my_data
    """

    filename: str | Path
    shape: tuple = None
    dtype: np.dtype = field(default=np.int16)
    metadata: dict = field(default_factory=dict)
    _file: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.filename = Path(self.filename)
        self.dtype = np.dtype(self.dtype)

        # If file exists and shape not provided, try to infer from ops.npy
        if self.filename.exists() and self.shape is None:
            ops_file = self.filename.parent / "ops.npy"
            if ops_file.exists():
                try:
                    ops = np.load(ops_file, allow_pickle=True).item()
                    Ly = ops.get("Ly")
                    Lx = ops.get("Lx")
                    nframes = ops.get("nframes", ops.get("n_frames"))
                    if all(x is not None for x in [Ly, Lx, nframes]):
                        self.shape = (nframes, Ly, Lx)
                        # Optionally copy metadata from ops
                        self.metadata.update(ops)
                        logger.debug(f"Inferred shape from ops.npy: {self.shape}")
                except Exception as e:
                    logger.warning(f"Could not read ops.npy: {e}")

            if self.shape is None:
                raise ValueError(
                    f"Cannot infer shape for {self.filename}. "
                    "Provide shape=(nframes, Ly, Lx) or ensure ops.npy exists."
                )

        # Creating new file
        if not self.filename.exists():
            if self.shape is None:
                raise ValueError(
                    "Must provide shape=(nframes, Ly, Lx) when creating new file"
                )
            mode = "w+"
        else:
            mode = "r+"

        self._file = np.memmap(
            self.filename, mode=mode, dtype=self.dtype, shape=self.shape
        )
        self.filenames = [self.filename]

    def __getitem__(self, key):
        return self._file[key]

    def __setitem__(self, key, value):
        """Allow assignment to the memmap."""
        if np.asarray(value).dtype != self.dtype:
            # Clip values to avoid overflow
            max_val = (
                np.iinfo(self.dtype).max - 1
                if np.issubdtype(self.dtype, np.integer)
                else None
            )
            if max_val:
                self._file[key] = np.clip(value, None, max_val).astype(self.dtype)
            else:
                self._file[key] = value.astype(self.dtype)
        else:
            self._file[key] = value

    def __len__(self):
        return self.shape[0]

    def __array__(self):
        """Return first 10 frames for quick inspection."""
        n = min(10, self.shape[0]) if self.shape[0] >= 10 else self.shape[0]
        return np.array([self._file[i] for i in range(n)])

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def min(self):
        return float(self._file[0].min())

    @property
    def max(self):
        return float(self._file[0].max())

    @property
    def nframes(self):
        return self.shape[0]

    @property
    def Ly(self):
        return self.shape[1]

    @property
    def Lx(self):
        return self.shape[2]

    @property
    def file(self):
        """Alias for _file, for backwards compatibility with BinaryFile API."""
        return self._file

    def flush(self):
        """Flush the memmap to disk."""
        self._file.flush()

    def close(self):
        """Close the memmap file."""
        if hasattr(self._file, "_mmap"):
            self._file._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _imwrite(
        self,
        outpath: Path | str,
        planes=None,
        target_chunk_mb: int = 50,
        ext: str = ".bin",
        progress_callback=None,
        debug: bool = False,
        overwrite: bool = False,
        output_name: str | None = None,
        **kwargs,
    ):
        """Write BinArray to disk in various formats."""
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)

        ext_clean = ext.lower().lstrip(".")

        # BinArray is always 3D (T, Y, X) - single plane
        # For binary output, use direct memmap copy (faster)
        if ext_clean == "bin":
            md = dict(self.metadata) if self.metadata else {}
            md["Ly"] = self.Ly
            md["Lx"] = self.Lx
            md["nframes"] = self.nframes
            md["num_frames"] = self.nframes

            if output_name is None:
                output_name = "data_raw.bin"
            outfile = outpath / output_name

            if not outfile.exists() or overwrite:
                logger.info(f"Writing binary to {outfile}")
                new_file = np.memmap(outfile, mode="w+", dtype=self.dtype, shape=self.shape)
                new_file[:] = self._file[:]
                new_file.flush()
                del new_file
            else:
                logger.info(f"Binary file already exists: {outfile}")

            # Write ops.npy
            ops_file = outpath / "ops.npy"
            np.save(ops_file, md)
            logger.info(f"Wrote ops.npy to {ops_file}")
            return outpath

        # For other formats, use common implementation
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            output_name=output_name,
            **kwargs,
        )

