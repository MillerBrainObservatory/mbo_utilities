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

    def _imwrite(self, outpath, **kwargs):
        _write_plane(
            self._d,
            Path(outpath),
            overwrite=kwargs.get("overwrite", False),
            metadata=self.metadata,
            target_chunk_mb=kwargs.get("target_chunk_mb", 20),
            progress_callback=kwargs.get("progress_callback", None),
            debug=kwargs.get("debug", False),
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
        **kwargs,
    ):
        from mbo_utilities._writers import _write_plane
        from mbo_utilities.file_io import get_plane_from_filename

        md = self.metadata.copy()
        plane = md.get("plane") or get_plane_from_filename(Path(outpath).stem, None)
        if plane is None:
            raise ValueError("Cannot determine plane from metadata.")

        outpath = Path(outpath)
        ext = ext.lower().lstrip(".")
        fname = f"plane{plane:03d}.{ext}" if ext != "bin" else "data_raw.bin"
        target = (
            outpath.joinpath(fname)
            if outpath.is_dir()
            else outpath.parent.joinpath(fname)
        )

        _write_plane(
            self,
            target,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            metadata=md,
            progress_callback=progress_callback,
            debug=debug,
            dshape=(self.shape[0], self.shape[-2], self.shape[-1]),  # (T, Y, X) - Fixed Y/X order
            plane_index=None,
            **kwargs,
        )


@dataclass
class NpyArray:
    filenames: list[Path]

    def __post_init__(self):
        if not self.filenames:
            raise ValueError("No filenames provided.")
        if len(self.filenames) > 1:
            raise ValueError("NpyArray only supports a single .npy file.")
        self.filenames = [Path(p) for p in self.filenames]
        self._file = np.load(self.filenames[0], mmap_mode="r")
        self.shape = self._file.shape
        self.dtype = self._file.dtype
        self.ndim = self._file.ndim


@dataclass
class TiffArray:
    filenames: List[Path] | List[str] | Path | str
    _chunks: Any = None
    _dask_array: da.Array = field(default=None, init=False, repr=False)
    _metadata: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.filenames, list):
            self.filenames = expand_paths(self.filenames)
        self.filenames = [Path(p) for p in self.filenames]
        self._metadata = _safe_get_metadata(self.filenames[0])
        self.num_rois = self._metadata.get("num_rois", 1)

    @property
    def chunks(self):
        return self._chunks or CHUNKS_4D

    @chunks.setter
    def chunks(self, value):
        self._chunks = value

    def _open_one(self, path: Path) -> da.Array:
        try:
            with tifffile.TiffFile(path) as tf:
                z = tf.aszarr()
                a = da.from_zarr(z, chunks=self.chunks)
                axes = tf.series[0].axes
        except Exception:
            try:
                mm = tifffile.memmap(path, mode="r")
                a = da.from_array(mm, chunks=self.chunks)
                axes = _axes_or_guess(mm.ndim)
            except Exception:
                arr = tifffile.imread(path)
                a = da.from_array(arr, chunks=self.chunks)
                axes = _axes_or_guess(arr.ndim)
        a = _to_tzyx(a, axes)
        if a.ndim == 3:
            a = da.expand_dims(a, 0)
        return a

    def _build_dask(self) -> da.Array:
        parts = [self._open_one(p) for p in self.filenames]
        if len(parts) == 1:
            return parts[0]
        return da.concatenate(parts, axis=0)

    @property
    def dask(self) -> da.Array:
        if self._dask_array is None:
            self._dask_array = self._build_dask()
        return self._dask_array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.dask.shape)

    @property
    def dtype(self):
        return self.dask.dtype

    @property
    def ndim(self):
        return self.dask.ndim

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __getitem__(self, key):
        return self.dask[key]

    def __getattr__(self, attr):
        if attr.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        try:
            return getattr(self.dask, attr)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __array__(self):
        n = min(10, self.dask.shape[0])
        return self.dask[:n].compute()

    def min(self) -> float:
        return float(self.dask[0].min().compute())

    def max(self) -> float:
        return float(self.dask[0].max().compute())

    def imshow(self, **kwargs):
        import fastplotlib as fpl

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        window_funcs = kwargs.get("window_funcs", None)
        return fpl.ImageWidget(
            data=self.dask,
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
        progress_callback=None,
        debug=None,
    ):
        outpath = Path(outpath)
        md = dict(self.metadata) if isinstance(self.metadata, dict) else {}
        _write_plane(
            self,
            outpath,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            metadata=md,
            progress_callback=progress_callback,
            debug=debug,
            dshape=(self.shape[0], self.shape[-2], self.shape[-1]),  # (T, Y, X) - Fixed Y/X order
            plane_index=None,
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
                "num_frames": self.num_frames,
                "use_fft": self.use_fft,
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

    def num_planes(self):
        """LBM alias for num_channels."""
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
        # convert to 0 based indexing
        if isinstance(planes, int):
            planes = [planes - 1]
        elif planes is None:
            planes = list(range(self.num_channels))
        elif isinstance(planes, (list, tuple)):
            planes = [p - 1 for p in planes]
        else:
            raise RuntimeError(
                f"Invalid values for requested z-plane type: {type(planes)}"
            )
        for roi in iter_rois(self):
            for plane in planes:
                if not isinstance(plane, int):
                    raise ValueError(f"Plane must be an integer, got {type(plane)}")
                self.roi = roi
                if roi is None:
                    fname = f"plane{plane + 1:02d}_stitched{ext}"
                else:
                    fname = f"plane{plane + 1:02d}_roi{roi}{ext}"

                if ext in [".bin", ".binary"]:
                    # saving to bin for suite2p
                    # we want the filename to be data_raw.bin
                    # so put the fname as the folder name
                    fname_bin_stripped = Path(fname).stem  # remove extension
                    if "structural" in kwargs and kwargs["structural"]:
                        target = outpath / fname_bin_stripped / "data_chan2.bin"
                    else:
                        target = outpath / fname_bin_stripped / "data_raw.bin"
                else:
                    target = outpath.joinpath(fname)

                target.parent.mkdir(exist_ok=True)
                if target.exists() and not overwrite:
                    logger.warning(f"File {target} already exists. Skipping write.")
                    continue

                md = self.metadata.copy()
                md["plane"] = plane + 1  # back to 1-based indexing
                md["mroi"] = roi
                md["roi"] = roi  # alias

                # Determine actual number of frames to write
                num_frames = kwargs.get("num_frames", self.shape[0])

                _write_plane(
                    self,
                    target,
                    overwrite=overwrite,
                    target_chunk_mb=target_chunk_mb,
                    metadata=md,
                    progress_callback=progress_callback,
                    debug=debug,
                    dshape=(num_frames, self.shape[-2], self.shape[-1]),  # (T, Y, X)
                    plane_index=plane,
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
    def __init__(self, array: np.ndarray | str | Path, metadata: dict | None = None):
        if isinstance(array, (str, Path)):
            self.path = Path(array)
            if not self.path.exists():
                raise FileNotFoundError(f"Numpy file not found: {self.path}")
            self.data = np.load(self.path, mmap_mode="r")
            self._tempfile = None
        elif isinstance(array, np.ndarray):
            logger.info(f"Creating temporary .npy file for array.")
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

    def __getitem__(self, item):
        return self.data[item]

    def __array__(self):
        return np.asarray(self.data)

    @property
    def filenames(self) -> list[Path]:
        return [self.path]

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError("metadata must be a dict")
        self._metadata = value

    def close(self):
        if self._tempfile:
            try:
                Path(self._tempfile.name).unlink(missing_ok=True)
            except Exception:
                pass
            self._tempfile = None

    def __del__(self):
        self.close()


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

    def __getitem__(self, item):
        return self.data[item]


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
        outpath = Path(outpath)

        # Normalize planes to 0-based indexing
        if isinstance(planes, int):
            planes = [planes - 1]
        elif planes is None:
            planes = list(range(self.shape[1]))  # all z-planes
        else:
            planes = [p - 1 for p in planes]

        for plane in planes:
            fname = f"plane{plane + 1:02d}{ext}"

            if ext in [".bin", ".binary"]:
                # Suite2p expects data_raw.bin under a folder
                # fname_bin_stripped = Path(fname).stem
                target = outpath / "data_raw.bin"
            else:
                target = outpath.joinpath(fname)

            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists() and not overwrite:
                logger.warning(f"File {target} already exists. Skipping write.")
                continue

            # Metadata per plane
            if isinstance(self.metadata, list):
                md = self.metadata[plane].copy()
            else:
                md = dict(self.metadata)
            md["plane"] = plane + 1  # back to 1-based
            md["z"] = plane

            _write_plane(
                self,
                target,
                overwrite=overwrite,
                target_chunk_mb=target_chunk_mb,
                metadata=md,
                progress_callback=progress_callback,
                debug=debug,
                dshape=(self.shape[0], self.shape[-2], self.shape[-1]),  # (T, Y, X) - Fixed Y/X order
                plane_index=plane,
                **kwargs,
            )


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


# class IsoviewArray:
#     """
#     Reader for Isoview lightsheet microscopy data stored as zarr arrays.

#     Expects a directory structure like:
#         base_dir/
#             TM000000/
#                 SPM00_TM000000_CM00_CHN01.zarr  (Z, Y, X)
#                 SPM00_TM000000_CM01_CHN01.zarr
#                 SPM00_TM000000_CM02_CHN00.zarr
#                 SPM00_TM000000_CM03_CHN00.zarr
#             TM000001/
#                 ...

#     Or a single TM folder:
#         TM000010/
#             SPM00_TM000010_CM00_CHN01.zarr
#             ...

#     Each .zarr file is 3D (Z, Y, X) for a single camera/channel at one timepoint.

#     Note: Not all camera/channel combinations may exist. The array tracks which
#     combinations are available and provides access by sequential index.

#     Shape depends on input:
#     - Multiple TM folders: (T, Z, Views, Y, X) - 5D
#     - Single TM folder: (Z, Views, Y, X) - 4D

#     Parameters
#     ----------
#     path : str or Path
#         Path to the base directory containing TM* folders, or a single TM folder

#     Examples
#     --------
#     >>> arr = IsoviewArray("path/to/zarr_output")  # Multiple timepoints
#     >>> print(arr.shape)  # (T, Z, NumViews, Y, X)
#     >>> arr = IsoviewArray("path/to/TM000010")  # Single timepoint
#     >>> print(arr.shape)  # (Z, NumViews, Y, X)
#     >>> print(arr.views)  # [(0, 1), (1, 1), (2, 0), (3, 0)] - (camera, channel) pairs
#     """

#     def __init__(self, path: str | Path):
#         try:
#             import zarr
#         except ImportError:
#             raise ImportError("zarr is required. Install with `pip install zarr>=3.0`")

#         self.base_path = Path(path)
#         if not self.base_path.exists():
#             raise FileNotFoundError(f"Path does not exist: {self.base_path}")

#         # Check if the input IS a TM folder (single timepoint)
#         self._single_timepoint = False
#         if self.base_path.name.startswith("TM"):
#             # Single TM folder provided
#             zarr_files_in_path = list(self.base_path.glob("*.zarr"))
#             if zarr_files_in_path:
#                 self._single_timepoint = True
#                 self.tm_folders = [self.base_path]
#             else:
#                 raise ValueError(f"TM folder {self.base_path} contains no .zarr files")
#         else:
#             # Find all TM folders and sort them
#             self.tm_folders = sorted(
#                 [d for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith("TM")],
#                 key=lambda x: int(x.name[2:])  # Sort by TM number
#             )

#             if not self.tm_folders:
#                 raise ValueError(f"No TM* folders found in {self.base_path}")

#         # Parse the first TM folder to discover camera/channel combinations
#         first_tm = self.tm_folders[0]
#         zarr_files = sorted(first_tm.glob("*.zarr"))

#         if not zarr_files:
#             raise ValueError(f"No .zarr files found in {first_tm}")

#         # Parse filenames to extract camera/channel info
#         # Expected format: SPM00_TM000000_CM{camera:02d}_CHN{channel:02d}.zarr
#         # Store as list of (camera, channel) tuples - "views"
#         self._views = []  # List of (camera, channel) tuples

#         for zf in sorted(zarr_files):
#             name = zf.stem  # e.g., SPM00_TM000000_CM00_CHN01
#             parts = name.split("_")

#             # Find CM and CHN parts
#             cm_idx = None
#             chn_idx = None
#             for part in parts:
#                 if part.startswith("CM"):
#                     cm_idx = int(part[2:])
#                 elif part.startswith("CHN"):
#                     chn_idx = int(part[3:])

#             if cm_idx is not None and chn_idx is not None:
#                 self._views.append((cm_idx, chn_idx))

#         if not self._views:
#             raise ValueError(f"No valid camera/channel combinations found in {first_tm}")

#         logger.info(
#             f"IsoviewArray: {len(self.tm_folders)} timepoints, "
#             f"{len(self._views)} views: {self._views}"
#         )

#         # Open first zarr to get Z, Y, X dimensions and metadata
#         first_zarr_path = zarr_files[0]
#         first_zarr = zarr.open(first_zarr_path, mode="r")

#         # Handle OME-Zarr groups
#         if isinstance(first_zarr, zarr.Group):
#             if "0" in first_zarr:
#                 first_arr = first_zarr["0"]
#             else:
#                 raise ValueError(f"OME-Zarr group missing '0' array: {first_zarr_path}")
#             # Store group attrs as metadata
#             self._zarr_attrs = dict(first_zarr.attrs)
#         else:
#             first_arr = first_zarr
#             self._zarr_attrs = dict(first_arr.attrs) if hasattr(first_arr, 'attrs') else {}

#         self._single_shape = first_arr.shape  # (Z, Y, X)
#         self._dtype = first_arr.dtype

#         # Cache for opened zarr arrays: (t_idx, view_idx) -> zarr array
#         self._zarr_cache = {}

#     def _get_zarr(self, t_idx: int, view_idx: int):
#         """Get or open a zarr array for the given timepoint and view index."""
#         import zarr

#         cache_key = (t_idx, view_idx)
#         if cache_key in self._zarr_cache:
#             return self._zarr_cache[cache_key]

#         camera, channel = self._views[view_idx]
#         tm_folder = self.tm_folders[t_idx]

#         # Find matching zarr file
#         pattern = f"*_CM{camera:02d}_CHN{channel:02d}.zarr"
#         matches = list(tm_folder.glob(pattern))

#         if not matches:
#             raise FileNotFoundError(
#                 f"No zarr file matching {pattern} in {tm_folder}"
#             )

#         zarr_path = matches[0]
#         z = zarr.open(zarr_path, mode="r")

#         # Handle OME-Zarr groups
#         if isinstance(z, zarr.Group):
#             if "0" in z:
#                 z = z["0"]
#             else:
#                 raise ValueError(f"OME-Zarr group missing '0' array: {zarr_path}")

#         self._zarr_cache[cache_key] = z
#         return z

#     @property
#     def shape(self) -> tuple[int, ...]:
#         """
#         Shape depends on input:
#         - Multiple TM folders: (T, Z, Views, Y, X) - 5D
#         - Single TM folder: (Z, Views, Y, X) - 4D
#         """
#         z, y, x = self._single_shape
#         if self._single_timepoint:
#             return (z, len(self._views), y, x)
#         else:
#             t = len(self.tm_folders)
#             return (t, z, len(self._views), y, x)

#     @property
#     def dtype(self):
#         return self._dtype

#     @property
#     def ndim(self):
#         return 4 if self._single_timepoint else 5

#     @property
#     def size(self):
#         return np.prod(self.shape)

#     @property
#     def views(self) -> list[tuple[int, int]]:
#         """List of (camera, channel) tuples for each view index."""
#         return self._views

#     @property
#     def num_views(self) -> int:
#         """Number of camera/channel views."""
#         return len(self._views)

#     @property
#     def num_timepoints(self) -> int:
#         """Number of timepoints."""
#         return len(self.tm_folders)

#     @property
#     def metadata(self) -> dict:
#         """
#         Metadata extracted from zarr attributes.

#         Returns a dict with keys like:
#         - timestamp, time_point, camera_type, wavelength, exposure_time
#         - dimensions, z_step, planes, detection_objective
#         - pixel_resolution, dz, frame_rate, num_frames
#         - Plus computed fields: num_timepoints, views, shape
#         """
#         # Start with zarr attrs
#         meta = dict(self._zarr_attrs)

#         # Remove nested OME metadata (keep it accessible separately)
#         ome = meta.pop("ome", None)

#         # Add computed fields
#         meta["num_timepoints"] = len(self.tm_folders)
#         meta["views"] = self._views
#         meta["shape"] = self.shape
#         meta["single_timepoint"] = self._single_timepoint

#         # Extract useful info from OME if present
#         if ome and "multiscales" in ome:
#             multiscales = ome["multiscales"]
#             if multiscales and len(multiscales) > 0:
#                 ms = multiscales[0]
#                 if "axes" in ms:
#                     meta["axes"] = ms["axes"]
#                 if "datasets" in ms and len(ms["datasets"]) > 0:
#                     ds = ms["datasets"][0]
#                     if "coordinateTransformations" in ds:
#                         for ct in ds["coordinateTransformations"]:
#                             if ct.get("type") == "scale":
#                                 meta["scale"] = ct.get("scale")

#         return meta

#     @property
#     def ome_metadata(self) -> dict | None:
#         """Raw OME-Zarr metadata if available."""
#         return self._zarr_attrs.get("ome")

#     def __len__(self):
#         return self.shape[0]

#     def view_index(self, camera: int, channel: int) -> int:
#         """Get view index for a specific camera/channel combination."""
#         try:
#             return self._views.index((camera, channel))
#         except ValueError:
#             raise ValueError(
#                 f"No view for camera={camera}, channel={channel}. "
#                 f"Available views: {self._views}"
#             )

#     def __getitem__(self, key):
#         """
#         Index the array.

#         Shape depends on input:
#         - Multiple TM folders: (T, Z, Views, Y, X) - 5D
#         - Single TM folder: (Z, Views, Y, X) - 4D

#         Supports integer indexing, slices, and combinations.
#         """
#         if not isinstance(key, tuple):
#             key = (key,)

#         # Normalize indices helper
#         def to_indices(k, max_val):
#             """Convert key to list of indices."""
#             if isinstance(k, int):
#                 if k < 0:
#                     k = max_val + k
#                 return [k]
#             elif isinstance(k, slice):
#                 return list(range(*k.indices(max_val)))
#             elif isinstance(k, (list, np.ndarray)):
#                 return list(k)
#             else:
#                 return list(range(max_val))

#         if self._single_timepoint:
#             # 4D indexing: (Z, Views, Y, X)
#             key = key + (slice(None),) * (4 - len(key))
#             z_key, view_key, y_key, x_key = key
#             t_indices = [0]  # Only one timepoint
#             t_key = 0  # For squeeze tracking
#         else:
#             # 5D indexing: (T, Z, Views, Y, X)
#             key = key + (slice(None),) * (5 - len(key))
#             t_key, z_key, view_key, y_key, x_key = key
#             t_indices = to_indices(t_key, len(self.tm_folders))

#         z_indices = to_indices(z_key, self._single_shape[0])
#         view_indices = to_indices(view_key, len(self._views))

#         # Build output array (always 5D internally for simplicity)
#         out_shape = (
#             len(t_indices),
#             len(z_indices),
#             len(view_indices),
#             *self._single_shape[1:],  # Y, X
#         )

#         # Handle Y, X slicing
#         if isinstance(y_key, int):
#             out_shape = out_shape[:3] + (1,) + out_shape[4:]
#         elif isinstance(y_key, slice):
#             y_size = len(range(*y_key.indices(self._single_shape[1])))
#             out_shape = out_shape[:3] + (y_size,) + out_shape[4:]

#         if isinstance(x_key, int):
#             out_shape = out_shape[:4] + (1,)
#         elif isinstance(x_key, slice):
#             x_size = len(range(*x_key.indices(self._single_shape[2])))
#             out_shape = out_shape[:4] + (x_size,)

#         result = np.empty(out_shape, dtype=self._dtype)

#         for ti, t_idx in enumerate(t_indices):
#             for vi, view_idx in enumerate(view_indices):
#                 zarr_arr = self._get_zarr(t_idx, view_idx)

#                 # Index the zarr array (Z, Y, X)
#                 data = zarr_arr[z_key, y_key, x_key]

#                 # Handle dimension reduction from integer indexing
#                 if isinstance(z_key, int):
#                     data = data[np.newaxis, ...]
#                 if isinstance(y_key, int):
#                     data = data[:, np.newaxis, :]
#                 if isinstance(x_key, int):
#                     data = data[:, :, np.newaxis]

#                 result[ti, :, vi, ...] = data

#         # Squeeze out singleton dimensions from integer indexing
#         if self._single_timepoint:
#             # 4D: always squeeze out the T dimension, then check Z, Views, Y, X
#             result = np.squeeze(result, axis=0)
#             int_indexed = [
#                 isinstance(z_key, int),
#                 isinstance(view_key, int),
#                 isinstance(y_key, int),
#                 isinstance(x_key, int),
#             ]
#             # Squeeze in reverse order to preserve axis indices
#             for ax in range(3, -1, -1):
#                 if int_indexed[ax] and ax < result.ndim and result.shape[ax] == 1:
#                     result = np.squeeze(result, axis=ax)
#         else:
#             # 5D: check T, Z, Views, Y, X
#             int_indexed = [
#                 isinstance(t_key, int),
#                 isinstance(z_key, int),
#                 isinstance(view_key, int),
#                 isinstance(y_key, int),
#                 isinstance(x_key, int),
#             ]
#             # Squeeze in reverse order to preserve axis indices
#             for ax in range(4, -1, -1):
#                 if int_indexed[ax] and ax < result.ndim and result.shape[ax] == 1:
#                     result = np.squeeze(result, axis=ax)

#         return result

#     def __array__(self):
#         """Materialize full array into memory."""
#         return self[:]

#     @property
#     def filenames(self) -> list[str]:
#         """Return list of TM folder paths as strings (for GUI compatibility)."""
#         return [str(f) for f in self.tm_folders]

#     def __repr__(self):
#         return (
#             f"IsoviewArray(shape={self.shape}, dtype={self.dtype}, "
#             f"views={self._views})"
#         )


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
        outpath: Path,
        planes=None,
        target_chunk_mb: int = 20,
        ext: str = ".bin",
        progress_callback=None,
        debug: bool = False,
        overwrite: bool = False,
        output_name: str | None = None,
        **kwargs,
    ):
        """Write BinArray to disk."""

        outpath = Path(outpath)
        if output_name is None:
            output_name = "data_raw.bin"

        outfile = outpath / output_name

        # Ensure directory exists
        outpath.mkdir(parents=True, exist_ok=True)

        # Write the binary file
        if not outfile.exists() or overwrite:
            logger.info(f"Writing binary to {outfile}")
            # Copy the memmap
            new_file = np.memmap(outfile, mode="w+", dtype=self.dtype, shape=self.shape)
            new_file[:] = self._file[:]
            new_file.flush()
            del new_file
        else:
            logger.info(f"Binary file already exists: {outfile}")

        # Write ops.npy if we have metadata
        if self.metadata:
            ops_file = outpath / "ops.npy"
            ops_data = {
                **self.metadata,
                "Ly": self.Ly,
                "Lx": self.Lx,
                "nframes": self.nframes,
            }
            np.save(ops_file, ops_data)
            logger.info(f"Wrote ops.npy to {ops_file}")

        return outpath


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
