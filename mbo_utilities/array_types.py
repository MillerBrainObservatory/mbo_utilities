from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, List, Sequence

import fastplotlib as fpl
import h5py
import numpy as np
import tifffile
import zarr
from dask import array as da

from mbo_utilities import log
from mbo_utilities._parsing import _make_json_serializable
from mbo_utilities._writers import _write_plane
from mbo_utilities.file_io import (
    _multi_tiff_to_fsspec,
    _convert_range_to_slice,
    expand_paths, files_to_dask, derive_tag_from_filename,
)
from mbo_utilities.metadata import get_metadata, clean_scanimage_metadata
from mbo_utilities.phasecorr import ALL_PHASECORR_METHODS, bidir_phasecorr
from mbo_utilities.roi import iter_rois
from mbo_utilities.scanreader import utils
from mbo_utilities.util import subsample_array

logger = log.get("array_types")

CHUNKS_4D = {0: 1, 1: "auto", 2: -1, 3: -1}
CHUNKS_3D = {0: 1, 1: -1, 2: -1}

class LazyArrayProtocol:
    """
    Protocol for lazy array types.

    Must implement:
    - __getitem__    (method)
    - __len__        (method)
    - min            (property)
    - max            (property)
    - ndim           (property)
    - shape          (property)
    - dtype          (property)
    - metadata       (property)

    Optionally implement:
    - __array__      (method)
    - imshow         (method)
    - _imwrite       (method)
    - close          (method)
    - chunks         (property)
    - dask           (property)
    """

    def __getitem__(self, key: int | slice | tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __array__(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def min(self) -> float:
        raise NotImplementedError

    @property
    def max(self) -> float:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError



def register_zplanes_s3d(
        filenames,
        metadata,
        outpath=None,
        progress_callback=None
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
        logger.warning("Missing required metadata for axial alignment: frame_rate / num_planes")
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
        return 1


def _safe_get_metadata(path: Path) -> dict:
    try:
        return get_metadata(path)
    except Exception:
        return {}


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

        # resolve both possible bins
        self.raw_file = Path(self.metadata.get("raw_file", path.with_name("data_raw.bin")))
        self.reg_file = Path(self.metadata.get("reg_file", path.with_name("data.bin")))

        # choose which one to use
        if path.suffix == ".bin":
            self.active_file = path
        else:
            self.active_file = self.reg_file if self.reg_file.exists() else self.raw_file

        # confirm
        if not self.active_file.exists():
            raise FileNotFoundError(f"Active binary not found: {self.active_file}")

        self.Ly = self.metadata["Ly"]
        self.Lx = self.metadata["Lx"]
        self.nframes = self.metadata.get("nframes", self.metadata.get("n_frames"))
        self.shape = (self.nframes, self.Ly, self.Lx)
        self.dtype = np.int16
        self._file = np.memmap(self.active_file, mode="r", dtype=self.dtype, shape=self.shape)
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

        # if both are available, and the same shape, show both
        if "raw_file" in self.metadata and "reg_file" in self.metadata:
            try:
                raw = Suite2pArray(self.metadata["raw_file"])
                reg = Suite2pArray(self.metadata["reg_file"])
                if raw.shape == reg.shape:
                    arrays.extend([raw, reg])
                    names.extend(["raw", "registered"])
                else:
                    arrays.append(reg)
                    names.append("registered")
            except Exception as e:
                logger.warning(f"Could not open raw_file or reg_file: {e}")
        if "reg_file" in self.metadata:
            try:
                reg = Suite2pArray(self.metadata["reg_file"])
                arrays.append(reg)
                names.append("registered")
            except Exception as e:
                logger.warning(f"Could not open reg_file: {e}")

        elif "raw_file" in self.metadata:
            try:
                raw = Suite2pArray(self.metadata["raw_file"])
                arrays.append(raw)
                names.append("raw")
            except Exception as e:
                logger.warning(f"Could not open raw_file: {e}")

        if not arrays:
            raise ValueError("No loadable raw_file or reg_file in ops")

        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        histogram_widget = kwargs.get("histogram_widget", True)
        window_funcs = kwargs.get("window_funcs", None)

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
    def __init__(self, filenames: Path | str, dataset: str = "mov"):
        self.filenames = Path(filenames)
        self._f = h5py.File(self.filenames, "r")
        self._d = self._f[dataset]
        self.shape = self._d.shape
        self.dtype = self._d.dtype
        self.ndim = self._d.ndim
        self._metadata = None

    @property
    def num_planes(self) -> int:
        # TODO: not sure what to do here
        return 14

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
        if self._metadata is not None:
            return self._metadata
        return dict(self._f.attrs)

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

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

        self.tags = [derive_tag_from_filename(f) for f in self.filenames]

    @property
    def metadata(self) -> dict:
        return self._metadata or {}

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    @property
    def chunks(self):
        return self._chunks or CHUNKS_4D

    @property
    def dask(self) -> da.Array:
        if self._dask_array is not None:
            return self._dask_array

        if len(self.filenames) == 1:
            arr = tifffile.imread(self.filenames[0], aszarr=True)
            darr = da.from_zarr(arr)
            if darr.ndim == 2:
                darr = darr[None, None, :, :]
            elif darr.ndim == 3:
                darr = darr[:, None, :, :]
        else:
            darr = files_to_dask(self.filenames)
            if darr.ndim == 3:
                darr = darr[None, :, :, :]
        self._dask_array = darr
        return darr

    @property
    def shape(self):
        return tuple(self.dask.shape)

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
        return getattr(self.dask, attr)

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
        from mbo_utilities.lazy_array import _write_plane
        from mbo_utilities.file_io import get_plane_from_filename

        md = self.metadata.copy()
        plane = md.get("plane")
        if plane is None:
            # Try to get from filename
            try:
                plane = get_plane_from_filename(Path(outpath).stem, None)
            except (ValueError, AttributeError):
                # Default to 0 if we can't determine plane number
                plane = 0
                logger.debug(f"Could not determine plane number, defaulting to {plane}")

        outpath = Path(outpath)
        ext = ext.lower().lstrip(".")
        fname = f"plane{plane:03d}.{ext}" if ext != "bin" else "data_raw.bin"
        target = outpath.joinpath(fname) if outpath.is_dir() else outpath.parent.joinpath(fname)

        _write_plane(
            self,
            target,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            metadata=md,
            progress_callback=progress_callback,
            debug=debug,
            dshape=(self.shape[0], self.shape[-1], self.shape[-2]),
            plane_index=None,
            **kwargs,
        )
        return outpath


@dataclass
class NpyArray:
    filenames: list[Path]
    _metadata: dict = field(default_factory=dict, init=False)

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

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    @property
    def min(self) -> float:
        return float(self._file.min())

    @property
    def max(self) -> float:
        return float(self._file.max())

    def __getitem__(self, key):
        return self._file[key]

    def __len__(self):
        return self.shape[0]

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

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    def __getitem__(self, key):
        return self.dask[key]

    def __getattr__(self, attr):
        return getattr(self.dask, attr)

    def __array__(self):
        n = min(10, self.dask.shape[0])
        return self.dask[:n].compute()

    def min(self) -> float:
        return float(self.dask[0].min().compute())

    def max(self) -> float:
        return float(self.dask[0].max().compute())

    def imshow(self, **kwargs) -> fpl.ImageWidget:
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
            dshape=(self.shape[0], self.shape[-1], self.shape[-2]),
            plane_index=None,
        )


class MboRawArray:
    """
    Lazy reader for raw ScanImage multi-ROI TIFF files without unnecessary class hierarchy.

    Handles:
    - Multi-ROI raw ScanImage acquisitions
    - ROI stitching (roi=None) or splitting (roi=0)
    - Bidirectional scan phase correction
    - Lazy indexing: __getitem__ only reads requested data

    Internal ROI representation:
    - roi_fields: list of dicts with keys:
      - height, width: ROI dimensions in pixels
      - yslice, xslice: where to cut from TIFF pages
      - output_yslice, output_xslice: where to paste in stitched output

    ROI Semantics:
    - roi=None: Return stitched full FOV (all ROIs merged)
    - roi=0: Return tuple of all individual ROIs
    - roi=int>0: Return specific ROI (1-based indexing)
    - roi=list[int]: Return tuple of specified ROIs
    """

    def __init__(
        self,
        files: str | Path | list = None,
        roi: int | Sequence[int] | None = None,
        fix_phase: bool = True,
        phasecorr_method: str = "mean",
        border: int | tuple[int, int, int, int] = 3,
        upsample: int = 5,
        max_offset: int = 4,
        use_fft: bool = False,
    ):
        """
        Initialize a lazy reader for raw ScanImage multi-ROI TIFFs.

        Parameters
        ----------
        files : str, Path, or list of str/Path, optional
            TIFF file(s) to load
        roi : int, list[int], or None, optional
            ROI selection: None=stitch, 0=split, int>0=single, list=multiple
        fix_phase : bool, default True
            Apply bidirectional scan phase correction
        phasecorr_method : str, default "mean"
            Phase correction method ("mean", "max", "std", "mean-sub")
        border : int or (top, bottom, left, right), default 3
            Border width for phase correction
        upsample : int, default 5
            Upsampling factor for phase correlation
        max_offset : int, default 4
            Maximum allowed phase offset in pixels
        use_fft : bool, default False
            Use FFT for phase correlation instead of direct
        """
        self._metadata = {"cleaned_scanimage_metadata": False}
        self._fix_phase = fix_phase
        self._phasecorr_method = phasecorr_method
        self.border: int | tuple[int, int, int, int] = border
        self.max_offset: int = max_offset
        self.upsample: int = upsample
        self.reference = ""
        self._roi = roi
        self._offset = 0.0
        self._use_fft = use_fft

        # Lazy-loaded TIFF files
        self._tiff_files = None
        self.filenames = None
        self._dtype = np.int16

        # Parsed ROI structure (list of dicts with slice info)
        self.roi_fields = []

        # Header and metadata from first file
        self.header = ""

        if files:
            self.read_data(files)

    @property
    def tiff_files(self):
        """Lazy-load TiffFile objects."""
        if self._tiff_files is None:
            self._tiff_files = [
                tifffile.TiffFile(str(f), mode="r") for f in self.filenames
            ]
        return self._tiff_files

    @property
    def num_channels(self) -> int:
        """Number of channels (aka planes in LBM terminology)."""
        import re
        match = re.search(r"hChannels\.channelSave = (?P<channels>.*)", self.header)
        if match:
            from mbo_utilities.scanreader.utils import matlabstr2py
            channels = matlabstr2py(match.group("channels"))
            return len(channels) if isinstance(channels, list) else 1
        return 1

    @property
    def num_frames(self) -> int:
        """Total time frames across all files."""
        return sum(len(tf.pages) // self.num_channels for tf in self.tiff_files)

    @property
    def _page_height(self) -> int:
        """Height of TIFF pages in pixels."""
        return self.tiff_files[0].pages[0].shape[0]

    @property
    def _page_width(self) -> int:
        """Width of TIFF pages in pixels."""
        return self.tiff_files[0].pages[0].shape[1]

    @property
    def num_rois(self) -> int:
        """Number of ROIs in this scan."""
        return len(self.roi_fields)

    @property
    def roi(self):
        """Get the current ROI index."""
        return self._roi

    @roi.setter
    def roi(self, value):
        """Set the current ROI index."""
        self._roi = value

    @property
    def fix_phase(self) -> bool:
        """Whether to apply phase correction."""
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("fix_phase must be a boolean value.")
        self._fix_phase = value

    @property
    def phasecorr_method(self) -> str:
        """Current phase correction method."""
        return self._phasecorr_method

    @phasecorr_method.setter
    def phasecorr_method(self, value: str | None):
        if value not in ALL_PHASECORR_METHODS:
            raise ValueError(
                f"Unsupported phase correction method: {value}. "
                f"Supported methods are: {ALL_PHASECORR_METHODS}"
            )
        if value is None:
            self.fix_phase = False
        self._phasecorr_method = value

    @property
    def use_fft(self) -> bool:
        """Whether to use FFT for phase correlation."""
        return self._use_fft

    @use_fft.setter
    def use_fft(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("use_fft must be a boolean value.")
        self._use_fft = value

    @property
    def offset(self) -> float | np.ndarray:
        """Phase offset result from last correction."""
        return self._offset

    @offset.setter
    def offset(self, value: float | np.ndarray):
        """Set phase offset."""
        if isinstance(value, int):
            self._offset = float(value)
        else:
            self._offset = value

    @property
    def metadata(self) -> dict:
        """Full metadata including phase correction settings."""
        self._metadata.update({
            "fix_phase": self.fix_phase,
            "phasecorr_method": self.phasecorr_method,
            "offset": self.offset,
            "border": self.border,
            "upsample": self.upsample,
            "max_offset": self.max_offset,
            "num_frames": self.num_frames,
            "use_fft": self.use_fft,
        })
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata.update(value)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape (time, channels, height, width) relative to current ROI."""
        if self.roi is not None:
            if not isinstance(self.roi, (list, tuple)):
                if self.roi > 0:
                    # Single ROI: use its width only
                    roi_field = self.roi_fields[self.roi - 1]
                    width = roi_field["xslice"].stop - roi_field["xslice"].start
                    return (
                        self.num_frames,
                        self.num_channels,
                        roi_field["height"],
                        width,
                    )
        # roi=None or list: return full FOV shape
        fov_height = self._get_fov_height()
        fov_width = self._get_fov_width()
        return (self.num_frames, self.num_channels, fov_height, fov_width)

    @property
    def shape_full(self) -> tuple[int, int, int, int]:
        """Full FOV shape regardless of ROI setting."""
        return (
            self.num_frames,
            self.num_channels,
            self._get_fov_height(),
            self._get_fov_width(),
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 4)."""
        return 4

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(np.prod(self.shape))

    @property
    def dtype(self) -> np.dtype:
        """Data type of the arrays."""
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def num_planes(self) -> int:
        """Alias for num_channels (LBM terminology)."""
        return self.num_channels

    def _get_fov_height(self) -> int:
        """Full field-of-view height (sum of all ROI heights with gaps)."""
        if not self.roi_fields:
            return self._page_height
        total = 0
        for roi_field in self.roi_fields:
            total = max(total, roi_field["output_yslice"].stop)
        return total

    def _get_fov_width(self) -> int:
        """Full field-of-view width (maximum x position of all ROIs)."""
        if not self.roi_fields:
            return self._page_width
        # Get the rightmost x position of any ROI
        return max(rf["output_xslice"].stop for rf in self.roi_fields)

    def read_data(self, filenames, dtype=np.int16):
        """Load TIFF files and parse ROI metadata."""
        self.filenames = expand_paths(filenames)
        self.dtype = dtype
        self.reference = None

        # Read header and ScanImage metadata from first TIFF
        first_tiff = self.tiff_files[0]
        self.header = first_tiff.scanimage_metadata.get("Header", "") if hasattr(
            first_tiff, "scanimage_metadata"
        ) else ""

        # Extract and clean metadata
        first_path = Path(self.filenames[0])
        self._metadata = get_metadata(first_path)
        self._metadata["si"] = _make_json_serializable(
            first_tiff.scanimage_metadata
        )
        self._metadata = clean_scanimage_metadata(self._metadata)
        self._metadata["cleaned_scanimage_metadata"] = True

        # Parse ROI fields from ScanImage metadata
        self._parse_roi_fields()

    def _parse_roi_fields(self):
        """Extract ROI geometry from ScanImage metadata into simple dicts."""
        try:
            roi_infos = self.tiff_files[0].scanimage_metadata["RoiGroups"][
                "imagingRoiGroup"
            ]["rois"]
        except KeyError:
            raise RuntimeError(
                "This file is not a raw-scanimage tiff or is missing tiff.scanimage_metadata."
            )

        # Ensure roi_infos is a list
        if not isinstance(roi_infos, list):
            roi_infos = [roi_infos]

        # Filter out malformed ROIs
        roi_infos = [r for r in roi_infos if isinstance(r.get("zs"), (int, float, list))]

        # Build ROI field structures
        # First pass: extract all ROI dimensions
        roi_dims = []
        for roi_idx, roi_info in enumerate(roi_infos):
            scanfields = roi_info.get("scanfields", [])
            if not isinstance(scanfields, list):
                scanfields = [scanfields]

            if not scanfields:
                continue

            # Since MBO forces z=[0], we only use the first scanfield
            sf = scanfields[0]
            height, width = sf["pixelResolutionXY"]
            roi_dims.append((roi_idx, int(height), int(width)))

        # Second pass: build field dicts with correct stitching positions
        self.roi_fields = []
        next_y_output_pos = 0  # Output Y position (accumulates for stacking)
        next_x_output_pos = 0  # Output X position (accumulates for horizontal placement)
        next_y_tiff_pos = 0    # TIFF page Y position (accumulates where ROIs are stored in TIFF)

        for roi_idx, height, width in roi_dims:
            roi_field = {
                "height": height,
                "width": width,
                # Where in TIFF page to read this ROI
                "yslice": slice(next_y_tiff_pos, next_y_tiff_pos + height),
                "xslice": slice(0, width),
                # Where to place this ROI in the output stitched image
                "output_yslice": slice(next_y_output_pos, next_y_output_pos + height),
                "output_xslice": slice(next_x_output_pos, next_x_output_pos + width),
                "roi_idx": roi_idx,
            }
            self.roi_fields.append(roi_field)
            # ROIs are stacked vertically in TIFF pages
            next_y_tiff_pos += height
            # ROIs are placed horizontally in output
            next_x_output_pos += width

    def save_fsspec(self, filenames):
        """Generate kerchunk references for cloud-friendly access."""
        base_dir = Path(filenames[0]).parent
        combined_json_path = base_dir / "combined_refs.json"

        if combined_json_path.is_file():
            logger.debug(f"Removing existing combined reference file: {combined_json_path}")
            combined_json_path.unlink()

        logger.debug(f"Generating combined kerchunk reference for {len(filenames)} files…")
        combined_refs = _multi_tiff_to_fsspec(tif_files=filenames, base_dir=base_dir)

        with open(combined_json_path, "w") as _f:
            json.dump(combined_refs, _f)

        logger.info(f"Combined kerchunk reference written to {combined_json_path}")
        self.reference = combined_json_path
        return combined_json_path

    def _read_pages(
        self,
        frames: list[int],
        chans: list[int],
        yslice: slice = slice(None),
        xslice: slice = slice(None),
    ) -> np.ndarray:
        """Read TIFF pages with optional phase correction.

        Parameters
        ----------
        frames : list[int]
            Frame indices to read
        chans : list[int]
            Channel indices to read
        yslice : slice
            Y-axis slice into each TIFF page
        xslice : slice
            X-axis slice into each TIFF page

        Returns
        -------
        np.ndarray
            Array of shape (len(frames), len(chans), height, width)
        """
        # Map (frame, channel) to TIFF page index
        pages = [f * self.num_channels + z for f in frames for z in chans]

        # Compute output dimensions
        tiff_width_px = len(utils.listify_index(xslice, self._page_width))
        tiff_height_px = len(utils.listify_index(yslice, self._page_height))
        buf = np.empty(
            (len(pages), tiff_height_px, tiff_width_px), dtype=self.dtype
        )

        # Read pages from TIFF files
        start = 0
        for tf in self.tiff_files:
            end = start + len(tf.pages)
            idxs = [i for i, p in enumerate(pages) if start <= p < end]
            if not idxs:
                start = end
                continue

            # Read chunk from this TIFF file
            frame_idx = [pages[i] - start for i in idxs]
            chunk = tf.asarray(key=frame_idx)[..., yslice, xslice]

            # Apply phase correction if enabled
            if self.fix_phase:
                corrected, offset = bidir_phasecorr(
                    chunk,
                    method=self.phasecorr_method,
                    upsample=self.upsample,
                    max_offset=self.max_offset,
                    border=self.border,
                    use_fft=self.use_fft,
                )
                buf[idxs] = corrected
                self.offset = offset
            else:
                buf[idxs] = chunk
                self.offset = 0.0

            start = end

        return buf.reshape(len(frames), len(chans), tiff_height_px, tiff_width_px)

    def _get_roi_indices(self) -> list[int]:
        """Get list of ROI indices to read based on self.roi setting.

        Returns
        -------
        list[int]
            0-based ROI indices to read
        """
        if self.roi is None:
            return None  # Signal to stitch all
        elif self.roi == 0:
            return list(range(self.num_rois))  # All ROIs individually
        elif isinstance(self.roi, int):
            return [self.roi - 1]  # Single ROI (convert to 0-based)
        elif isinstance(self.roi, list):
            return [r - 1 for r in self.roi]  # Multiple ROIs (convert to 0-based)
        else:
            raise ValueError(f"Invalid roi value: {self.roi}")

    def __getitem__(self, key: int | slice | tuple) -> np.ndarray | tuple:
        """Lazy indexing with ROI handling.

        Supports:
        - arr[t] -> single frame
        - arr[t:t+n] -> time slice
        - arr[t, z] -> frame and channel
        - arr[t, z, y, x] -> full indexing

        Returns
        -------
        np.ndarray or tuple[np.ndarray, ...]
            Data array(s) - tuple if roi=0 or roi=list
        """
        if not isinstance(key, tuple):
            key = (key,)

        # Pad key with slice(None) for missing dimensions
        t_key, z_key, _, _ = tuple(_convert_range_to_slice(k) for k in key) + (
            slice(None),
        ) * (4 - len(key))

        # Convert to frame/channel indices
        frames = utils.listify_index(t_key, self.num_frames)
        chans = utils.listify_index(z_key, self.num_channels)

        if not frames or not chans:
            return np.empty(0)

        logger.debug(
            f"Phase-corrected: {self.fix_phase}/{self.phasecorr_method}, "
            f"channels: {chans}, roi: {self.roi}",
        )

        # Get ROI indices to read
        roi_indices = self._get_roi_indices()

        if roi_indices is None:
            # Stitch all ROIs
            out = self._read_and_stitch(frames, chans)
        else:
            # Return tuple of individual ROIs or single ROI
            roi_arrays = [
                self._read_single_roi(roi_idx, frames, chans)
                for roi_idx in roi_indices
            ]
            out = tuple(roi_arrays) if len(roi_arrays) > 1 else roi_arrays[0]

        # Squeeze dimensions that were indexed with int
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

        return out

    def _read_single_roi(self, roi_idx: int, frames: list[int], chans: list[int]) -> np.ndarray:
        """Read a single ROI."""
        roi_field = self.roi_fields[roi_idx]
        return self._read_pages(
            frames, chans,
            yslice=roi_field["yslice"],
            xslice=roi_field["xslice"],
        )

    def _read_and_stitch(self, frames: list[int], chans: list[int]) -> np.ndarray:
        """Read all ROIs and stitch them into a single FOV."""
        # First, read all ROI data
        roi_data_list = []
        max_height = 0
        total_width = 0

        for roi_field in self.roi_fields:
            roi_data = self._read_pages(
                frames, chans,
                yslice=roi_field["yslice"],
                xslice=roi_field["xslice"],
            )
            roi_data_list.append(roi_data)
            # Get dimensions: roi_data is (n_frames, n_chans, roi_height, roi_width)
            roi_height = roi_data.shape[2]
            roi_width = roi_data.shape[3]
            max_height = max(max_height, roi_height)
            total_width += roi_width

        # Create output array with correct dimensions
        # ROIs are placed side-by-side horizontally
        out = np.zeros(
            (len(frames), len(chans), max_height, total_width), dtype=self.dtype
        )

        # Place each ROI in the output
        x_offset = 0
        for roi_data in roi_data_list:
            roi_height = roi_data.shape[2]
            roi_width = roi_data.shape[3]
            out[:, :, :roi_height, x_offset:x_offset+roi_width] = roi_data
            x_offset += roi_width

        return out

    def __len__(self) -> int:
        """Number of time frames."""
        return self.num_frames

    def __array__(self) -> np.ndarray:
        """Materialize to NumPy array with intelligent subsampling."""
        return subsample_array(self, ignore_dims=[-1, -2, -3])

    def min(self) -> float:
        """Minimum value in first TIFF page."""
        page = self.tiff_files[0].pages[0]
        return float(np.min(page.asarray()))

    def max(self) -> float:
        """Maximum value in first TIFF page."""
        page = self.tiff_files[0].pages[0]
        return float(np.max(page.asarray()))

    def close(self):
        """Close all open TIFF files."""
        if self._tiff_files is not None:
            for tf in self._tiff_files:
                tf.close()
            self._tiff_files = None

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite: bool = False,
        target_chunk_mb: int = 50,
        ext: str = ".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        """Write array data to file(s).

        Parameters
        ----------
        outpath : Path or str
            Output directory or file path
        overwrite : bool
            Whether to overwrite existing files
        target_chunk_mb : int
            Target chunk size in MB for efficient I/O
        ext : str
            File extension (".tiff", ".zarr", ".bin", ".h5", ".nwb")
        progress_callback : callable, optional
            Callback for progress updates
        planes : int or list[int], optional
            Plane indices to write (default: all)
        """
        # Convert plane indices to 0-based
        if isinstance(planes, int):
            planes = [planes - 1]
        elif planes is None:
            planes = list(range(self.num_planes))
        else:
            planes = [p - 1 for p in planes]

        outpath = Path(outpath)
        for roi in iter_rois(self):
            for plane in planes:
                if not isinstance(plane, int):
                    raise ValueError(f"Plane must be an integer, got {type(plane)}")

                self.roi = roi

                # Generate filename
                if roi is None:
                    fname = f"plane{plane + 1:02d}_stitched{ext}"
                else:
                    fname = f"plane{plane + 1:02d}_roi{roi}{ext}"

                # Determine output path
                if ext in [".bin", ".binary"]:
                    fname_base = Path(fname).stem
                    if "structural" in kwargs and kwargs["structural"]:
                        target = outpath / fname_base / "data_chan2.bin"
                    else:
                        target = outpath / fname_base / "data_raw.bin"
                else:
                    target = outpath / fname if outpath.is_dir() else outpath

                target.parent.mkdir(exist_ok=True, parents=True)

                if target.exists() and not overwrite:
                    logger.warning(f"File {target} already exists. Skipping write.")
                    continue

                # Write the data
                md = self.metadata.copy()
                md["plane"] = plane + 1  # Back to 1-based indexing
                md["mroi"] = roi
                md["roi"] = roi

                _write_plane(
                    self,
                    target,
                    overwrite=overwrite,
                    target_chunk_mb=target_chunk_mb,
                    metadata=md,
                    progress_callback=progress_callback,
                    debug=debug,
                    dshape=(self.shape[0], self.shape[-1], self.shape[-2]),
                    plane_index=plane,
                    **kwargs,
                )

    def imshow(self, **kwargs):
        """Create interactive visualization with fastplotlib.

        Creates ImageWidget with one pane per ROI (or all stitched if roi=None).

        Returns
        -------
        fastplotlib.ImageWidget
            Interactive image viewer
        """
        arrays = []
        names = []

        for roi in iter_rois(self):
            arr = copy.copy(self)
            arr.roi = roi
            arr.fix_phase = False  # Disable phase correction for display
            arr.use_fft = False
            arrays.append(arr)
            names.append(f"ROI {roi}" if roi else "Stitched mROIs")

        figure_shape = (1, len(arrays))

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get(
            "figure_kwargs",
            {"size": (1000, 1200)},
        )
        window_funcs = kwargs.get("window_funcs", None)

        return fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            figure_shape=figure_shape,
            graphic_kwargs={"vmin": arrays[0].min(), "vmax": arrays[0].max()},
            window_funcs=window_funcs,
        )


class NumpyArray:
    def __init__(
            self,
            array: np.ndarray | str | Path,
            metadata: dict | None = None
    ):
        if isinstance(array, (str, Path)):
            self.path = Path(array)
            if not self.path.exists():
                raise FileNotFoundError(f"Numpy file not found: {self.path}")
            self.data = np.load(self.path, mmap_mode="r")
            self._tempfile = None
        elif isinstance(array, np.ndarray):
            logger.info(f"Creating temporary .npy file for array.")
            tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
            np.save(tmp, array)
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

    @property
    def min(self) -> float:
        return float(self.data.min())

    @property
    def max(self) -> float:
        return float(self.data.max())

    def __len__(self):
        return self.shape[0]

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
        self._metadata = {}

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    @property
    def min(self) -> float:
        return float(self.data[0].min())

    @property
    def max(self) -> float:
        return float(self.data[0].max())

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.shape[0]


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
        if isinstance(filenames, (str, Path)):
            filenames = [filenames]

        self.filenames = [Path(p).with_suffix(".zarr") for p in filenames]
        self.rois = rois
        for p in self.filenames:
            if not p.exists():
                raise FileNotFoundError(f"No zarr store at {p}")

        self.zs = [zarr.open(p, mode="r") for p in self.filenames]

        shapes = [z.shape for z in self.zs]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent shapes across zarr stores: {shapes}")

        self._metadata = [dict(z.attrs) for z in self.zs]
        self.compressor = compressor

    @property
    def metadata(self):
        # if one store, return dict, if many, return the first
        # TODO: zarr consolidate metadata
        return self._metadata[0] if len(self._metadata) >= 1 else self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        # Update the first metadata entry (or all of them)
        if len(self._metadata) >= 1:
            self._metadata[0] = value
        else:
            self._metadata = [value]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        t, h, w = self.zs[0].shape
        return t, len(self.zs), h, w

    @property
    def dtype(self):
        return self.zs[0].dtype

    @property
    def size(self):
        return np.prod(self.shape)

    def __array__(self):
        """Materialize full array into memory: (T, Z, H, W)."""
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
            # convert contiguous lists to slices for zarr
            if isinstance(idx, list) and len(idx) > 0:
                if all(idx[i] + 1 == idx[i+1] for i in range(len(idx)-1)):
                    return slice(idx[0], idx[-1] + 1)
                else:
                    return np.array(idx)  # will require looping later
            return idx

        y_key = normalize(y_key)
        x_key = normalize(x_key)

        if len(self.zs) == 1:
            if isinstance(z_key, int) and z_key != 0:
                raise IndexError("Z dimension has size 1, only index 0 is valid")
            return self.zs[0][t_key, y_key, x_key]

        # multi-zarr
        if isinstance(z_key, int):
            return self.zs[z_key][t_key, y_key, x_key]

        if isinstance(z_key, slice):
            z_indices = range(len(self.zs))[z_key]
        else:
            raise IndexError("Z indexing must be int or slice")

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
                dshape=(self.shape[0], self.shape[-1], self.shape[-2]),
                plane_index=plane,
                **kwargs,
            )
