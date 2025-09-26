from __future__ import annotations

import os
import tempfile
import copy
import json
import time

from dataclasses import field
from typing import Any, Tuple, List, Sequence
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import h5py
import zarr
import tifffile
from dask import array as da
import fastplotlib as fpl

from mbo_utilities.metadata import get_metadata
from mbo_utilities._parsing import _make_json_serializable
from mbo_utilities._writers import _write_plane
from mbo_utilities.file_io import (
    _multi_tiff_to_fsspec,
    HAS_ZARR,
    _convert_range_to_slice,
    expand_paths,
)
from mbo_utilities.util import subsample_array

from mbo_utilities import log
from mbo_utilities.roi import iter_rois
from mbo_utilities.phasecorr import ALL_PHASECORR_METHODS, nd_windowed
from mbo_utilities.scanreader import scans, utils
from mbo_utilities.scanreader.multiroi import ROI

try:
    from suite3d.job import Job  # noqa
    HAS_SUITE3D = True
except ImportError:
    HAS_SUITE3D = False

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

logger = log.get("array_types")

CHUNKS_4D = {0: 1, 1: "auto", 2: -1, 3: -1}
CHUNKS_3D = {0: 1, 1: -1, 2: -1}


def register_zplanes(filenames, metadata, outpath = None) -> Path | None:
    if not HAS_SUITE3D:
        print(
            "Suite3D is not installed. Cannot preprocess."
            "Install with `pip install mbo_utilities[suite3d, cuda12] # CUDA 12.x or"
            "             `pip install mbo_utilities[suite3d, cuda11] # CUDA 11.x"
        )
    if not HAS_CUPY:
        print(
            "CuPy is not installed. Cannot preprocess."
            "Install with `pip install cupy-cuda12x` # CUDA 12.x or"
            "             `pip install cupy-cuda11x` # CUDA 11.x"
        )

    if "frame_rate" not in metadata or "num_planes" not in metadata:
        print("Missing required metadata for axial alignment: frame_rate / num_planes")
        return None

    if outpath is not None:
        job_path = Path(outpath)
    else:
        job_path = Path(str(filenames[0].parent) + ".summary")

    job_id = metadata.get("job_id", "preprocessed")

    params = {
        'fs': metadata["frame_rate"],
        'planes': np.arange(metadata["num_planes"]),
        'n_ch_tif': metadata["num_planes"],
        'tau': metadata.get("tau", 1.3),
        'lbm': metadata.get("lbm", True),
        'fuse_strips': metadata.get("fuse_planes", False),
        'subtract_crosstalk': metadata.get("subtract_crosstalk", False),
        'init_n_frames': metadata.get("init_n_frames", 500),
        'n_init_files': metadata.get("n_init_files", 1),
        'n_proc_corr': metadata.get("n_proc_corr", 15),
        'max_rigid_shift_pix': metadata.get("max_rigid_shift_pix", 150),
        '3d_reg': metadata.get("3d_reg", True),
        'gpu_reg': metadata.get("gpu_reg", True),
        'block_size': metadata.get("block_size", [64, 64]),
    }

    job = Job(
        str(job_path),
        job_id,
        create=True,
        overwrite=True,
        verbosity=-1,
        tifs=filenames,
        params=params
    )
    print("Running Suite3D job...")
    start = time.time()
    job.run_init_pass()
    end = time.time()
    print(f"Suite 3D init pass done in {end - start:.1f} seconds.")
    out_dir = job_path / job_id
    metadata["s3d-job"] = str(out_dir)
    metadata["s3d-params"] = params
    print(f"Preprocessed data saved to {out_dir}")
    return out_dir


def apply_zshifts(base_dir, inplace=False, metadata=None):
    """
    Apply plane shifts from summary.npy to TIFF stacks.

    Parameters
    ----------
    base_dir : str or Path
        Directory containing stitched plane TIFFs and summary.npy.
    inplace : bool, default=False
        If True, overwrite original TIFFs safely using a temporary file.
        If False, save new files with `_aligned` suffix.
    metadata : list of dict, optional
        Per-plane metadata to write into output TIFFs.

    Returns
    -------
    list of Path
        Paths to aligned TIFF files.
    """
    base_dir = Path(base_dir)
    tiffs = sorted(base_dir.rglob("plane*_stitched.tif"))
    summary_file = list(base_dir.rglob("*summary.npy"))[0]

    summary = np.load(summary_file, allow_pickle=True).item()
    plane_shifts = summary["plane_shifts"]

    with tifffile.TiffFile(tiffs[0]) as tf:
        nframes, H, W = tf.series[0].shape

    # Compute padding
    dy_min, dx_min = plane_shifts.min(axis=0)
    dy_max, dx_max = plane_shifts.max(axis=0)
    pad_top, pad_left = max(0, -dy_min), max(0, -dx_min)
    pad_bottom, pad_right = max(0, dy_max), max(0, dx_max)
    target_shape = (nframes, H + pad_top + pad_bottom, W + pad_left + pad_right)

    print("Final shape:", target_shape)

    outputs = []
    for i, (tif, (dy, dx)) in enumerate(zip(tiffs, plane_shifts)):
        meta = metadata[i] if metadata is not None else {}

        if inplace:
            fd, tmpname = tempfile.mkstemp(suffix=".tif", dir=tif.parent)
            os.close(fd)
            tmpfile = Path(tmpname)
            outpath = tif
        else:
            outpath = tif.with_name(tif.stem + "_aligned.tif")
            if outpath.exists():
                outpath.unlink()
            tmpfile = outpath

        with tifffile.TiffFile(tif) as tf:
            with tifffile.TiffWriter(tmpfile, bigtiff=True) as tw:
                iy, ix = int(dy), int(dx)
                for page in tf.pages:
                    frame = page.asarray()
                    canvas = np.zeros(target_shape[1:], dtype=frame.dtype)
                    yy = slice(pad_top + iy, pad_top + iy + H)
                    xx = slice(pad_left + ix, pad_left + ix + W)
                    canvas[yy, xx] = frame
                    tw.write(canvas, contiguous=True, photometric="minisblack", metadata=meta)

        if inplace:
            for _ in range(6):
                try:
                    os.replace(str(tmpfile), str(outpath))
                    break
                except PermissionError:
                    time.sleep(0.2)
            else:
                raise
        print("Wrote:", outpath)
        outputs.append(outpath)
    return outputs


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
        newshape = [tdim if have_T else 1, int(np.prod([a.shape[i] for i in rest[:-2]])), a.shape[-2], a.shape[-1]]
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


def _axes_or_guess(path: Path, arr_ndim: int) -> str:
    try:
        with tifffile.TiffFile(path) as tf:
            return tf.series[0].axes
    except Exception:
        if arr_ndim == 2:
            return "YX"
    if arr_ndim == 3:
        return "ZYX"
    if arr_ndim == 4:
        return "TZYX"
    return "YX"


def _safe_get_metadata(path: Path) -> dict:
    try:
        return get_metadata(path)
    except Exception:
        return {}


@dataclass
class Suite2pArray:
    metadata: dict | str | Path
    filename: str | Path = field(default=None)  # data.bin, data_raw.bin, etc

    def __post_init__(self):
        if isinstance(self.metadata, (str, Path)):
            ops_filename = self.metadata
            self.metadata = np.load(ops_filename, allow_pickle=True).item()
            self.filename = self.metadata["raw_file"]

        if self.filename is None:
            if "raw_file" in self.metadata:
                self.filename = self.metadata["raw_file"]
            elif "reg_file" in self.metadata:
                self.filename = self.metadata["reg_file"]
            else:
                raise ValueError(
                    "No data file found for Suite2p results. Please provide filename explicitly e.g. data_raw.bin"
                )

        self.Ly = self.metadata["Ly"]
        self.Lx = self.metadata["Lx"]
        self.nframes = self.metadata.get("nframes", self.metadata.get("n_frames"))
        if self.nframes is None:
            raise ValueError("Missing 'nframes' or 'n_frames' in metadata")

        self.shape = (self.nframes, self.Ly, self.Lx)
        self.dtype = np.int16
        self._file = np.memmap(
            self.filename, mode="r", dtype=self.dtype, shape=self.shape
        )
        self.filenames = [Path(self.filename)]

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


class H5Array:
    def __init__(self, filenames: Path | str, dataset: str = "mov"):
        self.filenames = Path(filenames)
        self._f = h5py.File(self.filenames, "r")
        self._d = self._f[dataset]
        self.shape = self._d.shape
        self.dtype = self._d.dtype
        self.ndim = self._d.ndim

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
        return dict(self._f.attrs)

    def _imwrite(self, outpath, **kwargs):
        _write_plane(
            self._d, Path(outpath),
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
    roi: Any = None
    _dask_array: da.Array = field(default=None, init=False, repr=False)

    @property
    def chunks(self) -> tuple[int, ...] | dict:
        return self._chunks or CHUNKS_4D

    @chunks.setter
    def chunks(self, value):
        self._chunks = value

    def _build_dask_array(self) -> da.Array:
        if len(self.filenames) == 1:
            arr = tifffile.memmap(self.filenames[0], mode="r")
            return da.from_array(arr, chunks=self.chunks)

        planes = []
        for p in self.filenames:
            mm = tifffile.memmap(p, mode="r")
            if mm.ndim == 3:
                mm = mm[None, ...]
            planes.append(da.from_array(mm, chunks=self.chunks))

        dstack = da.concatenate(planes, axis=0)  # (Z, T, Y, X)
        return dstack.transpose(1, 0, 2, 3)  # (T, Z, Y, X)

    @property
    def dask(self) -> da.Array:
        if self._dask_array is None:
            self._dask_array = self._build_dask_array()
        return self._dask_array

    def __getitem__(self, key: int | slice | tuple[int, ...]) -> np.ndarray:
        return self.dask[key]

    def __getattr__(self, attr):
        return getattr(self.dask, attr)

    @property
    def ndim(self) -> int:
        return self.dask.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.dask.shape)

    @property
    def metadata(self) -> dict:
        """
        Return metadata from the first TIFF file.
        Assumes all files have the same metadata structure.
        """
        if not self.filenames:
            return {}
        return get_metadata(self.filenames[0])

    def imshow(self, **kwargs) -> fpl.ImageWidget:
        if len(self.filenames) == 1:
            data = tifffile.memmap(self.filenames[0], mode="r")
        else:
            data = self.dask
        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get(
            "figure_kwargs",
            {
                "size": (800, 1000),
            },
        )
        window_funcs = kwargs.get("window_funcs", None)
        return fpl.ImageWidget(
            data=data,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,  # "canvas": canvas},
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
    ):
        if "plane" in self.metadata.keys():
            plane = self.metadata["plane"]
        else:
            from mbo_utilities import get_plane_from_filename
            plane = get_plane_from_filename(Path(outpath).stem, None)
            if plane is None:
                raise ValueError("Cannot determine plane from metadata.")
            else:
                self.metadata["plane"] = plane

        outpath = Path(outpath)
        ext = ext.lower().lstrip(".")

        if ext in {"bin"}:
            fname = "data_raw.bin"
        else:
            fname = f"plane{plane:03d}.{ext}"

        if outpath.is_dir():
            target = outpath.joinpath(fname)
        else:
            target = outpath.parent.joinpath(fname)

        _write_plane(
            self,
            target,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            metadata=self.metadata,
            progress_callback=progress_callback,
            debug=debug,
            dshape=(self.shape[0], self.shape[-1], self.shape[-2]),
            plane_index=None,  # convert to 0-based index
        )


@dataclass
class NpyArray:
    filenames: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = np.load(str(self.filenames), mmap_mode="r")
        return arr


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
                axes = _axes_or_guess(path, mm.ndim)
            except Exception:
                arr = tifffile.imread(path)
                a = da.from_array(arr, chunks=self.chunks)
                axes = _axes_or_guess(path, arr.ndim)
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
            ext=".tiff",
            progress_callback=None,
            debug=None,
            planes=None,
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


class MboRawArray(scans.ScanMultiROI):
    """
    A subclass of ScanMultiROI that ignores the num_fields dimension
    and reorders the output to [time, z, x, y].
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
    ):
        super().__init__(join_contiguous=True)
        self._metadata = {}  # set when pages are read
        self._fix_phase = fix_phase
        self._phasecorr_method = phasecorr_method
        self.border: int | tuple[int, int, int, int] = border
        self.max_offset: int = max_offset
        self.upsample: int = upsample
        self.use_zarr = False
        self.reference = ""
        self.roi = roi  # alias
        self._roi = roi
        self.pbar = None
        self.show_pbar = False
        self._offset = 0.0

        # Debugging toggles
        self.debug_flags = {
            "frame_idx": True,
            "roi_array_shape": False,
            "phase_offset": False,
        }
        self.logger = logger
        if files:
            self.read_data(files)

    def save_fsspec(self, filenames):
        base_dir = Path(filenames[0]).parent

        combined_json_path = base_dir / "combined_refs.json"

        if combined_json_path.is_file():
            # delete it, its cheap to create
            logger.debug(
                f"Removing existing combined reference file: {combined_json_path}"
            )
            combined_json_path.unlink()

        print(f"Generating combined kerchunk reference for {len(filenames)} files…")
        combined_refs = _multi_tiff_to_fsspec(tif_files=filenames, base_dir=base_dir)

        with open(combined_json_path, "w") as _f:
            json.dump(combined_refs, _f)

        print(f"Combined kerchunk reference written to {combined_json_path}")
        self.reference = combined_json_path
        return combined_json_path

    def as_dask(self):
        """
        Convert the current scan data to a Dask array.
        This will create a Dask array in the same directory as the reference file.
        """
        if not HAS_ZARR:
            raise ImportError(
                "Zarr is not installed. Please install it to use this method."
            )
        if not Path(self.reference).is_file():
            raise FileNotFoundError(
                f"Reference file {self.reference} does not exist. "
                "Please call save_fsspec() first."
            )
        raise NotImplementedError("Attempted to convert to Dask, but not implemented.")

    def as_zarr(self):
        """
        Convert the current scan data to a Zarr array.
        This will create a Zarr store in the same directory as the reference file.
        """
        if not HAS_ZARR:
            raise ImportError(
                "Zarr is not installed. Please install it to use this method."
            )
        if not Path(self.reference).is_file():
            return None
        return NotImplementedError("Attempted to convert to Zarr, but not implemented.")

    def read_data(self, filenames, dtype=np.int16):
        filenames = expand_paths(filenames)
        self.use_zarr = False
        self.reference = None
        super().read_data(filenames, dtype)
        self._metadata = get_metadata(
            self.tiff_files[0].filehandle.path
        )  # from the file
        self.metadata["si"] = _make_json_serializable(self.tiff_files[0].scanimage_metadata)
        self._rois = self._create_rois()
        self.fields = self._create_fields()
        if self.join_contiguous:
            self._join_contiguous_fields()

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
            }
        )
        return self._metadata

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
    def phasecorr_method(self):
        """
        Get the current phase correction method.
        Options are 'subpix' or 'mean'.
        """
        return self._phasecorr_method

    @phasecorr_method.setter
    def phasecorr_method(self, value: str | None):
        """
        Set the phase correction method.
        Options are 'two_step', 'subpix', or 'crosscorr'.
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
        self._roi = value

    @property
    def num_rois(self) -> int:
        return len(self.rois)

    @property
    def xslices(self):
        return self.fields[0].xslices

    @property
    def yslices(self):
        return self.fields[0].yslices

    @property
    def output_xslices(self):
        return self.fields[0].output_xslices

    @property
    def output_yslices(self):
        return self.fields[0].output_yslices

    def _read_pages(
        self, frames, chans, yslice=slice(None), xslice=slice(None), **kwargs
    ):
        pages = [
            frame * self.num_channels + zplane
            for frame in frames
            for zplane in chans
        ]

        tiff_width_px = len(utils.listify_index(xslice, self._page_width))
        tiff_height_px = len(utils.listify_index(yslice, self._page_height))
        #
        # if getattr(self, "use_zarr", False):
        #     zarray = self.as_zarr()
        #     if zarray is not None:
        #         buf = np.empty((len(pages), H, W), dtype=self.dtype)
        #         for i, page in enumerate(pages):
        #             f, c = divmod(page, C)
        #             buf[i] = zarray[f, c, yslice, xslice]
        #
        #         if self.fix_phase:
        #             self.logger.debug(
        #                 f"Applying phase correction with strategy: {self.phasecorr_method}"
        #             )
        #             buf, self.offset = nd_windowed(
        #                 buf,
        #                 method=self.phasecorr_method,
        #                 upsample=self.upsample,
        #                 max_offset=self.max_offset,
        #                 border=self.border,
        #             )
        #         return buf.reshape(len(frames), len(chans), H, W)
        #
        # TIFF path
        buf = np.empty(
            (len(pages),
             tiff_height_px,
             tiff_width_px),
            dtype=self.dtype
        )
        start = 0
        for tf in self.tiff_files:
            end = start + len(tf.pages)
            idxs = [i for i, p in enumerate(pages) if start <= p < end]
            if not idxs:
                start = end
                continue

            frame_idx = [pages[i] - start for i in idxs]
            chunk = tf.asarray(key=frame_idx)[..., yslice, xslice]

            if self.fix_phase:
                self.logger.debug(
                    f"Applying phase correction with strategy: {self.phasecorr_method}"
                )
                buf[idxs], self.offset = nd_windowed(
                    chunk,
                    method=self.phasecorr_method,
                    upsample=self.upsample,
                    max_offset=self.max_offset,
                    border=self.border,
                )
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
        frames = utils.listify_index(t_key, self.num_frames)
        chans = utils.listify_index(z_key, self.num_channels)
        if not frames or not chans:
            return np.empty(0)

        logger.debug(
            f"Phase-corrected: {self.fix_phase}/{self.phasecorr_method},"
            f" channels: {chans},"
            f" roi: {self.roi}",
        )
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
        return out

    def process_rois(self, frames, chans):
        if self.roi is not None:
            if isinstance(self.roi, list):
                return tuple(
                    self.process_single_roi(roi_idx - 1, frames, chans)
                    for roi_idx in self.roi
                )
            elif self.roi == 0:
                return tuple(
                    self.process_single_roi(roi_idx, frames, chans)
                    for roi_idx in range(self.num_rois)
                )
            elif isinstance(self.roi, int):
                return self.process_single_roi(self.roi - 1, frames, chans)
        else:
            H_out, W_out = self.field_heights[0], self.field_widths[0]
            out = np.zeros((len(frames), len(chans), H_out, W_out), dtype=self.dtype)
            for roi_idx in range(self.num_rois):
                roi_data = self.process_single_roi(roi_idx, frames, chans)
                oys, oxs = (
                    self.fields[0].output_yslices[roi_idx],
                    self.fields[0].output_xslices[roi_idx],
                )
                out[:, :, oys, oxs] = roi_data
            return out

    def process_single_roi(self, roi_idx, frames, chans):
        return self._read_pages(
            frames,
            chans,
            yslice=self.fields[0].yslices[roi_idx],
            xslice=self.fields[0].xslices[roi_idx],
        )

    @property
    def total_frames(self):
        return sum(len(tf.pages) // self.num_channels for tf in self.tiff_files)

    @property
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
                    s = self.fields[0].output_xslices[self.roi - 1]
                    width = s.stop - s.start
                    return (
                        self.total_frames,
                        self.num_channels,
                        self.field_heights[0],
                        width,
                    )
        # roi = None, or a list/tuple indicates the shape should be relative to the full FOV
        return (
            self.total_frames,
            self.num_channels,
            self.field_heights[0],
            self.field_widths[0],
        )

    @property
    def shape_full(self):
        return (
            self.total_frames,
            self.num_channels,
            self.field_heights[0],
            self.field_widths[0],
        )

    @property
    def ndim(self):
        return 4

    @property
    def size(self):
        return (
            self.num_frames
            * self.num_channels
            * self.field_heights[0]
            * self.field_widths[0]
        )

    @property
    def scanning_depths(self):
        """
        We override this because LBM should always be at a single scanning depth.
        """
        return [0]

    def _create_rois(self):
        """
        Create scan rois from the configuration file. Override the base method to force
        ROI's that have multiple 'zs' to a single depth.
        """
        try:
            roi_infos = self.tiff_files[0].scanimage_metadata["RoiGroups"][
                "imagingRoiGroup"
            ]["rois"]
        except KeyError:
            raise RuntimeError(
                "This file is not a raw-scanimage tiff or is missing tiff.scanimage_metadata."
            )
        roi_infos = roi_infos if isinstance(roi_infos, list) else [roi_infos]

        # discard empty/malformed ROIs
        roi_infos = list(
            filter(lambda r: isinstance(r["zs"], (int, float, list)), roi_infos)
        )

        # LBM uses a single depth that is not stored in metadata,
        # so force this to be 0.
        for roi_info in roi_infos:
            roi_info["zs"] = [0]

        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois

    def _create_fields(self):
        """Go over each slice depth and each roi generating the scanned fields."""
        fields = []
        previous_lines = 0
        for slice_id, scanning_depth in enumerate(self.scanning_depths):
            next_line_in_page = 0  # each slice is one tiff page
            for roi_id, roi in enumerate(self.rois):
                new_field = roi.get_field_at(scanning_depth)
                if new_field is not None:
                    # Set xslice and yslice (from where in the page to cut it)
                    new_field.yslices = [
                        slice(next_line_in_page, next_line_in_page + new_field.height)
                    ]
                    new_field.xslices = [slice(0, new_field.width)]

                    # Set output xslice and yslice (where to paste it in output)
                    new_field.output_yslices = [slice(0, new_field.height)]
                    new_field.output_xslices = [slice(0, new_field.width)]

                    # Set slice and roi id
                    new_field.slice_id = slice_id
                    new_field.roi_ids = [roi_id]

                    offsets = self._compute_offsets(
                        new_field.height, previous_lines + next_line_in_page
                    )
                    new_field.offsets = [offsets]
                    next_line_in_page += new_field.height + self._num_fly_to_lines
                    fields.append(new_field)
            previous_lines += self._num_lines_between_fields
        return fields

    def preprocess(self) -> Path | None:
        if not HAS_SUITE3D:
            print(
            "Suite3D is not installed. Cannot preprocess."
            "Install with `pip install mbo_utilities[suite3d, cuda12] # CUDA 12.x or"
            "             `pip install mbo_utilities[suite3d, cuda11] # CUDA 11.x"
            )
        if not HAS_CUPY:
            print(
                "CuPy is not installed. Cannot preprocess."
                "Install with `pip install cupy-cuda12x` # CUDA 12.x or"
                "             `pip install cupy-cuda11x` # CUDA 11.x"
            )

        parent_dir = self.filenames[0].parent
        job_path = Path(str(parent_dir) + ".summary")
        job_id = self.metadata.get("job_id", "preprocessed")

        params = {
            'fs': self.metadata["frame_rate"],
            'planes': np.arange(self.metadata["num_planes"]),
            'n_ch_tif': self.metadata["num_planes"],
            'tau': self.metadata.get("tau", 1.3),
            'lbm': self.metadata.get("lbm", True),
            'fuse_strips': self.metadata.get("fuse_planes", False),
            'subtract_crosstalk': self.metadata.get("subtract_crosstalk", False),
            'init_n_frames': self.metadata.get("init_n_frames", 500),
            'n_init_files': self.metadata.get("n_init_files", 1),
            'n_proc_corr': self.metadata.get("n_proc_corr", 15),
            'max_rigid_shift_pix': self.metadata.get("max_rigid_shift_pix", 150),
            '3d_reg': self.metadata.get("3d_reg", True),
            'gpu_reg': self.metadata.get("gpu_reg", True),
            'block_size': self.metadata.get("block_size", [64, 64]),
        }

        job = Job(
            str(job_path),
            job_id,
            create=True,
            overwrite=True,
            verbosity=-1,
            tifs=self.filenames,
            params=params
        )
        job.run_init_pass()
        out_dir = job_path / job_id
        self.metadata["s3d-job"] = str(out_dir)
        self.metadata["s3d-params"] = params
        self.logger.info(f"Preprocessed data saved to {out_dir}")
        return out_dir


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
    ):
        # convert to 0 based indexing
        if isinstance(planes, int):
            planes = [planes - 1]
        elif planes is None:
            planes = list(range(self.num_planes))
        else:
            planes = [p - 1 for p in planes]
        for roi in iter_rois(self):
            for plane in planes:
                self.roi = roi
                if roi is None:
                    fname = f"plane{plane+1:02d}_stitched{ext}"
                else:
                    fname = f"plane{plane+1:02d}_roi{roi}{ext}"

                if ext in [".bin", ".binary"]:
                    # saving to bin for suite2p
                    # we want the filename to be data_raw.bin
                    # so put the fname as the folder name
                    fname_bin_stripped = Path(fname).stem  # remove extension
                    target = outpath / fname_bin_stripped / "data_raw.bin"
                else:
                    target = outpath.joinpath(fname)

                target.parent.mkdir(exist_ok=True)
                if target.exists() and not overwrite:
                    logger.warning(
                        f"File {target} already exists. Skipping write."
                    )
                    continue

                md = self.metadata.copy()
                md["plane"] = plane + 1  # back to 1-based indexing
                md["mroi"] = roi
                md["roi"] = roi  # alias
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
                )

    def imshow(self, **kwargs):
        arrays = []
        names = []
        # if roi is None, use a single array.roi = None
        # if roi is 0, get a list of all ROIs by deeepcopying the array and setting each roi
        for roi in iter_rois(self):
            arr = copy.copy(self)
            arr.roi = roi
            arrays.append(arr)
            names.append(f"ROI {roi}" if roi else "Full Image")

        figure_shape = (1, len(arrays))

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get(
            "figure_kwargs",
            {
                "size": (800, 1000),
            },
        )
        window_funcs = kwargs.get("window_funcs", None)
        return fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,  # "canvas": canvas},
            figure_shape=figure_shape,
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
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

        nwbfile = read_nwb(path)
        self.data = nwbfile.acquisition['TwoPhotonSeries'].data
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.ndim = self.data.ndim

    def __getitem__(self, item):
        return self.data[item]

class ZarrArray:
    """
    Lightweight reader for _write_zarr outputs.
    Presents data as (T, Z, H, W) with Z=1.
    """
    def __init__(self, paths: str | Path | Sequence[str | Path]):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p).with_suffix(".zarr") for p in paths]
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(f"No zarr store at {p}")

        self.zs = [zarr.open(p, mode="r") for p in self.paths]

        shapes = [z.shape for z in self.zs]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent shapes across zarr stores: {shapes}")

        self._metadata = [dict(z.attrs) for z in self.zs]

    @property
    def metadata(self):
        # if one store, return dict, if many, return list of dicts
        return self._metadata[0] if len(self._metadata) >= 1 else self._metadata

    @property
    def shape(self) -> tuple[int, int, int, int]:
        t, h, w = self.zs[0].shape
        return (t, len(self.zs), h, w)

    @property
    def dtype(self):
        return self.zs[0].dtype

    @property
    def ndim(self):
        return 4

    @property
    def size(self):
        return np.prod(self.shape)

    def __array__(self):
        """Materialize full array into memory: (T, Z, H, W)."""
        arrs = [z[:] for z in self.zs]
        stacked = np.stack(arrs, axis=1)  # (T, Z, H, W)
        return stacked

    def __getitem__(self, key):
        """
        Index like a 4D array: (t, z, y, x).
        Handles one or multiple zarr stores.
        """
        # Normalize key to 4 components
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        t_key, z_key, y_key, x_key = key

        if len(self.zs) == 1:
            if isinstance(z_key, int) and z_key != 0:
                raise IndexError("Z dimension has size 1, only index 0 is valid")
            data = self.zs[0][t_key, y_key, x_key]
            return data[..., None, :, :] if data.ndim == 3 else data

        # multi-zarr case
        if isinstance(z_key, int):
            return self.zs[z_key][t_key, y_key, x_key]
        else:
            # fancy indexing across multiple z
            z_indices = range(len(self.zs))[z_key]
            arrs = [self.zs[i][t_key, y_key, x_key] for i in z_indices]
            return np.stack(arrs, axis=1)
