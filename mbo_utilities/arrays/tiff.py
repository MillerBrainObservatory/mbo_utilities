"""
TIFF array readers.

This module provides array readers for TIFF files:
- TiffArray: Lazy TIFF reader, auto-detects single file vs volume directory
- MBOTiffArray: Lazy TIFF reader for MBO processed TIFFs
- MboRawArray: Raw ScanImage TIFF reader with phase correction
"""

from __future__ import annotations

import copy
import json
import re
import time
import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import numpy as np
from tifffile import TiffFile

from mbo_utilities import log
from mbo_utilities.arrays._base import (
    _imwrite_base,
    iter_rois,
    ReductionMixin,
)
from mbo_utilities.file_io import derive_tag_from_filename, expand_paths
from mbo_utilities.metadata import get_metadata, get_param, extract_roi_slices
from mbo_utilities.phasecorr import bidir_phasecorr, ALL_PHASECORR_METHODS
from mbo_utilities.util import listify_index, index_length

if TYPE_CHECKING:
    pass

logger = log.get("arrays.tiff")


def _convert_range_to_slice(k):
    """Convert range objects to slices for indexing."""
    if isinstance(k, range):
        return slice(k.start, k.stop, k.step)
    return k


def _extract_tiff_plane_number(name: str) -> int | None:
    """Extract plane number from filename like 'plane01.tiff' or 'plane14_stitched.tif'."""
    match = re.search(r"plane(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def find_tiff_plane_files(directory: Path) -> list[Path]:
    """
    Find TIFF plane files in a directory.

    Looks for files matching the pattern 'planeXX.tiff' or 'planeXX.tif',
    sorted by plane number.

    Parameters
    ----------
    directory : Path
        Directory to search.

    Returns
    -------
    list[Path]
        List of TIFF files sorted by plane number, or empty list if not found.
    """
    plane_files = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in (".tif", ".tiff"):
            plane_num = _extract_tiff_plane_number(f.stem)
            if plane_num is not None:
                plane_files.append(f)

    if not plane_files:
        return []

    def sort_key(p):
        num = _extract_tiff_plane_number(p.stem)
        return num if num is not None else float("inf")

    return sorted(plane_files, key=sort_key)


class _SingleTiffPlaneReader:
    """Internal reader for a single TIFF plane."""

    def __init__(self, files: list[Path]):
        from mbo_utilities.metadata import query_tiff_pages

        self.filenames = files
        self.tiff_files = [TiffFile(f) for f in files]
        self._tiff_lock = threading.Lock()

        tf = self.tiff_files[0]
        page0 = tf.pages.first
        self._page_shape = page0.shape
        self._dtype = page0.dtype

        self._frames_per_file = []
        self._num_frames = 0

        for i, (tfile, fpath) in enumerate(zip(self.tiff_files, self.filenames)):
            nframes = None

            if i == 0:
                desc = page0.description
            else:
                desc = tfile.pages.first.description

            if desc:
                try:
                    meta = json.loads(desc)
                    if "shape" in meta and isinstance(meta["shape"], list):
                        if len(meta["shape"]) >= 3:
                            nframes = meta["shape"][0]
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

            if nframes is None:
                try:
                    est = query_tiff_pages(fpath)
                    if est > 1:
                        nframes = est
                except Exception:
                    pass

            if nframes is None:
                nframes = len(tfile.pages)

            self._frames_per_file.append(nframes)
            self._num_frames += nframes

    @property
    def shape(self) -> tuple:
        return (self._num_frames, self._page_shape[0], self._page_shape[1])

    @property
    def nframes(self) -> int:
        return self._num_frames

    @property
    def Ly(self) -> int:
        return self._page_shape[0]

    @property
    def Lx(self) -> int:
        return self._page_shape[1]

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        t_key = key[0] if len(key) > 0 else slice(None)
        y_key = key[1] if len(key) > 1 else slice(None)
        x_key = key[2] if len(key) > 2 else slice(None)

        t_key = _convert_range_to_slice(t_key)

        frames = listify_index(t_key, self._num_frames)
        if not frames:
            return np.empty((0,) + self._page_shape, dtype=self._dtype)

        out = self._read_frames(frames)

        if y_key != slice(None) or x_key != slice(None):
            out = out[:, y_key, x_key]

        if isinstance(t_key, int):
            out = out[0]

        return out

    def _read_frames(self, frames: list[int]) -> np.ndarray:
        buf = np.empty((len(frames), self._page_shape[0], self._page_shape[1]), dtype=self._dtype)

        start = 0
        frame_to_buf_idx = {f: i for i, f in enumerate(frames)}

        for tf, nframes in zip(self.tiff_files, self._frames_per_file):
            end = start + nframes
            file_frames = [f for f in frames if start <= f < end]
            if not file_frames:
                start = end
                continue

            local_indices = [f - start for f in file_frames]

            with self._tiff_lock:
                try:
                    chunk = tf.asarray(key=local_indices)
                except Exception as e:
                    raise IOError(
                        f"Failed to read frames {local_indices} from {tf.filename}: {e}"
                    ) from e

            if chunk.ndim == 2:
                chunk = chunk[np.newaxis, ...]

            for local_idx, global_frame in zip(local_indices, file_frames):
                buf_idx = frame_to_buf_idx[global_frame]
                chunk_idx = local_indices.index(local_idx)
                buf[buf_idx] = chunk[chunk_idx]

            start = end

        return buf

    def close(self):
        for tf in self.tiff_files:
            tf.close()


class TiffArray(ReductionMixin):
    """
    Lazy TIFF array reader with auto-detection of single file vs volume.

    Auto-detects:
    - Single TIFF file(s): returns 4D (T, 1, Y, X)
    - Directory with planeXX.tiff files: returns 4D (T, Z, Y, X) where Z > 1

    Parameters
    ----------
    files : str, Path, or list
        TIFF file path(s), or a directory containing plane TIFF files.

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape in TZYX format.
    dtype : np.dtype
        Data type.
    is_volumetric : bool
        True if multiple planes were detected.
    num_planes : int
        Number of Z-planes.

    Examples
    --------
    >>> # Single TIFF file
    >>> arr = TiffArray("data.tiff")
    >>> arr.shape
    (10000, 1, 512, 512)

    >>> # Volume directory with plane files
    >>> arr = TiffArray("tiff_volume/")
    >>> arr.shape
    (10000, 14, 512, 512)
    >>> arr.is_volumetric
    True
    """

    def __init__(self, files: str | Path | List[str] | List[Path]):
        self._planes: list[_SingleTiffPlaneReader] = []
        self._is_volumetric = False
        self._target_dtype = None

        # Normalize input
        if isinstance(files, (str, Path)):
            path = Path(files)
            if path.is_dir():
                plane_files = find_tiff_plane_files(path)
                if plane_files:
                    self._init_volume(plane_files)
                else:
                    tiff_files = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
                    if tiff_files:
                        self._init_single_plane(tiff_files)
                    else:
                        raise ValueError(f"No TIFF files found in {path}")
            else:
                self._init_single_plane(expand_paths(files))
        else:
            self._init_single_plane([Path(f) for f in files])

    def _init_single_plane(self, files: list[Path]):
        """Initialize as single-plane array."""
        self._is_volumetric = False
        reader = _SingleTiffPlaneReader(files)
        self._planes = [reader]

        self._nframes = reader.nframes
        self._nz = 1
        self._ly = reader.Ly
        self._lx = reader.Lx
        self._dtype = reader.dtype
        self.filenames = files

        self._metadata = {
            "shape": self.shape,
            "dtype": str(self._dtype),
            "nframes": self._nframes,
            "num_frames": self._nframes,
            "file_paths": [str(p) for p in files],
            "num_files": len(files),
        }
        self.num_rois = 1

    def _init_volume(self, plane_files: list[Path]):
        """Initialize as volumetric array."""
        self._is_volumetric = True

        for pfile in plane_files:
            reader = _SingleTiffPlaneReader([pfile])
            self._planes.append(reader)

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
        self.filenames = plane_files

        self._metadata = {
            "shape": self.shape,
            "dtype": str(self._dtype),
            "nframes": self._nframes,
            "num_frames": self._nframes,
            "num_planes": self._nz,
            "plane_files": [str(p) for p in plane_files],
        }
        self.num_rois = 1

        logger.info(
            f"Loaded TIFF volume: {self._nframes} frames, {self._nz} planes, "
            f"{self._ly}x{self._lx} px"
        )

    @property
    def is_volumetric(self) -> bool:
        """True if this array represents multi-plane data."""
        return self._is_volumetric

    @property
    def num_planes(self) -> int:
        """Number of Z-planes."""
        return self._nz

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._nframes, self._nz, self._ly, self._lx)

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    @property
    def ndim(self) -> int:
        return 4

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    def _compute_frame_vminmax(self):
        if not hasattr(self, '_cached_vmin'):
            frame = np.asarray(self[0, 0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        self._compute_frame_vminmax()
        return self._cached_vmax

    def __len__(self) -> int:
        return self._nframes

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        t_key, z_key, y_key, x_key = key

        t_key = _convert_range_to_slice(t_key)
        z_key = _convert_range_to_slice(z_key)

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

        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def astype(self, dtype, copy=True):
        self._target_dtype = np.dtype(dtype)
        return self

    def __array__(self):
        n = min(10, self._nframes)
        return self[:n]

    def close(self):
        for plane in self._planes:
            plane.close()

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


class MBOTiffArray(ReductionMixin):
    """
    Lazy TIFF array reader for MBO processed TIFFs.

    Uses TiffFile handles for lazy, memory-efficient access without dask.
    Reads frames on-demand using metadata to determine array structure.
    Output is always in TZYX format.

    Parameters
    ----------
    filenames : list[Path]
        List of TIFF file paths.
    roi : int, optional
        ROI index (not used for processed TIFFs).

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape in TZYX format.
    dtype : np.dtype
        Data type.
    """

    def __init__(self, filenames: list[Path], roi: int | None = None):
        from mbo_utilities.metadata import query_tiff_pages

        if not filenames:
            raise ValueError("No filenames provided.")

        self.filenames = [Path(f) for f in filenames]
        self.roi = roi

        self.tiff_files = [TiffFile(f) for f in self.filenames]
        self._tiff_lock = threading.Lock()

        tf = self.tiff_files[0]
        page0 = tf.pages.first
        self._page_shape = page0.shape
        self._dtype = page0.dtype

        self._frames_per_file = []
        self._num_frames = 0

        for i, (tfile, fpath) in enumerate(zip(self.tiff_files, self.filenames)):
            nframes = None

            if i == 0:
                desc = page0.description
            else:
                desc = tfile.pages.first.description

            if desc:
                try:
                    meta = json.loads(desc)
                    if "shape" in meta and isinstance(meta["shape"], list):
                        if len(meta["shape"]) >= 3:
                            nframes = meta["shape"][0]
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

            if nframes is None:
                try:
                    est = query_tiff_pages(fpath)
                    if est > 1:
                        nframes = est
                except Exception:
                    pass

            if nframes is None:
                logger.debug(f"Using len(pages) fallback for {fpath}")
                nframes = len(tfile.pages)

            self._frames_per_file.append(nframes)
            self._num_frames += nframes

        self._metadata = get_metadata(self.filenames)
        self._metadata.update({
            "shape": self.shape,
            "dtype": str(self._dtype),
            "nframes": self._num_frames,
            "num_frames": self._num_frames,
            "frames_per_file": self._frames_per_file,
            "file_paths": [str(p) for p in self.filenames],
            "num_files": len(self.filenames),
        })
        self.num_rois = get_param(self._metadata, "num_mrois", default=1)
        self.tags = [derive_tag_from_filename(f) for f in self.filenames]
        self._target_dtype = None

    @property
    def metadata(self) -> dict:
        return self._metadata or {}

    @metadata.setter
    def metadata(self, metadata: dict):
        self._metadata = metadata

    @property
    def shape(self) -> tuple[int, ...]:
        return self._num_frames, 1, self._page_shape[0], self._page_shape[1]

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def _compute_frame_vminmax(self):
        if not hasattr(self, '_cached_vmin'):
            frame = np.asarray(self[0, 0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        self._compute_frame_vminmax()
        return self._cached_vmax

    @property
    def ndim(self) -> int:
        return 4

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        t_key = key[0] if len(key) > 0 else slice(None)
        z_key = key[1] if len(key) > 1 else slice(None)
        y_key = key[2] if len(key) > 2 else slice(None)
        x_key = key[3] if len(key) > 3 else slice(None)

        t_key = _convert_range_to_slice(t_key)
        z_key = _convert_range_to_slice(z_key)
        y_key = _convert_range_to_slice(y_key)
        x_key = _convert_range_to_slice(x_key)

        frames = listify_index(t_key, self._num_frames)
        if not frames:
            return np.empty((0, 1) + self._page_shape, dtype=self.dtype)

        out = self._read_frames(frames)

        if y_key != slice(None) or x_key != slice(None):
            out = out[:, :, y_key, x_key]

        z_indices = listify_index(z_key, 1)
        if z_indices != [0]:
            out = out[:, z_indices, :, :]

        squeeze_axes = []
        if isinstance(t_key, int):
            squeeze_axes.append(0)
        if isinstance(z_key, int):
            squeeze_axes.append(1)
        if squeeze_axes:
            out = np.squeeze(out, axis=tuple(squeeze_axes))

        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)

        return out

    def _read_frames(self, frames: list[int]) -> np.ndarray:
        buf = np.empty(
            (len(frames), 1, self._page_shape[0], self._page_shape[1]), dtype=self._dtype
        )

        start = 0
        frame_to_buf_idx = {f: i for i, f in enumerate(frames)}

        for tf, nframes in zip(self.tiff_files, self._frames_per_file):
            end = start + nframes
            file_frames = [f for f in frames if start <= f < end]
            if not file_frames:
                start = end
                continue

            local_indices = [f - start for f in file_frames]

            with self._tiff_lock:
                try:
                    chunk = tf.asarray(key=local_indices)
                except Exception as e:
                    raise IOError(
                        f"MBOTiffArray: Failed to read frames {local_indices} from {tf.filename}\n"
                        f"File may be corrupted or incomplete.\n"
                        f": {type(e).__name__}: {e}"
                    ) from e

            if chunk.ndim == 2:
                chunk = chunk[np.newaxis, ...]

            for local_idx, global_frame in zip(local_indices, file_frames):
                buf_idx = frame_to_buf_idx[global_frame]
                chunk_idx = local_indices.index(local_idx)
                buf[buf_idx, 0] = chunk[chunk_idx]

            start = end

        return buf

    def astype(self, dtype, copy=True):
        self._target_dtype = np.dtype(dtype)
        return self

    def __array__(self):
        n = min(10, self._num_frames)
        return self[:n]

    def close(self):
        for tf in self.tiff_files:
            tf.close()

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


class MboRawArray(ReductionMixin):
    """
    Raw ScanImage TIFF reader with phase correction support.

    Handles multi-ROI ScanImage data with bidirectional scanning phase correction.
    Supports ROI stitching, splitting, and individual ROI access.

    Parameters
    ----------
    files : str, Path, or list
        TIFF file path(s).
    roi : int or Sequence[int], optional
        ROI selection:
        - None: Stitch all ROIs horizontally
        - 0: Split all ROIs into separate outputs
        - int > 0: Select specific ROI (1-indexed)
        - list: Select multiple specific ROIs
    fix_phase : bool, default True
        Apply bidirectional scanning phase correction.
    phasecorr_method : str, default "mean"
        Phase correction method ("mean", "median", "max").
    border : int or tuple, default 3
        Border pixels to exclude from phase estimation.
    upsample : int, default 5
        Upsampling factor for subpixel phase estimation.
    max_offset : int, default 4
        Maximum phase offset to search.
    use_fft : bool, default False
        Use FFT-based 2D phase correction (more accurate but slower).

    Attributes
    ----------
    shape : tuple[int, int, int, int]
        Shape as (nframes, num_planes, height, width).
    dtype : np.dtype
        Data type.
    num_channels : int
        Number of Z-planes/channels.
    num_rois : int
        Number of ROIs in the data.
    """

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
    ):
        self.filenames = [files] if isinstance(files, (str, Path)) else list(files)
        self.tiff_files = [TiffFile(f) for f in self.filenames]
        self._tiff_lock = threading.Lock()

        self._metadata = get_metadata(self.filenames)
        self.num_channels = get_param(self._metadata, "nplanes", default=1)
        self.num_rois = get_param(self._metadata, "num_rois", default=1)

        self._roi = roi
        self.roi = roi

        self._fix_phase = fix_phase
        self._use_fft = use_fft
        self._phasecorr_method = phasecorr_method
        self._border = border
        self._max_offset = max_offset
        self._upsample = upsample
        self._offset = 0.0
        self._mean_subtraction = False
        self.pbar = None
        self.show_pbar = False
        self.logger = logger

        self.debug_flags = {
            "frame_idx": True,
            "roi_array_shape": False,
            "phase_offset": False,
        }
        self.num_frames = get_param(self._metadata, "nframes")
        self._source_dtype = get_param(self._metadata, "dtype", default="int16")
        self._target_dtype = None
        self._ndim = self._metadata.get("ndim", 3)

        self._frames_per_file = self._metadata.get("frames_per_file", None)

        self._rois = self._extract_roi_info()

    def _extract_roi_info(self):
        """Extract ROI slice information using centralized metadata function."""
        rois = extract_roi_slices(self._metadata)
        if rois:
            logger.debug(
                f"ROI structure: {[(r['y_start'], r['y_end'], r['height']) for r in rois]}"
            )
        return rois

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return (
            self._target_dtype if self._target_dtype is not None else self._source_dtype
        )

    def _compute_frame_vminmax(self):
        if not hasattr(self, '_cached_vmin'):
            frame = self[0, 0]
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        self._compute_frame_vminmax()
        return self._cached_vmax

    @property
    def metadata(self):
        self._metadata.update(
            {
                "dtype": self.dtype,
                "fix_phase": self.fix_phase,
                "phasecorr_method": self.phasecorr_method,
                "offset": self.offset,
                "border": self.border,
                "upsample": self.upsample,
                "max_offset": self.max_offset,
                "nframes": self.num_frames,
                "num_frames": self.num_frames,
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
        return self._rois

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value: float | np.ndarray):
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
    def fix_phase(self):
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("do_phasecorr must be a boolean value.")
        self._fix_phase = value

    @property
    def border(self):
        return self._border

    @border.setter
    def border(self, value: int):
        self._border = value

    @property
    def max_offset(self):
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value: int):
        self._max_offset = value

    @property
    def upsample(self):
        return self._upsample

    @upsample.setter
    def upsample(self, value: int):
        self._upsample = value

    @property
    def mean_subtraction(self):
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("mean_subtraction must be a boolean value.")
        self._mean_subtraction = value

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, value):
        if value is not None and value != 0:
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
        tiff_width_px = index_length(xslice, self._page_width)
        tiff_height_px = index_length(yslice, self._page_height)
        buf = np.empty((len(pages), tiff_height_px, tiff_width_px), dtype=self.dtype)

        start = 0
        tiff_iterator = (
            zip(
                self.tiff_files, (f * self.num_channels for f in self._frames_per_file)
            )
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

            if len(frame_idx) <= 4:
                with self._tiff_lock:
                    try:
                        chunk = tf.asarray(key=frame_idx)
                    except Exception as e:
                        raise IOError(
                            f"MboRawArray: Failed to read pages {frame_idx} from {tf.filename}\n"
                            f"File may be corrupted or incomplete.\n"
                            f": {type(e).__name__}: {e}"
                        ) from e
            else:
                chunks = []
                for fi in frame_idx:
                    with self._tiff_lock:
                        try:
                            c = tf.asarray(key=fi)
                        except Exception as e:
                            raise IOError(
                                f"MboRawArray: Failed to read page {fi} from {tf.filename}\n"
                                f"File may be corrupted or incomplete.\n"
                                f": {type(e).__name__}: {e}"
                            ) from e
                    chunks.append(c if c.ndim == 3 else c[np.newaxis, ...])
                chunk = np.concatenate(chunks, axis=0)

            if chunk.ndim == 2:
                chunk = chunk[np.newaxis, ...]
            chunk = chunk[..., yslice, xslice]

            if self.fix_phase:
                import time as _t
                _t0 = _t.perf_counter()
                corrected, offset = bidir_phasecorr(
                    chunk,
                    method=self.phasecorr_method,
                    upsample=self.upsample,
                    max_offset=self.max_offset,
                    border=self.border,
                    use_fft=self.use_fft,
                )
                buf[idxs] = corrected
                self._offset = offset
                _t1 = _t.perf_counter()
                logger.debug(
                    f"phase_corr: offset={offset:.2f}, method={self.phasecorr_method}, "
                    f"fft={self.use_fft}, chunk={chunk.shape}, took {(_t1-_t0)*1000:.1f}ms"
                )
            else:
                buf[idxs] = chunk
                self._offset = 0.0
            start = end

        logger.debug(f"_read_pages: {len(frames)} frames, {len(chans)} chans -> {buf.shape}")
        return buf.reshape(len(frames), len(chans), tiff_height_px, tiff_width_px)

    def __getitem__(self, key):
        t0 = time.perf_counter()
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
        t1 = time.perf_counter()
        self.logger.debug(f"__getitem__ took {(t1-t0)*1000:.1f}ms")

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

        if self._target_dtype is not None:
            if isinstance(out, tuple):
                out = tuple(x.astype(self._target_dtype) for x in out)
            else:
                out = out.astype(self._target_dtype)

        return out

    def process_rois(self, frames, chans):
        if self.roi is not None and isinstance(self.roi, int) and self.roi != 0:
            return self._read_pages(
                frames, chans,
                yslice=self._rois[self.roi - 1]["slice"],
                xslice=slice(None),
            )

        full_data = self._read_pages(frames, chans, yslice=slice(None), xslice=slice(None))

        if self.roi is not None:
            if isinstance(self.roi, list):
                return tuple(
                    full_data[:, :, self._rois[r - 1]["slice"], :]
                    for r in self.roi
                )
            elif self.roi == 0:
                return tuple(
                    full_data[:, :, self._rois[r]["slice"], :]
                    for r in range(self.num_rois)
                )

        total_width = sum(roi["width"] for roi in self._rois)
        max_height = max(roi["height"] for roi in self._rois)
        out = np.zeros(
            (len(frames), len(chans), max_height, total_width), dtype=self.dtype
        )

        for roi_idx in range(self.num_rois):
            yslice = self._rois[roi_idx]["slice"]
            oys = self.output_yslices[roi_idx]
            oxs = self.output_xslices[roi_idx]
            out[:, :, oys, oxs] = full_data[:, :, yslice, :]

        return out

    @property
    def num_planes(self):
        return self.num_channels

    @property
    def shape(self):
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
        total_width = sum(roi["width"] for roi in self._rois)
        max_height = max(roi["height"] for roi in self._rois)
        return (
            self.num_frames,
            self.num_channels,
            max_height,
            total_width,
        )

    def size(self):
        total_width = sum(roi["width"] for roi in self._rois)
        max_height = max(roi["height"] for roi in self._rois)
        return self.num_frames * self.num_channels * max_height * total_width

    def astype(self, dtype, copy=True):
        self._target_dtype = np.dtype(dtype)
        return self

    @property
    def _page_height(self):
        return self._metadata.get("page_height")

    @property
    def _page_width(self):
        return self._metadata.get("page_width")

    def __array__(self, max_frames: int = 100):
        if self.num_frames <= max_frames:
            return np.asarray(self[:])

        rng = np.random.default_rng(42)
        sample_frames = sorted(rng.choice(self.num_frames, max_frames, replace=False).tolist())
        return np.asarray(self[sample_frames])

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
        import fastplotlib as fpl

        arrays = []
        names = []
        for roi in iter_rois(self):
            arr = copy.copy(self)
            arr.roi = roi
            arr.fix_phase = False
            arr.use_fft = False
            arrays.append(arr)
            names.append(f"ROI {roi}" if roi else "Stitched mROIs")

        figure_shape = (1, len(arrays))
        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get("figure_kwargs", {"size": (600, 600)})
        window_funcs = kwargs.get("window_funcs", None)

        sample_frame = arrays[0][0]
        vmin, vmax = float(sample_frame.min()), float(sample_frame.max())

        return fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            figure_shape=figure_shape,
            graphic_kwargs={"vmin": vmin, "vmax": vmax},
            window_funcs=window_funcs,
        )
