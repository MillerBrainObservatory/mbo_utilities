"""
TIFF array readers.

This module provides array readers for TIFF files:
- TiffArray: Lazy TIFF reader, auto-detects single file vs volume directory
- MBOTiffArray: Lazy TIFF reader for MBO processed TIFFs
- ScanImageArray: Base class for raw ScanImage TIFFs with phase correction
- LBMArray: LBM (Light Beads Microscopy) stacks, z-planes as channels
- PiezoArray: Piezo z-stacks with optional frame averaging
- CalibrationArray: Pollen/bead calibration stacks (LBM + piezo)
- SinglePlaneArray: Single-plane time series
- open_scanimage: Factory function that auto-detects stack type

Legacy alias:
- MboRawArray: Alias for ScanImageArray (backwards compatibility)
"""

from __future__ import annotations

import copy
import json
import re
import time
import threading
from pathlib import Path

import numpy as np
from tifffile import TiffFile

from mbo_utilities import log
from mbo_utilities.arrays._base import (
    _imwrite_base,
    ReductionMixin,
)
from mbo_utilities.file_io import derive_tag_from_filename, expand_paths
from mbo_utilities.metadata import get_metadata, get_param, extract_roi_slices
from mbo_utilities.metadata.scanimage import (
    StackType,
    detect_stack_type,
    get_frames_per_slice,
    get_log_average_factor,
)
from mbo_utilities.analysis.phasecorr import bidir_phasecorr
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline
from mbo_utilities.util import listify_index, index_length
from mbo_utilities.arrays.features import (
    DimLabels,
    PhaseCorrectionFeature,
    RoiFeatureMixin,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = log.get("arrays.tiff")

# register tiff reader pipeline info
_TIFF_INFO = PipelineInfo(
    name="tiff",
    description="Generic TIFF files (BigTIFF, OME-TIFF)",
    input_patterns=[
        "**/*.tif",
        "**/*.tiff",
    ],
    output_patterns=[
        "**/*.tif",
        "**/*.tiff",
    ],
    input_extensions=["tif", "tiff"],
    output_extensions=["tif", "tiff"],
    marker_files=[],
    category="reader",
)
register_pipeline(_TIFF_INFO)

# register scanimage raw tiff reader
_SCANIMAGE_INFO = PipelineInfo(
    name="scanimage_raw",
    description="Raw ScanImage TIFF files with multi-ROI support",
    input_patterns=[
        "**/*.tif",
        "**/*.tiff",
    ],
    output_patterns=[],
    input_extensions=["tif", "tiff"],
    output_extensions=[],
    marker_files=[],
    category="reader",
)
register_pipeline(_SCANIMAGE_INFO)


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

        for i, (tfile, fpath) in enumerate(zip(self.tiff_files, self.filenames, strict=False)):
            nframes = None

            desc = page0.description if i == 0 else tfile.pages.first.description

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
        from mbo_utilities.util import get_dtype

        return get_dtype(self._dtype)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        t_key = key[0] if len(key) > 0 else slice(None)
        y_key = key[1] if len(key) > 1 else slice(None)
        x_key = key[2] if len(key) > 2 else slice(None)

        t_key = _convert_range_to_slice(t_key)

        frames = listify_index(t_key, self._num_frames)
        if not frames:
            return np.empty((0, *self._page_shape), dtype=self._dtype)

        out = self._read_frames(frames)

        if y_key != slice(None) or x_key != slice(None):
            out = out[:, y_key, x_key]

        if isinstance(t_key, int):
            out = out[0]

        return out

    def _read_frames(self, frames: list[int]) -> np.ndarray:
        buf = np.empty(
            (len(frames), self._page_shape[0], self._page_shape[1]), dtype=self._dtype
        )

        start = 0
        frame_to_buf_idx = {f: i for i, f in enumerate(frames)}

        for tf, nframes in zip(self.tiff_files, self._frames_per_file, strict=False):
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
                    raise OSError(
                        f"Failed to read frames {local_indices} from {tf.filename}: {e}"
                    ) from e

            if chunk.ndim == 2:
                chunk = chunk[np.newaxis, ...]

            for local_idx, global_frame in zip(local_indices, file_frames, strict=False):
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
    dims : str | Sequence[str] | None, optional
        Dimension labels. If None, inferred from shape (TZYX).

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
    dims : tuple[str, ...]
        Dimension labels.

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

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """
        Check if this file can be opened by TiffArray.

        Returns True for any valid TIFF file. This is the fallback
        array type for TIFF files.

        Parameters
        ----------
        file : Path or str
            Path to check.

        Returns
        -------
        bool
            True if file is a TIFF file.
        """
        if not file:
            return False
        path = Path(file)
        return path.suffix.lower() in (".tif", ".tiff")

    def __init__(
        self,
        files: str | Path | list[str] | list[Path],
        dims: str | Sequence[str] | None = None,
    ):
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
            paths = [Path(f) for f in files]

            # Try to group by plane number
            plane_groups = {}
            for p in paths:
                pnum = _extract_tiff_plane_number(p.name)
                if pnum is not None:
                    plane_groups.setdefault(pnum, []).append(p)

            if len(plane_groups) > 1:
                # Multiple planes detected in file list! Load as volume.
                # Sort planes and ensure files within each plane are sorted
                sorted_pnums = sorted(plane_groups.keys())
                plane_files_list = []
                for pnum in sorted_pnums:
                    files_for_plane = sorted(plane_groups[pnum])
                    plane_files_list.append(files_for_plane)

                # Check if each plane has the same number of files (optional, but safer)
                # For now, let's just use _init_volume logic but handle lists of files
                self._init_volume_from_groups(plane_files_list)
            else:
                self._init_single_plane(paths)

        self._dim_labels = DimLabels(dims, ndim=self.ndim)

    @property
    def dims(self) -> tuple[str, ...]:
        return self._dim_labels.value

    def _init_volume_from_groups(self, plane_groups: list[list[Path]]):
        """Initialize as volumetric array from groups of files (one group per plane)."""
        self._is_volumetric = True
        self._planes = []

        for files in plane_groups:
            reader = _SingleTiffPlaneReader(files)
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
        self.filenames = [f for group in plane_groups for f in group]

        # Use metadata from first plane
        try:
            from mbo_utilities.metadata import get_metadata_single

            self._metadata = get_metadata_single(plane_groups[0][0])
        except Exception:
            self._metadata = {}

        self._metadata.update(
            {
                "shape": self.shape,
                "dtype": str(self._dtype),
                "nframes": self._nframes,
                "num_frames": self._nframes,
                "num_planes": self._nz,
                "file_paths": [str(p) for p in self.filenames],
            }
        )
        self.num_rois = 1

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

        # try to read metadata from file first, fall back to basic info
        try:
            from mbo_utilities.metadata import get_metadata_single

            self._metadata = get_metadata_single(files[0])
        except Exception:
            self._metadata = {}

        # ensure basic shape info is present
        self._metadata.update(
            {
                "shape": self.shape,
                "dtype": str(self._dtype),
                "nframes": self._nframes,
                "num_frames": self._nframes,
                "file_paths": [str(p) for p in files],
                "num_files": len(files),
            }
        )
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

        # try to read metadata from first plane file, fall back to basic info
        try:
            from mbo_utilities.metadata import get_metadata_single

            self._metadata = get_metadata_single(plane_files[0])
        except Exception:
            self._metadata = {}

        # ensure basic shape info is present
        self._metadata.update(
            {
                "shape": self.shape,
                "dtype": str(self._dtype),
                "nframes": self._nframes,
                "num_frames": self._nframes,
                "num_planes": self._nz,
                "plane_files": [str(p) for p in plane_files],
            }
        )
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
        from mbo_utilities.util import get_dtype

        return (
            self._target_dtype
            if self._target_dtype is not None
            else get_dtype(self._dtype)
        )

    @property
    def ndim(self) -> int:
        return 4

    @property
    def metadata(self) -> dict:
        """Return metadata as dict. Always returns dict, never None."""
        return self._metadata if self._metadata is not None else {}

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    def _compute_frame_vminmax(self):
        if not hasattr(self, "_cached_vmin"):
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
                raise IndexError(
                    f"Time index {t_key} out of bounds for {self._nframes} frames"
                )

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

    def __array__(self, dtype=None, copy=None):
        # return single frame for fast histogram/preview (prevents accidental full load)
        data = self[0]
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def close(self):
        for plane in self._planes:
            plane.close()

    def imshow(self, **kwargs):
        import fastplotlib as fpl

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        window_funcs = kwargs.get("window_funcs")
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
    dims : str | Sequence[str] | None, optional
        Dimension labels. If None, inferred from shape (TZYX).

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape in TZYX format.
    dtype : np.dtype
        Data type.
    dims : tuple[str, ...]
        Dimension labels.
    """

    # Keys that indicate MBO-processed TIFF (in shaped_metadata)
    _MBO_MARKER_KEYS = {"num_planes", "num_rois", "frame_rate", "Ly", "Lx", "fov", "pixel_resolution"}

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """
        Check if this file can be opened by MBOTiffArray.

        Returns True for TIFF files with MBO-specific shaped_metadata,
        indicating they were processed/assembled by mbo_utilities.

        Parameters
        ----------
        file : Path or str
            Path to check.

        Returns
        -------
        bool
            True if file has MBO shaped_metadata.
        """
        if not file or not isinstance(file, (str, Path)):
            return False
        path = Path(file)
        if path.suffix.lower() not in (".tif", ".tiff"):
            return False
        try:
            with TiffFile(file) as tf:
                if not hasattr(tf, "shaped_metadata") or tf.shaped_metadata is None:
                    return False
                meta = tf.shaped_metadata[0] if tf.shaped_metadata else {}
                if not isinstance(meta, dict):
                    return False
                # Check if any MBO-specific keys are present
                return bool(cls._MBO_MARKER_KEYS & meta.keys())
        except Exception:
            return False

    def __init__(
        self,
        filenames: list[Path],
        roi: int | None = None,
        dims: str | Sequence[str] | None = None,
    ):
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

        for i, (tfile, fpath) in enumerate(zip(self.tiff_files, self.filenames, strict=False)):
            nframes = None

            desc = page0.description if i == 0 else tfile.pages.first.description

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
        self._metadata.update(
            {
                "shape": self.shape,
                "dtype": str(self._dtype),
                "nframes": self._num_frames,
                "num_frames": self._num_frames,
                "frames_per_file": self._frames_per_file,
                "file_paths": [str(p) for p in self.filenames],
                "num_files": len(self.filenames),
            }
        )
        self.num_rois = get_param(self._metadata, "num_mrois", default=1)
        self.tags = [derive_tag_from_filename(f) for f in self.filenames]
        self._target_dtype = None
        self._dim_labels = DimLabels(dims, ndim=self.ndim)

    @property
    def dims(self) -> tuple[str, ...]:
        return self._dim_labels.value

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
    def shape(self) -> tuple[int, ...]:
        return self._num_frames, 1, self._page_shape[0], self._page_shape[1]

    @property
    def dtype(self):
        from mbo_utilities.util import get_dtype

        return (
            self._target_dtype
            if self._target_dtype is not None
            else get_dtype(self._dtype)
        )

    def _compute_frame_vminmax(self):
        if not hasattr(self, "_cached_vmin"):
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
            return np.empty((0, 1, *self._page_shape), dtype=self.dtype)

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
            (len(frames), 1, self._page_shape[0], self._page_shape[1]),
            dtype=self._dtype,
        )

        start = 0
        frame_to_buf_idx = {f: i for i, f in enumerate(frames)}

        for tf, nframes in zip(self.tiff_files, self._frames_per_file, strict=False):
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
                    raise OSError(
                        f"MBOTiffArray: Failed to read frames {local_indices} from {tf.filename}\n"
                        f"File may be corrupted or incomplete.\n"
                        f": {type(e).__name__}: {e}"
                    ) from e

            if chunk.ndim == 2:
                chunk = chunk[np.newaxis, ...]

            for local_idx, global_frame in zip(local_indices, file_frames, strict=False):
                buf_idx = frame_to_buf_idx[global_frame]
                chunk_idx = local_indices.index(local_idx)
                buf[buf_idx, 0] = chunk[chunk_idx]

            start = end

        return buf

    def astype(self, dtype, copy=True):
        self._target_dtype = np.dtype(dtype)
        return self

    def __array__(self, dtype=None, copy=None):
        # return single frame for fast histogram/preview (prevents accidental full load)
        data = self[0]
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def close(self):
        for tf in self.tiff_files:
            tf.close()

    def imshow(self, **kwargs):
        import fastplotlib as fpl

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        window_funcs = kwargs.get("window_funcs")
        return fpl.ImageWidget(
            data=self,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )

    def save(self, outpath, **kwargs):
        """Save array to disk."""
        return self._imwrite(outpath, **kwargs)

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


class ScanImageArray(RoiFeatureMixin, ReductionMixin):
    """
    Base class for raw ScanImage TIFF readers with phase correction support.

    Handles multi-ROI ScanImage data with bidirectional scanning phase correction.
    Supports ROI stitching, splitting, and individual ROI access.

    For automatic stack type detection, use the `open_scanimage()` factory function
    which returns the appropriate subclass (LBMArray, PiezoArray, or SinglePlaneArray).

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
    stack_type : StackType
        Detected stack type: "lbm", "piezo", or "single_plane".
    """

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """
        Check if this file can be opened by ScanImageArray.

        Returns True for raw ScanImage TIFFs that have scanimage_metadata
        and no shaped_metadata (indicating unprocessed acquisition data).

        Parameters
        ----------
        file : Path or str
            Path to check.

        Returns
        -------
        bool
            True if file is a raw ScanImage TIFF.
        """
        if not file or not isinstance(file, (str, Path)):
            return False
        path = Path(file)
        if path.suffix.lower() not in (".tif", ".tiff"):
            return False
        try:
            with TiffFile(file) as tf:
                # If shaped_metadata exists, it's a processed file (not raw)
                if (
                    hasattr(tf, "shaped_metadata")
                    and tf.shaped_metadata is not None
                    and isinstance(tf.shaped_metadata, (list, tuple))
                    and len(tf.shaped_metadata) > 0
                ):
                    return False
                # Must have ScanImage metadata
                return tf.scanimage_metadata is not None
        except Exception:
            return False

    # contextual descriptions for metadata params (shown in GUI tooltips)
    # subclasses can override to provide array-type-specific context
    METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Raw TIFF page height in pixels.",
        "Lx": "Raw TIFF page width in pixels.",
        "num_zplanes": "Number of z-planes in the stack.",
        "dz": "Z-step size in micrometers.",
    }

    # metadata fields that require user input (not available in file metadata)
    # subclasses override this to specify required fields
    # format: list of canonical param names
    REQUIRED_METADATA: list[str] = []

    def get_required_metadata(self) -> list[dict]:
        """
        Get list of required metadata fields with their current values.

        Returns a list of dicts with 'canonical', 'description', 'label',
        'unit', 'dtype', and 'value' for each required field.
        """
        from mbo_utilities.metadata import METADATA_PARAMS, get_param

        fields = []
        for param in self.REQUIRED_METADATA:
            value = get_param(self._metadata, param, default=None)
            mp = METADATA_PARAMS.get(param)
            desc = self.get_param_description(param)
            fields.append(
                {
                    "canonical": param,
                    "description": desc,
                    "label": mp.label if mp else param,
                    "unit": mp.unit if mp else "",
                    "dtype": mp.dtype if mp else float,
                    "value": value,
                }
            )
        return fields

    def get_param_description(self, param: str) -> str:
        """
        Get description for a metadata parameter with array-type context.

        Searches the class hierarchy for contextual descriptions, falling back
        to the global METADATA_PARAMS registry if no context is found.

        Parameters
        ----------
        param : str
            Parameter name (e.g., "Ly", "dz", "num_zplanes").

        Returns
        -------
        str
            Contextual description for the parameter.
        """
        # check class hierarchy for context
        for cls in type(self).__mro__:
            ctx = getattr(cls, "METADATA_CONTEXT", None)
            if ctx and param in ctx:
                return ctx[param]
        # fallback to METADATA_PARAMS
        from mbo_utilities.metadata import METADATA_PARAMS

        mp = METADATA_PARAMS.get(param)
        return mp.description if mp else ""

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
        metadata: dict | None = None,
        dims: str | Sequence[str] | None = None,
    ):
        self.filenames = [files] if isinstance(files, (str, Path)) else list(files)
        self.tiff_files = [TiffFile(f) for f in self.filenames]
        self._tiff_lock = threading.Lock()

        # Use provided metadata if available to avoid re-scanning
        if metadata is not None:
            self._metadata = metadata
        else:
            self._metadata = get_metadata(self.filenames)

        self.num_channels = get_param(self._metadata, "nplanes", default=1)
        self.num_frames = get_param(self._metadata, "nframes")
        self._source_dtype = get_param(self._metadata, "dtype", default="int16")
        self._target_dtype = None
        self._ndim = self._metadata.get("ndim", 3)
        self._frames_per_file = self._metadata.get("frames_per_file", None)

        # Extract ROI info before setting roi (needed for validation)
        self._rois = self._extract_roi_info()

        # Initialize ROI state (mixin provides roi property with validation)
        self._roi = None
        if roi is not None:
            self.roi = roi  # validates via mixin setter

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

        # Initialize PhaseCorrectionFeature
        self.phase_correction = PhaseCorrectionFeature(
            enabled=fix_phase,
            method=phasecorr_method,
            shift=None,  # auto-compute by default
            use_fft=use_fft,
            upsample=upsample,
            border=border if isinstance(border, int) else 3,  # feature checks int
            max_offset=max_offset,
        )

        self.phase_correction.add_event_handler(self._on_feature_change)

        self._dim_labels = DimLabels(dims, ndim=self.ndim)

    @property
    def dims(self) -> tuple[str, ...]:
        return self._dim_labels.value

    def _on_feature_change(self, event):
        # Optional: handle feature changes (log, etc)
        pass

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
    def stack_type(self) -> StackType:
        """Detected stack type: 'lbm', 'piezo', or 'single_plane'."""
        return detect_stack_type(self._metadata)

    @property
    def dtype(self):
        return (
            self._target_dtype if self._target_dtype is not None else self._source_dtype
        )

    def _compute_frame_vminmax(self):
        if not hasattr(self, "_cached_vmin"):
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
    def metadata(self) -> dict:
        """Return metadata as dict. Always returns dict, never None."""
        if self._metadata is None:
            self._metadata = {}
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
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        if self._metadata is None:
            self._metadata = {}
        self._metadata.update(value)

    @property
    def rois(self):
        """Alias for roi_slices (backwards compatibility)."""
        return self.roi_slices

    @property
    def offset(self):
        return self.phase_correction.effective_shift or self._offset

    @offset.setter
    def offset(self, value: float | np.ndarray):
        # If user manually sets offset, treat it as setting fixed shift
        if isinstance(value, (int, float)):
            self.phase_correction.shift = float(value)
        self._offset = value

    @property
    def use_fft(self):
        return self.phase_correction.use_fft

    @use_fft.setter
    def use_fft(self, value: bool):
        self.phase_correction.use_fft = value

    @property
    def phasecorr_method(self):
        return self.phase_correction.method.value

    @phasecorr_method.setter
    def phasecorr_method(self, value: str | None):
        if value is None:
            self.phase_correction.enabled = False
        else:
            self.phase_correction.method = value

    @property
    def fix_phase(self):
        return self.phase_correction.enabled

    @fix_phase.setter
    def fix_phase(self, value: bool):
        self.phase_correction.enabled = value

    @property
    def border(self):
        return self.phase_correction.border

    @border.setter
    def border(self, value: int):
        self.phase_correction.border = value

    @property
    def max_offset(self):
        return self.phase_correction.max_offset

    @max_offset.setter
    def max_offset(self, value: int):
        self.phase_correction.max_offset = value

    @property
    def upsample(self):
        return self.phase_correction.upsample

    @upsample.setter
    def upsample(self, value: int):
        self.phase_correction.upsample = value

    @property
    def mean_subtraction(self):
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("mean_subtraction must be a boolean value.")
        self._mean_subtraction = value

    # roi property is provided by RoiFeatureMixin

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
            zip(self.tiff_files, (f * self.num_channels for f in self._frames_per_file), strict=False)
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
                        raise OSError(
                            f"ScanImageArray: Failed to read pages {frame_idx} from {tf.filename}\n"
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
                            raise OSError(
                                f"ScanImageArray: Failed to read page {fi} from {tf.filename}\n"
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

                # If we have a fixed shift, use it
                shift = self.phase_correction.effective_shift

                if shift is not None:
                    from mbo_utilities.analysis.phasecorr import _apply_offset
                    # Use _apply_offset directly or feature.apply
                    # Note: feature.apply returns 2D, but we have 3D (Z/T) chunk (N, Y, X)
                    # Bidirectional phase correction applies to rows (X axis)
                    # _apply_offset handles N-D arrays if applied along last axis?
                    # Let's inspect source or assume it works like bidir_phasecorr

                    # Fallback to applying manually using computed shift
                    corrected = _apply_offset(chunk, shift, use_fft=self.use_fft)
                    offset = shift
                else:
                    # No fixed shift, compute on this chunk (Legacy behavior)
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
                    f"fft={self.use_fft}, chunk={chunk.shape}, took {(_t1 - _t0) * 1000:.1f}ms"
                )
            else:
                buf[idxs] = chunk
                self._offset = 0.0
            start = end

        logger.debug(
            f"_read_pages: {len(frames)} frames, {len(chans)} chans -> {buf.shape}"
        )
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
        self.logger.debug(f"__getitem__ took {(t1 - t0) * 1000:.1f}ms")

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
                frames,
                chans,
                yslice=self._rois[self.roi - 1]["slice"],
                xslice=slice(None),
            )

        full_data = self._read_pages(
            frames, chans, yslice=slice(None), xslice=slice(None)
        )

        if self.roi is not None:
            if isinstance(self.roi, list):
                return tuple(
                    full_data[:, :, self._rois[r - 1]["slice"], :] for r in self.roi
                )
            if self.roi == 0:
                pass  # Already handled by splitting in calling code or higher level

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
        """Write ScanImage array to disk."""
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

    def save(self, outpath, **kwargs):
        """Save array to disk."""
        return self._imwrite(outpath, **kwargs)

    @property
    def num_planes(self):
        return self.num_channels

    @property
    def shape(self):
        if self.roi is not None and not isinstance(self.roi, (list, tuple)):
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

    def __array__(self, dtype=None, copy=None):
        # return single frame for fast histogram/preview (prevents accidental full load)
        data = self[0]
        if dtype is not None:
            data = data.astype(dtype)
        return data

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
            roi_iterator=self.iter_rois(),
            **kwargs,
        )

    def imshow(self, **kwargs):
        import fastplotlib as fpl

        arrays = []
        names = []
        for roi in self.iter_rois():
            arr = copy.copy(self)
            arr.roi = roi
            # Need to disable feature on copies to show raw? Or valid?
            # Original code disabled correction.
            arr.fix_phase = False
            # Note: setting fix_phase=False on copy also sets feature.enabled=False
            # due to property delegation.
            arrays.append(arr)
            names.append(f"ROI {roi}" if roi else "Stitched mROIs")

        figure_shape = (1, len(arrays))
        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get("figure_kwargs", {"size": (600, 600)})
        window_funcs = kwargs.get("window_funcs")

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


# backwards compatibility alias
MboRawArray = ScanImageArray


class LBMArray(ScanImageArray):
    """
    LBM (Light Beads Microscopy) array reader.

    For LBM stacks, z-planes are interleaved as channels in ScanImage.
    Each TIFF frame represents one timepoint with all z-planes.

    This class validates that the data is actually an LBM stack and
    provides LBM-specific defaults.

    Parameters
    ----------
    files : str, Path, or list
        TIFF file path(s).
    **kwargs
        Additional arguments passed to ScanImageArray.

    Raises
    ------
    ValueError
        If the data is not an LBM stack.
    """

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """
        Check if this file can be opened by LBMArray.

        Returns True for raw ScanImage TIFFs with LBM stack type.

        Parameters
        ----------
        file : Path or str
            Path to check.

        Returns
        -------
        bool
            True if file is an LBM stack.
        """
        if not ScanImageArray.can_open(file):
            return False
        try:
            meta = get_metadata(file)
            return detect_stack_type(meta) == "lbm"
        except Exception:
            return False

    METADATA_CONTEXT: dict[str, str] = {
        "Ly": (
            "For LBM with multi-ROIs, this is the total vertical height of all ROI strips "
            "including fly-to deadspace between them (num_fly_to_lines pixels)."
        ),
        "Lx": "ROI width in pixels (same for all mROIs in the scan).",
        "num_zplanes": "Number of z-planes encoded as ScanImage channels for LBM.",
        "dz": "Z-step size (µm). Must be user-supplied for LBM - not in ScanImage metadata.",
        "fs": "Volume rate in Hz (frame rate / num_zplanes for LBM).",
    }

    # dz is not stored in ScanImage metadata for LBM, must be user-supplied
    REQUIRED_METADATA: list[str] = ["dz"]

    def __init__(
        self, files: str | Path | list, metadata: dict | None = None, **kwargs
    ):
        super().__init__(files, metadata=metadata, **kwargs)
        if self.stack_type != "lbm":
            raise ValueError(
                f"LBMArray requires LBM stack data, but detected '{self.stack_type}'. "
                f"Use open_scanimage() for automatic detection or ScanImageArray directly."
            )


class PiezoArray(ScanImageArray):
    """
    Piezo z-stack array reader with optional frame averaging.

    For piezo stacks, the z-piezo moves sequentially through slices.
    May have multiple frames per slice (framesPerSlice > 1) that can
    optionally be averaged together.

    Parameters
    ----------
    files : str, Path, or list
        TIFF file path(s).
    average_frames : bool, default False
        If True and framesPerSlice > 1, average frames at each z-slice.
        Only applies when logAverageFactor == 1 (not pre-averaged).
    **kwargs
        Additional arguments passed to ScanImageArray.

    Raises
    ------
    ValueError
        If the data is not a piezo stack.

    Attributes
    ----------
    frames_per_slice : int
        Number of frames acquired per z-slice.
    log_average_factor : int
        Averaging factor from acquisition (>1 means pre-averaged).
    average_frames : bool
        Whether to average frames per slice.
    can_average : bool
        True if frame averaging is possible (not pre-averaged, >1 frame/slice).
    """

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """
        Check if this file can be opened by PiezoArray.

        Returns True for raw ScanImage TIFFs with piezo stack type.

        Parameters
        ----------
        file : Path or str
            Path to check.

        Returns
        -------
        bool
            True if file is a piezo stack.
        """
        if not ScanImageArray.can_open(file):
            return False
        try:
            meta = get_metadata(file)
            return detect_stack_type(meta) == "piezo"
        except Exception:
            return False

    METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Frame height in pixels.",
        "Lx": "Frame width in pixels.",
        "num_zplanes": "Number of z-slices per volume (from hStackManager.numSlices).",
        "dz": "Z-step size in µm (from hStackManager.stackZStepSize).",
        "frames_per_slice": "Frames acquired at each z-position before piezo moves.",
        "log_average_factor": "If >1, frames were averaged during acquisition.",
        "fs": "Frame rate in Hz.",
    }

    def __init__(
        self,
        files: str | Path | list,
        average_frames: bool = False,
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(files, metadata=metadata, **kwargs)
        if self.stack_type != "piezo":
            raise ValueError(
                f"PiezoArray requires piezo stack data, but detected '{self.stack_type}'. "
                f"Use open_scanimage() for automatic detection or ScanImageArray directly."
            )

        self.frames_per_slice = get_frames_per_slice(self._metadata)
        self.log_average_factor = get_log_average_factor(self._metadata)
        self._average_frames = average_frames and self.can_average

    @property
    def frames_per_slice(self) -> int:
        """Number of frames acquired per z-slice (from hStackManager.framesPerSlice)."""
        return self._frames_per_slice

    @property
    def log_average_factor(self) -> int:
        """Averaging factor from acquisition (>1 means frames were pre-averaged)."""
        return self._log_average_factor

    @property
    def can_average(self) -> bool:
        """True if frame averaging is possible (not pre-averaged, >1 frame/slice)."""
        return self.log_average_factor == 1 and self.frames_per_slice > 1

    @property
    def average_frames(self) -> bool:
        """Whether to average frames per slice when reading data."""
        return self._average_frames

    @average_frames.setter
    def average_frames(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("average_frames must be a boolean value.")
        if value and not self.can_average:
            if self.log_average_factor > 1:
                logger.warning(
                    f"Frame averaging disabled: data was pre-averaged at acquisition "
                    f"(logAverageFactor={self.log_average_factor})."
                )
            elif self.frames_per_slice <= 1:
                logger.warning(
                    f"Frame averaging disabled: only 1 frame per slice "
                    f"(framesPerSlice={self.frames_per_slice})."
                )
            value = False
        self._average_frames = value

    @property
    def shape(self):
        base_shape = super().shape
        if self.average_frames and self.can_average:
            # when averaging, we have fewer timepoints
            # original: (total_frames, num_planes, H, W)
            # where total_frames = num_volumes * frames_per_slice
            # after averaging: (num_volumes, num_planes, H, W)
            n_frames, n_planes, h, w = base_shape
            n_volumes = n_frames // self.frames_per_slice
            return (n_volumes, n_planes, h, w)
        return base_shape

    def __getitem__(self, key):
        if not self.average_frames or not self.can_average:
            return super().__getitem__(key)

        # frame averaging mode: need to read multiple frames and average
        if not isinstance(key, tuple):
            key = (key,)

        t_key = key[0] if len(key) > 0 else slice(None)
        rest_key = key[1:] if len(key) > 1 else ()

        # convert t_key to list of volume indices
        n_volumes = self.shape[0]
        frames = listify_index(t_key, n_volumes)

        if not frames:
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)

        fps = self.frames_per_slice
        results = []

        for vol_idx in frames:
            # read all frames for this volume and average
            start_frame = vol_idx * fps
            end_frame = start_frame + fps
            # get data for all frames in this volume
            vol_data = super().__getitem__((slice(start_frame, end_frame), *rest_key))
            # average over the first axis (frames within volume)
            averaged = np.mean(vol_data, axis=0, keepdims=True)
            results.append(averaged)

        out = np.concatenate(results, axis=0)

        # handle integer indexing (squeeze first dim)
        if isinstance(t_key, int):
            out = out[0]

        return out


class SinglePlaneArray(ScanImageArray):
    """
    Single-plane time series array reader.

    For single-plane acquisitions without z-stack.

    Parameters
    ----------
    files : str, Path, or list
        TIFF file path(s).
    **kwargs
        Additional arguments passed to ScanImageArray.

    Raises
    ------
    ValueError
        If the data is not a single-plane acquisition.
    """

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """
        Check if this file can be opened by SinglePlaneArray.

        Returns True for raw ScanImage TIFFs with single-plane stack type.

        Parameters
        ----------
        file : Path or str
            Path to check.

        Returns
        -------
        bool
            True if file is a single-plane acquisition.
        """
        if not ScanImageArray.can_open(file):
            return False
        try:
            meta = get_metadata(file)
            return detect_stack_type(meta) == "single_plane"
        except Exception:
            return False

    METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Frame height in pixels.",
        "Lx": "Frame width in pixels.",
        "num_zplanes": "Single plane (no z-stack).",
        "fs": "Frame rate in Hz.",
    }

    def __init__(
        self, files: str | Path | list, metadata: dict | None = None, **kwargs
    ):
        super().__init__(files, metadata=metadata, **kwargs)
        if self.stack_type != "single_plane":
            raise ValueError(
                f"SinglePlaneArray requires single-plane data, but detected '{self.stack_type}'. "
                f"Use open_scanimage() for automatic detection or ScanImageArray directly."
            )


class CalibrationArray(ScanImageArray):
    """
    Calibration array reader for pollen/bead calibration data.

    For calibration stacks that combine LBM beamlet channels with piezo z-scanning.
    This is specifically for pollen calibration where the z-piezo scans through
    different focal planes while LBM channels capture individual beamlet data.

    The data has dimensions ZCYX where:
    - Z: piezo z-positions (focal planes)
    - C: LBM beamlet channels
    - Y: spatial height
    - X: spatial width

    Parameters
    ----------
    files : str, Path, or list
        TIFF file path(s).
    **kwargs
        Additional arguments passed to ScanImageArray.

    Raises
    ------
    ValueError
        If the data is not a pollen/calibration stack.

    Attributes
    ----------
    num_zplanes : int
        Number of z-positions (piezo steps).
    num_beamlets : int
        Number of LBM beamlet channels.
    """

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """
        Check if this file can be opened by CalibrationArray.

        Returns True for raw ScanImage TIFFs with pollen/calibration stack type.

        Parameters
        ----------
        file : Path or str
            Path to check.

        Returns
        -------
        bool
            True if file is a pollen/calibration stack.
        """
        if not ScanImageArray.can_open(file):
            return False
        try:
            meta = get_metadata(file)
            return detect_stack_type(meta) == "pollen"
        except Exception:
            return False

    METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Frame height in pixels.",
        "Lx": "Frame width in pixels.",
        "num_zplanes": "Number of z-slices (piezo positions).",
        "num_beamlets": "Number of LBM beamlet channels.",
        "dz": "Z-step size in µm (from hStackManager.stackZStepSize).",
        "fs": "Frame rate in Hz.",
    }

    def __init__(
        self, files: str | Path | list, metadata: dict | None = None, **kwargs
    ):
        super().__init__(files, metadata=metadata, **kwargs)
        if self.stack_type != "pollen":
            raise ValueError(
                f"CalibrationArray requires pollen calibration data, but detected '{self.stack_type}'. "
                f"Use open_scanimage() for automatic detection or ScanImageArray directly."
            )

    @property
    def num_zplanes(self) -> int:
        """Number of z-positions (piezo steps)."""
        return self.num_frames

    @property
    def num_beamlets(self) -> int:
        """Number of LBM beamlet channels."""
        return self.num_channels


def open_scanimage(files: str | Path | list, **kwargs) -> ScanImageArray:
    """
    Open ScanImage TIFF file(s), automatically detecting stack type.

    Factory function that returns the appropriate array subclass based on
    the detected acquisition type (LBM, piezo, pollen, or single-plane).

    Parameters
    ----------
    files : str, Path, or list
        TIFF file path(s).
    **kwargs
        Additional arguments passed to the array class.
        For PiezoArray, can include `average_frames=True`.

    Returns
    -------
    ScanImageArray
        One of LBMArray, PiezoArray, CalibrationArray, or SinglePlaneArray.
        Pollen calibration stacks (LBM + piezo) return CalibrationArray.

    Examples
    --------
    >>> arr = open_scanimage("data.tif")
    >>> print(arr.stack_type)
    'piezo'
    >>> print(type(arr).__name__)
    'PiezoArray'

    >>> # with frame averaging for piezo stacks
    >>> arr = open_scanimage("piezo_data.tif", average_frames=True)
    >>> if hasattr(arr, 'can_average') and arr.can_average:
    ...     print("Frames will be averaged per slice")
    """
    # get metadata to detect type
    file_list = [files] if isinstance(files, (str, Path)) else list(files)
    metadata = get_metadata(file_list)
    stack_type = detect_stack_type(metadata)

    logger.debug(f"open_scanimage: detected stack_type='{stack_type}'")

    # Pass metadata to prevent re-fetching/counting
    try:
        if stack_type == "lbm":
            # LBMArray doesn't use average_frames
            kwargs.pop("average_frames", None)
            return LBMArray(files, metadata=metadata, **kwargs)
        if stack_type == "pollen":
            # Pollen calibration: LBM beamlets + piezo z-scanning
            kwargs.pop("average_frames", None)
            return CalibrationArray(files, metadata=metadata, **kwargs)
        if stack_type == "piezo":
            return PiezoArray(files, metadata=metadata, **kwargs)
        # single_plane
        kwargs.pop("average_frames", None)
        return SinglePlaneArray(files, metadata=metadata, **kwargs)
    except TypeError:
        # Fallback if subclasses don't support metadata arg (safety)
        if stack_type == "lbm":
            kwargs.pop("average_frames", None)
            return LBMArray(files, **kwargs)
        if stack_type == "pollen":
            kwargs.pop("average_frames", None)
            return CalibrationArray(files, **kwargs)
        if stack_type == "piezo":
            return PiezoArray(files, **kwargs)
        kwargs.pop("average_frames", None)
        return SinglePlaneArray(files, **kwargs)
