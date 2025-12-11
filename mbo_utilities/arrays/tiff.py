"""
TIFF array readers.

This module provides array readers for TIFF files:
- TiffArray: Generic TIFF reader using TiffFile handles
- TiffVolumeArray: Reader for directories with plane TIFF files (4D volumes)
- MBOTiffArray: Lazy TIFF reader for MBO processed TIFFs (no dask)
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
from mbo_utilities.metadata import get_metadata
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

    # Sort by plane number
    def sort_key(p):
        num = _extract_tiff_plane_number(p.stem)
        return num if num is not None else float("inf")

    return sorted(plane_files, key=sort_key)


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

        # Allow string paths
        self.filenames = [Path(f) for f in filenames]
        self.roi = roi

        # Open TiffFile handles (no data read yet)
        self.tiff_files = [TiffFile(f) for f in self.filenames]
        self._tiff_lock = threading.Lock()

        # Extract info from first file's first page
        tf = self.tiff_files[0]
        page0 = tf.pages.first
        self._page_shape = page0.shape  # (Y, X) for 2D page
        self._dtype = page0.dtype

        # Try to get frame count from metadata without seeking
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
            if nframes is None:
                try:
                    est = query_tiff_pages(fpath)
                    if est > 1:
                        nframes = est
                except Exception:
                    pass

            # Method 3: Fallback to len(pages) - triggers seek but guaranteed
            if nframes is None:
                logger.debug(f"Using len(pages) fallback for {fpath}")
                nframes = len(tfile.pages)

            self._frames_per_file.append(nframes)
            self._num_frames += nframes

        # Collect metadata from first TIFF
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
        self.num_rois = self._metadata.get("num_rois", 1)
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
        # Return TZYX format: (frames, 1, Y, X)
        return self._num_frames, 1, self._page_shape[0], self._page_shape[1]

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame (frame 0, plane 0)."""
        if not hasattr(self, '_cached_vmin'):
            frame = np.asarray(self[0, 0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        """Min from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        """Max from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmax

    @property
    def ndim(self) -> int:
        return 4

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, key):
        """Read frames lazily using tf.asarray(key=frame_indices)."""
        if not isinstance(key, tuple):
            key = (key,)

        # Parse the key into frame indices
        t_key = key[0] if len(key) > 0 else slice(None)
        z_key = key[1] if len(key) > 1 else slice(None)
        y_key = key[2] if len(key) > 2 else slice(None)
        x_key = key[3] if len(key) > 3 else slice(None)

        # Convert range to slice
        t_key = _convert_range_to_slice(t_key)
        z_key = _convert_range_to_slice(z_key)
        y_key = _convert_range_to_slice(y_key)
        x_key = _convert_range_to_slice(x_key)

        # Convert to list of frame indices
        frames = listify_index(t_key, self._num_frames)
        if not frames:
            return np.empty((0, 1) + self._page_shape, dtype=self.dtype)

        # Read the requested frames
        out = self._read_frames(frames)

        # Apply spatial slicing if needed
        if y_key != slice(None) or x_key != slice(None):
            out = out[:, :, y_key, x_key]

        # Handle z dimension (always 1 for processed TIFFs)
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
        buf = np.empty(
            (len(frames), 1, self._page_shape[0], self._page_shape[1]), dtype=self._dtype
        )

        start = 0
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
                        f"MBOTiffArray: Failed to read frames {local_indices} from {tf.filename}\n"
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

    def close(self):
        """Close all TIFF file handles."""
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


class TiffArray(ReductionMixin):
    """
    Lazy TIFF array reader using TiffFile handles.

    Opens TiffFile handles on init (no data read), extracts shape from
    metadata/first page, and reads data lazily via tf.asarray() when indexed.
    Output is always in TZYX format.

    Parameters
    ----------
    files : str, Path, or list
        TIFF file path(s).

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape in TZYX format (nframes, 1, Y, X).
    dtype : np.dtype
        Data type.
    """

    def __init__(self, files: str | Path | List[str] | List[Path]):
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
            if nframes is None:
                try:
                    est = query_tiff_pages(fpath)
                    if est > 1:
                        nframes = est
                except Exception:
                    pass

            # Method 3: Fallback to len(pages) - triggers seek but guaranteed
            if nframes is None:
                logger.warning(f"Could not query TIFF pages for {fpath}")
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
        return self._num_frames, 1, self._page_shape[0], self._page_shape[1]

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    @property
    def ndim(self) -> int:
        return 4

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame (frame 0, plane 0)."""
        if not hasattr(self, '_cached_vmin'):
            frame = np.asarray(self[0, 0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        """Min from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        """Max from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmax

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
        buf = np.empty(
            (len(frames), 1, self._page_shape[0], self._page_shape[1]), dtype=self._dtype
        )

        start = 0
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


class TiffVolumeArray(ReductionMixin):
    """
    Reader for directories containing plane TIFF files as a 4D volume.

    Presents data as (T, Z, Y, X) by stacking individual plane TIFFs along
    the Z dimension. Each plane is loaded lazily via TiffArray.

    Parameters
    ----------
    directory : str or Path
        Path to directory containing plane TIFF files (e.g., plane01.tiff).
    plane_files : list[Path], optional
        Explicit list of plane files to use. If not provided, auto-detected.

    Attributes
    ----------
    shape : tuple[int, int, int, int]
        Shape as (T, Z, Y, X).
    dtype : np.dtype
        Data type.
    planes : list[TiffArray]
        Individual plane arrays.

    Examples
    --------
    >>> arr = TiffVolumeArray("tiff_output/")
    >>> arr.shape
    (10000, 14, 512, 512)
    >>> frame = arr[0]  # Get first frame across all planes
    >>> plane7 = arr[:, 6]  # Get all frames from plane 7 (0-indexed)
    """

    def __init__(
        self,
        directory: str | Path,
        plane_files: Sequence[Path] | None = None,
    ):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        # Find plane files
        if plane_files is None:
            plane_files = find_tiff_plane_files(self.directory)

        if not plane_files:
            raise ValueError(
                f"No TIFF plane files found in {self.directory}. "
                "Expected files matching pattern 'planeXX.tiff'."
            )

        # Load each plane as TiffArray
        self.planes: list[TiffArray] = []
        self.filenames = []
        for pfile in plane_files:
            arr = TiffArray(pfile)
            self.planes.append(arr)
            self.filenames.append(pfile)

        # Validate consistent shapes across planes
        # TiffArray shape is (T, 1, Y, X) - we check Y, X dimensions
        shapes = [(p.shape[2], p.shape[3]) for p in self.planes]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent spatial shapes across planes: {shapes}")

        nframes = [p.shape[0] for p in self.planes]
        if len(set(nframes)) != 1:
            logger.warning(
                f"Inconsistent frame counts across planes: {nframes}. "
                f"Using minimum: {min(nframes)}"
            )

        self._nframes = min(nframes)
        self._nz = len(self.planes)
        self._ly, self._lx = shapes[0]
        self._dtype = self.planes[0]._dtype
        self._target_dtype = None

        # Aggregate metadata from first plane
        self._metadata = dict(self.planes[0].metadata)
        self._metadata["num_planes"] = self._nz
        self._metadata["plane_files"] = [str(p) for p in plane_files]

        logger.info(
            f"Loaded TIFF volume: {self._nframes} frames, {self._nz} planes, "
            f"{self._ly}x{self._lx} px"
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self._nframes, self._nz, self._ly, self._lx)

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    @property
    def ndim(self) -> int:
        return 4

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def num_planes(self) -> int:
        return self._nz

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame (frame 0, plane 0)."""
        if not hasattr(self, '_cached_vmin'):
            frame = np.asarray(self[0, 0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        """Min from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        """Max from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmax

    def __len__(self) -> int:
        return self._nframes

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        t_key, z_key, y_key, x_key = key

        # Normalize t_key to respect _nframes limit
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

        # Handle single z index
        if isinstance(z_key, int):
            if z_key < 0:
                z_key = self._nz + z_key
            if z_key < 0 or z_key >= self._nz:
                raise IndexError(f"Z index {z_key} out of bounds for {self._nz} planes")
            # TiffArray returns (T, 1, Y, X), squeeze out the singleton Z
            out = self.planes[z_key][t_key, 0, y_key, x_key]
        else:
            # Handle z slice or full z
            if isinstance(z_key, slice):
                z_indices = range(self._nz)[z_key]
            elif isinstance(z_key, (list, np.ndarray)):
                z_indices = z_key
            else:
                z_indices = range(self._nz)

            # Stack data from selected planes
            # TiffArray returns (T, 1, Y, X), squeeze out the singleton Z before stacking
            arrs = [self.planes[i][t_key, 0, y_key, x_key] for i in z_indices]
            out = np.stack(arrs, axis=1)

        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def __array__(self) -> np.ndarray:
        """Materialize full array into memory: (T, Z, Y, X)."""
        arrs = [p[: self._nframes, 0] for p in self.planes]
        return np.stack(arrs, axis=1)

    def close(self):
        """Close all TIFF file handles."""
        for plane in self.planes:
            for tf in plane.tiff_files:
                tf.close()

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
        """Write TiffVolumeArray to disk in various formats."""
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
    use_fft : bool, default True
        Use FFT-based 2D phase correction (more accurate).

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
        use_fft: bool = True,
    ):
        self.filenames = [files] if isinstance(files, (str, Path)) else list(files)
        self.tiff_files = [TiffFile(f) for f in self.filenames]
        self._tiff_lock = threading.Lock()

        # Initialize data attributes first (needed for roi setter validation)
        self._metadata = get_metadata(self.filenames)
        self.num_channels = self._metadata["num_planes"]
        self.num_rois = self._metadata.get("num_rois", 1)

        # Now set roi (this will call the setter which validates)
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

        # Debug flags
        self.debug_flags = {
            "frame_idx": True,
            "roi_array_shape": False,
            "phase_offset": False,
        }
        self.num_frames = self._metadata["num_frames"]
        self._source_dtype = self._metadata["dtype"]
        self._target_dtype = None
        self._ndim = self._metadata["ndim"]

        self._frames_per_file = self._metadata.get("frames_per_file", None)

        self._rois = self._extract_roi_info()

    def _extract_roi_info(self):
        """Extract ROI positions and dimensions from metadata."""
        roi_groups = self._metadata["roi_groups"]
        if isinstance(roi_groups, dict):
            roi_groups = [roi_groups]

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

        # Calculate actual heights proportionally
        total_metadata_height = sum(heights_from_metadata)
        total_available_height = (
            actual_page_height - (len(roi_groups) - 1) * num_fly_to_lines
        )

        actual_heights = []
        remaining_height = total_available_height
        for i, metadata_height in enumerate(heights_from_metadata):
            if i == len(heights_from_metadata) - 1:
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
                "y_end": y_offset + height,
                "width": actual_page_width,
                "height": height,
                "x": 0,
                "slice": slice(y_offset, y_offset + height),
            }
            rois.append(roi_info)
            y_offset += height + num_fly_to_lines

        logger.debug(
            f"ROI structure: {[(r['y_start'], r['y_end'], r['height']) for r in rois]}"
        )

        return rois

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        """Return target dtype if set via astype(), otherwise source dtype."""
        return (
            self._target_dtype if self._target_dtype is not None else self._source_dtype
        )

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame (frame 0, plane 0)."""
        if not hasattr(self, '_cached_vmin'):
            frame = np.asarray(self[0, 0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        """Min from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        """Max from first frame for display (avoids full data read)."""
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
        """ROI info dict list."""
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
        """Border pixels to exclude from phase correlation."""
        return self._border

    @border.setter
    def border(self, value: int):
        self._border = value

    @property
    def max_offset(self):
        """Maximum allowed pixel offset for phase correction."""
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value: int):
        self._max_offset = value

    @property
    def upsample(self):
        """Upsampling factor for subpixel phase correlation."""
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
        # Validate ROI bounds
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
        tiff_width_px = len(listify_index(xslice, self._page_width))
        tiff_height_px = len(listify_index(yslice, self._page_height))
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
            with self._tiff_lock:
                try:
                    chunk = tf.asarray(key=frame_idx)
                except Exception as e:
                    raise IOError(
                        f"MboRawArray: Failed to read pages {frame_idx} from {tf.filename}\n"
                        f"File may be corrupted or incomplete.\n"
                        f": {type(e).__name__}: {e}"
                    ) from e
            if chunk.ndim == 2:
                chunk = chunk[np.newaxis, ...]
            chunk = chunk[..., yslice, xslice]

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
                self._offset = offset
            else:
                buf[idxs] = chunk
                self._offset = 0.0
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
        """Dispatch ROI processing."""
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
                yslice=self._rois[roi_idx]["slice"],
                xslice=slice(None),
            )
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
            xslice=slice(None),
        )

    @property
    def num_planes(self):
        """Alias for num_channels (ScanImage terminology)."""
        return self.num_channels

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
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    @property
    def _page_height(self):
        return self._metadata["page_height"]

    @property
    def _page_width(self):
        return self._metadata["page_width"]

    def __array__(self, max_frames: int = 100):
        """
        Return array representation using random frame subsampling.

        For large arrays, randomly samples frames to avoid loading entire dataset.
        This is more representative than strided subsampling for min/max estimation.

        Parameters
        ----------
        max_frames : int, default 100
            Maximum number of frames to sample. If array has more frames,
            random sampling is used.
        """
        if self.num_frames <= max_frames:
            # Small enough to load entirely
            return np.asarray(self[:])

        # Random sampling for large arrays
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
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

        # Fast vmin/vmax from single frame (no phase correction overhead)
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
