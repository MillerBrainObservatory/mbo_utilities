"""
NumPy array wrapper.

This module provides NumpyArray for wrapping NumPy arrays and .npy files
as lazy arrays conforming to LazyArrayProtocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin, Shape5DMixin
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline
import contextlib

logger = log.get("arrays.numpy")

# register numpy pipeline info
_NUMPY_INFO = PipelineInfo(
    name="numpy",
    description="NumPy .npy files",
    input_patterns=[
        "**/*.npy",
    ],
    output_patterns=[
        "**/*.npy",
    ],
    input_extensions=["npy"],
    output_extensions=["npy"],
    marker_files=[],
    category="reader",
)
register_pipeline(_NUMPY_INFO)


class NumpyArray(ReductionMixin, Shape5DMixin):
    """
    Lazy array wrapper for NumPy arrays and .npy files.

    Conforms to LazyArrayProtocol for compatibility with mbo_utilities I/O
    and processing pipelines. Supports 2D (image), 3D (time series), and
    4D (volumetric) data.

    Parameters
    ----------
    array : np.ndarray, str, or Path
        Either a numpy array (kept in memory, no temp file created)
        or a path to a .npy file (memory-mapped for lazy loading).
    metadata : dict, optional
        Metadata dictionary. If not provided, basic metadata is inferred
        from array shape.

    Examples
    --------
    >>> # From .npy file (memory-mapped, lazy)
    >>> arr = NumpyArray("data.npy")
    >>> arr.shape
    (100, 512, 512)

    >>> # From in-memory array (wraps directly, no temp file)
    >>> data = np.random.randn(100, 512, 512).astype(np.float32)
    >>> arr = NumpyArray(data)
    >>> arr[0:10]  # Slicing

    >>> # 4D volumetric data
    >>> vol = NumpyArray("volume.npy")  # shape: (T, Z, Y, X)
    >>> vol.ndim
    4

    >>> # Use with imwrite
    >>> from mbo_utilities import imread, imwrite
    >>> arr = imread(my_numpy_array)  # Returns NumpyArray
    >>> imwrite(arr, "output", ext=".zarr")  # Full write support
    """

    def __init__(self, array: np.ndarray | str | Path, metadata: dict | None = None):
        self._tempfile = None
        self._npz_file = None
        self._is_in_memory = False

        if isinstance(array, (str, Path)):
            self.path = Path(array)
            if not self.path.exists():
                raise FileNotFoundError(f"Numpy file not found: {self.path}")

            # Try loading - could be pure .npy or npz with embedded metadata
            loaded = np.load(self.path, mmap_mode="r", allow_pickle=True)

            if isinstance(loaded, np.lib.npyio.NpzFile):
                # NPZ format with embedded data and metadata
                self.data = loaded["data"]
                if "metadata" in loaded.files:
                    # Extract metadata dict from numpy array
                    meta_arr = loaded["metadata"]
                    if meta_arr.ndim == 0:
                        # Scalar array containing dict
                        self._metadata = meta_arr.item()
                    else:
                        self._metadata = {}
                else:
                    self._metadata = {}
                self._npz_file = loaded  # Keep reference to prevent closing
            else:
                # Pure .npy file
                self.data = loaded
                self._metadata = {}

        elif isinstance(array, np.ndarray):
            # Keep array in memory - no temp file needed
            # This is more efficient and avoids disk I/O
            self.data = array
            self.path = None
            self._metadata = {}
            self._is_in_memory = True
            logger.debug(f"Wrapping in-memory array with shape {array.shape}")
        else:
            raise TypeError(f"Expected np.ndarray or path, got {type(array)}")

        # Override with explicit metadata if provided
        if metadata is not None:
            self._metadata = metadata

        self._raw_shape = self.data.shape
        self._dtype = self.data.dtype
        self._target_dtype = None

        # Set dimension labels based on array shape
        self._dims = self._infer_dims()

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return self._shape5d()

    @property
    def ndim(self) -> int:
        return 5

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        s = self._raw_shape
        if len(s) == 5:
            return s
        if len(s) == 4:
            return (s[0], 1, s[1], s[2], s[3])
        if len(s) == 3:
            return (s[0], 1, 1, s[1], s[2])
        if len(s) == 2:
            return (1, 1, 1, s[0], s[1])
        return (1, 1, 1, 1, s[0]) if len(s) == 1 else (1, 1, 1, 1, 1)

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def _infer_dims(self) -> str:
        return "TCZYX"

    def __getitem__(self, item):
        # translate 5D indexing to the underlying array's actual ndim
        # shape5d pads missing dims with singletons at the front
        # e.g., 3D TYX data has shape5d (T, 1, 1, Y, X)
        # key (t, c, z, y, x) maps to (t, y, x) by dropping singleton dims
        if not isinstance(item, tuple):
            item = (item,)
        raw_ndim = len(self._raw_shape)
        skip = set()
        if len(item) > raw_ndim:
            if raw_ndim <= 3:
                skip = {1, 2}  # C, Z
            elif raw_ndim == 4:
                skip = {1}  # C
            raw_key = tuple(k for i, k in enumerate(item) if i not in skip)
            raw_key = raw_key[:raw_ndim]
        else:
            raw_key = item

        out = self.data[raw_key]
        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)

        # re-expand stripped singleton dims (only for slice keys, not int keys)
        for axis in sorted(skip):
            if axis < len(item) and not isinstance(item[axis], (int, np.integer)):
                out = np.expand_dims(out, axis=axis)

        return out

    def __len__(self) -> int:
        return self.nt

    def __array__(self, dtype=None, copy=None):
        # return single frame for fast preview
        if len(self._raw_shape) <= 2:
            data = self.data
        else:
            data = self.data[0]
        if dtype is not None:
            data = np.asarray(data).astype(dtype)
        return np.asarray(data)

    def __repr__(self) -> str:
        mem_str = " (in-memory)" if self._is_in_memory else ""
        return f"NumpyArray(shape={self.shape}, dtype={self.dtype}, dims='{self.dims}'{mem_str})"

    @property
    def dims(self) -> tuple[str, ...]:
        from mbo_utilities.arrays._base import DIMS
        return DIMS

    @dims.setter
    def dims(self, value):
        """Set dimension labels (ignored, always TCZYX)."""
        if len(value) != self.ndim:
            raise ValueError(f"dims length {len(value)} doesn't match ndim {self.ndim}")
        self._dims = value

    @property
    def num_planes(self) -> int:
        """Number of Z-planes (index 2 in 5D TCZYX)."""
        return self.shape[2]

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame (frame 0, channel 0, plane 0)."""
        if not hasattr(self, "_cached_vmin"):
            frame = np.asarray(self[0, 0, 0])
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
    def filenames(self) -> list[Path]:
        """Return list of source files (empty for in-memory arrays)."""
        if self.path is not None:
            return [self.path]
        return []

    @property
    def metadata(self) -> dict:
        """Return metadata as dict. Always returns dict, never None."""
        # ensure basic metadata is always present
        md = dict(self._metadata) if self._metadata is not None else {}
        if "num_timepoints" not in md:
            md["num_timepoints"] = self.shape[0]
        if "nframes" not in md:
            md["nframes"] = md["num_timepoints"]  # suite2p alias
        if "num_frames" not in md:
            md["num_frames"] = md["num_timepoints"]  # legacy alias
        if "Ly" not in md:
            md["Ly"] = self.shape[-2]
        if "Lx" not in md:
            md["Lx"] = self.shape[-1]
        return md

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value


    def close(self):
        """Release resources and clean up temporary files."""
        if self._npz_file is not None:
            with contextlib.suppress(Exception):
                self._npz_file.close()
            self._npz_file = None

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

        histogram_widget = kwargs.pop("histogram_widget", True)
        figure_kwargs = kwargs.pop("figure_kwargs", {"size": (800, 800)})
        # get min/max from first frame for contrast scaling
        first_frame = self.data[0]
        graphic_kwargs = kwargs.pop(
            "graphic_kwargs", {"vmin": float(first_frame.min()), "vmax": float(first_frame.max())}
        )

        # always 5D TCZYX: sliders for t, c, z
        slider_dim_names = ("t", "c", "z")
        window_funcs = kwargs.pop("window_funcs", (np.mean, None, None))
        window_sizes = kwargs.pop("window_sizes", (1, None, None))

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
