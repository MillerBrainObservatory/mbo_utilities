"""
NumPy array wrapper.

This module provides NumpyArray for wrapping NumPy arrays and .npy files
as lazy arrays conforming to LazyArrayProtocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, _index_5d_into_raw, DIMS, ReductionMixin, Shape5DMixin
from mbo_utilities.arrays.features._dim_labels import DEFAULT_DIMS
from mbo_utilities.lazy_array import register_array_class
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


_CANONICAL_DIMS = "TCZYX"


def _canonicalize_to_5d(arr: np.ndarray, dim_order: str | Sequence[str]) -> np.ndarray:
    """Permute and expand `arr` to canonical 5D TCZYX given user's `dim_order`."""
    labels = "".join(str(c).upper() for c in dim_order)
    if len(labels) != arr.ndim:
        raise ValueError(
            f"dim_order length {len(labels)} does not match array.ndim {arr.ndim}"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"dim_order has duplicate axes: {labels!r}")
    bad = [c for c in labels if c not in _CANONICAL_DIMS]
    if bad:
        raise ValueError(
            f"dim_order chars must be from {_CANONICAL_DIMS!r}, got {bad!r}"
        )

    positions = [_CANONICAL_DIMS.index(c) for c in labels]
    perm = sorted(range(len(positions)), key=lambda i: positions[i])
    arr = np.transpose(arr, perm)
    sorted_labels = "".join(labels[i] for i in perm)

    for i, c in enumerate(_CANONICAL_DIMS):
        if c not in sorted_labels:
            arr = np.expand_dims(arr, axis=i)
            sorted_labels = sorted_labels[:i] + c + sorted_labels[i:]
    return arr


def _normalize_declared(value: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize a declared dim order to a tuple of single uppercase chars."""
    if isinstance(value, str):
        return tuple(c.upper() for c in value if c.isalpha())
    return tuple(str(c).upper() for c in value)


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
        from array shape. A ``"dims"`` or ``"dimension_names"`` key is
        treated as ``dims`` below.
    dims : str or sequence of str, optional
        Axis order of the wrapped array (e.g. ``"TCYX"``, ``("Z","Y","X")``),
        length should equal the array's ndim, chars from ``TCZYX``. The array
        is canonicalized to 5D TCZYX. If omitted — or if the declared order is
        unusable (wrong length, duplicate/unknown axis) — the axes are
        chain-guessed from the rank and a warning is logged (never raises):
        2D=YX, 3D=TYX, 4D=TZYX, 5D=TCZYX. Settable after construction via
        ``arr.dims = ...``, ``arr.metadata = {"dims": ...}``, or
        ``arr.metadata = {"dimension_names": [...]}`` (NGFF lowercase form).
        ``imwrite`` uses whatever ``dims`` resolved to, so label channel-vs-Z
        before writing if the rank guess is wrong.
    dim_order : str or sequence of str, optional
        Alias for ``dims`` (back-compat).

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

    def __init__(
        self,
        array: np.ndarray | str | Path,
        metadata: dict | None = None,
        dim_order: str | Sequence[str] | None = None,
        dims: str | Sequence[str] | None = None,
    ):
        self._tempfile = None
        self._npz_file = None
        self._is_in_memory = False
        self._metadata = {}

        if isinstance(array, (str, Path)):
            self.path = Path(array)
            if not self.path.exists():
                raise FileNotFoundError(f"Numpy file not found: {self.path}")

            # Try loading - could be pure .npy or npz with embedded metadata
            loaded = np.load(self.path, mmap_mode="r", allow_pickle=True)

            if isinstance(loaded, np.lib.npyio.NpzFile):
                # NPZ format with embedded data and metadata
                source = loaded["data"]
                if "metadata" in loaded.files:
                    meta_arr = loaded["metadata"]
                    if meta_arr.ndim == 0:
                        self._metadata = meta_arr.item()
                self._npz_file = loaded  # Keep reference to prevent closing
            else:
                # Pure .npy file
                source = loaded

        elif isinstance(array, np.ndarray):
            source = array
            self.path = None
            self._is_in_memory = True
            logger.debug(f"Wrapping in-memory array with shape {array.shape}")
        else:
            raise TypeError(f"Expected np.ndarray or path, got {type(array)}")

        # Override with explicit metadata if provided
        if metadata is not None:
            self._metadata = metadata

        # `dims` (preferred) / `dim_order` (alias) declare the axes of the
        # wrapped array; metadata["dims"] or metadata["dimension_names"] are
        # honored when neither is passed. Canonicalized to 5D TCZYX. An
        # unusable declaration warns and chain-guesses by rank (never raises).
        declared = dims if dims is not None else dim_order
        if declared is None and isinstance(self._metadata, dict):
            declared = self._metadata.get("dims") or self._metadata.get(
                "dimension_names"
            )

        self._source = source
        self._dims_inferred = declared is None
        self._target_dtype = None
        self._apply_dim_order(declared)
        self._dtype = self.data.dtype

    PRIORITY = 50

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        p = Path(file)
        if p.suffix.lower() not in (".npy", ".npz"):
            return False
        if (p.parent / "pmd_demixer.npy").is_file():
            return False  # PMD demixer arrays are not supported (legacy raises)
        return p.is_file()

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

    def _apply_dim_order(self, declared: str | Sequence[str] | None) -> None:
        """Canonicalize ``self._source`` to 5D TCZYX given a declared axis order.

        ``declared`` describes the source axes (length == source.ndim). When
        None it is inferred from the source rank (3D=TYX, 4D=TZYX, 5D=TCZYX).
        Recomputes ``self.data``/``self._raw_shape`` and invalidates cached
        dimension specs so the reactive metadata recomputes.
        """
        source = self._source
        inferred = declared is None
        if declared is not None:
            declared = _normalize_declared(declared)
        else:
            declared = DEFAULT_DIMS.get(source.ndim)

        data = source
        if declared is not None:
            try:
                data = _canonicalize_to_5d(source, declared)
            except ValueError as e:
                # never raise: warn and chain-guess by rank (YX/TYX/TZYX/TCZYX)
                fallback = DEFAULT_DIMS.get(source.ndim)
                log.get().warning(
                    "dims %r unusable for shape %r (%s); inferring %s",
                    tuple(declared), source.shape, e,
                    "".join(fallback) if fallback else "?",
                )
                declared = fallback
                data = _canonicalize_to_5d(source, declared) if declared else source

        self.data = data
        self._declared_dims = tuple(declared) if declared else None
        self._raw_shape = data.shape
        self.invalidate_dimension_specs()

        # tell the user what was set, right when it happens
        suffix = "  (pass dims= to override)" if inferred and source.ndim >= 3 else ""
        log.get().info(
            "dims %s -> %s  shape %s%s",
            "".join(self._declared_dims) if self._declared_dims else "?",
            "".join(DIMS), self._shape5d(), suffix,
        )

    def __getitem__(self, item):
        # 5D TCZYX indexing translated onto the underlying array's stored
        # rank (_shape5d front-pads singleton T/C/Z).
        out = _index_5d_into_raw(self.data, item, len(self._raw_shape))
        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def __len__(self) -> int:
        return self.nt

    def __array__(self, dtype=None, copy=None):
        # return a single 2D plane for fast preview
        data = self.data
        while getattr(data, "ndim", 0) > 2:
            data = data[0]
        if dtype is not None:
            data = np.asarray(data).astype(dtype)
        return np.asarray(data)

    def __repr__(self) -> str:
        mem_str = " (in-memory)" if self._is_in_memory else ""
        return (
            f"NumpyArray(shape={self.shape}, dtype={self.dtype}, "
            f"dims='{''.join(self.dims)}'{mem_str})"
        )

    @property
    def dims(self) -> tuple[str, ...]:
        """Canonical 5D dimension labels (always TCZYX, matching .shape)."""
        return DIMS

    @dims.setter
    def dims(self, value):
        """Declare the axis order of the wrapped array.

        ``value`` describes the source array's axes (length == source.ndim,
        chars from TCZYX). The array is re-canonicalized to 5D TCZYX, so
        reading ``.dims`` back returns the canonical order while ``.shape``
        places each source axis on T/C/Z accordingly. Reactive: downstream
        metadata (dimension_specs, OME axes) recomputes. An unusable value
        warns and chain-guesses by rank (never raises).
        """
        declared = _normalize_declared(value)
        if declared == self.dims:
            return  # canonical 5D echo (e.g. metadata round-trip); nothing to remap
        self._apply_dim_order(declared)
        self._dims_inferred = False

    @property
    def input_dims(self) -> tuple[str, ...] | None:
        """Declared axis order of the wrapped array (None if unknown)."""
        return self._declared_dims

    @property
    def num_planes(self) -> int:
        """Number of Z-planes (index 2 in 5D TCZYX)."""
        return self.shape[2]

    # _compute_frame_vminmax / vmin / vmax inherited from ReductionMixin

    @property
    def filenames(self) -> list[Path]:
        """Return list of source files (empty for in-memory arrays)."""
        if self.path is not None:
            return [self.path]
        return []

    @property
    def metadata(self) -> dict:
        """Return metadata as dict. Always returns dict, never None.

        Shape-derived keys (dims, Ly, Lx, num_timepoints) are computed from
        the current canonical layout, so they stay reactive after .dims is
        changed. User-set keys take precedence over the derived defaults.
        """
        md = dict(self._metadata) if self._metadata is not None else {}
        md["dims"] = self.dims
        md["dimension_names"] = [d.lower() for d in self.dims]  # NGFF alias
        md["num_timepoints"] = md.get("num_timepoints", self.shape[0])
        md.setdefault("nframes", md["num_timepoints"])  # suite2p alias
        md.setdefault("num_frames", md["num_timepoints"])  # legacy alias
        md["Ly"] = md.get("Ly", self.shape[-2])
        md["Lx"] = md.get("Lx", self.shape[-1])
        return md

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        value = dict(value)
        # `dims` / `dimension_names` are owned by the dim-order machinery, not
        # stored loose in the dict, so the labels and layout never diverge.
        declared = value.pop("dims", None) or value.pop("dimension_names", None)
        self._metadata = value
        if declared is not None:
            self.dims = declared


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


register_array_class(NumpyArray)
