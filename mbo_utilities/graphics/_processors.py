"""
Custom NDImageProcessor subclasses for mbo_utilities arrays.

This module provides processors compatible with fastplotlib's iw-array branch,
supporting MBO-specific array types like MboRawArray and MBOTiffArray.

Processor hierarchy:
- BaseImageProcessor: Basic processing (mean subtraction)
- RasterScanProcessor: Adds bidirectional scan phase correction for raster scanning microscopes
- MultiSessionImageProcessor: Adds dynamic shape support for multi-session data

Note: Gaussian blur is handled via ImageWidget.spatial_func API, not on the processor.

This follows the fastplotlib pattern where processors own processing logic
and widgets handle UI orchestration.
"""

from typing import Callable, Literal
import numpy as np
from numpy.typing import ArrayLike

# Force rendercanvas to use Qt backend if PySide6 is available
# This must happen BEFORE importing fastplotlib to avoid glfw selection
import os
import importlib.util

if importlib.util.find_spec("PySide6") is not None:
    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
    import PySide6  # noqa: F401 - Must be imported before rendercanvas.qt can load

try:
    from fastplotlib.widgets.image_widget import NDImageProcessor
except ImportError:
    raise ImportError(
        "fastplotlib iw-array branch required. "
        "Install with: pip install git+https://github.com/fastplotlib/fastplotlib.git@iw-array"
    )

from mbo_utilities.phasecorr import apply_scan_phase_offsets, _phase_corr_2d
from mbo_utilities.array_types import MboRawArray

# Type alias for window functions (not exported by fastplotlib)
WindowFuncCallable = Callable[[ArrayLike, int, bool], ArrayLike]


class BaseImageProcessor(NDImageProcessor):
    """
    Base image processor with common preprocessing features.

    This processor extends NDImageProcessor with:
    - Mean subtraction for background removal

    Note: Gaussian blur should be set via ImageWidget.spatial_func API.

    Parameters
    ----------
    data : ArrayLike
        Image array data

    mean_image : np.ndarray, optional
        2d mean image to subtract (for mean-subtracted display)

    """

    def __init__(
        self,
        data: ArrayLike | None,
        mean_image: np.ndarray | None = None,
        n_display_dims: Literal[2, 3] = 2,
        rgb: bool = False,
        window_funcs: tuple[WindowFuncCallable | None, ...] | WindowFuncCallable = None,
        window_sizes: tuple[int | None, ...] | int = None,
        window_order: tuple[int, ...] = None,
        spatial_func: Callable[[ArrayLike], ArrayLike] = None,
        compute_histogram: bool = False,
        **kwargs,
    ):
        self._mean_image = mean_image

        # Initialize parent NDImageProcessor
        super().__init__(
            data=data,
            n_display_dims=n_display_dims,
            rgb=rgb,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
            window_order=window_order,
            spatial_func=spatial_func,
            compute_histogram=compute_histogram,
        )

    @property
    def mean_image(self) -> np.ndarray | None:
        """Mean image for subtraction (set by widget after z-stats computation)."""
        return self._mean_image

    @mean_image.setter
    def mean_image(self, value: np.ndarray | None) -> None:
        self._mean_image = value

    def get(self, indices: tuple[int, ...]) -> np.ndarray:
        """
        Get a frame at the specified indices with preprocessing applied.

        Processing order:
        1. Parent class get() - handles window_funcs and spatial_func (including gaussian blur)
        2. Mean subtraction (if mean_image is set)

        Parameters
        ----------
        indices : tuple[int, ...]
            Indices for each slider dimension

        Returns
        -------
        np.ndarray
            Processed frame (2D for n_display_dims=2)
        """
        # Use parent class get() which handles indexing, window funcs, and spatial_func
        frame = super().get(indices)

        if frame is None:
            return frame

        frame = np.asarray(frame)

        # Apply mean subtraction
        if self._mean_image is not None:
            frame = frame - self._mean_image

        return frame


class RasterScanProcessor(BaseImageProcessor):
    """
    Image processor for raster scanning microscopes with bidirectional scan correction.

    This processor extends BaseImageProcessor with phase correction for
    bidirectional scanning artifacts common in raster scanning microscopes
    (e.g., two-photon, confocal with resonant scanners).

    Parameters
    ----------
    data : ArrayLike
        Image array data (MboRawArray, TiffArray, etc.)

    fix_phase : bool, default False
        Apply bidirectional phase correction

    use_fft : bool, default False
        Use FFT-based subpixel phase correlation (slower, more accurate)

    border : int, default 3
        Border pixels to exclude from phase correlation

    max_offset : int, default 3
        Maximum pixel offset for phase correction

    phase_upsample : int, default 5
        Upsampling factor for subpixel phase correlation

    **kwargs
        Additional arguments passed to BaseImageProcessor
    """

    def __init__(
        self,
        data: ArrayLike | None,
        fix_phase: bool = False,
        use_fft: bool = False,
        border: int = 3,
        max_offset: int = 3,
        phase_upsample: int = 5,
        **kwargs
    ):
        # Raster scan specific parameters
        self._fix_phase = fix_phase
        self._use_fft = use_fft
        self._border = border
        self._max_offset = max_offset
        self._phase_upsample = phase_upsample

        # Cache for computed phase offset (invalidated when params change)
        self._cached_offset: float | None = None
        self._cached_offset_indices: tuple[int, ...] | None = None

        # Check if data is an MboRawArray (handles its own phase correction)
        self._is_mbo_array = isinstance(data, MboRawArray)

        # Sync processing flags with MboRawArray if applicable
        self._sync_array_flags(data)

        super().__init__(data=data, **kwargs)

    def _sync_array_flags(self, data: ArrayLike | None) -> None:
        """Sync processing flags with MboRawArray if applicable."""
        if data is None:
            return

        if hasattr(data, "fix_phase"):
            data.fix_phase = self._fix_phase
        if hasattr(data, "use_fft"):
            data.use_fft = self._use_fft
        if hasattr(data, "border"):
            data.border = self._border
        if hasattr(data, "max_offset"):
            data.max_offset = self._max_offset
        if hasattr(data, "upsample"):
            data.upsample = self._phase_upsample

    def _invalidate_cache(self) -> None:
        """Invalidate cached phase offset."""
        self._cached_offset = None
        self._cached_offset_indices = None

    # -------------------------------------------------------------------------
    # Raster scan properties
    # -------------------------------------------------------------------------

    @property
    def fix_phase(self) -> bool:
        """Whether bidirectional phase correction is enabled."""
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value: bool) -> None:
        if value != self._fix_phase:
            self._fix_phase = value
            self._sync_array_flags(self.data)
            self._invalidate_cache()

    @property
    def use_fft(self) -> bool:
        """Whether FFT-based phase correlation is used."""
        return self._use_fft

    @use_fft.setter
    def use_fft(self, value: bool) -> None:
        if value != self._use_fft:
            self._use_fft = value
            self._sync_array_flags(self.data)
            self._invalidate_cache()

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase correlation."""
        return self._border

    @border.setter
    def border(self, value: int) -> None:
        if value != self._border:
            self._border = value
            self._sync_array_flags(self.data)
            self._invalidate_cache()

    @property
    def max_offset(self) -> int:
        """Maximum pixel offset for phase correction."""
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value: int) -> None:
        if value != self._max_offset:
            self._max_offset = value
            self._sync_array_flags(self.data)
            self._invalidate_cache()

    @property
    def phase_upsample(self) -> int:
        """Upsampling factor for subpixel phase correlation."""
        return self._phase_upsample

    @phase_upsample.setter
    def phase_upsample(self, value: int) -> None:
        if value != self._phase_upsample:
            self._phase_upsample = value
            self._sync_array_flags(self.data)
            self._invalidate_cache()

    @property
    def current_offset(self) -> float:
        """Current cached phase offset (read-only)."""
        return self._cached_offset if self._cached_offset is not None else 0.0

    # -------------------------------------------------------------------------
    # Processing methods
    # -------------------------------------------------------------------------

    def _compute_phase_offset(self, frame: np.ndarray) -> float:
        """
        Compute phase correction offset for a frame.

        Uses 2D phase correlation to find the horizontal shift between
        even and odd rows caused by bidirectional scanning.

        Parameters
        ----------
        frame : np.ndarray
            2D frame to compute offset for

        Returns
        -------
        float
            Horizontal pixel offset
        """
        try:
            offset = _phase_corr_2d(
                frame,
                upsample=self._phase_upsample,
                border=self._border,
                max_offset=self._max_offset,
                use_fft=self._use_fft,
            )
            return float(offset)
        except Exception:
            return 0.0

    def get(self, indices: tuple[int, ...]) -> np.ndarray:
        """
        Get a frame at the specified indices with raster scan preprocessing.

        Processing order:
        1. Parent class get() - handles window_funcs, spatial_func, gaussian, mean_sub
        2. Phase correction (if enabled and not MboRawArray)

        Parameters
        ----------
        indices : tuple[int, ...]
            Indices for each slider dimension

        Returns
        -------
        np.ndarray
            Processed frame (2D for n_display_dims=2)
        """
        # Use parent class get() which handles base processing
        frame = super().get(indices)

        if frame is None:
            return frame

        # Apply phase correction for non-MboRawArray data
        # MboRawArray handles phase correction in __getitem__
        if self._fix_phase and not self._is_mbo_array:
            # Check if we need to recompute offset
            if (
                self._cached_offset is None
                or self._cached_offset_indices != indices
            ):
                self._cached_offset = self._compute_phase_offset(frame)
                self._cached_offset_indices = indices

            if self._cached_offset != 0.0:
                frame = apply_scan_phase_offsets(frame, self._cached_offset)

        return frame


# Alias for backwards compatibility
MboImageProcessor = RasterScanProcessor


class MultiSessionImageProcessor(RasterScanProcessor):
    """
    Image processor for multi-session data with dynamic shapes.

    This processor supports arrays where different sessions may have different
    numbers of timepoints or z-planes, implementing the dynamic shape proposal
    from the iw-array PR.

    Shape format: [session, time, z-plane, y, x]

    Each session can have unique (time, z-plane) dimensions.

    Parameters
    ----------
    data : ArrayLike
        Multi-session array data

    session_shapes : list[tuple[int, ...]]
        List of shapes for each session, e.g. [(100, 5), (150, 3), ...]
        where each tuple is (n_timepoints, n_zplanes)

    **kwargs
        Additional arguments passed to RasterScanProcessor
    """

    def __init__(
        self,
        data: ArrayLike | None,
        session_shapes: list[tuple[int, ...]] | None = None,
        **kwargs
    ):
        self.session_shapes = session_shapes or []
        self._current_session_idx = 0
        self._last_indices = None

        super().__init__(data=data, **kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return shape based on current session index.

        If a get() has been called, uses the session from those indices.
        Otherwise returns the shape of session 0.
        """
        if self._last_indices is not None and len(self._last_indices) > 0:
            session_idx = self._last_indices[0]
        else:
            session_idx = self._current_session_idx

        if session_idx < len(self.session_shapes):
            # Get session-specific shape
            session_t, session_z = self.session_shapes[session_idx]
            # Assuming spatial dims are same across sessions
            base_shape = self.data.shape
            return (base_shape[0], session_t, session_z, *base_shape[-2:])

        # Fallback to data shape
        return super().shape

    def get(self, indices: tuple[int, ...]) -> np.ndarray:
        """
        Get frame for multi-session data, tracking session index for dynamic shapes.

        Parameters
        ----------
        indices : tuple[int, ...]
            Indices where first index is session

        Returns
        -------
        np.ndarray
            Processed frame from specified session
        """
        # Store last indices to determine current session
        self._last_indices = indices

        if len(indices) > 0:
            self._current_session_idx = indices[0]

        # Use parent class get()
        return super().get(indices)
