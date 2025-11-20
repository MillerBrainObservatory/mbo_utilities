"""
Custom NDImageProcessor subclasses for mbo_utilities arrays.

This module provides processors compatible with fastplotlib's iw-array branch,
supporting MBO-specific array types like MboRawArray and MBOTiffArray.
"""

from typing import Callable, Literal
import numpy as np
from numpy.typing import ArrayLike

try:
    from fastplotlib.widgets.image_widget import NDImageProcessor, WindowFuncCallable
except ImportError:
    raise ImportError(
        "fastplotlib iw-array branch required. "
        "Install with: pip install git+https://github.com/fastplotlib/fastplotlib.git@iw-array"
    )


class MboImageProcessor(NDImageProcessor):
    """
    NDImageProcessor subclass for MBO arrays with phase correction and FFT support.

    This processor extends NDImageProcessor to handle MBO-specific preprocessing:
    - Bidirectional phase correction for scanning artifacts
    - FFT-based motion correction
    - ROI-specific processing

    Parameters
    ----------
    data : ArrayLike
        MBO array (MboRawArray or MBOTiffArray)

    fix_phase : bool, default False
        Apply bidirectional phase correction

    use_fft : bool, default False
        Use FFT-based motion correction

    fft_method : str, default "2d"
        FFT method to use ("2d" or "1d")

    phasecorr_method : str, default "mean"
        Phase correlation method

    **kwargs
        Additional arguments passed to NDImageProcessor
    """

    def __init__(
        self,
        data: ArrayLike | None,
        fix_phase: bool = False,
        use_fft: bool = False,
        fft_method: str = "2d",
        phasecorr_method: str = "mean",
        n_display_dims: Literal[2, 3] = 2,
        rgb: bool = False,
        window_funcs: tuple[WindowFuncCallable | None, ...] | WindowFuncCallable = None,
        window_sizes: tuple[int | None, ...] | int = None,
        window_order: tuple[int, ...] = None,
        spatial_func: Callable[[ArrayLike], ArrayLike] = None,
        compute_histogram: bool = True,
    ):
        # Store MBO-specific parameters
        self.fix_phase = fix_phase
        self.use_fft = use_fft
        self.fft_method = fft_method
        self.phasecorr_method = phasecorr_method

        # Set MBO array processing flags if it's an MboRawArray
        if data is not None and hasattr(data, 'fix_phase'):
            data.fix_phase = fix_phase
            data.use_fft = use_fft
            if hasattr(data, 'fft_method'):
                data.fft_method = fft_method
            if hasattr(data, 'phasecorr_method'):
                data.phasecorr_method = phasecorr_method

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

    def get(self, indices: tuple[int, ...]) -> np.ndarray:
        """
        Get a frame at the specified indices with MBO preprocessing applied.

        Parameters
        ----------
        indices : tuple[int, ...]
            Indices for each slider dimension

        Returns
        -------
        np.ndarray
            Processed frame
        """
        # Use parent class get() which handles window funcs and spatial_func
        frame = super().get(indices)

        # Additional MBO-specific processing could go here if needed
        # (currently handled by the array itself via fix_phase/use_fft flags)

        return frame


class MultiSessionImageProcessor(MboImageProcessor):
    """
    NDImageProcessor for multi-session data with dynamic shapes.

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
        Additional arguments passed to MboImageProcessor
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
