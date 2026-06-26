"""
Pyramid generation for ome-zarr multiscale images.

Provides functions for generating resolution pyramids compatible with
ome-ngff v0.5 specification and napari-ome-zarr plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DownsampleMethod = Literal[
    "mean", "nearest", "gaussian", "local_mean", "median", "mode"
]


@dataclass
class PyramidLevel:
    """single resolution level in a pyramid."""

    level: int  # 0 = full resolution
    shape: tuple[int, ...]
    scale: tuple[float, ...]  # physical scale per axis
    path: str  # zarr path (e.g., "0", "1", "2")


@dataclass
class PyramidConfig:
    """
    Configuration for pyramid generation.

    Parameters
    ----------
    max_layers : int
        maximum number of additional resolution levels beyond level 0.
        default 4 means levels 0, 1, 2, 3, 4 (5 total).
    scale_factors : tuple[int, ...]
        per-axis downsampling factors. default (1, 1, 2, 2) for TZYX
        means T and Z unchanged, Y and X downsampled by 2 per level.
    method : DownsampleMethod
        downsampling method: "mean" (default), "nearest", "gaussian", "local_mean"
    min_size : int
        stop adding levels when any spatial dimension falls below this.
        default 64 pixels.
    """

    max_layers: int = 4
    scale_factors: tuple[int, ...] = (1, 1, 2, 2)  # TZYX: only downsample Y, X
    method: DownsampleMethod = "mean"
    min_size: int = 64

    def get_scale_factors_for_ndim(self, ndim: int) -> tuple[int, ...]:
        """Get scale factors padded/trimmed for array ndim."""
        if ndim == len(self.scale_factors):
            return self.scale_factors
        if ndim == 3:
            return (1, 2, 2)
        if ndim == 4:
            return self.scale_factors
        if ndim == 5:
            return (1, 1, 1, 2, 2)
        # fallback: only downsample last 2 dims
        return (1,) * (ndim - 2) + (2, 2)


def compute_pyramid_shapes(
    base_shape: tuple[int, ...],
    config: PyramidConfig | None = None,
) -> list[PyramidLevel]:
    """
    Compute shapes for all pyramid levels without generating data.

    Parameters
    ----------
    base_shape : tuple
        shape of the full-resolution array (e.g., TZYX).
    config : PyramidConfig, optional
        pyramid configuration. uses defaults if not provided.

    Returns
    -------
    list[PyramidLevel]
        list of pyramid levels from level 0 (full res) to lowest resolution.
    """
    if config is None:
        config = PyramidConfig()

    ndim = len(base_shape)
    scale_factors = config.get_scale_factors_for_ndim(ndim)

    levels = []
    current_shape = base_shape
    base_scale = (1.0,) * ndim

    for level in range(config.max_layers + 1):
        # compute cumulative scale (physical size per pixel at this level)
        cumulative_scale = tuple(
            base_scale[i] * (scale_factors[i] ** level) for i in range(ndim)
        )

        levels.append(
            PyramidLevel(
                level=level,
                shape=current_shape,
                scale=cumulative_scale,
                path=str(level),
            )
        )

        # compute next level shape
        next_shape = tuple(
            max(1, s // scale_factors[i]) for i, s in enumerate(current_shape)
        )

        # stop if any spatial dimension falls below min_size
        # spatial dims are last 2 (Y, X)
        if any(next_shape[i] < config.min_size for i in range(-2, 0)):
            break

        current_shape = next_shape

    return levels


def downsample_block(
    data: np.ndarray,
    factors: tuple[int, ...],
    method: DownsampleMethod = "mean",
) -> np.ndarray:
    """
    Downsample a data block by given factors per axis.

    Parameters
    ----------
    data : np.ndarray
        input data block.
    factors : tuple[int, ...]
        downsampling factor per axis. must match data.ndim.
    method : DownsampleMethod
        "mean" - average pooling (best for intensity data)
        "median" - windowed median (webknossos default for intensity)
        "mode" - windowed mode (webknossos default for labels/masks)
        "nearest" - nearest neighbor
        "gaussian" - gaussian blur then subsample
        "local_mean" - local mean with antialiasing

    Returns
    -------
    np.ndarray
        downsampled data.
    """
    if len(factors) != data.ndim:
        raise ValueError(f"factors {factors} must match data.ndim {data.ndim}")

    # skip if all factors are 1
    if all(f == 1 for f in factors):
        return data

    if method == "nearest":
        # simple slicing - fastest
        slices = tuple(slice(None, None, f) for f in factors)
        return data[slices].copy()

    if method == "mean":
        return _downsample_mean(data, factors)

    if method == "median":
        return _downsample_median(data, factors)

    if method == "mode":
        return _downsample_mode(data, factors)

    if method == "gaussian":
        return _downsample_gaussian(data, factors)

    if method == "local_mean":
        return _downsample_local_mean(data, factors)

    raise ValueError(f"unknown downsampling method: {method}")


def _downsample_mean(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample using mean pooling (reshape + mean approach)."""
    # compute output shape
    out_shape = tuple(s // f for s, f in zip(data.shape, factors, strict=True))

    # reshape to group pixels for averaging
    new_shape = []
    for s, f in zip(out_shape, factors, strict=True):
        new_shape.extend([s, f])

    # axes to average over (odd indices)
    axes_to_mean = tuple(range(1, 2 * len(factors), 2))

    # trim data to be evenly divisible
    slices = tuple(slice(None, s * f) for s, f in zip(out_shape, factors, strict=True))
    trimmed = data[slices]

    # reshape and mean
    reshaped = trimmed.reshape(new_shape)
    return reshaped.mean(axis=axes_to_mean).astype(data.dtype)


def _downsample_median(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample using the windowed median (matches webknossos MEDIAN).

    Same block grouping as ``_downsample_mean``; ``np.median`` reduces over
    the per-axis factor axes. Median is an order statistic, so the grouping
    order within a block is irrelevant. Cast back to the source dtype, which
    truncates toward zero on even-count blocks exactly as webknossos does.
    """
    out_shape = tuple(s // f for s, f in zip(data.shape, factors, strict=True))
    new_shape = []
    for s, f in zip(out_shape, factors, strict=True):
        new_shape.extend([s, f])
    axes_to_reduce = tuple(range(1, 2 * len(factors), 2))
    slices = tuple(slice(None, s * f) for s, f in zip(out_shape, factors, strict=True))
    reshaped = data[slices].reshape(new_shape)
    return np.median(reshaped, axis=axes_to_reduce).astype(data.dtype)


def _downsample_mode(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample using the windowed mode (matches webknossos MODE).

    Best for label/segmentation data: each output voxel is the most frequent
    value in its block, so no value that is not present in the source is ever
    produced. Ties go to the value seen first in raster order, identical to
    webknossos' numba kernel.
    """
    out_shape = tuple(s // f for s, f in zip(data.shape, factors, strict=True))
    new_shape = []
    for s, f in zip(out_shape, factors, strict=True):
        new_shape.extend([s, f])
    slices = tuple(slice(None, s * f) for s, f in zip(out_shape, factors, strict=True))
    reshaped = data[slices].reshape(new_shape)

    nd = len(factors)
    data_axes = tuple(range(0, 2 * nd, 2))
    factor_axes = tuple(range(1, 2 * nd, 2))
    window = int(np.prod(factors))
    moved = np.ascontiguousarray(np.transpose(reshaped, data_axes + factor_axes))
    flat = moved.reshape(-1, window)
    result = _mode_rows(flat)
    return result.reshape(out_shape).astype(data.dtype)


def _mode_rows_py(flat: np.ndarray) -> np.ndarray:
    """Row-wise mode over a 2D ``(n_blocks, window)`` array.

    Counts values in first-seen order and returns the value with the highest
    count, ties kept at the first-seen value (numba-compiled on first use).
    """
    m, w = flat.shape
    out = np.empty(m, dtype=flat.dtype)
    values = np.empty(w, dtype=flat.dtype)
    counts = np.empty(w, dtype=np.int64)
    for r in range(m):
        n = 0
        for c in range(w):
            v = flat[r, c]
            hit = False
            for k in range(n):
                if values[k] == v:
                    counts[k] += 1
                    hit = True
                    break
            if not hit:
                values[n] = v
                counts[n] = 1
                n += 1
        best = 0
        for k in range(1, n):
            if counts[k] > counts[best]:
                best = k
        out[r] = values[best]
    return out


_mode_rows_compiled = None


def _mode_rows(flat: np.ndarray) -> np.ndarray:
    """numba-compiled :func:`_mode_rows_py`, imported lazily on first use."""
    global _mode_rows_compiled
    if _mode_rows_compiled is None:
        import numba

        _mode_rows_compiled = numba.njit(cache=True, nogil=True)(_mode_rows_py)
    return _mode_rows_compiled(flat)


def _downsample_gaussian(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample with gaussian blur for antialiasing."""
    from scipy.ndimage import gaussian_filter

    # sigma proportional to downsampling factor
    sigma = tuple(0.5 * (f - 1) for f in factors)
    blurred = gaussian_filter(data.astype(np.float32), sigma=sigma)

    # subsample
    slices = tuple(slice(None, None, f) for f in factors)
    return blurred[slices].astype(data.dtype)


def _downsample_local_mean(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample using skimage local_mean if available, else fall back to mean."""
    try:
        from skimage.transform import downscale_local_mean

        return downscale_local_mean(data, factors).astype(data.dtype)
    except ImportError:
        return _downsample_mean(data, factors)


