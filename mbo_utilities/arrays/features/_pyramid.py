"""
pyramid generation for ome-zarr multiscale images.

provides functions for generating resolution pyramids compatible with
ome-ngff v0.5 specification and napari-ome-zarr plugin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


DownsampleMethod = Literal["mean", "nearest", "gaussian", "local_mean"]


@dataclass
class PyramidLevel:
    """single resolution level in a pyramid."""

    level: int  # 0 = full resolution
    shape: tuple[int, ...]
    scale: tuple[float, ...]  # physical scale per axis
    path: str  # zarr path (e.g., "0", "1", "2")

    @property
    def scale_factor(self) -> int:
        """downsampling factor relative to level 0."""
        return 2**self.level


@dataclass
class PyramidConfig:
    """
    configuration for pyramid generation.

    parameters
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
        """get scale factors padded/trimmed for array ndim."""
        if ndim == len(self.scale_factors):
            return self.scale_factors
        if ndim == 3:
            # TYX: (1, 2, 2)
            return (1, 2, 2)
        if ndim == 4:
            return self.scale_factors
        if ndim == 5:
            # TCZYX: (1, 1, 1, 2, 2)
            return (1, 1, 1, 2, 2)
        # fallback: only downsample last 2 dims
        return (1,) * (ndim - 2) + (2, 2)


def compute_pyramid_shapes(
    base_shape: tuple[int, ...],
    config: PyramidConfig | None = None,
) -> list[PyramidLevel]:
    """
    compute shapes for all pyramid levels without generating data.

    parameters
    ----------
    base_shape : tuple
        shape of the full-resolution array (e.g., TZYX).
    config : PyramidConfig, optional
        pyramid configuration. uses defaults if not provided.

    returns
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
    downsample a data block by given factors per axis.

    parameters
    ----------
    data : np.ndarray
        input data block.
    factors : tuple[int, ...]
        downsampling factor per axis. must match data.ndim.
    method : DownsampleMethod
        "mean" - average pooling (default, best for intensity data)
        "nearest" - nearest neighbor (best for labels/masks)
        "gaussian" - gaussian blur then subsample
        "local_mean" - local mean with antialiasing

    returns
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

    if method == "gaussian":
        return _downsample_gaussian(data, factors)

    if method == "local_mean":
        return _downsample_local_mean(data, factors)

    raise ValueError(f"unknown downsampling method: {method}")


def _downsample_mean(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """downsample using mean pooling (reshape + mean approach)."""
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


def _downsample_gaussian(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """downsample with gaussian blur for antialiasing."""
    from scipy.ndimage import gaussian_filter

    # sigma proportional to downsampling factor
    sigma = tuple(0.5 * (f - 1) for f in factors)
    blurred = gaussian_filter(data.astype(np.float32), sigma=sigma)

    # subsample
    slices = tuple(slice(None, None, f) for f in factors)
    return blurred[slices].astype(data.dtype)


def _downsample_local_mean(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """downsample using skimage local_mean if available, else fall back to mean."""
    try:
        from skimage.transform import downscale_local_mean

        return downscale_local_mean(data, factors).astype(data.dtype)
    except ImportError:
        return _downsample_mean(data, factors)


def generate_pyramid(
    data: np.ndarray,
    config: PyramidConfig | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    generate pyramid levels from full-resolution data.

    yields (level_index, downsampled_data) tuples.
    level 0 is the original data.

    parameters
    ----------
    data : np.ndarray
        full-resolution data.
    config : PyramidConfig, optional
        pyramid configuration.

    yields
    ------
    tuple[int, np.ndarray]
        (level_index, data_at_level)
    """
    if config is None:
        config = PyramidConfig()

    levels = compute_pyramid_shapes(data.shape, config)
    scale_factors = config.get_scale_factors_for_ndim(data.ndim)

    current = data
    for level_info in levels:
        if level_info.level == 0:
            yield 0, data
        else:
            current = downsample_block(current, scale_factors, config.method)
            yield level_info.level, current


def build_multiscales_metadata(
    levels: list[PyramidLevel],
    base_scale: tuple[float, ...],
    axes: list[dict],
    name: str = "",
    downsample_type: str = "mean",
) -> dict:
    """
    build ome-ngff v0.5 multiscales metadata for a pyramid.

    parameters
    ----------
    levels : list[PyramidLevel]
        pyramid levels from compute_pyramid_shapes().
    base_scale : tuple[float, ...]
        physical scale at level 0 (e.g., (1/fs, dz, dy, dx) in seconds/micrometers).
    axes : list[dict]
        ome-ngff axes specification (e.g., [{"name": "t", "type": "time", ...}]).
    name : str, optional
        name for the multiscale image.
    downsample_type : str
        downsampling method used (for metadata).

    returns
    -------
    dict
        ome-ngff v0.5 "multiscales" metadata ready for zarr attrs.
    """
    datasets = []
    for level in levels:
        # compute physical scale for this level
        physical_scale = [
            base_scale[i] * level.scale[i] for i in range(len(base_scale))
        ]

        datasets.append(
            {
                "path": level.path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": physical_scale}
                ],
            }
        )

    multiscale = {
        "version": "0.5",
        "axes": axes,
        "datasets": datasets,
        "type": downsample_type,
    }

    if name:
        multiscale["name"] = name

    return [multiscale]


def build_napari_scale_attrs(
    levels: list[PyramidLevel],
    base_scale: tuple[float, ...],
) -> list[list[float]]:
    """
    build napari-compatible scale attributes for each pyramid level.

    napari reads the 'scale' attr from each array for proper display.

    parameters
    ----------
    levels : list[PyramidLevel]
        pyramid levels.
    base_scale : tuple[float, ...]
        physical scale at level 0.

    returns
    -------
    list[list[float]]
        scale arrays for each level, ordered by level index.
    """
    scales = []
    for level in levels:
        physical_scale = [
            base_scale[i] * level.scale[i] for i in range(len(base_scale))
        ]
        scales.append(physical_scale)
    return scales
