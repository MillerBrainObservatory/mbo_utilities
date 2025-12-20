"""
parameter access and normalization utilities.

provides functions to get/set metadata parameters using canonical names
and their aliases, with type conversion and defaults.
"""
from __future__ import annotations

from typing import Any

from .base import METADATA_PARAMS, ALIAS_MAP, VoxelSize


def get_param(
    metadata: dict | None,
    name: str,
    default: Any = None,
    *,
    override: Any = None,
    shape: tuple | None = None,
) -> Any:
    """
    Get a metadata parameter, checking all known aliases.

    This provides a unified way to access metadata values without needing to
    know which alias was used to store it. The function checks the canonical
    name first, then all registered aliases.

    Parameters
    ----------
    metadata : dict or None
        Metadata dictionary to search.
    name : str
        Canonical parameter name (e.g., "dx", "fs", "num_planes").
        Case-insensitive; will be resolved to canonical form.
    default : Any, optional
        Override default value. If None, uses the parameter's registered default.
    override : Any, optional
        If provided, returns this value directly (for user-specified overrides).
    shape : tuple, optional
        Array shape for fallback dimension extraction (Lx, Ly from shape[-1], shape[-2]).

    Returns
    -------
    Any
        Parameter value converted to the correct dtype, or default if not found.

    Examples
    --------
    >>> meta = {"umPerPixX": 0.5, "frame_rate": 30.0}
    >>> get_param(meta, "dx")
    0.5
    >>> get_param(meta, "fs")
    30.0
    >>> get_param(meta, "num_planes")  # uses default
    1
    >>> get_param(meta, "dx", override=0.3)  # override wins
    0.3
    """
    # if override provided, use it directly
    if override is not None:
        return override

    # resolve canonical name (case-insensitive lookup)
    canonical = ALIAS_MAP.get(name.lower())
    if canonical is None:
        # not a registered parameter - just do simple dict lookup
        if metadata is not None and name in metadata:
            return metadata[name]
        return default

    param = METADATA_PARAMS[canonical]

    # determine final default
    final_default = default if default is not None else param.default

    if metadata is None:
        # try shape fallback for dimensions
        if shape is not None:
            if canonical == "Lx" and len(shape) >= 1:
                return int(shape[-1])
            if canonical == "Ly" and len(shape) >= 2:
                return int(shape[-2])
        return final_default

    # check canonical name first, then all aliases
    keys_to_check = (param.canonical,) + param.aliases
    for key in keys_to_check:
        val = metadata.get(key)
        if val is not None:
            try:
                if param.dtype == tuple:
                    # handle tuple specially
                    if isinstance(val, (list, tuple)):
                        return tuple(val)
                    return val
                return param.dtype(val)
            except (TypeError, ValueError):
                continue

    # special handling for pixel_resolution tuple -> dx/dy
    if canonical in ("dx", "dy"):
        pixel_res = metadata.get("pixel_resolution")
        if pixel_res is not None:
            if isinstance(pixel_res, (list, tuple)) and len(pixel_res) >= 2:
                try:
                    idx = 0 if canonical == "dx" else 1
                    return float(pixel_res[idx])
                except (TypeError, ValueError, IndexError):
                    pass
            elif isinstance(pixel_res, (int, float)):
                return float(pixel_res)

    # fallback: extract Lx/Ly from shape
    if shape is not None:
        if canonical == "Lx" and len(shape) >= 1:
            return int(shape[-1])
        if canonical == "Ly" and len(shape) >= 2:
            return int(shape[-2])

    # special: try to get dtype from shape metadata
    if canonical == "dtype":
        arr_dtype = metadata.get("dtype")
        if arr_dtype is not None:
            return str(arr_dtype)

    return final_default


def get_voxel_size(
    metadata: dict | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
) -> VoxelSize:
    """
    Extract voxel size from metadata with optional user overrides.

    Resolution values are resolved in priority order:
    1. User-provided parameter (highest priority)
    2. Canonical keys (dx, dy, dz)
    3. pixel_resolution tuple
    4. Legacy keys (umPerPixX, umPerPixY, umPerPixZ)
    5. OME keys (PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)
    6. ScanImage SI keys
    7. Default: 1.0 micrometers

    Parameters
    ----------
    metadata : dict, optional
        Metadata dictionary to extract resolution from.
    dx : float, optional
        Override X resolution (micrometers per pixel).
    dy : float, optional
        Override Y resolution (micrometers per pixel).
    dz : float, optional
        Override Z resolution (micrometers per z-step).

    Returns
    -------
    VoxelSize
        Named tuple with (dx, dy, dz) in micrometers.

    Examples
    --------
    >>> meta = {"pixel_resolution": (0.5, 0.5)}
    >>> vs = get_voxel_size(meta, dz=5.0)
    >>> vs.dz
    5.0

    >>> vs = get_voxel_size({"dx": 0.3, "dy": 0.3, "dz": 2.0})
    >>> vs.pixel_resolution
    (0.3, 0.3)
    """
    if metadata is None:
        metadata = {}

    # helper to get first non-None value from a list of keys
    def _get_first(keys: list[str], default: float = 1.0) -> float:
        for key in keys:
            val = metadata.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return default

    # extract pixel_resolution tuple if present
    pixel_res = metadata.get("pixel_resolution")
    px_x, px_y = None, None
    if pixel_res is not None:
        if isinstance(pixel_res, (list, tuple)) and len(pixel_res) >= 2:
            try:
                px_x = float(pixel_res[0])
                px_y = float(pixel_res[1])
            except (TypeError, ValueError):
                pass
        elif isinstance(pixel_res, (int, float)):
            # single value: use for both X and Y
            px_x = px_y = float(pixel_res)

    # try to extract dz from ScanImage nested structure
    si_dz = None
    si = metadata.get("si", {})
    if isinstance(si, dict):
        # try hStackManager.actualStackZStepSize
        h_stack = si.get("hStackManager", {})
        if isinstance(h_stack, dict):
            si_dz = h_stack.get("actualStackZStepSize")
            if si_dz is None:
                si_dz = h_stack.get("stackZStepSize")

    # resolve dx
    resolved_dx = dx
    if resolved_dx is None:
        resolved_dx = _get_first(["dx", "umPerPixX", "PhysicalSizeX"], default=None)
    if resolved_dx is None and px_x is not None:
        resolved_dx = px_x
    if resolved_dx is None:
        resolved_dx = 1.0

    # resolve dy
    resolved_dy = dy
    if resolved_dy is None:
        resolved_dy = _get_first(["dy", "umPerPixY", "PhysicalSizeY"], default=None)
    if resolved_dy is None and px_y is not None:
        resolved_dy = px_y
    if resolved_dy is None:
        resolved_dy = 1.0

    # resolve dz (more aliases for z-step)
    resolved_dz = dz
    if resolved_dz is None:
        resolved_dz = _get_first(
            ["dz", "z_step", "umPerPixZ", "PhysicalSizeZ", "spacing"],
            default=None,
        )
    if resolved_dz is None and si_dz is not None:
        try:
            resolved_dz = float(si_dz)
        except (TypeError, ValueError):
            pass
    if resolved_dz is None:
        resolved_dz = 1.0

    return VoxelSize(dx=resolved_dx, dy=resolved_dy, dz=resolved_dz)


def normalize_resolution(
    metadata: dict,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
) -> dict:
    """
    Normalize resolution metadata by adding all standard aliases.

    This function ensures that resolution information is available under
    all commonly-used keys for different tools and formats:

    - Canonical: dx, dy, dz
    - Legacy: pixel_resolution (tuple), z_step, umPerPixX/Y/Z
    - OME: PhysicalSizeX/Y/Z with units
    - Convenience: voxel_size (3-tuple)

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to normalize. Modified in-place AND returned.
    dx : float, optional
        Override X resolution (micrometers per pixel).
    dy : float, optional
        Override Y resolution (micrometers per pixel).
    dz : float, optional
        Override Z resolution (micrometers per z-step).

    Returns
    -------
    dict
        The same metadata dict with resolution aliases added.

    Examples
    --------
    >>> meta = {"pixel_resolution": (0.5, 0.5)}
    >>> normalize_resolution(meta, dz=5.0)
    >>> meta["dz"]
    5.0
    >>> meta["voxel_size"]
    (0.5, 0.5, 5.0)
    >>> meta["PhysicalSizeZ"]
    5.0
    """
    vs = get_voxel_size(metadata, dx=dx, dy=dy, dz=dz)
    metadata.update(vs.to_dict(include_aliases=True))
    return metadata


def normalize_metadata(
    metadata: dict,
    shape: tuple | None = None,
    **overrides,
) -> dict:
    """
    Normalize metadata by adding all standard parameter aliases.

    This ensures that metadata values are accessible under all commonly-used
    keys for different tools and formats. Modifies the dictionary in-place.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to normalize. Modified in-place AND returned.
    shape : tuple, optional
        Array shape for inferring Lx, Ly if not present in metadata.
    **overrides
        Override values for specific parameters (e.g., dx=0.5, fs=30.0).

    Returns
    -------
    dict
        The same metadata dict with all standard aliases added.

    Examples
    --------
    >>> meta = {"umPerPixX": 0.5, "frame_rate": 30.0}
    >>> normalize_metadata(meta)
    >>> meta["dx"]
    0.5
    >>> meta["fs"]
    30.0
    >>> meta["PhysicalSizeX"]
    0.5
    """
    # handle VoxelSize (existing comprehensive resolution handling)
    vs = get_voxel_size(
        metadata,
        dx=overrides.get("dx"),
        dy=overrides.get("dy"),
        dz=overrides.get("dz"),
    )
    metadata.update(vs.to_dict(include_aliases=True))

    # normalize other parameters
    for name, param in METADATA_PARAMS.items():
        if name in ("dx", "dy", "dz"):
            continue  # already handled by VoxelSize

        value = get_param(
            metadata, name, override=overrides.get(name), shape=shape
        )
        if value is not None:
            # set canonical key
            metadata[name] = value
            # set all aliases
            for alias in param.aliases:
                metadata[alias] = value

    return metadata
