"""
Dimension labels feature for arrays.

Provides a flexible system for labeling array dimensions with sensible
defaults while allowing custom configurations like ZYX, sTZYX, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# common dimension label characters and their meanings
DIM_DESCRIPTIONS = {
    "T": "time",
    "Z": "z-plane/depth",
    "Y": "height",
    "X": "width",
    "C": "channel",
    "S": "session",
    "R": "roi/region",
    "V": "view",
    "B": "batch",
}

# default dimension mappings by ndim (ngff 0.5 compliant: T -> C -> Z -> Y -> X)
DEFAULT_DIMS = {
    2: ("Y", "X"),
    3: ("T", "Y", "X"),
    4: ("T", "Z", "Y", "X"),
    5: ("T", "C", "Z", "Y", "X"),  # ngff: time -> channel -> space
}

# alternative common dimension orderings
KNOWN_ORDERINGS = {
    # 3D alternatives
    "ZYX": ("Z", "Y", "X"),
    "TYX": ("T", "Y", "X"),
    "CYX": ("C", "Y", "X"),
    # 4D alternatives
    "TZYX": ("T", "Z", "Y", "X"),
    "ZTYX": ("Z", "T", "Y", "X"),
    "TCYX": ("T", "C", "Y", "X"),
    "CZYX": ("C", "Z", "Y", "X"),
    # 5D alternatives (ngff compliant: T -> C -> Z -> Y -> X)
    "TCZYX": ("T", "C", "Z", "Y", "X"),
    "TZCYX": ("T", "Z", "C", "Y", "X"),
    "STZYX": ("S", "T", "Z", "Y", "X"),
    "VTZYX": ("V", "T", "Z", "Y", "X"),
}


def parse_dims(
    dims: str | Sequence[str] | None, ndim: int, *, strict: bool = True
) -> tuple[str, ...]:
    """
    Parse dimension labels from various input formats.

    Parameters
    ----------
    dims : str | Sequence[str] | None
        dimension labels as string ("TZYX"), tuple/list, or None for default
    ndim : int
        number of dimensions (used for validation and defaults)

    Returns
    -------
    tuple[str, ...]
        normalized dimension labels

    Examples
    --------
    >>> parse_dims("TZYX", 4)
    ('T', 'Z', 'Y', 'X')
    >>> parse_dims(["T", "Z", "Y", "X"], 4)
    ('T', 'Z', 'Y', 'X')
    >>> parse_dims(None, 4)
    ('T', 'Z', 'Y', 'X')
    >>> parse_dims("ZYX", 3)
    ('Z', 'Y', 'X')
    """
    if dims is None:
        if ndim not in DEFAULT_DIMS:
            raise ValueError(f"no default dims for {ndim}D arrays")
        return DEFAULT_DIMS[ndim]

    if isinstance(dims, str):
        # check known orderings first
        if dims.upper() in KNOWN_ORDERINGS:
            result = KNOWN_ORDERINGS[dims.upper()]
        else:
            # parse character by character
            result = tuple(c.upper() for c in dims if c.isalpha())
    else:
        result = tuple(str(d).upper() for d in dims)

    if len(result) != ndim:
        if strict:
            raise ValueError(
                f"dimension labels {result} have {len(result)} elements, "
                f"but array has {ndim} dimensions"
            )
        fallback = DEFAULT_DIMS.get(ndim, result)
        from mbo_utilities import log

        log.get().warning(
            "dims %s have %d elements but array is %dD; inferring %s",
            result, len(result), ndim, "".join(fallback),
        )
        return fallback

    return result


def get_slider_dims(arr_or_dims) -> tuple[str, ...] | None:
    """
    Get the dimensions that should have sliders in a viewer.

    Convention: Y and X are spatial display dims, everything else gets sliders.

    Parameters
    ----------
    arr_or_dims : array-like or tuple[str, ...]
        array with dims property, or tuple of dimension labels

    Returns
    -------
    tuple[str, ...] | None
        dimensions that need sliders (excludes Y, X), lowercase for fastplotlib.
        returns None if dims cannot be determined.
    """
    # if it's already a tuple of strings, use it directly
    if isinstance(arr_or_dims, tuple) and all(isinstance(d, str) for d in arr_or_dims):
        dims = arr_or_dims
    else:
        # assume it's an array, try to get dims
        # use normalize=False to preserve descriptive names for slider labels
        dims = get_dims(arr_or_dims, normalize=False)

    if dims is None:
        return None

    # filter out spatial dims and singleton dims, convert to lowercase for fastplotlib
    # skip size-1 dims since fastplotlib doesn't handle singleton sliders well
    shape = getattr(arr_or_dims, "shape", None)
    slider_dims = []
    for i, d in enumerate(dims):
        if d in ("Y", "X"):
            continue
        if shape is not None and i < len(shape) and shape[i] <= 1:
            continue
        slider_dims.append(d.lower())
    return tuple(slider_dims) if slider_dims else None


# aliases used when resolving a slider name back to a canonical T/C/Z key.
# Isoview arrays expose descriptive labels like "Tile", "Cam", "Zplane";
# LBM/ScanImage arrays use lowercase "t"/"c"/"z". Consumers that need to
# locate a specific axis should go through `find_slider_name` so both
# label styles resolve to the same slider.
_SLIDER_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "t": ("t", "tile", "timepoint", "timepoints", "tp"),
    "c": ("c", "cam", "view", "channel", "channels", "cm"),
    "z": ("z", "zplane", "zplanes", "plane", "planes", "depth"),
}


def find_slider_name(names, canonical: str) -> str | None:
    """Return the slider name in ``names`` matching a canonical T/C/Z key.

    Matches both single-letter forms (``"t"``, ``"c"``, ``"z"``) and the
    descriptive labels Isoview arrays expose (``"Tile"``, ``"Timepoint"``,
    ``"Cam"``, ``"View"``, ``"Zplane"``) — case-insensitive. Returns the
    original-case name (which is what fastplotlib's ``Indices`` is keyed
    by) or ``None`` when no axis matches.
    """
    if not names:
        return None
    aliases = _SLIDER_NAME_ALIASES.get(canonical.lower(), (canonical.lower(),))
    for n in names:
        if n.lower() in aliases:
            return n
    return None


def get_dim_index(dims: tuple[str, ...], label: str) -> int | None:
    """
    Get the index of a dimension label.

    Parameters
    ----------
    dims : tuple[str, ...]
        dimension labels
    label : str
        dimension to find (case-insensitive)

    Returns
    -------
    int | None
        index of the dimension, or None if not found
    """
    label = label.upper()
    try:
        return dims.index(label)
    except ValueError:
        return None


# convenience functions for use outside feature system


def infer_dims(ndim: int) -> tuple[str, ...]:
    """
    Infer dimension labels from array dimensionality.

    Parameters
    ----------
    ndim : int
        number of dimensions

    Returns
    -------
    tuple[str, ...]
        default dimension labels for that ndim
    """
    if ndim not in DEFAULT_DIMS:
        raise ValueError(f"cannot infer dims for {ndim}D array")
    return DEFAULT_DIMS[ndim]


def get_dims(arr, *, normalize: bool = True) -> tuple[str, ...]:
    """
    Get dimension labels from an array in canonical form.

    Always returns uppercase single-letter labels (T, Z, C, Y, X, etc.)
    regardless of how the array's dims property is defined. Uses the array's
    `dims` property when present, else infers from ndim.

    Parameters
    ----------
    arr : array-like
        array with shape and optionally dims
    normalize : bool, default True
        if True, normalize descriptive names to canonical single-letter form

    Returns
    -------
    tuple[str, ...]
        dimension labels in canonical form (e.g., ("T", "Z", "Y", "X"))

    Examples
    --------
    >>> arr.dims  # LBMArray
    ('timepoints', 'z-planes', 'Y', 'X')
    >>> get_dims(arr)
    ('T', 'Z', 'Y', 'X')
    """
    from mbo_utilities.arrays.features._dim_tags import normalize_dims

    if hasattr(arr, "dims") and arr.dims is not None:
        dims = arr.dims
        if isinstance(dims, str):
            dims = parse_dims(dims, arr.ndim)
        else:
            dims = tuple(dims)
        return normalize_dims(dims) if normalize else dims

    # fallback to inference (already canonical)
    return infer_dims(arr.ndim)


def get_num_planes(arr) -> int:
    """
    Get number of Z-planes from an array.

    Always 5D TCZYX, so Z is at index 2.

    Parameters
    ----------
    arr : array-like
        array with shape (always 5D TCZYX)

    Returns
    -------
    int
        number of z-planes
    """
    return arr.shape[2]
