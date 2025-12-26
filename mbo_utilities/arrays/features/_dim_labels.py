"""
Dimension labels feature for arrays.

Provides a flexible system for labeling array dimensions with sensible
defaults while allowing custom configurations like ZYX, sTZYX, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent

if TYPE_CHECKING:
    from typing import Sequence

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

# default dimension mappings by ndim
# these are the "slider dimensions" - dims that get sliders in viewers
DEFAULT_DIMS = {
    2: ("Y", "X"),
    3: ("T", "Y", "X"),
    4: ("T", "Z", "Y", "X"),
    5: ("T", "Z", "C", "Y", "X"),
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
    # 5D alternatives
    "TZCYX": ("T", "Z", "C", "Y", "X"),
    "TCZVX": ("T", "C", "Z", "Y", "X"),
    "sTZYX": ("S", "T", "Z", "Y", "X"),
    "VTZYX": ("V", "T", "Z", "Y", "X"),
}


def parse_dims(dims: str | Sequence[str] | None, ndim: int) -> tuple[str, ...]:
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
        raise ValueError(
            f"dimension labels {result} have {len(result)} elements, "
            f"but array has {ndim} dimensions"
        )

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
        dims = get_dims(arr_or_dims)

    if dims is None:
        return None

    # filter out spatial dims and convert to lowercase for fastplotlib
    slider_dims = tuple(d.lower() for d in dims if d not in ("Y", "X"))
    return slider_dims if slider_dims else None


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


class DimLabels(ArrayFeature):
    """
    Dimension labels feature for arrays.

    Manages dimension labels with sensible defaults and validation.
    Emits events when labels change.

    Parameters
    ----------
    dims : str | Sequence[str] | None
        initial dimension labels. if None, inferred from ndim.
    ndim : int
        number of dimensions (required for validation)

    Attributes
    ----------
    value : tuple[str, ...]
        current dimension labels
    slider_dims : tuple[str, ...]
        dimensions that should have sliders (non-spatial)
    spatial_dims : tuple[str, ...]
        spatial dimensions (Y, X)

    Examples
    --------
    >>> labels = DimLabels(None, ndim=4)
    >>> labels.value
    ('T', 'Z', 'Y', 'X')

    >>> labels = DimLabels("ZYX", ndim=3)
    >>> labels.value
    ('Z', 'Y', 'X')

    >>> labels = DimLabels("sTZYX", ndim=5)
    >>> labels.value
    ('S', 'T', 'Z', 'Y', 'X')
    """

    def __init__(
        self,
        dims: str | Sequence[str] | None = None,
        ndim: int = 0,
        property_name: str = "dim_labels",
    ):
        super().__init__(property_name=property_name)
        self._ndim = ndim
        self._dims = parse_dims(dims, ndim) if ndim > 0 else ()

    @property
    def value(self) -> tuple[str, ...]:
        """current dimension labels"""
        return self._dims

    @property
    def slider_dims(self) -> tuple[str, ...]:
        """dimensions that should have sliders (non-spatial)"""
        return get_slider_dims(self._dims)

    @property
    def spatial_dims(self) -> tuple[str, ...]:
        """spatial dimensions (Y, X)"""
        return tuple(d for d in self._dims if d in ("Y", "X"))

    @property
    def ndim(self) -> int:
        """number of dimensions"""
        return self._ndim

    def set_value(self, array, value: str | Sequence[str] | None) -> None:
        """
        Set dimension labels.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to (for ndim validation)
        value : str | Sequence[str] | None
            new dimension labels
        """
        ndim = getattr(array, "ndim", self._ndim)
        old_dims = self._dims
        self._dims = parse_dims(value, ndim)
        self._ndim = ndim

        if old_dims != self._dims:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"value": self._dims, "old_value": old_dims},
            )
            self._call_event_handlers(event)

    def index(self, label: str) -> int | None:
        """
        Get index of a dimension label.

        Parameters
        ----------
        label : str
            dimension to find (case-insensitive)

        Returns
        -------
        int | None
            index or None if not found
        """
        return get_dim_index(self._dims, label)

    def has(self, label: str) -> bool:
        """check if a dimension label exists"""
        return self.index(label) is not None

    def __getitem__(self, idx: int) -> str:
        """get dimension label by index"""
        return self._dims[idx]

    def __len__(self) -> int:
        return len(self._dims)

    def __iter__(self):
        return iter(self._dims)

    def __contains__(self, item: str) -> bool:
        return item.upper() in self._dims

    def __repr__(self) -> str:
        return f"DimLabels({self._dims!r})"

    def __str__(self) -> str:
        return "".join(self._dims)


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


def get_dims(arr) -> tuple[str, ...]:
    """
    Get dimension labels from an array.

    Checks for:
    1. DimLabels feature (_dim_labels attribute)
    2. dims property
    3. Falls back to inference from ndim

    Parameters
    ----------
    arr : array-like
        array with shape and optionally dims

    Returns
    -------
    tuple[str, ...]
        dimension labels
    """
    # check for DimLabels feature
    if hasattr(arr, "_dim_labels") and arr._dim_labels is not None:
        return arr._dim_labels.value

    # check for dims property
    if hasattr(arr, "dims") and arr.dims is not None:
        dims = arr.dims
        if isinstance(dims, str):
            return parse_dims(dims, arr.ndim)
        return tuple(dims)

    # fallback to inference
    return infer_dims(arr.ndim)


def get_num_planes(arr) -> int:
    """
    Get number of Z-planes from an array.

    Checks:
    1. explicit num_planes property
    2. num_channels property (alias)
    3. shape at Z dimension index

    Parameters
    ----------
    arr : array-like
        array with shape

    Returns
    -------
    int
        number of planes (1 if no Z dimension)
    """
    # check explicit properties
    if hasattr(arr, "num_planes") and arr.num_planes is not None:
        return arr.num_planes
    if hasattr(arr, "num_channels") and arr.num_channels is not None:
        return arr.num_channels

    # infer from dims
    dims = get_dims(arr)
    if "Z" in dims:
        return arr.shape[dims.index("Z")]
    if "C" in dims:
        return arr.shape[dims.index("C")]

    return 1
