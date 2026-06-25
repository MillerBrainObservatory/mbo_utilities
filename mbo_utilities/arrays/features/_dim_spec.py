"""
dimension specification system for arrays.

provides a formal way for arrays to declare their dimension structure:
- spatial dimensions (Y, X): the 2D image plane
- iteratable dimensions (T, Z, C): slider/scroll dimensions
- batch dimensions (camera, trial): separate output files

this enables reactive metadata computation where output dimensions
are adjusted based on selections/slicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DimRole(str, Enum):
    """Role of a dimension in the array structure."""

    SPATIAL = "spatial"
    """spatial display dimensions (Y, X) - always form the 2D image plane."""

    ITERATABLE = "iteratable"
    """iteratable dimensions (T, Z, C) - can be scrolled/indexed with sliders."""

    BATCH = "batch"
    """batch dimensions (camera, trial) - produce separate output files."""

    @classmethod
    def from_dim_name(cls, name: str) -> DimRole:
        """infer role from dimension name."""
        name_upper = name.upper()
        if name_upper in ("Y", "X"):
            return cls.SPATIAL
        if name_upper in ("CAM", "CAMERA", "TRIAL", "RUN"):
            return cls.BATCH
        return cls.ITERATABLE


# map dimension names to default units
DIM_UNITS: dict[str, str | None] = {
    "T": "second",
    "Z": "micrometer",
    "Y": "micrometer",
    "X": "micrometer",
    "C": None,  # channel has no unit
    "V": "micrometer",  # volume (piezo)
    "R": "micrometer",  # roi
    "B": "micrometer",  # beamlet
    "S": "micrometer",  # slice
}


@dataclass
class DimensionSpec:
    """
    specification for a single dimension.

    parameters
    ----------
    name : str
        canonical dimension name: "T", "Z", "Y", "X", "C", etc.
    role : DimRole
        whether this is spatial, iteratable, or batch
    size : int
        current size in this dimension
    scale : float
        physical size per index (e.g., dz=15um, dt=0.033s, dx=0.5um)
    unit : str | None
        physical unit: "second", "micrometer", None (for channels)

    examples
    --------
    >>> spec = DimensionSpec("Z", DimRole.ITERATABLE, size=28, scale=15.0)
    >>> spec.unit
    'micrometer'
    """

    name: str
    role: DimRole
    size: int
    scale: float = 1.0
    unit: str | None = field(default=None)

    def __post_init__(self):
        # normalize name to uppercase
        self.name = self.name.upper()
        # set default unit if not provided
        if self.unit is None:
            self.unit = DIM_UNITS.get(self.name)

    @property
    def is_spatial(self) -> bool:
        return self.role == DimRole.SPATIAL

    @property
    def is_iteratable(self) -> bool:
        return self.role == DimRole.ITERATABLE

    @property
    def is_batch(self) -> bool:
        return self.role == DimRole.BATCH


@dataclass
class DimensionSpecs:
    """
    ordered collection of dimension specifications.

    provides convenient access to dimensions by name or role.

    parameters
    ----------
    specs : list of DimensionSpec
        ordered dimension specs matching array shape

    examples
    --------
    >>> specs = DimensionSpecs.from_array(arr)
    >>> specs.num_timepoints
    300
    >>> specs.spatial_dims
    ('Y', 'X')
    """

    specs: list[DimensionSpec]

    def __post_init__(self):
        # validate: must have at least Y, X
        names = [s.name for s in self.specs]
        if "Y" not in names or "X" not in names:
            pass  # allow for flexibility, but log warning?

    def __getitem__(self, idx: int | str) -> DimensionSpec:
        if isinstance(idx, int):
            return self.specs[idx]
        # lookup by name
        for spec in self.specs:
            if spec.name == idx.upper():
                return spec
        raise KeyError(f"No dimension named '{idx}'")

    def get(
        self, name: str, default: DimensionSpec | None = None
    ) -> DimensionSpec | None:
        """get dimension spec by name, or default if not found."""
        try:
            return self[name]
        except KeyError:
            return default

    @classmethod
    def from_array(
        cls,
        arr,
        dims: tuple[str, ...] | None = None,
        metadata: dict | None = None,
    ) -> DimensionSpecs:
        """
        create dimension specs from an array.

        parameters
        ----------
        arr : array-like
            array with shape attribute
        dims : tuple of str, optional
            dimension names (e.g., ("T", "Z", "Y", "X"))
            if None, inferred from array
        metadata : dict, optional
            metadata for scale values (dx, dy, dz, fs, etc.)

        returns
        -------
        DimensionSpecs
        """
        from mbo_utilities.arrays.features._dim_labels import get_dims
        from mbo_utilities.metadata.params import get_param

        if dims is None:
            dims = get_dims(arr)

        shape = arr.shape
        metadata = metadata or getattr(arr, "metadata", {}) or {}

        specs = []
        for i, dim_name in enumerate(dims):
            name = dim_name.upper()
            role = DimRole.from_dim_name(name)
            size = shape[i] if i < len(shape) else 1

            # get scale from metadata
            scale = 1.0
            if name == "X":
                scale = get_param(metadata, "dx", default=1.0) or 1.0
            elif name == "Y":
                scale = get_param(metadata, "dy", default=1.0) or 1.0
            elif name == "Z":
                scale = get_param(metadata, "dz", default=1.0) or 1.0
            elif name == "T":
                fs = get_param(metadata, "fs")
                if fs and fs > 0:
                    scale = 1.0 / fs  # time interval
                else:
                    scale = 1.0

            specs.append(
                DimensionSpec(
                    name=name,
                    role=role,
                    size=size,
                    scale=scale,
                )
            )

        return cls(specs)

    # convenience properties for common dimensions

    @property
    def spatial_dims(self) -> tuple[str, ...]:
        """names of spatial dimensions."""
        return tuple(s.name for s in self.specs if s.is_spatial)

    @property
    def iteratable_dims(self) -> tuple[str, ...]:
        """names of iteratable dimensions."""
        return tuple(s.name for s in self.specs if s.is_iteratable)

    @property
    def batch_dims(self) -> tuple[str, ...]:
        """names of batch dimensions."""
        return tuple(s.name for s in self.specs if s.is_batch)

    @property
    def num_timepoints(self) -> int:
        """size of T dimension (1 if no T)."""
        spec = self.get("T")
        return spec.size if spec else 1

    @property
    def num_zplanes(self) -> int:
        """size of Z dimension (1 if no Z)."""
        spec = self.get("Z")
        return spec.size if spec else 1

    @property
    def num_channels(self) -> int:
        """size of C dimension (1 if no C)."""
        spec = self.get("C")
        return spec.size if spec else 1

    @property
    def dx(self) -> float:
        """pixel size in X."""
        spec = self.get("X")
        return spec.scale if spec else 1.0

    @property
    def dy(self) -> float:
        """pixel size in Y."""
        spec = self.get("Y")
        return spec.scale if spec else 1.0

    @property
    def dz(self) -> float | None:
        """z-step size (None if no Z dimension)."""
        spec = self.get("Z")
        return spec.scale if spec else None

    @property
    def fs(self) -> float | None:
        """frame rate in Hz (None if no T or dt=0)."""
        spec = self.get("T")
        if spec and spec.scale > 0:
            return 1.0 / spec.scale
        return None

    @property
    def finterval(self) -> float | None:
        """frame interval in seconds (None if no T)."""
        spec = self.get("T")
        return spec.scale if spec else None
