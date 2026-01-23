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
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from mbo_utilities.metadata.base import VoxelSize


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
    >>> spec.total_extent
    420.0
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
    def total_extent(self) -> float:
        """total physical extent of this dimension."""
        return self.size * self.scale

    @property
    def is_spatial(self) -> bool:
        return self.role == DimRole.SPATIAL

    @property
    def is_iteratable(self) -> bool:
        return self.role == DimRole.ITERATABLE

    @property
    def is_batch(self) -> bool:
        return self.role == DimRole.BATCH

    def with_selection(self, indices: Sequence[int]) -> DimensionSpec:
        """
        create new spec adjusted for a selection.

        parameters
        ----------
        indices : sequence of int
            0-based indices that were selected

        returns
        -------
        DimensionSpec
            new spec with adjusted size and scale
        """
        if not indices:
            return DimensionSpec(
                name=self.name,
                role=self.role,
                size=0,
                scale=self.scale,
                unit=self.unit,
            )

        new_size = len(indices)

        # compute step factor if uniform spacing
        step_factor = 1
        if len(indices) > 1:
            steps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
            unique_steps = set(steps)
            if len(unique_steps) == 1:
                step_factor = steps[0]

        new_scale = self.scale * step_factor

        return DimensionSpec(
            name=self.name,
            role=self.role,
            size=new_size,
            scale=new_scale,
            unit=self.unit,
        )

    def to_ome_axis(self) -> dict[str, Any]:
        """convert to OME-NGFF axis dict."""
        axis = {"name": self.name.lower()}

        # set type based on role and name
        if self.name == "T":
            axis["type"] = "time"
        elif self.name == "C":
            axis["type"] = "channel"
        else:
            axis["type"] = "space"

        # add unit if present
        if self.unit:
            axis["unit"] = self.unit

        return axis


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
    >>> specs.Lx
    512
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

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self):
        return iter(self.specs)

    def __getitem__(self, idx: int | str) -> DimensionSpec:
        if isinstance(idx, int):
            return self.specs[idx]
        # lookup by name
        for spec in self.specs:
            if spec.name == idx.upper():
                return spec
        raise KeyError(f"No dimension named '{idx}'")

    def get(self, name: str, default: DimensionSpec | None = None) -> DimensionSpec | None:
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

            specs.append(DimensionSpec(
                name=name,
                role=role,
                size=size,
                scale=scale,
            ))

        return cls(specs)

    # convenience properties for common dimensions

    @property
    def shape(self) -> tuple[int, ...]:
        """shape tuple from dimension sizes."""
        return tuple(s.size for s in self.specs)

    @property
    def ndim(self) -> int:
        return len(self.specs)

    @property
    def dims(self) -> tuple[str, ...]:
        """dimension names as tuple."""
        return tuple(s.name for s in self.specs)

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
    def Lx(self) -> int:
        """size of X dimension."""
        spec = self.get("X")
        return spec.size if spec else 0

    @property
    def Ly(self) -> int:
        """size of Y dimension."""
        spec = self.get("Y")
        return spec.size if spec else 0

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

    def with_selections(self, selections: dict[str, Sequence[int]]) -> DimensionSpecs:
        """
        create new specs adjusted for selections.

        parameters
        ----------
        selections : dict
            mapping of dim_name -> indices (0-based)

        returns
        -------
        DimensionSpecs
            new specs with adjusted sizes and scales
        """
        new_specs = []
        for spec in self.specs:
            if spec.name in selections:
                new_specs.append(spec.with_selection(selections[spec.name]))
            else:
                new_specs.append(spec)
        return DimensionSpecs(new_specs)

    def to_dict(self, include_aliases: bool = True) -> dict[str, Any]:
        """
        export as metadata dict.

        parameters
        ----------
        include_aliases : bool
            if True, includes all standard aliases

        returns
        -------
        dict
            metadata dictionary with reactive values
        """
        result = {
            "shape": self.shape,
            "dims": self.dims,
            "ndim": self.ndim,
        }

        # spatial dimensions
        result["Lx"] = self.Lx
        result["Ly"] = self.Ly
        result["dx"] = self.dx
        result["dy"] = self.dy

        # z dimension
        if self.get("Z"):
            result["dz"] = self.dz
            result["num_zplanes"] = self.num_zplanes
            if include_aliases:
                result["nplanes"] = self.num_zplanes
                result["num_planes"] = self.num_zplanes
                result["n_planes"] = self.num_zplanes
                result["Z"] = self.num_zplanes
                result["nz"] = self.num_zplanes
                result["slices"] = self.num_zplanes
                result["z_step"] = self.dz
                result["PhysicalSizeZ"] = self.dz

        # time dimension
        if self.get("T"):
            result["num_timepoints"] = self.num_timepoints
            if self.fs:
                result["fs"] = self.fs
                result["finterval"] = self.finterval
            if include_aliases:
                result["nframes"] = self.num_timepoints
                result["num_frames"] = self.num_timepoints
                result["n_frames"] = self.num_timepoints
                result["T"] = self.num_timepoints
                result["nt"] = self.num_timepoints
                result["timepoints"] = self.num_timepoints
                if self.fs:
                    result["frame_rate"] = self.fs

        # channel dimension
        if self.get("C"):
            result["num_channels"] = self.num_channels
            if include_aliases:
                result["nchannels"] = self.num_channels
                result["C"] = self.num_channels

        # resolution aliases
        if include_aliases:
            result["PhysicalSizeX"] = self.dx
            result["PhysicalSizeY"] = self.dy
            result["pixel_resolution"] = (self.dx, self.dy)
            if self.dz:
                result["voxel_size"] = (self.dx, self.dy, self.dz)

        return result

    def to_ome_axes(self) -> list[dict[str, Any]]:
        """convert to OME-NGFF axes list."""
        return [spec.to_ome_axis() for spec in self.specs]

    def to_ome_scale(self) -> list[float]:
        """get scale values in dimension order for OME-NGFF."""
        return [spec.scale for spec in self.specs]


class DimensionSpecMixin:
    """
    mixin providing dimension specification interface for arrays.

    arrays can override _build_dimension_specs() to customize their
    dimension structure.
    """

    _dimension_specs: DimensionSpecs | None = None

    def _build_dimension_specs(self) -> DimensionSpecs:
        """
        build dimension specs for this array.

        override in subclasses for custom dimension handling.
        default implementation infers from shape and dims.
        """
        return DimensionSpecs.from_array(self)

    @property
    def dimension_specs(self) -> DimensionSpecs:
        """dimension specifications for this array."""
        if self._dimension_specs is None:
            self._dimension_specs = self._build_dimension_specs()
        return self._dimension_specs

    def invalidate_dimension_specs(self):
        """clear cached dimension specs (call when shape/dims change)."""
        self._dimension_specs = None

    # convenience properties that delegate to dimension_specs

    @property
    def spatial_dims(self) -> tuple[str, ...]:
        """dimensions that form the 2D display plane."""
        return self.dimension_specs.spatial_dims

    @property
    def iteratable_dims(self) -> tuple[str, ...]:
        """dimensions that can be scrolled/indexed."""
        return self.dimension_specs.iteratable_dims

    @property
    def batch_dims(self) -> tuple[str, ...]:
        """dimensions that should produce separate output files."""
        return self.dimension_specs.batch_dims
