"""
Voxel size feature for arrays.

Provides physical dimensions (dx, dy, dz) in micrometers.
"""

from __future__ import annotations


from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent
from mbo_utilities.metadata.base import VoxelSize


class VoxelSizeFeature(ArrayFeature):
    """
    Voxel size feature for arrays.

    Manages physical pixel/voxel dimensions in micrometers.

    Parameters
    ----------
    dx : float
        pixel size in X dimension (µm)
    dy : float
        pixel size in Y dimension (µm)
    dz : float | None
        voxel size in Z dimension (µm), None for 2D/3D data

    Examples
    --------
    >>> vs = VoxelSizeFeature(0.5, 0.5, 5.0)
    >>> vs.value
    VoxelSize(dx=0.5, dy=0.5, dz=5.0)
    >>> vs.dx
    0.5
    >>> vs.pixel_resolution
    (0.5, 0.5)
    """

    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float | None = None,
        property_name: str = "voxel_size",
    ):
        super().__init__(property_name=property_name)
        self._value = VoxelSize(float(dx), float(dy), dz if dz is None else float(dz))

    @property
    def value(self) -> VoxelSize:
        """Current voxel size as VoxelSize namedtuple."""
        return self._value

    @property
    def dx(self) -> float:
        """Pixel size in X (µm)."""
        return self._value.dx

    @property
    def dy(self) -> float:
        """Pixel size in Y (µm)."""
        return self._value.dy

    @property
    def dz(self) -> float | None:
        """Voxel size in Z (µm), None if not set."""
        return self._value.dz

    @property
    def pixel_resolution(self) -> tuple[float, float]:
        """(dx, dy) tuple for 2D resolution."""
        return self._value.pixel_resolution

    @property
    def is_isotropic_xy(self) -> bool:
        """True if dx == dy."""
        return abs(self._value.dx - self._value.dy) < 1e-9

    def set_value(self, array, value) -> None:
        """
        Set voxel size.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : VoxelSize | tuple | dict
            new voxel size as VoxelSize, (dx, dy, dz) tuple, or dict
        """
        old_value = self._value

        if isinstance(value, VoxelSize):
            self._value = value
        elif isinstance(value, (tuple, list)):
            if len(value) == 2:
                self._value = VoxelSize(float(value[0]), float(value[1]), None)
            elif len(value) == 3:
                self._value = VoxelSize(
                    float(value[0]),
                    float(value[1]),
                    None if value[2] is None else float(value[2]),
                )
            else:
                raise ValueError(f"expected 2 or 3 values, got {len(value)}")
        elif isinstance(value, dict):
            self._value = VoxelSize(
                float(value.get("dx", 1.0)),
                float(value.get("dy", 1.0)),
                value.get("dz"),
            )
        else:
            raise TypeError(f"expected VoxelSize, tuple, or dict, got {type(value)}")

        if old_value != self._value:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"value": self._value, "old_value": old_value},
            )
            self._call_event_handlers(event)

    def to_scale(self, dims: tuple[str, ...] | None = None) -> tuple[float, ...]:
        """
        Convert to scale tuple for napari/viewers.

        Parameters
        ----------
        dims : tuple[str, ...] | None
            dimension labels to determine scale order.
            if None, returns (dz, dy, dx) for 3D or (dy, dx) for 2D.

        Returns
        -------
        tuple[float, ...]
            scale values in dimension order
        """
        if dims is None:
            if self._value.dz is not None:
                return (self._value.dz, self._value.dy, self._value.dx)
            return (self._value.dy, self._value.dx)

        scale = []
        for d in dims:
            if d == "X":
                scale.append(self._value.dx)
            elif d == "Y":
                scale.append(self._value.dy)
            elif d in ("Z", "C"):
                scale.append(self._value.dz if self._value.dz is not None else 1.0)
            else:
                scale.append(1.0)  # T, S, etc. default to 1
        return tuple(scale)

    def __repr__(self) -> str:
        return f"VoxelSizeFeature(dx={self.dx}, dy={self.dy}, dz={self.dz})"


def get_voxel_size_from_metadata(metadata: dict) -> VoxelSize:
    """
    Extract voxel size from metadata dict.

    Parameters
    ----------
    metadata : dict
        metadata dictionary

    Returns
    -------
    VoxelSize
        extracted voxel size with defaults for missing values
    """
    from mbo_utilities.metadata import get_voxel_size

    return get_voxel_size(metadata)


class VoxelSizeMixin:
    """
    Mixin class that adds voxel size support to array classes.

    This mixin provides convenience properties that delegate to a
    `voxel_size` VoxelSizeFeature instance. Feature detection uses
    duck typing: presence of `voxel_size` attribute indicates the
    array supports physical dimension operations.

    Usage
    -----
    Check for voxel size support::

        if hasattr(arr, 'voxel_size'):
            # array supports voxel size operations
            print(f"Pixel size: {arr.dx} x {arr.dy} µm")

    Required attributes (set by implementing class):
        voxel_size : VoxelSizeFeature
            The feature instance managing voxel size state

    Properties provided (all delegate to voxel_size):
        dx : float
            Pixel size in X dimension (µm)
        dy : float
            Pixel size in Y dimension (µm)
        dz : float | None
            Voxel size in Z dimension (µm)
        pixel_resolution : tuple[float, float]
            (dx, dy) tuple for 2D resolution
        is_isotropic_xy : bool
            True if dx == dy
    """

    @property
    def dx(self) -> float:
        """Pixel size in X dimension (µm)."""
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            return vs.dx
        return 1.0

    @dx.setter
    def dx(self, value: float):
        """Set pixel size in X dimension."""
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            current = vs.value
            vs.set_value(self, VoxelSize(float(value), current.dy, current.dz))

    @property
    def dy(self) -> float:
        """Pixel size in Y dimension (µm)."""
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            return vs.dy
        return 1.0

    @dy.setter
    def dy(self, value: float):
        """Set pixel size in Y dimension."""
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            current = vs.value
            vs.set_value(self, VoxelSize(current.dx, float(value), current.dz))

    @property
    def dz(self) -> float | None:
        """Voxel size in Z dimension (µm), None if not set."""
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            return vs.dz
        return None

    @dz.setter
    def dz(self, value: float | None):
        """Set voxel size in Z dimension."""
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            current = vs.value
            vs.set_value(
                self, VoxelSize(current.dx, current.dy, float(value) if value else None)
            )

    @property
    def pixel_resolution(self) -> tuple[float, float]:
        """(dx, dy) tuple for 2D resolution."""
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            return vs.pixel_resolution
        return (1.0, 1.0)

    @property
    def is_isotropic_xy(self) -> bool:
        """True if dx == dy."""
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            return vs.is_isotropic_xy
        return True

    def get_scale(self, dims: tuple[str, ...] | None = None) -> tuple[float, ...]:
        """
        Get scale tuple for napari/viewers.

        Parameters
        ----------
        dims : tuple[str, ...] | None
            dimension labels to determine scale order.
            if None, returns (dz, dy, dx) for 3D or (dy, dx) for 2D.

        Returns
        -------
        tuple[float, ...]
            scale values in dimension order
        """
        vs = getattr(self, "voxel_size", None)
        if vs is not None:
            return vs.to_scale(dims)
        return (1.0, 1.0)
