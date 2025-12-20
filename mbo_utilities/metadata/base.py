"""
base types and data structures for metadata handling.

this module contains the core types used across the metadata system:
- MetadataParameter: standardized parameter definition
- VoxelSize: named tuple for voxel dimensions
- METADATA_PARAMS: central registry of known parameters
- alias lookup utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Any


@dataclass
class MetadataParameter:
    """
    Standardized metadata parameter.

    Provides a central registry for parameter names, their aliases across
    different formats (ScanImage, Suite2p, OME, TIFF tags), and type information.

    Attributes
    ----------
    canonical : str
        The standard key name (e.g., "dx", "fs", "nplanes").
    aliases : tuple[str, ...]
        All known aliases for this parameter.
    dtype : type
        Expected Python type (float, int, str).
    unit : str, optional
        Physical unit if applicable (e.g., "micrometer", "Hz").
    default : Any
        Default value if parameter is not found in metadata.
    description : str
        Human-readable description of the parameter.
    """

    canonical: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    dtype: type = float
    unit: str | None = None
    default: Any = None
    description: str = ""


class VoxelSize(NamedTuple):
    """
    Voxel size in micrometers (dx, dy, dz).

    This class represents the physical size of a voxel in 3D space.
    All values are in micrometers.

    Attributes
    ----------
    dx : float
        Pixel size in X dimension (micrometers per pixel).
    dy : float
        Pixel size in Y dimension (micrometers per pixel).
    dz : float
        Pixel/voxel size in Z dimension (micrometers per z-step).

    Examples
    --------
    >>> vs = VoxelSize(0.5, 0.5, 5.0)
    >>> vs.dx
    0.5
    >>> vs.dz
    5.0
    >>> tuple(vs)
    (0.5, 0.5, 5.0)
    """

    dx: float
    dy: float
    dz: float

    @property
    def pixel_resolution(self) -> tuple[float, float]:
        """Return (dx, dy) tuple for backward compatibility."""
        return (self.dx, self.dy)

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """Return (dx, dy, dz) tuple."""
        return (self.dx, self.dy, self.dz)

    def to_dict(self, include_aliases: bool = True) -> dict:
        """
        Convert to dictionary with optional aliases.

        Parameters
        ----------
        include_aliases : bool
            If True, includes all standard aliases (OME, ImageJ, legacy).

        Returns
        -------
        dict
            Dictionary with resolution values and aliases.
        """
        result = {
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "pixel_resolution": self.pixel_resolution,
            "voxel_size": self.voxel_size,
        }

        if include_aliases:
            # legacy aliases
            result["umPerPixX"] = self.dx
            result["umPerPixY"] = self.dy
            result["umPerPixZ"] = self.dz

            # OME format
            result["PhysicalSizeX"] = self.dx
            result["PhysicalSizeY"] = self.dy
            result["PhysicalSizeZ"] = self.dz
            result["PhysicalSizeXUnit"] = "micrometer"
            result["PhysicalSizeYUnit"] = "micrometer"
            result["PhysicalSizeZUnit"] = "micrometer"

            # additional aliases
            result["z_step"] = self.dz  # backward compat

        return result


# metadata params registry
# dimensions: TZYX (4D), TYX (3D), or YX (2D)
METADATA_PARAMS: dict[str, MetadataParameter] = {
    # spatial resolution (micrometers per pixel)
    "dx": MetadataParameter(
        canonical="dx",
        aliases=(
            "Dx",
            "umPerPixX",
            "PhysicalSizeX",
            "pixelResolutionX",
            "pixel_size_x",
            "XResolution",
            "pixel_resolution_um",
        ),
        dtype=float,
        unit="micrometer",
        default=1.0,
        description="Pixel size in X dimension (µm/pixel)",
    ),
    "dy": MetadataParameter(
        canonical="dy",
        aliases=(
            "Dy",
            "umPerPixY",
            "PhysicalSizeY",
            "pixelResolutionY",
            "pixel_size_y",
            "YResolution",
        ),
        dtype=float,
        unit="micrometer",
        default=1.0,
        description="Pixel size in Y dimension (µm/pixel)",
    ),
    "dz": MetadataParameter(
        canonical="dz",
        aliases=(
            "Dz",
            "umPerPixZ",
            "PhysicalSizeZ",
            "z_step",
            "spacing",
            "pixelResolutionZ",
            "ZResolution",
        ),
        dtype=float,
        unit="micrometer",
        default=1.0,
        description="Voxel size in Z dimension (µm/z-step)",
    ),
    # temporal
    "fs": MetadataParameter(
        canonical="fs",
        aliases=(
            "frame_rate",
            "fr",
            "sampling_frequency",
            "frameRate",
            "scanFrameRate",
            "fps",
            "vps",
        ),
        dtype=float,
        unit="Hz",
        default=None,
        description="Frame rate / sampling frequency (Hz)",
    ),
    # image dimensions (pixels)
    "Lx": MetadataParameter(
        canonical="Lx",
        aliases=(
            "lx",
            "LX",
            "width",
            "nx",
            "size_x",
            "image_width",
            "fov_x",
            "num_px_x",
        ),
        dtype=int,
        unit="pixels",
        default=None,
        description="Image width in pixels",
    ),
    "Ly": MetadataParameter(
        canonical="Ly",
        aliases=(
            "ly",
            "LY",
            "height",
            "ny",
            "size_y",
            "image_height",
            "fov_y",
            "num_px_y",
        ),
        dtype=int,
        unit="pixels",
        default=None,
        description="Image height in pixels",
    ),
    # frame/plane/channel counts
    "nframes": MetadataParameter(
        canonical="nframes",
        aliases=("num_frames", "n_frames", "frames", "T", "nt"),
        dtype=int,
        default=None,
        description="Number of frames (time points) in the dataset",
    ),
    "nplanes": MetadataParameter(
        canonical="nplanes",
        aliases=(
            "num_planes",
            "n_planes",
            "planes",
            "Z",
            "nz",
            "num_z",
            "numPlanes",
            "zplanes",
        ),
        dtype=int,
        default=1,
        description="Number of z-planes",
    ),
    "nchannels": MetadataParameter(
        canonical="nchannels",
        aliases=(
            "num_channels",
            "n_channels",
            "channels",
            "C",
            "nc",
            "numChannels",
        ),
        dtype=int,
        default=1,
        description="Number of channels (typically 1 for calcium imaging)",
    ),
    # data type
    "dtype": MetadataParameter(
        canonical="dtype",
        aliases=("data_type", "pixel_type", "datatype"),
        dtype=str,
        default="int16",
        description="Data type of pixel values",
    ),
    # shape (tuple - special handling)
    "shape": MetadataParameter(
        canonical="shape",
        aliases=("array_shape", "data_shape", "size"),
        dtype=tuple,
        default=None,
        description="Array shape as tuple (T, Z, Y, X) or (T, Y, X) or (Y, X)",
    ),
}


def _build_alias_map() -> dict[str, str]:
    """build reverse lookup: alias (lowercase) -> canonical name."""
    alias_map = {}
    for param in METADATA_PARAMS.values():
        alias_map[param.canonical.lower()] = param.canonical
        for alias in param.aliases:
            alias_map[alias.lower()] = param.canonical
    return alias_map


ALIAS_MAP: dict[str, str] = _build_alias_map()


def get_canonical_name(name: str) -> str | None:
    """
    Get the canonical parameter name for an alias.

    Parameters
    ----------
    name : str
        Parameter name or alias.

    Returns
    -------
    str or None
        Canonical name, or None if not a registered parameter.
    """
    return ALIAS_MAP.get(name.lower())
