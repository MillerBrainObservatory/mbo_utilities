"""
base types and data structures for metadata handling.

this module contains the core types used across the metadata system:
- MetadataParameter: standardized parameter definition
- VoxelSize: named tuple for voxel dimensions
- RoiMode: enum for multi-ROI handling modes
- METADATA_PARAMS: central registry of known parameters
- alias lookup utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Any


class RoiMode(str, Enum):
    """
    Mode for handling multi-ROI (mROI) data from ScanImage.

    ScanImage can acquire multiple ROIs in a single scan. This enum controls
    how those ROIs are processed during read/write operations.

    Attributes
    ----------
    concat_y : str
        Horizontally concatenate ROIs along the Y axis into a single FOV.
        This is the default mode - all ROIs are stitched together.
    separate : str
        Keep ROIs as separate files/arrays. Each ROI is written to its
        own file with "_roiN" suffix.

    Examples
    --------
    >>> from mbo_utilities.metadata import RoiMode
    >>> mode = RoiMode.concat_y
    >>> mode.value
    'concat_y'
    >>> RoiMode.from_string("CONCAT_Y")  # case-insensitive
    <RoiMode.concat_y: 'concat_y'>
    """

    concat_y = "concat_y"
    separate = "separate"

    # future modes:
    # concat_y_blend = "concat_y_blend"  # blend at seams
    # concat_y_reg = "concat_y_reg"      # register at seams

    @classmethod
    def from_string(cls, value: str) -> "RoiMode":
        """
        Case-insensitive lookup of RoiMode from string.

        Parameters
        ----------
        value : str
            String representation of the mode (e.g., "concat_y", "SEPARATE").

        Returns
        -------
        RoiMode
            The matching enum member.

        Raises
        ------
        ValueError
            If no matching mode is found.
        """
        value_lower = value.lower().strip()
        for member in cls:
            if member.value.lower() == value_lower:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown RoiMode: {value!r}. Valid modes: {valid}")

    @property
    def description(self) -> str:
        """Human-readable description of the mode."""
        descriptions = {
            RoiMode.concat_y: "horizontally concatenate ROIs",
            RoiMode.separate: "separate ROI files",
        }
        return descriptions.get(self, self.value)


@dataclass
class MetadataParameter:
    """
    Standardized metadata parameter.

    Provides a central registry for parameter names, their aliases across
    different formats (ScanImage, Suite2p, OME, TIFF tags), and type information.

    Attributes
    ----------
    canonical : str
        The standard key name (e.g., "dx", "fs", "num_zplanes").
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
    label : str, optional
        Display label for GUI (e.g., "Frame Rate" for "fs").
    """

    canonical: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    dtype: type = float
    unit: str | None = None
    default: Any = None
    description: str = ""
    label: str = ""


class VoxelSize(NamedTuple):
    """
    Voxel size in micrometers (dx, dy, dz).

    This class represents the physical size of a voxel in 3D space.
    All values are in micrometers.

    Attributes
    ----------
    dx : float
        Pixel size in X dimension (µm / px).
    dy : float
        Pixel size in Y dimension (µm / px).
    dz : float | None, optional
        Pixel/voxel size in Z dimension (µm / px).
        For LBM configurations, this must be supplied by the user.

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
    dz: float | None

    @property
    def pixel_resolution(self) -> tuple[float, float]:
        """Return (dx, dy) tuple for backward compatibility."""
        return (self.dx, self.dy)

    @property
    def voxel_size(self) -> tuple[float, float, float | None]:
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
            "PhysicalSizeX",
            "pixelResolutionX",
            "pixel_size_x",
            "XResolution",
            "pixel_resolution_um",
        ),
        dtype=float,
        unit="µm",
        default=1.0,
        description="Pixel size in X dimension (µm/pixel)",
        label="Pixel Size X",
    ),
    "dy": MetadataParameter(
        canonical="dy",
        aliases=(
            "Dy",
            "PhysicalSizeY",
            "pixelResolutionY",
            "pixel_size_y",
            "YResolution",
        ),
        dtype=float,
        unit="µm",
        default=1.0,
        description="Pixel size in Y dimension (µm/pixel)",
        label="Pixel Size Y",
    ),
    "dz": MetadataParameter(
        canonical="dz",
        aliases=(
            "Dz",
            "PhysicalSizeZ",
            "z_step",
            "spacing",
            "pixelResolutionZ",
            "ZResolution",
        ),
        dtype=float,
        unit="µm",
        default=None,
        description="Voxel size in Z dimension (µm/z-step). Must be user-supplied for LBM.",
        label="Z Step",
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
        label="Frame Rate",
    ),
    # ImageJ frame interval (seconds between frames, inverse of fs)
    "finterval": MetadataParameter(
        canonical="finterval",
        aliases=(
            "frame_interval",
            "FrameInterval",
            "dt",
            "time_interval",
        ),
        dtype=float,
        unit="s",
        default=None,
        description="Frame interval in seconds (1/fs). Used by ImageJ/Fiji.",
        label="Frame Interval",
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
            "page_width",
            "ImageWidth",
        ),
        dtype=int,
        unit="px",
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
            "page_height",
            "ImageLength",
        ),
        dtype=int,
        unit="px",
        default=None,
        description="Image height in pixels",
    ),
    # frame/plane/channel counts
    # note: in suite2p ops.npy, "nframes" means timepoints (post-registration), not per-slice frames
    "num_timepoints": MetadataParameter(
        canonical="num_timepoints",
        aliases=(
            "nframes",        # suite2p ops.npy compatibility
            "num_frames",     # legacy alias
            "n_frames",
            "frames",
            "T",
            "nt",
            "timepoints",
            "n_timepoints",
        ),
        dtype=int,
        default=None,
        description="Number of timepoints (T dimension) in the dataset",
        label="Timepoints",
    ),
    "num_zplanes": MetadataParameter(
        canonical="num_zplanes",
        aliases=(
            "num_planes",
            "nplanes",
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
        label="Num Z-Planes",
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
    # total number of elements
    "size": MetadataParameter(
        canonical="size",
        aliases=("num_elements", "total_elements"),
        dtype=int,
        default=None,
        description="Total number of elements in the array (product of dimensions)",
    ),
    # array shape tuple
    "shape": MetadataParameter(
        canonical="shape",
        aliases=("array_shape", "data_shape"),
        dtype=tuple,
        default=None,
        description="Array shape as tuple (T, Z, Y, X) or (T, Y, X) or (Y, X)",
    ),
    # stack detection (ScanImage-derived)
    "stack_type": MetadataParameter(
        canonical="stack_type",
        aliases=("stackType",),
        dtype=str,
        default="single_plane",
        description="Stack type: lbm, piezo, or single_plane",
    ),
    "lbm_stack": MetadataParameter(
        canonical="lbm_stack",
        aliases=("is_lbm", "lbmStack"),
        dtype=bool,
        default=False,
        description="True if Light Beads Microscopy stack",
    ),
    "piezo_stack": MetadataParameter(
        canonical="piezo_stack",
        aliases=("is_piezo", "piezoStack"),
        dtype=bool,
        default=False,
        description="True if piezo-driven z-stack",
    ),
    "num_color_channels": MetadataParameter(
        canonical="num_color_channels",
        aliases=("color_channels", "ncolors", "num_colors"),
        dtype=int,
        default=1,
        description="Number of color channels (1 or 2)",
    ),
    # ROI/FOV parameters
    "num_mrois": MetadataParameter(
        canonical="num_mrois",
        aliases=("num_rois", "scanimage_multirois", "numROIs", "nrois", "n_rois"),
        dtype=int,
        default=1,
        description="Number of mROIs (ScanImage multi-ROI scan regions)",
    ),
    "roi": MetadataParameter(
        canonical="roi",
        aliases=("roi_size", "roi_px"),
        dtype=tuple,
        unit="px",
        default=None,
        description="ROI dimensions as (width, height) in pixels",
    ),
    "fov": MetadataParameter(
        canonical="fov",
        aliases=("fov_px", "fov_pixels"),
        dtype=tuple,
        unit="px",
        default=None,
        description="Field of view as (x, y) in pixels",
    ),
    "fov_um": MetadataParameter(
        canonical="fov_um",
        aliases=("fov_micrometers",),
        dtype=tuple,
        unit="µm",
        default=None,
        description="Field of view as (x, y) in µm",
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


# core imaging metadata keys - always shown in metadata viewers/editors
# these are the essential parameters for calcium imaging data
IMAGING_METADATA_KEYS: tuple[str, ...] = (
    "fs",
    "dx",
    "dy",
    "dz",
    "Lx",
    "Ly",
    "num_zplanes",
    "num_timepoints",
    "dtype",
)


def get_imaging_metadata_info() -> list[dict]:
    """
    Get display info for core imaging metadata parameters.

    Returns a list of dicts with keys: canonical, label, unit, aliases, dtype.
    Used by GUI widgets to display/edit imaging metadata.

    Returns
    -------
    list[dict]
        List of metadata info dicts for each imaging parameter.
    """
    result = []
    for key in IMAGING_METADATA_KEYS:
        param = METADATA_PARAMS.get(key)
        if param:
            # format aliases as comma-separated string
            all_aliases = [param.canonical] + list(param.aliases[:3])  # limit to 3 aliases
            aliases_str = ", ".join(all_aliases)
            result.append({
                "canonical": param.canonical,
                "label": param.label or param.canonical,
                "unit": param.unit or "",
                "aliases": aliases_str,
                "dtype": param.dtype,
            })
    return result
