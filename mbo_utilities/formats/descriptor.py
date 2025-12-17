"""
Data source descriptor types.

Provides enums and dataclass for describing data sources without loading them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path


class DataFormat(Enum):
    """supported data formats"""
    SUITE2P = auto()
    TIFF = auto()
    MBO_TIFF = auto()
    RAW_SCANIMAGE = auto()
    ZARR = auto()
    OME_ZARR = auto()
    HDF5 = auto()
    BINARY = auto()
    NUMPY = auto()
    ISOVIEW = auto()
    UNKNOWN = auto()


class DataStructure(Enum):
    """data organization structure"""
    SINGLE_PLANE = auto()   # TYX - single z plane
    VOLUME = auto()         # TZYX - multiple z planes
    MULTI_ROI = auto()      # multiple scanfields (can be stitched)
    TIMESERIES = auto()     # T is primary axis
    UNKNOWN = auto()


@dataclass
class DataSourceDescriptor:
    """
    Describes what exists at a path without loading the data.

    Use `describe(path)` to create a descriptor for any path.

    Attributes
    ----------
    path : Path
        The path that was analyzed.
    format : DataFormat
        Detected data format.
    structure : DataStructure
        Detected data structure.
    is_readable : bool
        Whether this path can be read as imaging data.
    is_writable : bool
        Whether outputs can be written to this path safely.
    plane_count : int | None
        Number of z-planes detected (if applicable).
    roi_count : int | None
        Number of ROIs detected (if applicable).
    detected_files : list[Path]
        Key files detected during analysis.
    metadata : dict
        Additional format-specific metadata.
    """
    path: Path
    format: DataFormat
    structure: DataStructure
    is_readable: bool = True
    is_writable: bool = True
    plane_count: int | None = None
    roi_count: int | None = None
    detected_files: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [
            f"DataSourceDescriptor(",
            f"  path={self.path},",
            f"  format={self.format.name},",
            f"  structure={self.structure.name},",
        ]
        if self.plane_count is not None:
            parts.append(f"  plane_count={self.plane_count},")
        if self.roi_count is not None:
            parts.append(f"  roi_count={self.roi_count},")
        parts.append(f"  is_readable={self.is_readable},")
        parts.append(f"  is_writable={self.is_writable},")
        if self.detected_files:
            parts.append(f"  detected_files=[{len(self.detected_files)} files],")
        parts.append(")")
        return "\n".join(parts)
