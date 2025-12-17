"""
Data format detection and handling.

This module provides a registry-based system for detecting and reading
various imaging data formats.

Usage
-----
>>> from mbo_utilities import describe
>>> desc = describe("D:/demo/results/")
>>> print(desc.format)
DataFormat.SUITE2P
>>> print(desc.structure)
DataStructure.VOLUME
>>> print(desc.plane_count)
14

Format Handlers
---------------
Each format has a handler class with a priority. Higher priority handlers
are checked first, allowing specific formats to take precedence over
generic ones (e.g., IsoviewFormat over ZarrFormat).

Current handlers (by priority):
- IsoviewFormat (120): Isoview lightsheet zarr structure
- Suite2pFormat (100): Suite2p binary output directories
- RawScanimageFormat (90): Raw ScanImage TIFFs
- MboTiffFormat (80): MBO-processed TIFFs
- ZarrFormat (50): Generic Zarr v3 stores
- TiffFormat (40): Generic TIFF files
- HDF5Format (30): HDF5 files
- BinaryFormat (20): Raw binary files
- NumpyFormat (10): NumPy .npy files
"""

from mbo_utilities.formats.descriptor import (
    DataFormat,
    DataSourceDescriptor,
    DataStructure,
)
from mbo_utilities.formats.registry import FormatRegistry, describe

__all__ = [
    "DataFormat",
    "DataSourceDescriptor",
    "DataStructure",
    "FormatRegistry",
    "describe",
]
