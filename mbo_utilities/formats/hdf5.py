"""
HDF5 format handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mbo_utilities.formats.base import FormatBase
from mbo_utilities.formats.descriptor import (
    DataFormat,
    DataSourceDescriptor,
    DataStructure,
)


class HDF5Format(FormatBase):
    """
    Handler for HDF5 files.

    Priority: 30
    """

    priority = 30

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        if path.is_file():
            return path.suffix.lower() in [".h5", ".hdf5", ".hdf"]

        return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        return DataSourceDescriptor(
            path=path,
            format=DataFormat.HDF5,
            structure=DataStructure.TIMESERIES,
            detected_files=[path],
            is_readable=True,
            is_writable=True,
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import H5Array
        from mbo_utilities.formats.base import filter_kwargs
        return H5Array(path, **filter_kwargs("H5Array", kwargs))
