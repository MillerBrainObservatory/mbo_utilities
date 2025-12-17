"""
NumPy format handler.
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


class NumpyFormat(FormatBase):
    """
    Handler for NumPy .npy files.

    Note: ops.npy files are handled by Suite2pFormat, not this handler.

    Priority: 10 (lowest)
    """

    priority = 10

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        if path.is_file() and path.suffix.lower() == ".npy":
            # don't match ops.npy (Suite2p handles that)
            if path.stem == "ops":
                return False
            return True

        return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        return DataSourceDescriptor(
            path=path,
            format=DataFormat.NUMPY,
            structure=DataStructure.UNKNOWN,  # need to load to know
            detected_files=[path],
            is_readable=True,
            is_writable=True,
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import NumpyArray
        from mbo_utilities.formats.base import filter_kwargs
        return NumpyArray(path, **filter_kwargs("NumpyArray", kwargs))
