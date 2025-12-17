"""
Raw binary format handler.
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


class BinaryFormat(FormatBase):
    """
    Handler for raw binary files without Suite2p ops.npy.

    Requires shape parameter to read.

    Priority: 20
    """

    priority = 20

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        if path.is_file() and path.suffix.lower() == ".bin":
            # only match if there's NO ops.npy (otherwise Suite2pFormat handles it)
            ops_file = path.parent / "ops.npy"
            return not ops_file.exists()

        return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        return DataSourceDescriptor(
            path=path,
            format=DataFormat.BINARY,
            structure=DataStructure.UNKNOWN,  # need shape to know
            detected_files=[path],
            is_readable=True,
            is_writable=True,
            metadata={"requires_shape": True},
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import BinArray
        from mbo_utilities.formats.base import filter_kwargs

        if "shape" not in kwargs:
            raise ValueError(
                f"Binary file {path} requires shape parameter. "
                "Provide shape=(nframes, Ly, Lx) as kwarg."
            )
        return BinArray(path, **filter_kwargs("BinArray", kwargs))
