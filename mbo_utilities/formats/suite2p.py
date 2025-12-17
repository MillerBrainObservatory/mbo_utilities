"""
Suite2p format handler.

Handles Suite2p binary output directories containing ops.npy and data.bin files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mbo_utilities.formats.base import FormatBase
from mbo_utilities.formats.descriptor import (
    DataFormat,
    DataSourceDescriptor,
    DataStructure,
)


def _extract_plane_number(name: str) -> int | None:
    """extract plane number from directory name like 'plane01_stitched'"""
    match = re.search(r"plane(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _find_plane_dirs(directory: Path) -> list[Path]:
    """find suite2p plane directories sorted by plane number"""
    plane_dirs = []
    for subdir in directory.iterdir():
        if subdir.is_dir():
            ops_file = subdir / "ops.npy"
            if ops_file.exists():
                plane_dirs.append(subdir)

    def sort_key(p: Path) -> float:
        num = _extract_plane_number(p.name)
        return num if num is not None else float("inf")

    return sorted(plane_dirs, key=sort_key)


class Suite2pFormat(FormatBase):
    """
    Handler for Suite2p binary output directories.

    Detects directories containing:
    - ops.npy + data.bin (single plane)
    - plane*/ subdirectories with ops.npy (volume)

    Priority: 100 (high - distinctive structure)
    """

    priority = 100

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        # direct file references
        if path.is_file():
            if path.suffix == ".npy" and path.stem == "ops":
                return True
            if path.suffix == ".bin":
                return (path.parent / "ops.npy").exists()
            return False

        # directory: check for ops.npy or plane subdirs
        if path.is_dir():
            if (path / "ops.npy").exists():
                return True
            plane_dirs = _find_plane_dirs(path)
            return len(plane_dirs) > 0

        return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        path = Path(path)

        # handle file references
        if path.is_file():
            if path.suffix == ".npy" and path.stem == "ops":
                ops_dir = path.parent
            elif path.suffix == ".bin":
                ops_dir = path.parent
            else:
                ops_dir = path
        else:
            ops_dir = path

        # check for volume structure
        plane_dirs = _find_plane_dirs(ops_dir) if ops_dir.is_dir() else []

        if len(plane_dirs) > 1:
            structure = DataStructure.VOLUME
            plane_count = len(plane_dirs)
            detected_files = [d / "ops.npy" for d in plane_dirs]
        elif (ops_dir / "ops.npy").exists():
            structure = DataStructure.SINGLE_PLANE
            plane_count = 1
            detected_files = [ops_dir / "ops.npy"]
        else:
            # single plane from file reference
            structure = DataStructure.SINGLE_PLANE
            plane_count = 1
            detected_files = []

        return DataSourceDescriptor(
            path=path,
            format=DataFormat.SUITE2P,
            structure=structure,
            plane_count=plane_count,
            detected_files=detected_files,
            is_readable=True,
            is_writable=False,  # don't write arbitrary files to suite2p dirs
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import Suite2pArray
        from mbo_utilities.formats.base import filter_kwargs
        return Suite2pArray(path, **filter_kwargs("Suite2pArray", kwargs))

    def can_write_to(self, path: Path) -> bool:
        # suite2p directories should not have arbitrary outputs written to them
        return not self.matches(path)
