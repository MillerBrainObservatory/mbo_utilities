"""
Zarr format handlers.

Provides handlers for:
- ZarrFormat: generic Zarr v3 stores
- IsoviewFormat: Isoview lightsheet microscopy zarr structure
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


class IsoviewFormat(FormatBase):
    """
    Handler for Isoview lightsheet microscopy data.

    Detects directories with TM* subfolders containing .zarr files,
    or consolidated zarr stores with camera_N groups.

    Priority: 120 (very specific structure, checked first)
    """

    priority = 120

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        if not path.is_dir():
            return False

        # check for TM* subfolders pattern
        tm_folders = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("TM")]
        if tm_folders:
            # verify at least one has zarr files
            for tm in tm_folders[:3]:  # check first few
                if list(tm.glob("*.zarr")):
                    return True

        # check if this IS a TM folder
        if path.name.startswith("TM"):
            if list(path.glob("*.zarr")):
                return True

        # check for consolidated zarr with camera groups
        if path.suffix.lower() == ".zarr":
            camera_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("camera_")]
            if camera_dirs:
                return True

        return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        tm_folders = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("TM")]

        if tm_folders:
            plane_count = len(tm_folders)
        else:
            plane_count = 1

        return DataSourceDescriptor(
            path=path,
            format=DataFormat.ISOVIEW,
            structure=DataStructure.VOLUME,
            plane_count=plane_count,
            is_readable=True,
            is_writable=False,  # complex structure, don't write into
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import IsoviewArray
        from mbo_utilities.formats.base import filter_kwargs
        return IsoviewArray(path, **filter_kwargs("IsoviewArray", kwargs))


class ZarrFormat(FormatBase):
    """
    Handler for generic Zarr v3 stores.

    Detects .zarr directories or directories containing .zarr stores.

    Priority: 50 (generic zarr, after specific zarr formats)
    """

    priority = 50

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        # .zarr suffix
        if path.suffix.lower() == ".zarr" and path.is_dir():
            return True

        # directory containing .zarr stores
        if path.is_dir():
            zarr_stores = list(path.glob("*.zarr"))
            if zarr_stores:
                return True

        return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        # check for nested zarrs
        if path.is_dir():
            sub_zarrs = list(path.glob("*.zarr"))
            if sub_zarrs and path.suffix.lower() != ".zarr":
                # directory containing multiple zarr stores
                detected_files = sub_zarrs
                plane_count = len(sub_zarrs)
                structure = DataStructure.VOLUME if plane_count > 1 else DataStructure.TIMESERIES
            else:
                # single zarr store
                detected_files = [path]
                plane_count = 1
                structure = DataStructure.TIMESERIES
        else:
            detected_files = [path]
            plane_count = 1
            structure = DataStructure.TIMESERIES

        # check for OME-Zarr
        data_format = DataFormat.ZARR
        if path.suffix.lower() == ".zarr":
            # check for OME metadata
            zattrs = path / ".zattrs"
            zarr_json = path / "zarr.json"
            if zattrs.exists() or zarr_json.exists():
                try:
                    import json
                    meta_file = zarr_json if zarr_json.exists() else zattrs
                    with open(meta_file) as f:
                        attrs = json.load(f)
                    if "multiscales" in attrs or "ome" in str(attrs).lower():
                        data_format = DataFormat.OME_ZARR
                except Exception:
                    pass

        return DataSourceDescriptor(
            path=path,
            format=data_format,
            structure=structure,
            plane_count=plane_count,
            detected_files=detected_files[:5],
            is_readable=True,
            is_writable=True,
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import ZarrArray
        from mbo_utilities.formats.base import filter_kwargs
        filtered = filter_kwargs("ZarrArray", kwargs)
        if path.suffix.lower() == ".zarr":
            return ZarrArray([path], **filtered)
        else:
            zarrs = list(path.glob("*.zarr"))
            return ZarrArray(zarrs, **filtered)
