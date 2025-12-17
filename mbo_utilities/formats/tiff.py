"""
TIFF format handlers.

Provides handlers for:
- TiffFormat: generic TIFF files
- MboTiffFormat: MBO-processed TIFFs with metadata
- RawScanimageFormat: raw ScanImage TIFFs with ROI support
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


def _find_plane_tiffs(directory: Path) -> list[Path]:
    """find planeXX.tiff files sorted by plane number"""
    plane_files = []
    for f in directory.glob("plane*.tif*"):
        if f.is_file():
            plane_files.append(f)

    def sort_key(p: Path) -> float:
        match = re.search(r"plane(\d+)", p.stem, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return float("inf")

    return sorted(plane_files, key=sort_key)


class RawScanimageFormat(FormatBase):
    """
    Handler for raw ScanImage TIFF files.

    Detects TIFFs with ScanImage metadata indicating raw acquisition data.
    Supports multi-ROI configurations.

    Priority: 90 (checked before generic TIFF)
    """

    priority = 90

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        # must be a tiff file or directory containing tiffs
        if path.is_file():
            if path.suffix.lower() not in [".tif", ".tiff"]:
                return False
            return self._is_raw_scanimage(path)
        elif path.is_dir():
            tiffs = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
            if tiffs:
                return self._is_raw_scanimage(tiffs[0])
        return False

    def _is_raw_scanimage(self, path: Path) -> bool:
        """check if tiff has raw scanimage metadata"""
        try:
            from mbo_utilities.metadata import is_raw_scanimage
            return is_raw_scanimage(path)
        except Exception:
            return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        from mbo_utilities.metadata import get_metadata

        if path.is_file():
            files = [path]
        else:
            files = sorted(path.glob("*.tif*"))

        # try to get roi info from metadata
        roi_count = None
        try:
            md = get_metadata(files[0])
            roi_count = md.get("num_rois", 1)
        except Exception:
            pass

        structure = DataStructure.MULTI_ROI if roi_count and roi_count > 1 else DataStructure.TIMESERIES

        return DataSourceDescriptor(
            path=path,
            format=DataFormat.RAW_SCANIMAGE,
            structure=structure,
            roi_count=roi_count,
            detected_files=files[:5],  # first few files
            is_readable=True,
            is_writable=False,  # don't overwrite raw data
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import MboRawArray
        from mbo_utilities.formats.base import filter_kwargs
        filtered = filter_kwargs("MboRawArray", kwargs)
        if path.is_file():
            return MboRawArray(files=[path], **filtered)
        else:
            files = sorted(path.glob("*.tif*"))
            return MboRawArray(files=files, **filtered)


class MboTiffFormat(FormatBase):
    """
    Handler for MBO-processed TIFF files.

    Detects TIFFs with MBO metadata indicating they were processed
    by the mbo_utilities pipeline.

    Priority: 80 (checked before generic TIFF)
    """

    priority = 80

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        if path.is_file():
            if path.suffix.lower() not in [".tif", ".tiff"]:
                return False
            return self._has_mbo_metadata(path)
        elif path.is_dir():
            tiffs = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
            if tiffs:
                return self._has_mbo_metadata(tiffs[0])
        return False

    def _has_mbo_metadata(self, path: Path) -> bool:
        """check if tiff has mbo processing metadata"""
        try:
            from mbo_utilities.metadata import has_mbo_metadata
            return has_mbo_metadata(path)
        except Exception:
            return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        if path.is_file():
            files = [path]
        else:
            files = sorted(path.glob("*.tif*"))

        # check for volume structure
        plane_files = _find_plane_tiffs(path) if path.is_dir() else []

        if len(plane_files) > 1:
            structure = DataStructure.VOLUME
            plane_count = len(plane_files)
        else:
            structure = DataStructure.TIMESERIES
            plane_count = 1

        return DataSourceDescriptor(
            path=path,
            format=DataFormat.MBO_TIFF,
            structure=structure,
            plane_count=plane_count,
            detected_files=files[:5],
            is_readable=True,
            is_writable=True,  # can add more outputs alongside
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import MBOTiffArray
        from mbo_utilities.formats.base import filter_kwargs
        filtered = filter_kwargs("MBOTiffArray", kwargs)
        if path.is_file():
            return MBOTiffArray([path], **filtered)
        else:
            files = sorted(path.glob("*.tif*"))
            return MBOTiffArray(files, **filtered)


class TiffFormat(FormatBase):
    """
    Handler for generic TIFF files.

    Handles standard TIFF and BigTIFF files without special metadata.
    For directories, only matches if there's a planeXX structure or
    a single TIFF file. Directories with multiple non-plane TIFFs
    are not matched (they may have different dimensions).

    Priority: 40 (fallback for TIFFs)
    """

    priority = 40

    def matches(self, path: Path) -> bool:
        if not path.exists():
            return False

        if path.is_file():
            return path.suffix.lower() in [".tif", ".tiff"]
        elif path.is_dir():
            # check for plane structure first (planeXX.tiff naming)
            plane_tiffs = _find_plane_tiffs(path)
            if plane_tiffs:
                return True

            # for directories without plane structure, only match if
            # there's exactly one tiff (multiple non-plane tiffs may
            # have different dimensions and shouldn't be loaded together)
            tiffs = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
            return len(tiffs) == 1
        return False

    def describe(self, path: Path) -> DataSourceDescriptor:
        if path.is_file():
            files = [path]
            structure = DataStructure.TIMESERIES
            plane_count = 1
        else:
            # check for volume structure
            plane_files = _find_plane_tiffs(path)
            if plane_files:
                files = plane_files
                structure = DataStructure.VOLUME
                plane_count = len(plane_files)
            else:
                files = sorted(path.glob("*.tif*"))
                structure = DataStructure.TIMESERIES
                plane_count = 1

        return DataSourceDescriptor(
            path=path,
            format=DataFormat.TIFF,
            structure=structure,
            plane_count=plane_count,
            detected_files=files[:5],
            is_readable=True,
            is_writable=True,
        )

    def read(self, path: Path, **kwargs: Any):
        from mbo_utilities.arrays import TiffArray
        from mbo_utilities.formats.base import filter_kwargs
        filtered = filter_kwargs("TiffArray", kwargs)
        if path.is_file():
            return TiffArray([path], **filtered)
        else:
            return TiffArray(path, **filtered)
