"""
Base format protocol and abstract class.

Defines the interface all format handlers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mbo_utilities.formats.descriptor import DataSourceDescriptor

if TYPE_CHECKING:
    from mbo_utilities._protocols import LazyArrayProtocol


# allowed kwargs per array class
_ALLOWED_KWARGS = {
    "MboRawArray": {
        "roi",
        "fix_phase",
        "phasecorr_method",
        "border",
        "upsample",
        "max_offset",
        "use_fft",
    },
    "ZarrArray": {"filenames", "compressor", "rois"},
    "MBOTiffArray": {"filenames", "_chunks"},
    "Suite2pArray": {"use_raw"},
    "BinArray": {"shape"},
    "H5Array": {"dataset"},
    "TiffArray": set(),
    "NumpyArray": {"metadata"},
    "IsoviewArray": set(),
}


def filter_kwargs(array_class_name: str, kwargs: dict) -> dict:
    """filter kwargs to only those accepted by the array class"""
    allowed = _ALLOWED_KWARGS.get(array_class_name, set())
    return {k: v for k, v in kwargs.items() if k in allowed}


class FormatBase(ABC):
    """
    Abstract base class for data format handlers.

    Each format (Suite2P, Tiff, Zarr, etc.) subclasses this to provide
    format-specific detection, description, and reading logic.

    Attributes
    ----------
    priority : int
        Higher priority handlers are checked first. Use higher values
        for more specific formats (e.g., IsoviewFormat > ZarrFormat).
    """

    priority: int = 0

    @abstractmethod
    def matches(self, path: Path) -> bool:
        """
        Check if this format can handle the given path.

        Parameters
        ----------
        path : Path
            Path to check (file or directory).

        Returns
        -------
        bool
            True if this format handler can read the path.
        """
        ...

    @abstractmethod
    def describe(self, path: Path) -> DataSourceDescriptor:
        """
        Create a descriptor for the path.

        Only called if matches() returned True.

        Parameters
        ----------
        path : Path
            Path to describe.

        Returns
        -------
        DataSourceDescriptor
            Description of the data source.
        """
        ...

    @abstractmethod
    def read(self, path: Path, **kwargs: Any) -> "LazyArrayProtocol":
        """
        Instantiate a lazy array from the path.

        Parameters
        ----------
        path : Path
            Path to read.
        **kwargs
            Format-specific options.

        Returns
        -------
        LazyArrayProtocol
            Lazy array instance.
        """
        ...

    def can_write_to(self, path: Path) -> bool:
        """
        Check if outputs can be safely written to this path.

        By default, returns False if the path matches this format
        (to prevent accidental overwrites of data directories).

        Parameters
        ----------
        path : Path
            Path to check.

        Returns
        -------
        bool
            True if safe to write outputs here.
        """
        return not self.matches(path)
