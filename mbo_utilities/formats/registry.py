"""
Format registry for data source detection.

Manages registered format handlers and provides detection/dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mbo_utilities.formats.descriptor import (
    DataFormat,
    DataSourceDescriptor,
    DataStructure,
)

if TYPE_CHECKING:
    from mbo_utilities.formats.base import FormatBase


class FormatRegistry:
    """
    Central registry for data format handlers.

    Handlers are checked in priority order (highest first) when detecting
    the format of a path.
    """

    _handlers: list["FormatBase"] = []
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """lazy-load handlers on first use"""
        if cls._initialized:
            return

        from mbo_utilities.formats.suite2p import Suite2pFormat
        from mbo_utilities.formats.tiff import TiffFormat, MboTiffFormat, RawScanimageFormat
        from mbo_utilities.formats.zarr import ZarrFormat, IsoviewFormat
        from mbo_utilities.formats.hdf5 import HDF5Format
        from mbo_utilities.formats.binary import BinaryFormat
        from mbo_utilities.formats.numpy import NumpyFormat

        # register in priority order (will be sorted anyway)
        for handler_cls in [
            IsoviewFormat,      # 120 - very specific zarr structure
            Suite2pFormat,      # 100 - distinctive ops.npy structure
            RawScanimageFormat, # 90 - raw scanimage tiffs
            MboTiffFormat,      # 80 - mbo-processed tiffs
            ZarrFormat,         # 50 - generic zarr
            TiffFormat,         # 40 - generic tiff
            HDF5Format,         # 30 - hdf5 files
            BinaryFormat,       # 20 - raw binary
            NumpyFormat,        # 10 - numpy arrays
        ]:
            cls.register(handler_cls())

        cls._initialized = True

    @classmethod
    def register(cls, handler: "FormatBase") -> None:
        """
        Register a format handler.

        Parameters
        ----------
        handler : FormatBase
            Handler instance to register.
        """
        cls._handlers.append(handler)
        cls._handlers.sort(key=lambda h: -h.priority)

    @classmethod
    def detect(cls, path: Path | str) -> DataSourceDescriptor:
        """
        Detect the format of a path.

        Parameters
        ----------
        path : Path | str
            Path to analyze.

        Returns
        -------
        DataSourceDescriptor
            Description of what exists at the path.
        """
        cls._ensure_initialized()
        path = Path(path)

        if not path.exists():
            return DataSourceDescriptor(
                path=path,
                format=DataFormat.UNKNOWN,
                structure=DataStructure.UNKNOWN,
                is_readable=False,
                is_writable=True,
            )

        for handler in cls._handlers:
            if handler.matches(path):
                return handler.describe(path)

        # no handler matched
        return DataSourceDescriptor(
            path=path,
            format=DataFormat.UNKNOWN,
            structure=DataStructure.UNKNOWN,
            is_readable=False,
            is_writable=True,
        )

    @classmethod
    def get_handler(cls, path: Path | str) -> "FormatBase | None":
        """
        Get the handler for a path.

        Parameters
        ----------
        path : Path | str
            Path to check.

        Returns
        -------
        FormatBase | None
            Handler that matches the path, or None.
        """
        cls._ensure_initialized()
        path = Path(path)

        for handler in cls._handlers:
            if handler.matches(path):
                return handler
        return None

    @classmethod
    def can_write_to(cls, path: Path | str) -> bool:
        """
        Check if outputs can be safely written to a path.

        Parameters
        ----------
        path : Path | str
            Path to check.

        Returns
        -------
        bool
            True if safe to write here.
        """
        cls._ensure_initialized()
        path = Path(path)

        # non-existent or empty directories are always writable
        if not path.exists():
            return True
        if path.is_dir() and not any(path.iterdir()):
            return True

        # check with matching handler
        handler = cls.get_handler(path)
        if handler:
            return handler.can_write_to(path)

        return True

    @classmethod
    def read(cls, path: Path | str, **kwargs) -> "LazyArrayProtocol":
        """
        Read data from a path using the appropriate handler.

        Parameters
        ----------
        path : Path | str
            Path to read.
        **kwargs
            Format-specific options.

        Returns
        -------
        LazyArrayProtocol
            Lazy array instance.

        Raises
        ------
        ValueError
            If no handler matches the path.
        """
        from mbo_utilities._protocols import LazyArrayProtocol

        cls._ensure_initialized()
        path = Path(path)

        handler = cls.get_handler(path)
        if handler is None:
            raise ValueError(f"No format handler found for: {path}")

        return handler.read(path, **kwargs)


def describe(path: Path | str) -> DataSourceDescriptor:
    """
    Inspect a path and return its data source descriptor.

    Useful for understanding what imread() will detect before loading.

    Parameters
    ----------
    path : Path | str
        Path to inspect.

    Returns
    -------
    DataSourceDescriptor
        Description of the data source.

    Examples
    --------
    >>> from mbo_utilities import describe
    >>> desc = describe("D:/demo/results/")
    >>> print(desc.format)
    DataFormat.SUITE2P
    >>> print(desc.structure)
    DataStructure.VOLUME
    >>> print(desc.plane_count)
    14
    """
    return FormatRegistry.detect(path)
