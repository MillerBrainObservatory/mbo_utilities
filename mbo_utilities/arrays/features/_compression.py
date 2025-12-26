"""
Compression feature for arrays.

Provides compression settings for zarr/h5/tiff formats.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent

if TYPE_CHECKING:
    pass


class Codec(str, Enum):
    """Supported compression codecs."""

    none = "none"
    gzip = "gzip"
    zstd = "zstd"
    lz4 = "lz4"
    blosc = "blosc"
    zlib = "zlib"
    lzma = "lzma"

    @classmethod
    def from_string(cls, value: str) -> "Codec":
        """Case-insensitive lookup."""
        value_lower = value.lower().strip()
        for member in cls:
            if member.value == value_lower:
                return member
        # try without case
        for member in cls:
            if member.name.lower() == value_lower:
                return member
        raise ValueError(f"unknown codec: {value}")


class CompressionSettings(NamedTuple):
    """Compression settings tuple."""

    codec: Codec
    level: int
    shuffle: bool = True

    @property
    def is_compressed(self) -> bool:
        """True if compression is enabled"""
        return self.codec != Codec.none


class CompressionFeature(ArrayFeature):
    """
    Compression feature for arrays.

    Manages compression settings for data storage.

    Parameters
    ----------
    codec : Codec | str
        compression codec
    level : int
        compression level (0-9 for most codecs)
    shuffle : bool
        enable byte shuffling (improves compression for numerical data)

    Examples
    --------
    >>> comp = CompressionFeature("gzip", level=5)
    >>> comp.codec
    <Codec.gzip: 'gzip'>
    >>> comp.level
    5
    >>> comp.is_compressed
    True
    """

    def __init__(
        self,
        codec: Codec | str = Codec.gzip,
        level: int = 1,
        shuffle: bool = True,
        property_name: str = "compression",
    ):
        super().__init__(property_name=property_name)

        if isinstance(codec, str):
            self._codec = Codec.from_string(codec)
        else:
            self._codec = codec

        self._level = max(0, min(9, level))
        self._shuffle = shuffle

    @property
    def value(self) -> CompressionSettings:
        """compression settings as namedtuple"""
        return CompressionSettings(self._codec, self._level, self._shuffle)

    @property
    def codec(self) -> Codec:
        """compression codec"""
        return self._codec

    @property
    def level(self) -> int:
        """compression level (0-9)"""
        return self._level

    @property
    def shuffle(self) -> bool:
        """byte shuffling enabled"""
        return self._shuffle

    @property
    def is_compressed(self) -> bool:
        """True if compression is enabled"""
        return self._codec != Codec.none

    def set_value(self, array, value) -> None:
        """
        Set compression settings.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : CompressionSettings | tuple | dict | str
            new compression settings
        """
        old_value = self.value

        if isinstance(value, CompressionSettings):
            self._codec = value.codec
            self._level = value.level
            self._shuffle = value.shuffle
        elif isinstance(value, str):
            self._codec = Codec.from_string(value)
        elif isinstance(value, (tuple, list)):
            self._codec = Codec.from_string(value[0]) if isinstance(value[0], str) else value[0]
            if len(value) > 1:
                self._level = int(value[1])
            if len(value) > 2:
                self._shuffle = bool(value[2])
        elif isinstance(value, dict):
            if "codec" in value:
                codec = value["codec"]
                self._codec = Codec.from_string(codec) if isinstance(codec, str) else codec
            if "level" in value:
                self._level = int(value["level"])
            if "shuffle" in value:
                self._shuffle = bool(value["shuffle"])
        else:
            raise TypeError(f"expected CompressionSettings, tuple, dict, or str, got {type(value)}")

        new_value = self.value
        if old_value != new_value:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"value": new_value, "old_value": old_value},
            )
            self._call_event_handlers(event)

    def disable(self) -> None:
        """Disable compression."""
        self._codec = Codec.none

    def to_zarr_codecs(self):
        """
        Convert to zarr v3 codec pipeline.

        Returns
        -------
        list
            list of zarr codecs
        """
        from zarr.codecs import BytesCodec, GzipCodec

        if self._codec == Codec.none:
            return [BytesCodec()]
        elif self._codec == Codec.gzip:
            return [BytesCodec(), GzipCodec(level=self._level)]
        else:
            # fallback to gzip for unsupported codecs
            return [BytesCodec(), GzipCodec(level=self._level)]

    def to_h5_kwargs(self) -> dict:
        """
        Convert to h5py create_dataset kwargs.

        Returns
        -------
        dict
            kwargs for h5py.create_dataset
        """
        if self._codec == Codec.none:
            return {}
        elif self._codec == Codec.gzip:
            return {"compression": "gzip", "compression_opts": self._level}
        elif self._codec == Codec.lzf:
            return {"compression": "lzf"}
        else:
            return {"compression": "gzip", "compression_opts": self._level}

    def __repr__(self) -> str:
        if self._codec == Codec.none:
            return "CompressionFeature(none)"
        return f"CompressionFeature({self._codec.value}, level={self._level})"
