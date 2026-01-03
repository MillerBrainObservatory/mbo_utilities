"""
Data type feature for arrays.

Provides dtype management with lazy conversion support.
"""

from __future__ import annotations


import numpy as np

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent



class DataTypeFeature(ArrayFeature):
    """
    Data type feature for arrays.

    Manages original and target dtypes for lazy conversion.

    Parameters
    ----------
    original : np.dtype | str
        original data type
    target : np.dtype | str | None
        target data type for conversion, None to use original

    Examples
    --------
    >>> dt = DataTypeFeature(np.int16)
    >>> dt.original
    dtype('int16')
    >>> dt.value
    dtype('int16')
    >>> dt.target = np.float32
    >>> dt.value
    dtype('float32')
    """

    def __init__(
        self,
        original: np.dtype | str,
        target: np.dtype | str | None = None,
        property_name: str = "dtype",
    ):
        super().__init__(property_name=property_name)
        self._original = np.dtype(original)
        self._target = np.dtype(target) if target is not None else None

    @property
    def value(self) -> np.dtype:
        """Effective dtype (target if set, else original)."""
        return self._target if self._target is not None else self._original

    @property
    def original(self) -> np.dtype:
        """Original data type."""
        return self._original

    @property
    def target(self) -> np.dtype | None:
        """Target data type for conversion."""
        return self._target

    @target.setter
    def target(self, value: np.dtype | str | None) -> None:
        """Set target dtype."""
        old_value = self._target
        self._target = np.dtype(value) if value is not None else None

        if old_value != self._target:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={
                    "value": self.value,
                    "target": self._target,
                    "old_target": old_value,
                },
            )
            self._call_event_handlers(event)

    @property
    def is_converted(self) -> bool:
        """True if target dtype differs from original."""
        return self._target is not None and self._target != self._original

    @property
    def itemsize(self) -> int:
        """Bytes per element for effective dtype."""
        return self.value.itemsize

    @property
    def is_float(self) -> bool:
        """True if effective dtype is floating point."""
        return np.issubdtype(self.value, np.floating)

    @property
    def is_integer(self) -> bool:
        """True if effective dtype is integer."""
        return np.issubdtype(self.value, np.integer)

    @property
    def is_unsigned(self) -> bool:
        """True if effective dtype is unsigned integer."""
        return np.issubdtype(self.value, np.unsignedinteger)

    def set_value(self, array, value: np.dtype | str | None) -> None:
        """
        Set target dtype.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : np.dtype | str | None
            new target dtype, None to use original
        """
        self.target = value

    def reset(self) -> None:
        """Reset to original dtype (clear target)."""
        self.target = None

    def convert(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to target dtype if needed.

        Parameters
        ----------
        data : np.ndarray
            data to convert

        Returns
        -------
        np.ndarray
            converted data (or original if no conversion needed)
        """
        if self._target is None or data.dtype == self._target:
            return data
        return data.astype(self._target)

    def info_dict(self) -> dict:
        """
        Get dtype info as dictionary.

        Returns
        -------
        dict
            dtype information
        """
        return {
            "original": str(self._original),
            "target": str(self._target) if self._target else None,
            "effective": str(self.value),
            "itemsize": self.itemsize,
            "is_float": self.is_float,
            "is_integer": self.is_integer,
        }

    def __repr__(self) -> str:
        if self._target is None:
            return f"DataTypeFeature({self._original})"
        return f"DataTypeFeature({self._original} -> {self._target})"
