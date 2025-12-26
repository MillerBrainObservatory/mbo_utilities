"""
Frame rate feature for arrays.

Provides temporal sampling frequency in Hz.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent

if TYPE_CHECKING:
    pass


class FrameRateFeature(ArrayFeature):
    """
    Frame rate feature for arrays.

    Manages temporal sampling frequency in Hz with derived properties.

    Parameters
    ----------
    fs : float | None
        sampling frequency in Hz

    Examples
    --------
    >>> fr = FrameRateFeature(30.0)
    >>> fr.value
    30.0
    >>> fr.period
    0.03333...
    >>> fr.period_ms
    33.33...
    """

    def __init__(
        self,
        fs: float | None = None,
        property_name: str = "frame_rate",
    ):
        super().__init__(property_name=property_name)
        self._value = float(fs) if fs is not None else None

    @property
    def value(self) -> float | None:
        """frame rate in Hz"""
        return self._value

    @property
    def fs(self) -> float | None:
        """alias for value - sampling frequency in Hz"""
        return self._value

    @property
    def period(self) -> float | None:
        """frame interval in seconds (1/fs)"""
        if self._value is None or self._value == 0:
            return None
        return 1.0 / self._value

    @property
    def period_ms(self) -> float | None:
        """frame interval in milliseconds"""
        if self._value is None or self._value == 0:
            return None
        return 1000.0 / self._value

    @property
    def finterval(self) -> float | None:
        """alias for period - frame interval in seconds (ImageJ convention)"""
        return self.period

    def set_value(self, array, value: float | None) -> None:
        """
        Set frame rate.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : float | None
            new frame rate in Hz
        """
        old_value = self._value
        self._value = float(value) if value is not None else None

        if old_value != self._value:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"value": self._value, "old_value": old_value},
            )
            self._call_event_handlers(event)

    def set_from_period(self, period: float) -> None:
        """
        Set frame rate from period (seconds).

        Parameters
        ----------
        period : float
            frame interval in seconds
        """
        if period <= 0:
            raise ValueError("period must be positive")
        self._value = 1.0 / period

    def time_to_frames(self, seconds: float) -> int:
        """
        Convert time in seconds to frame count.

        Parameters
        ----------
        seconds : float
            time in seconds

        Returns
        -------
        int
            number of frames
        """
        if self._value is None:
            raise ValueError("frame rate not set")
        return int(seconds * self._value)

    def frames_to_time(self, frames: int) -> float:
        """
        Convert frame count to time in seconds.

        Parameters
        ----------
        frames : int
            number of frames

        Returns
        -------
        float
            time in seconds
        """
        if self._value is None:
            raise ValueError("frame rate not set")
        return frames / self._value

    def __repr__(self) -> str:
        if self._value is None:
            return "FrameRateFeature(None)"
        return f"FrameRateFeature({self._value:.2f} Hz)"
