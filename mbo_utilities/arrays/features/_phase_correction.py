"""
Phase correction feature for arrays.

Provides bidirectional scan phase correction settings.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent

if TYPE_CHECKING:
    pass


class PhaseCorrMethod(str, Enum):
    """Phase correction methods."""

    none = "none"
    frame = "frame"  # per-frame 2D correlation
    mean = "mean"  # mean projection
    max = "max"  # max projection
    std = "std"  # std projection
    mean_sub = "mean-sub"  # mean subtraction

    @classmethod
    def from_string(cls, value: str | None) -> "PhaseCorrMethod":
        """Case-insensitive lookup."""
        if value is None:
            return cls.none
        value_lower = value.lower().strip()
        if value_lower in ("", "none", "false"):
            return cls.none
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"unknown phase correction method: {value}")

    @property
    def is_3d(self) -> bool:
        """True if method uses 3D window."""
        return self in (
            PhaseCorrMethod.mean,
            PhaseCorrMethod.max,
            PhaseCorrMethod.std,
            PhaseCorrMethod.mean_sub,
        )


class PhaseCorrectionFeature(ArrayFeature):
    """
    Phase correction feature for arrays.

    Manages bidirectional scan phase correction settings.

    Parameters
    ----------
    enabled : bool
        whether correction is enabled
    method : PhaseCorrMethod | str
        correction method
    shift : float | None
        fixed shift value (None for auto-compute)
    use_fft : bool
        use FFT-based subpixel correction
    upsample : int
        upsampling factor for subpixel phase estimation
    border : int
        border pixels to exclude from phase estimation
    max_offset : int
        maximum phase offset to search

    Examples
    --------
    >>> pc = PhaseCorrectionFeature(enabled=True, method="mean")
    >>> pc.enabled
    True
    >>> pc.method
    <PhaseCorrMethod.mean: 'mean'>
    """

    def __init__(
        self,
        enabled: bool = False,
        method: PhaseCorrMethod | str = PhaseCorrMethod.mean,
        shift: float | None = None,
        use_fft: bool = False,
        upsample: int = 10,
        border: int = 10,
        max_offset: int = 4,
        property_name: str = "phase_correction",
    ):
        super().__init__(property_name=property_name)
        self._enabled = enabled

        if isinstance(method, str):
            self._method = PhaseCorrMethod.from_string(method)
        else:
            self._method = method

        self._shift = shift
        self._use_fft = use_fft
        self._upsample = upsample
        self._border = border
        self._max_offset = max_offset
        self._computed_shift: float | None = None

    @property
    def value(self) -> dict:
        """phase correction settings as dict"""
        return {
            "enabled": self._enabled,
            "method": self._method.value,
            "shift": self.effective_shift,
            "use_fft": self._use_fft,
            "upsample": self._upsample,
            "border": self._border,
            "max_offset": self._max_offset,
        }

    @property
    def enabled(self) -> bool:
        """whether correction is enabled"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """enable or disable correction"""
        old_enabled = self._enabled
        self._enabled = bool(value)
        if old_enabled != self._enabled:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"enabled": self._enabled, "old_enabled": old_enabled},
            )
            self._call_event_handlers(event)

    @property
    def method(self) -> PhaseCorrMethod:
        """correction method"""
        return self._method

    @method.setter
    def method(self, value: PhaseCorrMethod | str) -> None:
        """set correction method"""
        if isinstance(value, str):
            self._method = PhaseCorrMethod.from_string(value)
        else:
            self._method = value
        # reset computed shift when method changes
        self._computed_shift = None

    @property
    def shift(self) -> float | None:
        """fixed shift value (None for auto)"""
        return self._shift

    @shift.setter
    def shift(self, value: float | None) -> None:
        """set fixed shift value"""
        self._shift = value

    @property
    def effective_shift(self) -> float | None:
        """effective shift (fixed or computed)"""
        if self._shift is not None:
            return self._shift
        return self._computed_shift

    @property
    def use_fft(self) -> bool:
        """use FFT-based subpixel correction"""
        return self._use_fft

    @use_fft.setter
    def use_fft(self, value: bool) -> None:
        """set FFT mode"""
        self._use_fft = bool(value)

    @property
    def upsample(self) -> int:
        """upsampling factor for subpixel phase estimation"""
        return self._upsample

    @upsample.setter
    def upsample(self, value: int) -> None:
        """set upsampling factor"""
        self._upsample = max(1, int(value))

    @property
    def border(self) -> int:
        """border pixels to exclude from phase estimation"""
        return self._border

    @border.setter
    def border(self, value: int) -> None:
        """set border pixels"""
        self._border = max(0, int(value))

    @property
    def max_offset(self) -> int:
        """maximum phase offset to search"""
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value: int) -> None:
        """set maximum offset"""
        self._max_offset = max(1, int(value))

    @property
    def is_auto_shift(self) -> bool:
        """True if shift is auto-computed"""
        return self._shift is None

    def set_value(self, array, value) -> None:
        """
        Set phase correction settings.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : dict | bool
            settings dict or bool to enable/disable
        """
        old_value = self.value

        if isinstance(value, bool):
            self._enabled = value
        elif isinstance(value, dict):
            if "enabled" in value:
                self._enabled = bool(value["enabled"])
            if "method" in value:
                method = value["method"]
                if isinstance(method, str):
                    self._method = PhaseCorrMethod.from_string(method)
                else:
                    self._method = method
            if "shift" in value:
                self._shift = value["shift"]
            if "use_fft" in value:
                self._use_fft = bool(value["use_fft"])
            if "upsample" in value:
                self._upsample = max(1, int(value["upsample"]))
            if "border" in value:
                self._border = max(0, int(value["border"]))
            if "max_offset" in value:
                self._max_offset = max(1, int(value["max_offset"]))
        else:
            raise TypeError(f"expected dict or bool, got {type(value)}")

        new_value = self.value
        if old_value != new_value:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"value": new_value, "old_value": old_value},
            )
            self._call_event_handlers(event)

    def compute_shift(self, array, window_size: int = 100) -> float:
        """
        Compute phase shift from array data.

        Parameters
        ----------
        array : array-like
            the array to compute shift from
        window_size : int
            number of frames to use for 3D methods

        Returns
        -------
        float
            computed shift value
        """
        from mbo_utilities.analysis.phasecorr import bidir_phasecorr

        # get a window of frames
        n_frames = len(array)
        n_sample = min(window_size, n_frames)
        start = max(0, (n_frames - n_sample) // 2)

        if array.ndim == 4:
            # use first plane
            window = array[start : start + n_sample, 0]
        else:
            window = array[start : start + n_sample]

        # compute using phasecorr module
        corrected, shift = bidir_phasecorr(
            window,
            method=self._method.value if self._method != PhaseCorrMethod.none else "mean",
            return_shift=True,
            use_fft=self._use_fft,
        )

        self._computed_shift = shift
        return shift

    def apply(self, frame):
        """
        Apply phase correction to a frame.

        Parameters
        ----------
        frame : np.ndarray
            2D frame to correct

        Returns
        -------
        np.ndarray
            corrected frame
        """
        if not self._enabled:
            return frame

        shift = self.effective_shift
        if shift is None or abs(shift) < 0.01:
            return frame

        from mbo_utilities.analysis.phasecorr import _apply_offset

        return _apply_offset(frame, shift, use_fft=self._use_fft)

    def reset(self) -> None:
        """Reset to defaults."""
        self._enabled = False
        self._method = PhaseCorrMethod.mean
        self._shift = None
        self._computed_shift = None
        self._use_fft = False
        self._upsample = 10
        self._border = 10
        self._max_offset = 4

    def __repr__(self) -> str:
        if not self._enabled:
            return "PhaseCorrectionFeature(disabled)"
        shift_str = f"{self.effective_shift:.2f}" if self.effective_shift else "auto"
        return f"PhaseCorrectionFeature({self._method.value}, shift={shift_str})"
