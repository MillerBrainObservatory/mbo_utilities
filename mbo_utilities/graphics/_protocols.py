"""
Protocols for capability detection in mbo_utilities arrays and processors.

These protocols define the interface contracts for various features.
UI components use these to determine what controls to show.
"""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class SupportsRasterScan(Protocol):
    """Protocol for arrays/processors that support raster scan phase correction."""

    @property
    def fix_phase(self) -> bool:
        """Whether bidirectional phase correction is enabled."""
        ...

    @fix_phase.setter
    def fix_phase(self, value: bool) -> None:
        ...

    @property
    def use_fft(self) -> bool:
        """Whether FFT-based phase correlation is used."""
        ...

    @use_fft.setter
    def use_fft(self, value: bool) -> None:
        ...

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase correlation."""
        ...

    @border.setter
    def border(self, value: int) -> None:
        ...

    @property
    def max_offset(self) -> int:
        """Maximum pixel offset for phase correction."""
        ...

    @max_offset.setter
    def max_offset(self, value: int) -> None:
        ...

    @property
    def phase_upsample(self) -> int:
        """Upsampling factor for subpixel phase correlation."""
        ...

    @phase_upsample.setter
    def phase_upsample(self, value: int) -> None:
        ...


@runtime_checkable
class SupportsMetadata(Protocol):
    """Protocol for arrays that provide metadata."""

    @property
    def metadata(self) -> dict:
        """Metadata dictionary."""
        ...


@runtime_checkable
class SupportsROI(Protocol):
    """Protocol for arrays that support ROI selection."""

    @property
    def roi(self) -> int | None:
        """Current ROI index."""
        ...

    @roi.setter
    def roi(self, value: int | None) -> None:
        ...

    @property
    def rois(self) -> list | None:
        """Available ROI indices."""
        ...


@runtime_checkable
class SupportsPhaseOffset(Protocol):
    """Protocol for arrays/processors that expose phase offset values."""

    @property
    def offset(self) -> float | np.ndarray:
        """Current computed phase offset."""
        ...


# Helper functions for capability detection


def supports_raster_scan(obj) -> bool:
    """Check if an object supports raster scan phase correction."""
    return isinstance(obj, SupportsRasterScan)


def supports_metadata(obj) -> bool:
    """Check if an object provides metadata."""
    return isinstance(obj, SupportsMetadata)


def supports_roi(obj) -> bool:
    """Check if an object supports ROI selection."""
    return isinstance(obj, SupportsROI)


def supports_phase_offset(obj) -> bool:
    """Check if an object exposes phase offset values."""
    return isinstance(obj, SupportsPhaseOffset)


def get_capabilities(obj) -> set[str]:
    """
    Get all capabilities of an object.

    Returns a set of capability names that the object supports.
    """
    caps = set()
    if supports_raster_scan(obj):
        caps.add("raster_scan")
    if supports_metadata(obj):
        caps.add("metadata")
    if supports_roi(obj):
        caps.add("roi")
    if supports_phase_offset(obj):
        caps.add("phase_offset")
    return caps
