"""
Per-slice statistics container.

SliceStats holds mean/std/SNR/min/max for a single slice (z-plane, camera,
roi, etc.).
"""

from __future__ import annotations

from typing import NamedTuple


class SliceStats(NamedTuple):
    """statistics for a single slice (plane, camera, roi, etc.)."""

    mean: float
    std: float
    snr: float
    min: float = 0.0
    max: float = 0.0


# backwards compatibility alias
PlaneStats = SliceStats
