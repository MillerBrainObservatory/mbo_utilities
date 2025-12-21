"""
Re-export availability flags from the root _installation module.

This module exists for backwards compatibility - imports should
preferably use `from mbo_utilities._installation import HAS_*`.
"""

from mbo_utilities._installation import (
    HAS_SUITE2P,
    HAS_SUITE3D,
    HAS_CUPY,
    HAS_TORCH,
    HAS_RASTERMAP,
    HAS_IMGUI,
    HAS_FASTPLOTLIB,
    HAS_PYSIDE6,
)

__all__ = [
    "HAS_SUITE2P",
    "HAS_SUITE3D",
    "HAS_CUPY",
    "HAS_TORCH",
    "HAS_RASTERMAP",
    "HAS_IMGUI",
    "HAS_FASTPLOTLIB",
    "HAS_PYSIDE6",
]
