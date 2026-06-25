"""Canonical colormap list shared by GUI widgets.

Both the summary-image widget and the video save dialog start from
DEFAULT_COLORMAPS and dynamically prepend the active fastplotlib cmap when
it falls outside this set, so any cmap-library name (gnuplot2, nipy_spectral,
etc.) still works after a Sync.
"""

from __future__ import annotations

DEFAULT_COLORMAPS: tuple[str, ...] = (
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "cividis",
    "gray",
    "turbo",
    "hot",
    "gnuplot2",
)

DEFAULT_COLORMAP: str = "viridis"
