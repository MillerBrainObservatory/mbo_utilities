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


def with_active(active: str | None, base: tuple[str, ...] = DEFAULT_COLORMAPS) -> list[str]:
    """Return the base list with `active` prepended if it's not already in it.

    Mirrors the pattern used by summary_image and the video dialog: keep a
    small curated list, but always include whatever cmap is currently active.
    """
    if not active:
        return list(base)
    if active in base:
        return list(base)
    return [active, *base]
