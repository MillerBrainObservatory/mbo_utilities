"""IsoView readers, writers, and helpers.

Public surface in one place — every symbol below is intended to
eventually move to ``~/repos/isoview`` as the standalone home for
IsoView pipeline integration. Until then this package is the
canonical location inside ``mbo_utilities``.

Public API:

- :class:`IsoviewArray` — lazy ``(T, C, Z, Y, X)`` reader for any of
  the four IsoView output trees (corrected / fused / raw / clusterpt).
- :func:`detect_isoview_kind` — auto-detect which kind a path belongs to.
- :func:`consolidate_isoview` — collapse one output tree into a single
  OME-NGFF 0.5 zarr group.
- :func:`to_bigstitcher` — produce a transient Zarr v2 / OME-NGFF 0.4
  mirror of a consolidated zarr for BigStitcher import.
- :func:`isoview_to_ome_zarr` — thin convenience wrapper to write any
  kind to OME-Zarr without the full consolidation step.

Internals (not exported, available via the submodules):

- ``.array``       — lazy reader, scanners, regexes, XML metadata
- ``.consolidate`` — writer, pyramid math, companion-file logic

See :doc:`README.md` for the on-disk file inventory and consolidated
OME-Zarr layout.
"""

from __future__ import annotations

from mbo_utilities.arrays.isoview.array import (
    ISOVIEW_ZARR_CHUNK_ZYX,
    IsoviewArray,
    detect_isoview_kind,
    isoview_to_ome_zarr,
    isoview_zarr_chunks,
    make_raw_projections,
)
from mbo_utilities.arrays.isoview.consolidate import (
    consolidate_isoview,
    to_bigstitcher,
)

__all__ = [
    "ISOVIEW_ZARR_CHUNK_ZYX",
    "IsoviewArray",
    "consolidate_isoview",
    "detect_isoview_kind",
    "isoview_to_ome_zarr",
    "isoview_zarr_chunks",
    "make_raw_projections",
    "to_bigstitcher",
]
