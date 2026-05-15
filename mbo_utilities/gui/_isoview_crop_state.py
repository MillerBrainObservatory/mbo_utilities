"""Per-dataset crop store for the IsoView pipeline.

The crop is logically attached to the RAW acquisition (the source the user
points correct_stack at). Both the corrected and fused outputs derive from
that raw root, so we key crops by the raw path — opening the corrected
or fused IsoviewArray later resolves to the same entry.

Two consumers:
- The Run tab's submit code: turns the store into
  ``crop_left``/``crop_top``/``crop_front``/``crop_width``/``crop_height``/
  ``crop_depth`` dicts that go into isoview's ``ProcessingConfig``.
- The standalone crop window: reads + writes per-view bounds as the user
  drags sliders.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# crops[<raw_root_str>][view_int] = {"z": (z0, z1), "y": (y0, y1), "x": (x0, x1),
#                                    "shape": (nz, ny, nx)}
_STORE: dict[str, dict[int, dict[str, tuple]]] = {}


def _raw_root_for(arr: Any) -> Path | None:
    """Map any IsoviewArray to the raw acquisition root that holds the
    ``SPC##_TM##_*.stack`` files.

    For ``kind="raw"`` it's ``arr.scan_root`` itself.
    For ``kind="corrected"`` / ``kind="fused"`` we use the existing
    :func:`mbo_utilities.arrays.isoview.array._sibling_raw_root` resolver,
    which already knows how to strip ``.corrected`` / ``.fused`` suffixes.
    """
    if arr is None:
        return None
    kind = getattr(arr, "kind", None)
    if kind == "raw":
        sr = getattr(arr, "scan_root", None)
        return Path(sr) if sr is not None else None
    try:
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root
        return _sibling_raw_root(arr)
    except Exception:
        return None


def _key(arr: Any) -> str | None:
    root = _raw_root_for(arr)
    return str(root) if root is not None else None


def get_crops(arr: Any) -> dict[int, dict[str, tuple]]:
    """Return the current crops for ``arr``'s raw root, or ``{}``."""
    k = _key(arr)
    if k is None:
        return {}
    return dict(_STORE.get(k, {}))


def set_view_bounds(
    arr: Any,
    view: int,
    *,
    z: tuple[int, int],
    y: tuple[int, int],
    x: tuple[int, int],
    shape: tuple[int, int, int],
) -> None:
    """Record ``view``'s crop. ``z``/``y``/``x`` are half-open ranges."""
    k = _key(arr)
    if k is None:
        return
    _STORE.setdefault(k, {})[int(view)] = {
        "z": (int(z[0]), int(z[1])),
        "y": (int(y[0]), int(y[1])),
        "x": (int(x[0]), int(x[1])),
        "shape": tuple(int(d) for d in shape),
    }


def clear(arr: Any) -> None:
    k = _key(arr)
    if k is not None:
        _STORE.pop(k, None)


def _is_full_extent(bounds: dict[str, tuple]) -> bool:
    """True iff every axis spans the full source shape (i.e. no real crop)."""
    nz, ny, nx = bounds.get("shape", (0, 0, 0))
    z0, z1 = bounds["z"]
    y0, y1 = bounds["y"]
    x0, x1 = bounds["x"]
    return (z0, z1) == (0, nz) and (y0, y1) == (0, ny) and (x0, x1) == (0, nx)


def to_config_args(arr: Any) -> dict[str, dict[int, int]]:
    """Project the per-view bounds onto isoview's ``ProcessingConfig`` fields.

    Returns up to six keys:
      crop_left / crop_top / crop_front  — starts
      crop_width / crop_height / crop_depth — spans
    Each is a ``{view_int: value}`` dict. Views whose bounds match the
    full source shape are omitted entirely (isoview leaves those fields
    None, falling back to the source dims).

    Returns ``{}`` when no crops are set (so callers can safely
    ``args.update(to_config_args(arr))``).
    """
    crops = get_crops(arr)
    if not crops:
        return {}

    left: dict[int, int] = {}
    top: dict[int, int] = {}
    front: dict[int, int] = {}
    width: dict[int, int] = {}
    height: dict[int, int] = {}
    depth: dict[int, int] = {}

    for view, b in crops.items():
        if _is_full_extent(b):
            continue
        z0, z1 = b["z"]
        y0, y1 = b["y"]
        x0, x1 = b["x"]
        left[view] = x0
        top[view] = y0
        front[view] = z0
        width[view] = x1 - x0
        height[view] = y1 - y0
        depth[view] = z1 - z0

    out: dict[str, dict[int, int]] = {}
    if left:
        out["crop_left"] = left
    if top:
        out["crop_top"] = top
    if front:
        out["crop_front"] = front
    if width:
        out["crop_width"] = width
    if height:
        out["crop_height"] = height
    if depth:
        out["crop_depth"] = depth
    return out


def summary(arr: Any) -> str:
    """Compact single-line summary for the Run-tab readout."""
    crops = get_crops(arr)
    if not crops:
        return "(none)"
    parts: list[str] = []
    for view in sorted(crops):
        b = crops[view]
        z0, z1 = b["z"]
        y0, y1 = b["y"]
        x0, x1 = b["x"]
        parts.append(
            f"VW{view:02d} z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}]"
            + ("" if not _is_full_extent(b) else " (full)")
        )
    return " | ".join(parts)
