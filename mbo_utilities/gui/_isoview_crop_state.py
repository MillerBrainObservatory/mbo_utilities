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
    """Resolve a stable per-acquisition key for ``arr``'s crops.

    Logically the key is the raw acquisition root (where the
    ``SPC##_TM##_*.stack`` files would live). When the raw stacks
    aren't on disk we synthesize that same path by walking ancestors
    of ``scan_root`` for the ``.corrected[_suffix]`` / ``.fused[_suffix]``
    directory and stripping the suffix — yielding the path the raw root
    WOULD have. This keeps the crop store keyed by acquisition identity
    regardless of which downstream tree the user opened.
    """
    if arr is None:
        return None
    kind = getattr(arr, "kind", None)
    if kind == "raw":
        sr = getattr(arr, "scan_root", None)
        return Path(sr) if sr is not None else None

    # First-choice: existing sibling raw root (works for legacy layouts
    # where the raw acquisition is still alongside the .corrected/ tree).
    try:
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root
        sibling = _sibling_raw_root(arr)
        if sibling is not None:
            return sibling
    except Exception:
        pass

    # Fallback: derive the canonical raw path by walking up from
    # scan_root, finding the .corrected*/.fused* ancestor, and stripping
    # that suffix. Stable even when the raw stacks have been deleted.
    sr = getattr(arr, "scan_root", None)
    if sr is None:
        return None
    scan_root = Path(sr)
    for ancestor in (scan_root, *scan_root.parents):
        n = ancestor.name
        for suf in (".corrected", ".fused"):
            idx = n.find(suf)
            if idx < 0:
                continue
            rest = n[idx + len(suf):]
            if rest == "" or rest.startswith("_"):
                return ancestor.parent / n[:idx]
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
