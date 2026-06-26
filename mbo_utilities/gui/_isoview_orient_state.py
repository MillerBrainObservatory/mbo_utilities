"""Per-view orientation store for the IsoView alignment widget.

The orientation (90-degree rotations + flips) that maps a view's adjustable
member onto its reference is attached to the raw acquisition root (the same
key the crop store uses), so corrected and fused views of one acquisition
resolve to the same entry.

Keyed by view label:
  - corrected: ``"VW00"`` (second camera onto first) and ``"VW90"`` (fourth
    camera onto third) — the adjustable member of each camera pair.
  - fused: ``"VW90"`` (the VW90 volume mapped onto VW00).
"""
from __future__ import annotations

from typing import Any

from mbo_utilities.gui._isoview_crop_state import _raw_root_for
from mbo_utilities.gui.widgets._orient import orientation_ops

# _ORIENT_STORE[<raw_root_str>][<view_key>] = {"rotations": [...], "flips": [...]}
_ORIENT_STORE: dict[str, dict[str, dict[str, list]]] = {}


def _key(arr: Any) -> str | None:
    rr = _raw_root_for(arr)
    return str(rr) if rr is not None else None


def get(arr: Any, view_key: str) -> dict | None:
    """Mutable ``{"rotations","flips"}`` entry for ``view_key``, created on
    first access. ``None`` when no stable per-acquisition key resolves."""
    k = _key(arr)
    if k is None:
        return None
    return _ORIENT_STORE.setdefault(k, {}).setdefault(
        view_key, {"rotations": [], "flips": []}
    )


def get_all(arr: Any) -> dict[str, dict[str, list]]:
    k = _key(arr)
    if k is None:
        return {}
    return {
        v: {"rotations": list(d["rotations"]), "flips": list(d["flips"])}
        for v, d in _ORIENT_STORE.get(k, {}).items()
    }


def clear(arr: Any) -> None:
    k = _key(arr)
    if k is not None:
        _ORIENT_STORE.pop(k, None)


def orientation_ops_for(arr: Any) -> dict[str, list]:
    """``{view_key: op_list}`` for every view with a non-identity orientation.

    Op lists are isoview's ``["rot", axis, deg]`` / ``["flip", axis]`` format,
    ready to feed a fusion / registration step.
    """
    out: dict[str, list] = {}
    for v, d in get_all(arr).items():
        ops = orientation_ops(d["rotations"], d["flips"])
        if ops:
            out[v] = ops
    return out


def summary(arr: Any) -> str:
    """One-line readout of the configured per-view orientations."""
    ops = orientation_ops_for(arr)
    if not ops:
        return "(none)"
    from mbo_utilities.gui.widgets._orient import ops_label

    return ", ".join(f"{v}: {ops_label(o)}" for v, o in sorted(ops.items()))


# Per-view orientation explicitly committed via the Align views "Apply" button.
# Read by the BigStitcher export to override its table default for that view.
_APPLIED_STORE: dict[str, dict[str, dict[str, list]]] = {}


def apply(arr: Any, view_key: str, state: dict[str, list]) -> None:
    k = _key(arr)
    if k is None:
        return
    _APPLIED_STORE.setdefault(k, {})[view_key] = {
        "rotations": [dict(r) for r in state.get("rotations", [])],
        "flips": list(state.get("flips", [])),
    }


def get_applied(arr: Any, view_key: str) -> dict | None:
    k = _key(arr)
    if k is None:
        return None
    return (_APPLIED_STORE.get(k) or {}).get(view_key)


def applied_ops(arr: Any, view_key: str) -> list | None:
    """Op list for a view's applied orientation, or ``None`` if never applied."""
    s = get_applied(arr, view_key)
    if s is None:
        return None
    return orientation_ops(s["rotations"], s["flips"])


def get_all_applied(arr: Any) -> dict:
    """``{view_key: {rotations, flips}}`` for every view the Align views widget
    has committed an orientation for (empty when none)."""
    k = _key(arr)
    if k is None:
        return {}
    return dict(_APPLIED_STORE.get(k) or {})
