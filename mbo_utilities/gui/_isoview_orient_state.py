"""Per-view orientation committed by the IsoView "Align views" widget.

The orientation (90-degree rotations + flips) a user commits via Apply is
attached to the raw acquisition root (the same key the crop store uses), so
corrected and fused views of one acquisition resolve to the same entry. Read by
the BigStitcher export to override its table default for that view.

Keyed by view label (``"VW00"``/``"VW90"``/...). The widget's seed defaults are
held in the widget, not here; only explicit Apply commits land in this store.
"""
from __future__ import annotations

from typing import Any

from mbo_utilities.gui._isoview_crop_state import _raw_root_for
from mbo_utilities.gui.widgets._orient import orientation_ops

# _APPLIED_STORE[<raw_root_str>][<view_key>] = {"rotations": [...], "flips": [...]}
_APPLIED_STORE: dict[str, dict[str, dict[str, list]]] = {}


def _key(arr: Any) -> str | None:
    rr = _raw_root_for(arr)
    return str(rr) if rr is not None else None


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
