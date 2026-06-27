"""Canonical selection conversion shared by the GUI, lbm-suite2p-python, isoview.

One entry point that turns a per-axis selection (the kind of value the GUI
selection table and CLI produce) into canonical 0-based index lists keyed by the
canonical ``T``/``C``/``Z`` axis, plus thin adapters to the keyword names each
consumer expects. Keeps the 1-based-public / 0-based-internal rule and the
view/cam/plane/timepoint aliasing in one place.

- ``selection_to_canonical(arr, {"View": "1:2", "Z": "1:14"})`` ->
  ``{"C": [0, 1], "Z": [0..13]}`` (0-based).
- ``to_lsp_kwargs`` emits 1-based ``timepoints``/``planes``/``channels`` (the
  canonical selection API lbm-suite2p-python and imwrite share).
- ``to_isoview_kwargs`` emits 0-based ``timepoints`` and camera ints
  (``cameras`` == the C-axis indices the isoview pipeline expects).
"""
from __future__ import annotations

from mbo_utilities.arrays.features._dim_labels import _SLIDER_NAME_ALIASES
from mbo_utilities.arrays.features._slicing import parse_selection


def _to_canonical_axis(key: str) -> str | None:
    """Resolve any axis label/alias (``view``/``cam``/``plane``/``timepoint``...)
    to the canonical ``"T"``/``"C"``/``"Z"``, or ``None`` if it isn't one."""
    k = str(key).lower()
    if k in ("t", "c", "z"):
        return k.upper()
    for canon, aliases in _SLIDER_NAME_ALIASES.items():
        if k in aliases:
            return canon.upper()
    return None


def canonical_axis_sizes(arr) -> dict[str, int]:
    """``{"T","C","Z"}`` sizes from a 5D ``(T, C, Z, Y, X)`` lazy array."""
    shape = tuple(int(s) for s in arr.shape)
    t, c, z = (shape + (1, 1, 1))[:3]
    return {"T": t, "C": c, "Z": z}


def selection_to_canonical(arr, selections, one_based: bool = True) -> dict[str, list[int]]:
    """Map ``{axis: selection}`` to ``{"T"|"C"|"Z": [0-based indices]}``.

    ``axis`` may be a canonical key or any alias (``View``/``Cam``/``Channel``,
    ``Plane``, ``Timepoint``/``Tile``...). ``selection`` is anything
    :func:`parse_selection` accepts (``"1:10:2"``, int, list, ``None``). Axes
    omitted from ``selections`` (or given ``None``) are left out of the result;
    the caller treats a missing axis as "all".
    """
    sizes = canonical_axis_sizes(arr)
    out: dict[str, list[int]] = {}
    for key, sel in selections.items():
        canon = _to_canonical_axis(key)
        if canon is None or sel is None:
            continue
        out[canon] = parse_selection(sel, sizes[canon], one_based=one_based)
    return out


def to_lsp_kwargs(canonical: dict[str, list[int]]) -> dict[str, list[int]]:
    """Canonical 0-based indices -> lbm-suite2p-python / imwrite 1-based kwargs.

    Emits ``timepoints``/``planes``/``channels`` (1-based, the canonical
    selection API). Only present axes are emitted.
    """
    out: dict[str, list[int]] = {}
    if "T" in canonical:
        out["timepoints"] = [i + 1 for i in canonical["T"]]
    if "Z" in canonical:
        out["planes"] = [i + 1 for i in canonical["Z"]]
    if "C" in canonical:
        out["channels"] = [i + 1 for i in canonical["C"]]
    return out


def to_isoview_kwargs(canonical: dict[str, list[int]]) -> dict[str, list[int]]:
    """Canonical 0-based indices -> isoview pipeline kwargs (0-based).

    Emits 0-based ``timepoints`` and ``cameras`` (camera ints == C-axis
    indices). Only present axes are emitted.
    """
    out: dict[str, list[int]] = {}
    if "T" in canonical:
        out["timepoints"] = list(canonical["T"])
    if "C" in canonical:
        out["cameras"] = list(canonical["C"])
    return out
