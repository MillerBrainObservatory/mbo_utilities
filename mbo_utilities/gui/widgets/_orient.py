"""Shared 90-degree orientation helpers (rotations + flips).

Used by the Tile Grid preview and the per-view alignment widget. Rotations
are applied first, then flips, matching the order ``pipelines.isoview``
composes for the BigStitcher export, so previews and any baked seed agree.

An orientation is an op list of ``["rot", axis, deg]`` / ``["flip", axis]``
entries (axis in ``"X"/"Y"/"Z"``, deg a 90-degree multiple). The live UI
state is two lists: ``rotations`` (dicts ``{"sign","axis","deg"}``) and
``flips`` (axis strings); :func:`orientation_ops` turns them into op lists.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

_WHITE = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)


def axis_rotation(axis: str, deg: float) -> np.ndarray:
    """3x3 right-hand rotation about X/Y/Z by a 90-degree multiple."""
    d = int(round(deg)) % 360
    c = {0: 1.0, 90: 0.0, 180: -1.0, 270: 0.0}[d]
    s = {0: 0.0, 90: 1.0, 180: 0.0, 270: -1.0}[d]
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def axis_flip(axis: str) -> np.ndarray:
    diag = {"X": (-1.0, 1, 1), "Y": (1, -1.0, 1), "Z": (1, 1, -1.0)}[axis]
    return np.diag(np.array(diag, dtype=float))


def compose_R(ops: list) -> np.ndarray:
    """Composed 3x3 signed-permutation matrix for an orientation op list.

    Prefers isoview's own ``_orientation_affine`` so the preview matches
    exactly what the BigStitcher export bakes; falls back to a local build
    when isoview isn't importable.
    """
    try:
        from isoview.views import _orientation_affine

        aff = _orientation_affine(ops)
        return np.eye(3) if aff is None else np.asarray(aff, dtype=float)[:3, :3]
    except Exception:
        R = np.eye(3)
        for op in ops:
            M = (
                axis_rotation(op[1], op[2])
                if op[0] == "rot"
                else axis_flip(op[1])
            )
            R = M @ R
        return R


def orient_2d_plan(R: np.ndarray) -> dict:
    """Reduce a 90-degree orientation to a 2D projection-display plan.

    For a signed axis permutation, the reoriented volume's max-projection
    down the new Z is one of the three source MIPs (xy/xz/yz) with an
    in-plane transpose + flips. Returns the source ``mip`` axis, whether to
    transpose, the horizontal/vertical flips, and which source axis
    (0=X, 1=Y, 2=Z) drives the displayed X/Y.
    """
    R = np.asarray(R, dtype=float)

    def _src(row: int) -> tuple[int, float]:
        j = int(np.argmax(np.abs(R[row])))
        return j, (1.0 if R[row, j] >= 0 else -1.0)

    try:
        xsrc, sx = _src(0)
        ysrc, sy = _src(1)
        zsrc, _sz = _src(2)
        if len({xsrc, ysrc, zsrc}) != 3:
            raise ValueError("not a clean axis permutation")
    except Exception:
        xsrc, ysrc, zsrc, sx, sy = 0, 1, 2, 1.0, 1.0

    mip = {2: "xy", 1: "xz", 0: "yz"}[zsrc]
    row_axis, _col_axis = {"xy": (1, 0), "xz": (2, 0), "yz": (2, 1)}[mip]
    return {
        "mip": mip,
        "transpose": row_axis == xsrc,
        "flip_h": sx < 0,
        "flip_v": sy < 0,
        "xsrc": xsrc,
        "ysrc": ysrc,
    }


def apply_plan(m: np.ndarray, plan: dict) -> np.ndarray:
    """Transpose + flip a source MIP into display orientation."""
    out = m
    if plan["transpose"]:
        out = out.T
    if plan["flip_v"]:
        out = out[::-1, :]
    if plan["flip_h"]:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def ops_label(ops: list) -> str:
    if not ops:
        return "identity"
    parts = []
    for op in ops:
        if op[0] == "rot":
            d = int(op[2])
            parts.append(f"{op[1]}{'+' if d >= 0 else ''}{d}")
        else:
            parts.append(f"flip{op[1]}")
    return " ".join(parts)


def orientation_ops(rotations: list, flips: list) -> list:
    """Op list from live UI state: rotations first, then flips."""
    ops: list = []
    for rot in rotations:
        deg = int(rot["deg"])
        if rot["sign"] == "-":
            deg = -deg
        ops.append(["rot", rot["axis"], deg])
    for axis in flips:
        ops.append(["flip", axis])
    return ops


def _orient_toggle(label: str, active: bool) -> bool:
    """Highlighted small button; returns True when clicked."""
    if active:
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.20, 0.45, 0.85, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.26, 0.52, 0.92, 1.0))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.16, 0.38, 0.75, 1.0))
    clicked = imgui.small_button(label)
    if active:
        imgui.pop_style_color(3)
    return clicked


def draw_orient_row(rotations: list, flips: list, key: str = "") -> bool:
    """Rotate X/Y/Z + Flip X/Y/Z control row.

    Mutates ``rotations`` / ``flips`` in place and returns True when an op
    changed this frame. ``key`` disambiguates imgui IDs when several rows are
    drawn in one frame (e.g. one per view).
    """
    changed = False
    imgui.align_text_to_frame_padding()
    imgui.text("Rotate")
    for axis in ("X", "Y", "Z"):
        imgui.same_line()
        if imgui.small_button(f"+{axis}##rot_{axis}_{key}"):
            rotations.append({"sign": "+", "axis": axis, "deg": 90})
            changed = True
    for axis in ("X", "Y", "Z"):
        imgui.same_line()
        if imgui.small_button(f"-{axis}##rotn_{axis}_{key}"):
            rotations.append({"sign": "-", "axis": axis, "deg": 90})
            changed = True

    imgui.same_line()
    imgui.text("Flip")
    for axis in ("X", "Y", "Z"):
        imgui.same_line()
        active = axis in flips
        if _orient_toggle(f"{axis}##flip_{axis}_{key}", active):
            if active:
                flips.remove(axis)
            else:
                flips.append(axis)
            changed = True

    imgui.same_line()
    if imgui.small_button(f"Reset##orient_{key}"):
        rotations.clear()
        flips.clear()
        changed = True
    imgui.same_line()
    imgui.text_colored(_WHITE, ops_label(orientation_ops(rotations, flips)))
    return changed
