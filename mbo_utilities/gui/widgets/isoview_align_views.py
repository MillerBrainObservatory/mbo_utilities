"""Align views widget for IsoView raw / corrected / fused stacks.

Overlays any two views in the VW00 reference frame so you can verify and tune
the orientation that maps a target view onto VW00 (the BigStitcher export
reference). The reference view is oriented by its export default; the target
view's rotations / flips are editable, seed from the same default, are made
isotropic, and overlaid on the reference. Apply commits the target's
orientation for the export to bake (see
:mod:`mbo_utilities.gui._isoview_orient_state`).

Replaces the former View Align + Camera Align widgets.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import tifffile
from imgui_bundle import imgui

from mbo_utilities.arrays.isoview.array import (
    camera_from_view_label,
    camera_view_label,
)
from mbo_utilities.gui import _isoview_orient_state as orient_state
from mbo_utilities.gui._imgui_helpers import draw_toolbar_row
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets._orient import (
    apply_plan,
    compose_R,
    draw_orient_row,
    ops_label,
    orient_2d_plan,
    orientation_ops,
)
from mbo_utilities.gui.widgets.summary_image import (
    _GpuImage,
    _auto_range,
    center_popup_on_open,
    draw_section_header,
)

_WHITE = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
_THUMB_MAX = 256
_AXES = ("xy", "xz", "yz")
# 0 reference alone, 1 target alone, 2 overlay (default).
_TOGGLE_LABELS = ("Reference", "Target", "Overlay")

# pre-multiplier that picks which oriented axis is projected (down its Z):
# xy -> Z, xz -> Y, yz -> X.
_SWAP = {
    "xy": np.eye(3),
    "xz": np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float),
    "yz": np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float),
}


def _unwrap(arr):
    seen: set[int] = set()
    while True:
        if id(arr) in seen:
            return arr
        seen.add(id(arr))
        inner = getattr(arr, "_arr", None)
        if inner is None or inner is arr:
            return arr
        arr = inner


def _resize_to(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if a.shape == shape:
        return a
    try:
        import cv2

        return cv2.resize(
            np.ascontiguousarray(a, dtype=np.float32),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    except Exception:
        ys = (np.linspace(0, a.shape[0] - 1, shape[0])).astype(int)
        xs = (np.linspace(0, a.shape[1] - 1, shape[1])).astype(int)
        return a[np.ix_(ys, xs)]


def _seed_for_view(cam: int, rotated: bool = False) -> dict:
    """Editable rotations/flips that map a view's camera onto VW00, from the
    export's CM->CM00 default, picking the Normal or Rotated mounting table."""
    try:
        from mbo_utilities.gui.widgets.pipelines.isoview import (
            _CM_ALIGN_DEFAULT,
            _CM_ALIGN_DEFAULT_ROTATED,
        )

        table = _CM_ALIGN_DEFAULT_ROTATED if rotated else _CM_ALIGN_DEFAULT
        seed = table.get(cam)
    except Exception:
        seed = None
    if not seed:
        return {"rotations": [], "flips": []}
    return {
        "rotations": [dict(r) for r in seed["rotations"]],
        "flips": list(seed["flips"]),
    }


class IsoviewAlignViews(Widget):
    """Overlay any reference + target view in the VW00 frame to align them."""

    name = "Align views"
    priority = 64

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._popup_open = False
        self._axis_idx = 0
        self._tile = 0
        self._toggle = 2  # 0 reference, 1 target, 2 overlay (default)
        self._ref_idx = 0
        self._target_idx = 1
        self._sig: str | None = None
        self._views_list: list[dict] = []  # {view_label, c, cam}, sorted by cam
        self._orient: dict[str, dict] = {}  # view_label -> editable {rotations, flips}
        self._is_rotated: bool = False
        self._projections: dict | None = None
        self._slots: list = []
        self._mip_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._rgba_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._gpu_cache: OrderedDict[tuple, _GpuImage] = OrderedDict()

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return cls._find(parent) is not None

    @staticmethod
    def _views(arr) -> list[dict]:
        """Available views as ``{view_label, c, cam}``, sorted by camera.

        Raw / corrected: each single-camera channel (``VW{angle}``). Fused: the
        ``VW00`` / ``VW90`` fused volumes. ``c`` is the C-axis index; ``cam`` is
        the underlying camera int that keys the CM->VW00 alignment table.
        """
        kind = str(getattr(arr, "kind", "") or "").lower()
        names = list(getattr(arr, "view_names", []) or [])
        out: list[dict] = []
        seen: set[str] = set()
        if kind == "fused":
            for c, nm in enumerate(names):
                label = "VW00" if "VW00" in nm else ("VW90" if "VW90" in nm else None)
                if label is None or label in seen:
                    continue
                cam = camera_from_view_label(label)
                if cam is None:
                    continue
                seen.add(label)
                out.append({"view_label": label, "c": c, "cam": cam})
        else:
            for c, nm in enumerate(names):
                if str(nm).endswith("_fused"):
                    continue
                cam = camera_from_view_label(nm)
                if cam is None:
                    continue
                label = camera_view_label(cam)
                if label in seen:
                    continue
                seen.add(label)
                out.append({"view_label": label, "c": c, "cam": cam})
        out.sort(key=lambda v: v["cam"])
        return out

    @classmethod
    def _find(cls, parent: Any):
        for raw in parent._get_data_arrays():
            arr = _unwrap(raw)
            if str(getattr(arr, "kind", "") or "").lower() not in (
                "raw",
                "corrected",
                "fused",
            ):
                continue
            if len(cls._views(arr)) >= 2:
                return arr
        return None

    def _active_array(self):
        return self._find(self.parent)

    def _backend(self):
        try:
            return self.parent._figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _ensure(self, arr) -> bool:
        sig = f"{id(arr)}:{getattr(arr, 'scan_root', '')}"
        if self._sig == sig and self._views_list:
            return True
        self._sig = sig
        self._reset_caches()
        self._orient = {}
        self._is_rotated = (
            str((getattr(arr, "metadata", {}) or {}).get("camera_orientation", ""))
            .strip()
            .lower()
            == "rotated"
        )
        views = self._views(arr)
        if len(views) < 2:
            self._views_list = []
            return False
        self._views_list = views
        n = len(views)
        self._ref_idx = next((i for i, v in enumerate(views) if v["cam"] == 0), 0)
        self._target_idx = next((i for i in range(n) if i != self._ref_idx), 0)
        try:
            self._projections = (
                arr.projections() if hasattr(arr, "projections") else None
            )
        except Exception:
            self._projections = None
        slots = {k[2] for k in (self._projections or {}).get("files", {})}
        self._slots = sorted(slots, key=lambda s: (isinstance(s, str), s))
        nt = int(arr.shape[0])
        self._tile = min(self._tile, max(0, nt - 1))
        return True

    def _target_state(self, view_label: str, cam: int) -> dict:
        st = self._orient.get(view_label)
        if st is None:
            st = _seed_for_view(cam, self._is_rotated)
            self._orient[view_label] = st
        return st

    def _ref_ops(self, cam: int) -> list:
        st = _seed_for_view(cam, self._is_rotated)
        return orientation_ops(st["rotations"], st["flips"])

    def _summary_text(self, arr) -> str:
        """Effective per-view orientation the export will bake: the Apply
        override when present, else the seed default. ``"(none)"`` only when
        every view resolves to identity."""
        parts = []
        for v in self._views_list:
            ops = orient_state.applied_ops(arr, v["view_label"])
            if ops is None:
                st = _seed_for_view(v["cam"], self._is_rotated)
                ops = orientation_ops(st["rotations"], st["flips"])
            if ops:
                parts.append(f"{v['view_label']}: {ops_label(ops)}")
        return ", ".join(parts) if parts else "(none)"

    def _slot_for_tile(self, arr):
        if self._slots:
            return self._slots[min(self._tile, len(self._slots) - 1)]
        tm = (getattr(arr, "tile_metadata", {}) or {}).get(self._tile, {})
        return tm.get("specimen_name", self._tile)

    def _reset_caches(self) -> None:
        for gpu in self._gpu_cache.values():
            gpu.destroy()
        self._gpu_cache.clear()
        self._rgba_cache.clear()
        self._mip_cache.clear()

    def _mip_params(self, arr) -> tuple[int, int]:
        """``(xy_stride, z_stride)`` so a decimated Z pixel spans about the same
        physical distance as a decimated Y/X pixel (isotropic thumbs)."""
        ny, nx = int(arr.shape[3]), int(arr.shape[4])
        s = max(1, -(-max(ny, nx) // _THUMB_MAX))  # ceil
        dxy = float(getattr(arr, "dx", 0.0) or 0.0)
        dz = float(getattr(arr, "dz", 0.0) or 0.0)
        sz = max(1, int(round(s * dxy / dz))) if dxy > 0 and dz > 0 else 1
        return s, sz

    def _load_mips(self, arr, view_label: str, c: int, slot) -> dict | None:
        """Three isotropic MIPs for one view, keyed by ``(view_label, slot)``.

        Loads the pipeline's precomputed xy/xz/yz projection TIFs (fast),
        falling back to reading + projecting the volume.
        """
        key = (id(arr), view_label, slot)
        cached = self._mip_cache.get(key)
        if cached is not None:
            self._mip_cache.move_to_end(key)
            return cached
        s, sz = self._mip_params(arr)
        mips = self._load_proj_mips(view_label, slot, s, sz)
        if mips is None:
            mips = self._load_volume_mips(arr, c, s, sz)
        if mips is None:
            return None
        self._mip_cache[key] = mips
        while len(self._mip_cache) > 24:
            self._mip_cache.popitem(last=False)
        return mips

    def _load_proj_mips(self, view_label: str, slot, s: int, sz: int) -> dict | None:
        files = (self._projections or {}).get("files", {})
        if not files:
            return None
        mips: dict = {}
        for axis in _AXES:
            p = files.get((axis, view_label, slot))
            if p is None:
                return None
            try:
                m = np.squeeze(np.asarray(tifffile.imread(str(p))))
            except Exception:
                return None
            if m.ndim != 2:
                return None
            # xy = (Y, X); xz = (Z, X); yz = (Z, Y). Z is axis 0 of xz/yz.
            m = m[::s, ::s] if axis == "xy" else m[::sz, ::s]
            mips[axis] = np.ascontiguousarray(m)
        return mips

    def _load_volume_mips(self, arr, c: int, s: int, sz: int) -> dict | None:
        try:
            block = np.asarray(arr[self._tile, c, :, ::s, ::s])
            block = np.squeeze(block)
            if block.ndim != 3:
                return None
            return {
                "xy": np.ascontiguousarray(block.max(axis=0)),
                "xz": np.ascontiguousarray(block.max(axis=1)[::sz]),
                "yz": np.ascontiguousarray(block.max(axis=2)[::sz]),
            }
        except Exception:
            return None

    @staticmethod
    def _oriented(mips: dict, ops: list, axis: str) -> np.ndarray:
        plan = orient_2d_plan(_SWAP[axis] @ compose_R(ops))
        return apply_plan(mips[plan["mip"]], plan)

    @staticmethod
    def _norm(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
        span = max(hi - lo, 1e-9)
        return np.clip((a.astype(np.float32) - lo) / span, 0.0, 1.0)

    def _composite(
        self, ref: np.ndarray, adj: np.ndarray, lo: float, hi: float
    ) -> np.ndarray:
        adj = _resize_to(adj, ref.shape)
        r = self._norm(ref, lo, hi)
        a = self._norm(adj, lo, hi)
        if self._toggle == 0:  # reference only (gray)
            rgb = np.stack([r, r, r], axis=-1)
        elif self._toggle == 1:  # target only (gray)
            rgb = np.stack([a, a, a], axis=-1)
        else:  # overlay: reference cyan, target red
            rgb = np.stack([a, r, r], axis=-1)
        return np.ascontiguousarray((rgb * 255).astype(np.uint8))

    def _ensure_gpu(self, key: tuple, rgba: np.ndarray) -> _GpuImage | None:
        backend = self._backend()
        if backend is None:
            return None
        gpu = self._gpu_cache.get(key)
        if gpu is None or gpu.arr is not rgba:
            if gpu is not None:
                gpu.destroy()
                self._gpu_cache.pop(key, None)
            gpu = _GpuImage(backend, rgba, "gray", 0.0, 255.0)
            self._gpu_cache[key] = gpu
            while len(self._gpu_cache) > 32:
                _, ev = self._gpu_cache.popitem(last=False)
                ev.destroy()
        else:
            self._gpu_cache.move_to_end(key)
        return gpu

    def draw(self) -> None:
        arr = self._active_array()
        if arr is None or not self._ensure(arr):
            return
        draw_section_header("Align views")
        imgui.indent(8)
        try:
            if imgui.button("Open##alignviews_open"):
                self._popup_open = True
            imgui.text_colored(_WHITE, self._summary_text(arr))
        finally:
            imgui.unindent(8)
        if self._popup_open:
            self._draw_popup(arr)

    def _draw_toolbar(self, arr) -> None:
        labels = [v["view_label"] for v in self._views_list]
        n = len(labels)

        def _ref():
            ch, v = imgui.combo("##alignviews_ref", self._ref_idx, labels)
            if ch:
                self._ref_idx = v
                if self._target_idx == v:
                    self._target_idx = next((i for i in range(n) if i != v), v)

        def _target():
            ch, v = imgui.combo("##alignviews_target", self._target_idx, labels)
            if ch:
                self._target_idx = v
                if self._ref_idx == v:
                    self._ref_idx = next((i for i in range(n) if i != v), v)

        def _axis():
            ch, v = imgui.combo("##alignviews_axis", self._axis_idx, list(_AXES))
            if ch:
                self._axis_idx = v

        def _show():
            ch, v = imgui.combo("##alignviews_show", self._toggle, list(_TOGGLE_LABELS))
            if ch:
                self._toggle = v

        items = [
            ("Reference", 90.0, _ref),
            ("Target", 90.0, _target),
            ("Axis", 90.0, _axis),
        ]
        nt = int(arr.shape[0])
        if nt > 1:

            def _tile():
                ch, v = imgui.slider_int("##alignviews_tile", self._tile, 0, nt - 1)
                if ch:
                    self._tile = int(v)

            items.append(("Tile", 160.0, _tile))
        items.append(("Show", 120.0, _show))
        draw_toolbar_row(items)
        ref_label = self._views_list[self._ref_idx]["view_label"]
        imgui.text_colored(_WHITE, f"overlay: {ref_label} (ref) = cyan, target = red")

    def _draw_overlay(self, arr, cell: float) -> None:
        axis = _AXES[self._axis_idx]
        slot = self._slot_for_tile(arr)
        ref = self._views_list[self._ref_idx]
        tgt = self._views_list[self._target_idx]
        st = self._target_state(tgt["view_label"], tgt["cam"])
        rotations, flips = st["rotations"], st["flips"]

        imgui.text_colored(
            _WHITE, f"{ref['view_label']} (ref)  <-  {tgt['view_label']} (target)"
        )
        draw_orient_row(rotations, flips, key="alignviews")
        imgui.same_line()
        if imgui.small_button("Default##alignviews_default"):
            self._orient[tgt["view_label"]] = _seed_for_view(
                tgt["cam"], self._is_rotated
            )
            st = self._orient[tgt["view_label"]]
            rotations, flips = st["rotations"], st["flips"]
        imgui.same_line()
        if imgui.small_button("Apply##alignviews_apply"):
            orient_state.apply(arr, tgt["view_label"], st)

        ref_ops = self._ref_ops(ref["cam"])
        ops = orientation_ops(rotations, flips)
        ops_sig = (
            tuple(tuple(o) for o in ref_ops),
            tuple(tuple(o) for o in ops),
        )
        key = (ref["view_label"], tgt["view_label"], axis, slot, self._toggle, ops_sig)
        rgba = self._rgba_cache.get(key)
        if rgba is None:
            ref_mips = self._load_mips(arr, ref["view_label"], ref["c"], slot)
            tgt_mips = self._load_mips(arr, tgt["view_label"], tgt["c"], slot)
            if ref_mips is None or tgt_mips is None:
                imgui.text_colored(_WHITE, "(projection unavailable)")
                return
            ref_img = self._oriented(ref_mips, ref_ops, axis)
            tgt_img = self._oriented(tgt_mips, ops, axis)
            lo, hi = _auto_range(np.concatenate([ref_img.ravel(), tgt_img.ravel()]))
            rgba = self._composite(ref_img, tgt_img, lo, hi)
            self._rgba_cache[key] = rgba
            while len(self._rgba_cache) > 24:
                self._rgba_cache.popitem(last=False)
        gpu = self._ensure_gpu(key, rgba)
        if gpu is None:
            return
        h, w = rgba.shape[:2]
        sc = min(cell / max(w, 1), cell / max(h, 1))
        dw, dh = w * sc, h * sc
        pos = imgui.get_cursor_screen_pos()
        imgui.get_window_draw_list().add_image(
            gpu.ref, pos, imgui.ImVec2(pos.x + dw, pos.y + dh)
        )
        imgui.dummy(imgui.ImVec2(dw, dh))

    def _draw_popup(self, arr) -> None:
        center_popup_on_open(default_em=(50.0, 48.0), min_em=(34.0, 26.0))
        opened, self._popup_open = imgui.begin(
            "Align views###alignviews_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            imgui.end()
            return
        self._draw_toolbar(arr)
        imgui.separator()
        avail = imgui.get_content_region_avail()
        cell = max(150.0, min(avail.x - 16.0, avail.y - 64.0))
        imgui.begin_child("##alignviews_body", imgui.ImVec2(0, 0), child_flags=0)
        try:
            self._draw_overlay(arr, cell)
        finally:
            imgui.end_child()
        imgui.end()

    def cleanup(self) -> None:
        self._reset_caches()
