"""Per-view alignment widget for IsoView corrected / fused stacks.

Within each view two members image the same sample from opposite directions
(corrected: cameras 0,1 = VW00 and 2,3 = VW90; fused: VW00 and VW90). The
second member must be flipped / rotated to register onto the first. This
widget overlays the reference and the (re)oriented second member so the user
can dial in that orientation with the same Rotate X/Y/Z + Flip X/Y/Z controls
as the Tile Grid. The chosen orientation is saved per view (see
:mod:`mbo_utilities.gui._isoview_orient_state`).

Generic over kinds:
  - corrected: pairs (cam0 ref, cam1 adj) -> "VW00" and (cam2 ref, cam3 adj)
    -> "VW90".
  - fused: pair (VW00 ref, VW90 adj) -> "VW90" (VW90 mapped onto VW00).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import tifffile
from imgui_bundle import imgui

from mbo_utilities.gui import _isoview_orient_state as orient_state
from mbo_utilities.gui._imgui_helpers import draw_toolbar_row
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets._orient import (
    compose_R,
    draw_orient_row,
    orient_2d_plan,
    orientation_ops,
    apply_plan,
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
# 0 reference (ref alone), 1 target (2nd member alone), 2 overlay (both)
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


class IsoviewViewAlign(Widget):
    """Overlay each view's reference + reoriented second member to align them."""

    name = "View Align"
    priority = 64  # just above Projections (65) / Tile Grid (66)

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._popup_open = False
        self._axis_idx = 0
        self._tile = 0
        self._toggle = 2  # 0 reference, 1 target, 2 overlay (default)
        self._sig: str | None = None
        self._pairs: list[dict] = []
        self._projections: dict | None = None
        self._slots: list = []
        self._mip_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._rgba_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._gpu_cache: OrderedDict[tuple, _GpuImage] = OrderedDict()

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return cls._find(parent) is not None

    @staticmethod
    def _find(parent: Any):
        for raw in parent._get_data_arrays():
            arr = _unwrap(raw)
            if str(getattr(arr, "kind", "") or "").lower() not in ("corrected", "fused"):
                continue
            if int(getattr(arr, "num_views", 0) or 0) >= 2:
                return arr
        return None

    def _active_array(self):
        return self._find(self.parent)

    def _backend(self):
        try:
            return self.parent._figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _build_pairs(self, arr) -> list[dict]:
        """Reference/adjustable channel pairs for this array.

        ``c`` indices are positions on the array's C axis; ``view_key`` is the
        orient-store key (the adjustable member's view).
        """
        kind = str(getattr(arr, "kind", "") or "").lower()
        names = list(getattr(arr, "channel_names", []) or [])
        if kind == "fused":
            vw = {}
            for c, nm in enumerate(names):
                if "VW00" in nm:
                    vw[0] = c
                elif "VW90" in nm:
                    vw[90] = c
            if 0 in vw and 90 in vw:
                return [{
                    "view_key": "VW90", "ref_c": vw[0], "adj_c": vw[90],
                    "ref_label": "VW00", "adj_label": "VW90",
                }]
            return []
        # corrected: cameras on the C axis, grouped into views by camera_view_map
        views = list(getattr(arr, "views", []) or [])  # sorted camera indices
        cvm = (getattr(arr, "metadata", {}) or {}).get("camera_view_map") or {}
        if not cvm and len(views) >= 4:
            cvm = {views[0]: 0, views[1]: 0, views[2]: 90, views[3]: 90}
        pairs: list[dict] = []
        for vw_angle, vw_key in ((0, "VW00"), (90, "VW90")):
            cams = sorted(cam for cam, v in cvm.items() if int(v) == vw_angle)
            cams = [cam for cam in cams if cam in views]
            if len(cams) >= 2:
                from mbo_utilities.arrays.isoview.array import camera_view_label
                pairs.append({
                    "view_key": vw_key,
                    "ref_c": views.index(cams[0]), "adj_c": views.index(cams[1]),
                    "ref_label": camera_view_label(cams[0]),
                    "adj_label": camera_view_label(cams[1]),
                })
        return pairs

    def _ensure(self, arr) -> bool:
        sig = f"{id(arr)}:{getattr(arr, 'scan_root', '')}"
        if self._sig == sig and self._pairs:
            return True
        self._sig = sig
        self._reset_caches()
        self._pairs = self._build_pairs(arr)
        try:
            self._projections = arr.projections() if hasattr(arr, "projections") else None
        except Exception:
            self._projections = None
        slots = {k[2] for k in (self._projections or {}).get("files", {})}
        self._slots = sorted(slots, key=lambda s: (isinstance(s, str), s))
        nt = int(arr.shape[0])
        self._tile = min(self._tile, max(0, nt - 1))
        return bool(self._pairs)

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

    @staticmethod
    def _decimate(m: np.ndarray) -> np.ndarray:
        s = max(1, -(-max(m.shape) // _THUMB_MAX))
        return np.ascontiguousarray(m[::s, ::s])

    def _load_mips(self, arr, view_label: str, c: int, slot) -> dict | None:
        """Three MIPs for one camera/view, keyed by ``(view_label, slot)``.

        Loads the pipeline's precomputed xy/xz/yz projection TIFs (fast); only
        falls back to reading + projecting the full volume when no projection
        sidecar exists for this camera/tile.
        """
        key = (id(arr), view_label, slot)
        cached = self._mip_cache.get(key)
        if cached is not None:
            self._mip_cache.move_to_end(key)
            return cached
        mips = self._load_proj_mips(view_label, slot)
        if mips is None:
            mips = self._load_volume_mips(arr, c)
        if mips is None:
            return None
        self._mip_cache[key] = mips
        while len(self._mip_cache) > 24:
            self._mip_cache.popitem(last=False)
        return mips

    def _load_proj_mips(self, view_label: str, slot) -> dict | None:
        files = (self._projections or {}).get("files", {})
        if not files:
            return None
        mips: dict = {}
        for axis in _AXES:
            p = files.get((axis, view_label, slot))
            if p is None:
                return None
            try:
                m = np.asarray(tifffile.imread(str(p)))
            except Exception:
                return None
            if m.ndim != 2:
                m = np.squeeze(m)
            if m.ndim != 2:
                return None
            mips[axis] = self._decimate(m)
        return mips

    def _load_volume_mips(self, arr, c: int) -> dict | None:
        ny, nx = int(arr.shape[3]), int(arr.shape[4])
        s = max(1, -(-max(ny, nx) // _THUMB_MAX))
        try:
            block = np.asarray(arr[self._tile, c, :, ::s, ::s])
            if block.ndim != 3:
                block = np.squeeze(block)
            if block.ndim != 3:
                return None
            return {
                "xy": np.ascontiguousarray(block.max(axis=0)),
                "xz": np.ascontiguousarray(block.max(axis=1)),
                "yz": np.ascontiguousarray(block.max(axis=2)),
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

    def _composite(self, ref: np.ndarray, adj: np.ndarray, lo: float, hi: float) -> np.ndarray:
        adj = _resize_to(adj, ref.shape)
        r = self._norm(ref, lo, hi)
        a = self._norm(adj, lo, hi)
        if self._toggle == 0:        # reference only (gray)
            rgb = np.stack([r, r, r], axis=-1)
        elif self._toggle == 1:      # target only (gray)
            rgb = np.stack([a, a, a], axis=-1)
        else:                         # overlay: reference cyan, target red
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
        draw_section_header("View Align")
        imgui.indent(8)
        try:
            if imgui.button("Open##viewalign_open"):
                self._popup_open = True
            imgui.text_colored(_WHITE, orient_state.summary(arr))
        finally:
            imgui.unindent(8)
        if self._popup_open:
            self._draw_popup(arr)

    def _draw_toolbar(self, arr) -> None:
        def _axis():
            ch, v = imgui.combo("##viewalign_axis", self._axis_idx, list(_AXES))
            if ch:
                self._axis_idx = v

        def _show():
            ch, v = imgui.combo("##viewalign_show", self._toggle, list(_TOGGLE_LABELS))
            if ch:
                self._toggle = v

        def _flip():
            if imgui.small_button("flip##viewalign"):
                self._toggle = 0 if self._toggle == 2 else 2

        items = [("Axis", 90.0, _axis)]
        nt = int(arr.shape[0])
        if nt > 1:
            def _tile():
                ch, v = imgui.slider_int("##viewalign_tile", self._tile, 0, nt - 1)
                if ch:
                    self._tile = int(v)
            items.append(("Tile", 160.0, _tile))
        items.append(("Show", 120.0, _show))
        items.append((None, 48.0, _flip))
        draw_toolbar_row(items)
        imgui.text_colored(_WHITE, "overlay: reference = cyan, target = red")

    def _draw_pair(self, arr, pair: dict, cell: float) -> None:
        axis = _AXES[self._axis_idx]
        slot = self._slot_for_tile(arr)
        store = orient_state.get(arr, pair["view_key"])
        rotations = store["rotations"] if store else []
        flips = store["flips"] if store else []

        imgui.text_colored(
            _WHITE,
            f"{pair['view_key']}:  {pair['ref_label']} (ref) <- {pair['adj_label']} (target)",
        )
        draw_orient_row(rotations, flips, key=pair["view_key"])

        # Everything below is keyed by what affects the image; on a cache hit
        # (the common idle frame) no MIP is read and nothing is recomputed.
        ops = orientation_ops(rotations, flips)
        ops_sig = tuple(tuple(o) for o in ops)
        key = (pair["view_key"], axis, slot, self._toggle, ops_sig)
        rgba = self._rgba_cache.get(key)
        if rgba is None:
            ref_mips = self._load_mips(arr, pair["ref_label"], pair["ref_c"], slot)
            adj_mips = self._load_mips(arr, pair["adj_label"], pair["adj_c"], slot)
            if ref_mips is None or adj_mips is None:
                imgui.text_colored(_WHITE, "(projection unavailable)")
                return
            ref = self._oriented(ref_mips, [], axis)
            adj = self._oriented(adj_mips, ops, axis)
            lo, hi = _auto_range(np.concatenate([ref.ravel(), adj.ravel()]))
            rgba = self._composite(ref, adj, lo, hi)
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
        center_popup_on_open(default_em=(58.0, 56.0), min_em=(36.0, 28.0))
        opened, self._popup_open = imgui.begin(
            "View Align###viewalign_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            imgui.end()
            return
        self._draw_toolbar(arr)
        imgui.separator()
        avail = imgui.get_content_region_avail()
        npairs = max(1, len(self._pairs))
        # split the height across pairs; ~60px per pair for label + controls
        cell = max(150.0, min(avail.x - 16.0, avail.y / npairs - 64.0))
        imgui.begin_child("##viewalign_body", imgui.ImVec2(0, 0), child_flags=0)
        try:
            for i, pair in enumerate(self._pairs):
                if i:
                    imgui.separator()
                self._draw_pair(arr, pair, cell)
        finally:
            imgui.end_child()
        imgui.end()

    def cleanup(self) -> None:
        self._reset_caches()
