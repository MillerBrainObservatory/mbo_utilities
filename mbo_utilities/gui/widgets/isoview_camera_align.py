"""Per-camera alignment preview for the BigStitcher export.

The BigStitcher export sends every camera reoriented onto CM00 (the
reference). This widget previews that mapping: CM00 is shown by default and
a selectable second camera (CM01/CM02/CM03) is rotated / flipped with the
same Rotate X/Y/Z + Flip X/Y/Z toolbar as the Tile Grid, made isotropic, and
overlaid on CM00 so you can dial in the orientation that registers it onto
CM00 (e.g. rotate CM02 90 about X, flip Z or not).

Preview only — nothing is saved. Each camera seeds from the export's default
CM->CM00 alignment (``pipelines.isoview._CM_ALIGN_DEFAULT``), so the overlay
opens at what the export would bake.

Activates for raw / corrected IsoviewArrays carrying >=2 single-camera
channels (CM##).
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

import numpy as np
import tifffile
from imgui_bundle import imgui

from mbo_utilities.arrays.isoview.array import (
    camera_from_view_label,
    camera_view_label,
)
from mbo_utilities.gui._imgui_helpers import draw_toolbar_row
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets._orient import (
    apply_plan,
    compose_R,
    draw_orient_row,
    orient_2d_plan,
    ops_label,
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
# 0 reference (CM00 alone), 1 target (the camera alone), 2 overlay (both)
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


def _seed_for_camera(cam: int, rotated: bool = False) -> dict:
    """Editable rotations/flips seeded from the export's CM->CM00 default,
    picking the Normal or Rotated table per the acquisition mounting."""
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


class IsoviewCameraAlign(Widget):
    """Overlay a selectable camera (reoriented) onto CM00 to align them."""

    name = "Camera Align"
    priority = 67  # right after Tile Grid (66)

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._popup_open = False
        self._axis_idx = 0
        self._tile = 0
        self._toggle = 2  # 0 reference, 1 target, 2 overlay (default)
        self._cam_idx = 0  # index into self._adj_cams
        self._sig: str | None = None
        self._ref_c: int = 0
        self._ref_label: str = "VW00"
        self._adj_cams: list[int] = []  # selectable camera ints (excludes ref)
        self._cam_c: dict[int, int] = {}  # camera int -> C-axis index
        self._orient: dict[int, dict] = {}  # camera int -> {rotations, flips}
        self._is_rotated: bool = False  # acquisition camera_orientation
        self._projections: dict | None = None
        self._slots: list = []
        self._mip_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._rgba_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._gpu_cache: OrderedDict[tuple, _GpuImage] = OrderedDict()

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return cls._find(parent) is not None

    @staticmethod
    def _cameras(arr) -> dict[int, int]:
        """``{camera_int: C-axis index}`` for single-camera views (``VW{angle}``).

        First channel index wins per camera. Fused pairs (``..._fused``) are
        skipped.
        """
        cams: dict[int, int] = {}
        for c, name in enumerate(getattr(arr, "channel_names", []) or []):
            if str(name).endswith("_fused"):
                continue
            cam = camera_from_view_label(name)
            if cam is not None:
                cams.setdefault(cam, c)
        return cams

    @classmethod
    def _find(cls, parent: Any):
        for raw in parent._get_data_arrays():
            arr = _unwrap(raw)
            if str(getattr(arr, "kind", "") or "").lower() not in ("raw", "corrected"):
                continue
            if len(cls._cameras(arr)) >= 2:
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
        if self._sig == sig and self._adj_cams:
            return True
        self._sig = sig
        self._reset_caches()
        self._orient = {}
        self._is_rotated = str(
            (getattr(arr, "metadata", {}) or {}).get("camera_orientation", "")
        ).strip().lower() == "rotated"
        cams = self._cameras(arr)
        if len(cams) < 2:
            self._adj_cams = []
            return False
        self._cam_c = cams
        ref = 0 if 0 in cams else min(cams)
        self._ref_c = cams[ref]
        self._ref_label = camera_view_label(ref)
        self._adj_cams = sorted(c for c in cams if c != ref)
        self._cam_idx = min(self._cam_idx, len(self._adj_cams) - 1)
        try:
            self._projections = arr.projections() if hasattr(arr, "projections") else None
        except Exception:
            self._projections = None
        slots = {k[2] for k in (self._projections or {}).get("files", {})}
        self._slots = sorted(slots, key=lambda s: (isinstance(s, str), s))
        nt = int(arr.shape[0])
        self._tile = min(self._tile, max(0, nt - 1))
        return True

    def _cam_state(self, cam: int) -> dict:
        st = self._orient.get(cam)
        if st is None:
            st = _seed_for_camera(cam, self._is_rotated)
            self._orient[cam] = st
        return st

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
        """``(xy_stride, z_stride)`` so a decimated Z pixel spans about the
        same physical distance as a decimated Y/X pixel (isotropic thumbs)."""
        ny, nx = int(arr.shape[3]), int(arr.shape[4])
        s = max(1, -(-max(ny, nx) // _THUMB_MAX))  # ceil
        dxy = float(getattr(arr, "dx", 0.0) or 0.0)
        dz = float(getattr(arr, "dz", 0.0) or 0.0)
        sz = max(1, int(round(s * dxy / dz))) if dxy > 0 and dz > 0 else 1
        return s, sz

    def _load_mips(self, arr, cam_label: str, c: int, slot) -> dict | None:
        """Three isotropic MIPs for one camera, keyed by ``(cam_label, slot)``.

        Loads the pipeline's precomputed xy/xz/yz projection TIFs (fast),
        falling back to reading + projecting the volume. The Z axis of the
        xz/yz MIPs is strided to match the lateral pixel pitch so a camera
        rotated into Z isn't squashed.
        """
        key = (id(arr), cam_label, slot)
        cached = self._mip_cache.get(key)
        if cached is not None:
            self._mip_cache.move_to_end(key)
            return cached
        s, sz = self._mip_params(arr)
        mips = self._load_proj_mips(cam_label, slot, s, sz)
        if mips is None:
            mips = self._load_volume_mips(arr, c, s, sz)
        if mips is None:
            return None
        self._mip_cache[key] = mips
        while len(self._mip_cache) > 24:
            self._mip_cache.popitem(last=False)
        return mips

    def _load_proj_mips(self, cam_label: str, slot, s: int, sz: int) -> dict | None:
        files = (self._projections or {}).get("files", {})
        if not files:
            return None
        mips: dict = {}
        for axis in _AXES:
            p = files.get((axis, cam_label, slot))
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
        draw_section_header("Camera Align")
        imgui.indent(8)
        try:
            if imgui.button("Open##camalign_open"):
                self._popup_open = True
            cam = self._adj_cams[self._cam_idx]
            st = self._cam_state(cam)
            label = ops_label(orientation_ops(st["rotations"], st["flips"]))
            imgui.text_colored(_WHITE, f"{camera_view_label(cam)}: {label}")
        finally:
            imgui.unindent(8)
        if self._popup_open:
            self._draw_popup(arr)

    def _draw_toolbar(self, arr) -> None:
        cam_labels = [camera_view_label(c) for c in self._adj_cams]

        def _camera():
            ch, v = imgui.combo("##camalign_cam", self._cam_idx, cam_labels)
            if ch:
                self._cam_idx = v

        def _axis():
            ch, v = imgui.combo("##camalign_axis", self._axis_idx, list(_AXES))
            if ch:
                self._axis_idx = v

        def _show():
            ch, v = imgui.combo("##camalign_show", self._toggle, list(_TOGGLE_LABELS))
            if ch:
                self._toggle = v

        items = [("Camera", 90.0, _camera), ("Axis", 90.0, _axis)]
        nt = int(arr.shape[0])
        if nt > 1:
            def _tile():
                ch, v = imgui.slider_int("##camalign_tile", self._tile, 0, nt - 1)
                if ch:
                    self._tile = int(v)
            items.append(("Tile", 160.0, _tile))
        items.append(("Show", 120.0, _show))
        draw_toolbar_row(items)
        imgui.text_colored(
            _WHITE, f"overlay: {self._ref_label} (ref) = cyan, target = red"
        )

    def _draw_overlay(self, arr, cam: int, cell: float) -> None:
        axis = _AXES[self._axis_idx]
        slot = self._slot_for_tile(arr)
        adj_c = self._cam_c[cam]
        adj_label = camera_view_label(cam)
        st = self._cam_state(cam)
        rotations, flips = st["rotations"], st["flips"]

        imgui.text_colored(
            _WHITE, f"{self._ref_label} (ref)  <-  {adj_label} (target)"
        )
        draw_orient_row(rotations, flips, key="camalign")
        imgui.same_line()
        if imgui.small_button("Default##camalign_default"):
            self._orient[cam] = _seed_for_camera(cam, self._is_rotated)
            st = self._orient[cam]
            rotations, flips = st["rotations"], st["flips"]

        ops = orientation_ops(rotations, flips)
        ops_sig = tuple(tuple(o) for o in ops)
        key = (cam, axis, slot, self._toggle, ops_sig)
        rgba = self._rgba_cache.get(key)
        if rgba is None:
            ref_mips = self._load_mips(arr, self._ref_label, self._ref_c, slot)
            adj_mips = self._load_mips(arr, adj_label, adj_c, slot)
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
        center_popup_on_open(default_em=(50.0, 48.0), min_em=(34.0, 26.0))
        opened, self._popup_open = imgui.begin(
            "Camera Align###camalign_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            imgui.end()
            return
        self._draw_toolbar(arr)
        imgui.separator()
        cam = self._adj_cams[self._cam_idx]
        avail = imgui.get_content_region_avail()
        cell = max(150.0, min(avail.x - 16.0, avail.y - 64.0))
        imgui.begin_child("##camalign_body", imgui.ImVec2(0, 0), child_flags=0)
        try:
            self._draw_overlay(arr, cam, cell)
        finally:
            imgui.end_child()
        imgui.end()

    def cleanup(self) -> None:
        self._reset_caches()
