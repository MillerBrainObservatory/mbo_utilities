"""TileGridViewer - fast preview of every tile of a tiled acquisition.

Tiled acquisitions fold spatial tiles into the T axis, so the main viewer
only shows one tile at a time on the scrollwheel. This widget lays all
tiles of a chosen z-block out in a grid (columns/rows from the per-tile
stage positions in ``metadata["tiles"]``) for a quick glimpse across the
whole acquisition.

Thumbnails are read straight from the raw data: the full depth strided
down in Y/X to ~128 px and max-projected. That decimated read is a few
ms/tile off a memmap (vs minutes to build full-resolution projections), so
the grid fills in over a few frames. Tiles load incrementally (a budget per
frame) so the UI never blocks, and clicking a cell drives the main viewer's
Tile slider.

An orientation control adds 90°-multiple rotations + X/Y/Z flips (the same
ops the BigStitcher export bakes onto VW00), applied as a live preview only
— nothing is saved. For a faithful out-of-plane view (rotate about X/Y, flip
Z) the displayed tile is the matching source MIP (xy/xz/yz) with a 2D
transpose + flips. One read per tile fills all three axes, so every
orientation — the default included — shows the same real projection.

Activates for any array that is tiled and carries per-tile metadata
(see :class:`mbo_utilities.arrays.isoview.IsoviewArray`).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from imgui_bundle import imgui

from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.summary_image import (
    _DEFAULT_COLORMAP,
    _DEFAULT_COLORMAPS,
    _GpuImage,
    _auto_range,
    center_popup_on_open,
    draw_section_header,
)

_WHITE = (1.0, 1.0, 1.0, 1.0)

# Longest thumbnail edge (px). The Y/X stride is chosen to land near this.
_THUMB_MAX = 128

# Uncached tiles read per frame, so opening a big z-block never blocks the
# UI — tiles pop in over a few frames.
_LOAD_PER_FRAME = 12

# Thumbnails are tiny (~32 KiB); cache enough to hold every tile of a large
# dataset (16x16x8 = 2048 tiles ~= 64 MiB) so re-navigation is instant.
_THUMB_CACHE_MAX = 4096
# GPU textures are bigger; bound to about one z-block plus slack.
_GPU_CACHE_MAX = 320

_CONTRAST_DISPLAY = 0
_CONTRAST_AUTO = 1
_CONTRAST_LABELS = ("Display", "Auto")


def _cluster_axis(values: list[float]) -> list[float]:
    """Group stage-coordinate values into ordered grid bins.

    Values within half the smallest gap collapse to one bin (absorbs
    acquisition jitter); the bin's mean is its representative center.
    """
    uniq = sorted({float(v) for v in values if v is not None})
    if not uniq:
        return []
    gaps = [b - a for a, b in zip(uniq, uniq[1:]) if b - a > 1e-9]
    tol = min(gaps) * 0.5 if gaps else 1.0
    clusters: list[list[float]] = [[uniq[0]]]
    for v in uniq[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _bin_index(centers: list[float], v: float | None) -> int:
    if not centers or v is None:
        return 0
    return int(np.argmin([abs(v - c) for c in centers]))


def _unwrap(arr):
    """Peel display proxies (`_SqueezeSingletonDims`, `_ScrubTimingProxy`)
    so reads hit the real array with unmodified 5D indexing.
    """
    seen: set[int] = set()
    while True:
        if id(arr) in seen:
            return arr
        seen.add(id(arr))
        inner = getattr(arr, "_arr", None)
        if inner is None or inner is arr:
            return arr
        arr = inner


def _axis_rot_local(axis: str, deg: float) -> np.ndarray:
    """3x3 right-hand rotation about X/Y/Z by a 90° multiple (mirrors
    isoview.views._axis_rotation; used only when isoview isn't importable)."""
    d = int(round(deg)) % 360
    c = {0: 1.0, 90: 0.0, 180: -1.0, 270: 0.0}[d]
    s = {0: 0.0, 90: 1.0, 180: 0.0, 270: -1.0}[d]
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _axis_flip_local(axis: str) -> np.ndarray:
    diag = {"X": (-1.0, 1, 1), "Y": (1, -1.0, 1), "Z": (1, 1, -1.0)}[axis]
    return np.diag(np.array(diag, dtype=float))


def _compose_R(ops: list) -> np.ndarray:
    """Composed 3x3 signed-permutation matrix for an orientation op list.

    Prefers isoview's own ``_orientation_affine`` so the preview matches
    exactly what the BigStitcher export bakes; falls back to a local build
    when isoview isn't importable. ``ops`` are ``["rot", axis, deg]`` /
    ``["flip", axis]`` entries (the format isoview accepts).
    """
    try:
        from isoview.views import _orientation_affine

        aff = _orientation_affine(ops)
        return np.eye(3) if aff is None else np.asarray(aff, dtype=float)[:3, :3]
    except Exception:
        R = np.eye(3)
        for op in ops:
            M = (
                _axis_rot_local(op[1], op[2])
                if op[0] == "rot"
                else _axis_flip_local(op[1])
            )
            R = M @ R
        return R


def _orient_2d_plan(R: np.ndarray) -> dict:
    """Reduce a 90°-multiple orientation to a 2D projection-display plan.

    For a signed axis permutation, the reoriented volume's max-projection
    down the new Z is one of the three source MIPs (xy/xz/yz) with an
    in-plane transpose + flips. Returns the source ``mip`` axis, whether to
    transpose, the horizontal/vertical flips, and which source axis
    (0=X, 1=Y, 2=Z) drives the displayed X/Y (so Z's anisotropy can be
    applied to the right screen axis).
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
    # source MIP layout in (row_axis, col_axis) of source-volume axes
    row_axis, _col_axis = {"xy": (1, 0), "xz": (2, 0), "yz": (2, 1)}[mip]
    return {
        "mip": mip,
        "transpose": row_axis == xsrc,  # want rows=ysrc, cols=xsrc
        "flip_h": sx < 0,
        "flip_v": sy < 0,
        "xsrc": xsrc,
        "ysrc": ysrc,
    }


def _ops_label(ops: list) -> str:
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


class TileGridViewer(Widget):
    """Preview all tiles of a tiled acquisition laid out per z-block."""

    name = "Tile Grid"
    priority = 66  # right after Projections (65)

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._popup_open: bool = False
        self._zblock: int = 0
        self._c_index: int = 0
        self._cmaps: list[str] = list(_DEFAULT_COLORMAPS)
        self._cmap_idx: int = self._cmaps.index(_DEFAULT_COLORMAP)
        self._contrast_mode: int = _CONTRAST_DISPLAY

        # live preview orientation (not persisted): rotations applied in
        # order, then flips. Composed to the same ops the export bakes.
        self._rotations: list[dict] = []
        self._flips: list[str] = []

        self._sig: str | None = None
        self._grid: dict | None = None
        self._channel_names: list[str] = []

        # source MIPs keyed by (ti, c, axis); oriented display thumbs and
        # GPU textures keyed by (ti, c, *plan_key) so changing orientation
        # swaps to its own cache slot instead of re-uploading every frame.
        self._mip_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._thumb_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._gpu_cache: OrderedDict[tuple, _GpuImage] = OrderedDict()
        self._range: tuple[float, float] | None = None
        self._range_sig: tuple | None = None

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return cls._find(parent) is not None

    @staticmethod
    def _find(parent: Any):
        for raw in parent._get_data_arrays():
            arr = _unwrap(raw)
            if bool(getattr(arr, "is_tiled", False)) and getattr(
                arr, "tile_metadata", None
            ):
                return arr
        return None

    def _active_array(self):
        return self._find(self.parent)

    def _backend(self):
        try:
            return self.parent._figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _ensure_grid(self, arr) -> bool:
        sig = str(getattr(arr, "scan_root", "")) or str(id(arr))
        if self._sig == sig and self._grid is not None:
            return True
        self._sig = sig
        self._reset_caches()
        self._channel_names = list(getattr(arr, "channel_names", []) or [])

        tiles = arr.tile_metadata or {}
        entries = []
        for ti, t in tiles.items():
            entries.append((
                int(ti), int(t.get("specimen", ti)),
                t.get("stage_x"), t.get("stage_y"), t.get("stage_z"),
            ))
        if not entries:
            self._grid = None
            return False

        cols = _cluster_axis([e[2] for e in entries])
        rows = _cluster_axis([e[3] for e in entries])
        zblocks = _cluster_axis([e[4] for e in entries])

        placed: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}
        for ti, spc, x, y, z in entries:
            zi = _bin_index(zblocks, z)
            ri = _bin_index(rows, y)
            ci = _bin_index(cols, x)
            placed.setdefault(zi, {})[(ri, ci)] = (ti, spc)

        self._grid = {
            "cols": cols, "rows": rows, "zblocks": zblocks, "placed": placed,
            "ntiles": len(entries),
        }
        self._zblock = min(self._zblock, max(0, len(zblocks) - 1))
        return True

    def _reset_caches(self) -> None:
        for gpu in self._gpu_cache.values():
            gpu.destroy()
        self._gpu_cache.clear()
        self._thumb_cache.clear()
        self._mip_cache.clear()
        self._range = None
        self._range_sig = None

    def _orientation_ops(self) -> list:
        """Current orientation as an op list: rotations first, then flips.

        Matches the order ``pipelines.isoview`` composes for the export, so
        the preview and the baked seed agree.
        """
        ops: list = []
        for rot in self._rotations:
            deg = int(rot["deg"])
            if rot["sign"] == "-":
                deg = -deg
            ops.append(["rot", rot["axis"], deg])
        for axis in self._flips:
            ops.append(["flip", axis])
        return ops

    def _mip_params(self, arr) -> tuple[int, int, float, float]:
        """``(xy_stride, z_stride, pix_xy_um, pix_z_um)`` for the MIP reads.

        ``z_stride`` is chosen so a decimated Z pixel spans about the same
        physical distance as a decimated Y/X pixel — that keeps xz/yz
        thumbnails close to square-pixelled. Falls back to no Z decimation
        and unit pixels when the voxel sizes aren't in the metadata.
        """
        ny, nx = int(arr.shape[3]), int(arr.shape[4])
        s = max(1, -(-max(ny, nx) // _THUMB_MAX))  # ceil
        md = getattr(arr, "metadata", {}) or {}
        dxy = float(md.get("dx") or md.get("pixel_resolution_um") or 0.0)
        dz = float(md.get("dz") or md.get("axial_step") or md.get("z_step") or 0.0)
        if dxy > 0 and dz > 0:
            sz = max(1, int(round(s * dxy / dz)))
            return s, sz, s * dxy, sz * dz
        return s, 1, 1.0, 1.0

    def _load_source_mip(self, arr, ti, c, axis) -> np.ndarray | None:
        """Decimated MIP for one axis, cached per ``(ti, c, axis)``.

        Maxes over the full depth (Y/X strided down to thumbnail size) so
        every orientation shows a real projection, and one read fills all
        three axes (xz/yz come for free from the same block). The default
        xy and the post-rotation xy are therefore the same image — no
        sparse-vs-full mismatch where the untransformed tile looks worse.
        """
        key = (ti, c, axis)
        cached = self._mip_cache.get(key)
        if cached is not None:
            self._mip_cache.move_to_end(key)
            return cached
        s, sz, _, _ = self._mip_params(arr)
        try:
            block = np.asarray(arr[ti, c, :, ::s, ::s])
            if block.ndim != 3:
                block = np.squeeze(block)
            if block.ndim != 3:
                return None
            self._mip_cache[(ti, c, "xy")] = np.ascontiguousarray(block.max(axis=0))
            self._mip_cache[(ti, c, "xz")] = np.ascontiguousarray(block.max(axis=1)[::sz])
            self._mip_cache[(ti, c, "yz")] = np.ascontiguousarray(block.max(axis=2)[::sz])
        except Exception:
            return None
        while len(self._mip_cache) > _THUMB_CACHE_MAX:
            self._mip_cache.popitem(last=False)
        return self._mip_cache.get(key)

    @staticmethod
    def _apply_plan(m: np.ndarray, plan: dict) -> np.ndarray:
        """Transpose + flip a source MIP into display orientation."""
        out = m
        if plan["transpose"]:
            out = out.T
        if plan["flip_v"]:
            out = out[::-1, :]
        if plan["flip_h"]:
            out = out[:, ::-1]
        return np.ascontiguousarray(out)

    def _contrast_range(
        self, arr, loaded: list[np.ndarray], plan_key: tuple
    ) -> tuple[float, float]:
        if self._contrast_mode == _CONTRAST_DISPLAY:
            lo = float(getattr(arr, "_cached_vmin", 0.0) or 0.0)
            hi = float(getattr(arr, "_cached_vmax", 1000.0) or 1000.0)
            return lo, (hi if hi > lo else lo + 1.0)
        sig = (self._zblock, self._c_index, plan_key, len(loaded))
        if sig == self._range_sig and self._range is not None:
            return self._range
        if loaded:
            sample = np.concatenate([t.ravel() for t in loaded])
            self._range = _auto_range(sample)
        else:
            self._range = (0.0, 1.0)
        self._range_sig = sig
        return self._range

    def _ensure_gpu(self, key: tuple[int, int], thumb: np.ndarray,
                    lo: float, hi: float) -> _GpuImage | None:
        backend = self._backend()
        if backend is None:
            return None
        cmap = self._cmaps[self._cmap_idx]
        gpu = self._gpu_cache.get(key)
        if gpu is None or gpu.arr is not thumb:
            if gpu is not None:
                gpu.destroy()
                self._gpu_cache.pop(key, None)
            gpu = _GpuImage(backend, thumb, cmap, lo, hi)
            self._gpu_cache[key] = gpu
            while len(self._gpu_cache) > _GPU_CACHE_MAX:
                _, evicted = self._gpu_cache.popitem(last=False)
                evicted.destroy()
        else:
            gpu.reupload_if_changed(cmap, lo, hi)
            self._gpu_cache.move_to_end(key)
        return gpu

    def _tile_slider_name(self) -> str | None:
        iw = getattr(self.parent, "image_widget", None)
        if iw is None:
            return None
        try:
            from mbo_utilities.arrays.features import find_slider_name
            names = getattr(iw, "_slider_dim_names", None) or ()
            return find_slider_name(names, "t")
        except Exception:
            return None

    def _current_tile(self) -> int | None:
        iw = getattr(self.parent, "image_widget", None)
        name = self._tile_slider_name()
        if iw is None or name is None:
            return None
        try:
            return int(iw.indices[name])
        except Exception:
            return None

    def _jump_to_tile(self, ti: int) -> None:
        iw = getattr(self.parent, "image_widget", None)
        name = self._tile_slider_name()
        if iw is None or name is None:
            return
        try:
            iw.indices[name] = int(ti)
        except Exception:
            pass

    def draw(self) -> None:
        arr = self._active_array()
        if arr is None or not self._ensure_grid(arr):
            return
        draw_section_header("Tile Grid")
        imgui.indent(8)
        try:
            if imgui.button("Open grid##tilegrid_open"):
                self._popup_open = True
            g = self._grid
            ncols = max(1, len(g["cols"]))
            nrows = max(1, len(g["rows"]))
            nz = max(1, len(g["zblocks"]))
            imgui.text_colored(_WHITE, f"{g['ntiles']} tiles")
            imgui.text_colored(_WHITE, f"grid: {ncols} x {nrows}")
            imgui.text_colored(_WHITE, f"z-blocks: {nz}")
        finally:
            imgui.unindent(8)

        if self._popup_open:
            self._draw_popup(arr)

    def _zblock_range(self, i: int) -> tuple[float, float]:
        """Stage-Z interval covered by z-block ``i``: from this block's
        position to the next block's (the last block mirrors the prior gap).
        """
        zb = self._grid["zblocks"]
        if not zb:
            return (0.0, 0.0)
        if len(zb) == 1:
            return (zb[0], zb[0])
        if i < len(zb) - 1:
            return (zb[i], zb[i + 1])
        return (zb[i], zb[i] + (zb[i] - zb[i - 1]))

    def _draw_toolbar(self) -> None:
        g = self._grid
        zblocks = g["zblocks"]
        nz = len(zblocks)
        if nz > 1:
            lo, hi = self._zblock_range(self._zblock)
            imgui.set_next_item_width(300)
            changed, new_z = imgui.slider_int(
                "Z-block##tilegrid", self._zblock, 0, nz - 1,
                f"%d / {nz - 1}   z {lo:.0f}-{hi:.0f} um",
            )
            if changed:
                self._zblock = int(new_z)
        else:
            lo, _ = self._zblock_range(0)
            imgui.text_colored(_WHITE, f"single z-block  (z {lo:.0f} um)")

        if len(self._channel_names) > 1:
            imgui.same_line()
            imgui.set_next_item_width(140)
            c_changed, new_c = imgui.combo(
                "View##tilegrid", self._c_index, list(self._channel_names)
            )
            if c_changed:
                self._c_index = new_c

        imgui.same_line()
        imgui.set_next_item_width(110)
        cmap_changed, new_cmap = imgui.combo(
            "Cmap##tilegrid", self._cmap_idx, list(self._cmaps)
        )
        if cmap_changed:
            self._cmap_idx = new_cmap

        imgui.same_line()
        imgui.set_next_item_width(90)
        ctr_changed, new_ctr = imgui.combo(
            "Contrast##tilegrid", self._contrast_mode, list(_CONTRAST_LABELS)
        )
        if ctr_changed:
            self._contrast_mode = new_ctr

        self._draw_orient_row()

    def _orient_toggle(self, label: str, active: bool) -> bool:
        """Highlighted small button; returns True when clicked."""
        if active:
            imgui.push_style_color(
                imgui.Col_.button, imgui.ImVec4(0.20, 0.45, 0.85, 1.0))
            imgui.push_style_color(
                imgui.Col_.button_hovered, imgui.ImVec4(0.26, 0.52, 0.92, 1.0))
            imgui.push_style_color(
                imgui.Col_.button_active, imgui.ImVec4(0.16, 0.38, 0.75, 1.0))
        clicked = imgui.small_button(label)
        if active:
            imgui.pop_style_color(3)
        return clicked

    def _draw_orient_row(self) -> None:
        """Preview-only orientation: 90° rotations + X/Y/Z flips (not saved)."""
        imgui.align_text_to_frame_padding()
        imgui.text("Rotate")
        for axis in ("X", "Y", "Z"):
            imgui.same_line()
            if imgui.small_button(f"+{axis}##rot_{axis}"):
                self._rotations.append({"sign": "+", "axis": axis, "deg": 90})
        for axis in ("X", "Y", "Z"):
            imgui.same_line()
            if imgui.small_button(f"-{axis}##rotn_{axis}"):
                self._rotations.append({"sign": "-", "axis": axis, "deg": 90})

        imgui.same_line()
        imgui.text("Flip")
        for axis in ("X", "Y", "Z"):
            imgui.same_line()
            active = axis in self._flips
            if self._orient_toggle(f"{axis}##flip_{axis}", active):
                if active:
                    self._flips.remove(axis)
                else:
                    self._flips.append(axis)

        imgui.same_line()
        if imgui.small_button("Reset##orient"):
            self._rotations = []
            self._flips = []
        imgui.same_line()
        imgui.text_colored(_WHITE, _ops_label(self._orientation_ops()))

    def _draw_grid(self, arr) -> None:
        g = self._grid
        ncols = max(1, len(g["cols"]))
        nrows = max(1, len(g["rows"]))
        placed = g["placed"].get(self._zblock, {})
        c = self._c_index

        ops = self._orientation_ops()
        plan = _orient_2d_plan(_compose_R(ops))
        plan_key = (plan["mip"], plan["transpose"], plan["flip_h"], plan["flip_v"])
        _s, _sz, pix_xy, pix_z = self._mip_params(arr)

        loaded = [
            self._thumb_cache[(ti, c, *plan_key)]
            for (ti, _spc) in placed.values()
            if (ti, c, *plan_key) in self._thumb_cache
        ]
        lo, hi = self._contrast_range(arr, loaded, plan_key)

        avail = imgui.get_content_region_avail()
        spacing = 4.0
        cell = max(48.0, (avail.x - spacing * (ncols - 1)) / ncols)
        cur_tile = self._current_tile()

        grey = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.25, 0.25, 0.25, 1.0))
        dark = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.08, 0.08, 0.08, 1.0))
        yellow = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.85, 0.2, 1.0))

        budget = _LOAD_PER_FRAME
        imgui.begin_child("##tilegrid_canvas", imgui.ImVec2(0, 0), child_flags=0)
        draw_list = imgui.get_window_draw_list()
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(spacing, spacing))
        try:
            for ri in range(nrows):
                for ci in range(ncols):
                    if ci > 0:
                        imgui.same_line()
                    pos = imgui.get_cursor_screen_pos()
                    cmax = imgui.ImVec2(pos.x + cell, pos.y + cell)
                    clicked = imgui.invisible_button(
                        f"##cell_{ri}_{ci}", imgui.ImVec2(cell, cell)
                    )
                    entry = placed.get((ri, ci))
                    if entry is None:
                        draw_list.add_rect(pos, cmax, grey)
                        continue
                    ti, spc = entry
                    gkey = (ti, c, *plan_key)
                    thumb = self._thumb_cache.get(gkey)
                    if thumb is None and budget > 0:
                        src = self._load_source_mip(arr, ti, c, plan["mip"])
                        if src is not None:
                            thumb = self._apply_plan(src, plan)
                            self._thumb_cache[gkey] = thumb
                            while len(self._thumb_cache) > _THUMB_CACHE_MAX:
                                self._thumb_cache.popitem(last=False)
                        budget -= 1
                    gpu = (
                        self._ensure_gpu(gkey, thumb, lo, hi)
                        if thumb is not None else None
                    )
                    if gpu is not None:
                        h, w = thumb.shape
                        psx = pix_z if plan["xsrc"] == 2 else pix_xy
                        psy = pix_z if plan["ysrc"] == 2 else pix_xy
                        pw = max(w * psx, 1e-6)
                        ph = max(h * psy, 1e-6)
                        sc = min(cell / pw, cell / ph)
                        dw, dh = pw * sc, ph * sc
                        ix = pos.x + (cell - dw) * 0.5
                        iy = pos.y + (cell - dh) * 0.5
                        draw_list.add_image(
                            gpu.ref,
                            imgui.ImVec2(ix, iy),
                            imgui.ImVec2(ix + dw, iy + dh),
                        )
                    else:
                        draw_list.add_rect_filled(pos, cmax, dark)
                    if clicked:
                        self._jump_to_tile(ti)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(f"SPM{spc:02d}  (tile {ti})")
                    is_current = ti == cur_tile
                    draw_list.add_rect(
                        pos, cmax, yellow if is_current else grey,
                        thickness=3.0 if is_current else 1.0,
                    )
                    self._draw_cell_label(draw_list, pos, spc)
        finally:
            imgui.pop_style_var(1)
            imgui.end_child()

    def _draw_cell_label(self, draw_list, pos, spc: int) -> None:
        txt = f"SPM{spc:02d}"
        tw = imgui.calc_text_size(txt)
        bg = imgui.color_convert_float4_to_u32(imgui.ImVec4(0, 0, 0, 0.55))
        draw_list.add_rect_filled(
            pos, imgui.ImVec2(pos.x + tw.x + 4, pos.y + tw.y + 2), bg,
        )
        draw_list.add_text(
            imgui.ImVec2(pos.x + 2, pos.y + 1),
            imgui.color_convert_float4_to_u32(imgui.ImVec4(*_WHITE)), txt,
        )

    def _draw_popup(self, arr) -> None:
        center_popup_on_open(default_em=(64.0, 60.0), min_em=(40.0, 30.0))
        opened, self._popup_open = imgui.begin(
            "Tile Grid###tilegrid_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            imgui.end()
            return
        self._draw_toolbar()
        imgui.separator()
        self._draw_grid(arr)
        imgui.end()

    def cleanup(self) -> None:
        self._reset_caches()
