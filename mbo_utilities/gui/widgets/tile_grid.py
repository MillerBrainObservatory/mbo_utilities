"""TileGridViewer - fast preview of every tile of a tiled acquisition.

Tiled acquisitions fold spatial tiles into the T axis, so the main viewer
only shows one tile at a time on the scrollwheel. This widget lays all
tiles of a chosen z-block out in a grid (columns/rows from the per-tile
stage positions in ``metadata["tiles"]``) for a quick glimpse across the
whole acquisition.

Thumbnails are read straight from the raw data: a few evenly-spaced Z
planes, strided down in Y/X to ~128 px, max-projected. That decimated read
is ~1 ms/tile off a memmap (vs minutes to build full-resolution
projections), so the grid fills in immediately. Tiles load incrementally
(a budget per frame) so the UI never blocks, and clicking a cell drives the
main viewer's Tile slider.

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

# Z planes sampled per tile for the sparse max-projection thumbnail.
_THUMB_MIP_PLANES = 3

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

        self._sig: str | None = None
        self._grid: dict | None = None
        self._channel_names: list[str] = []

        # thumbnails keyed by (tile_index, c_index); GPU textures by the same
        self._thumb_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._gpu_cache: OrderedDict[tuple[int, int], _GpuImage] = OrderedDict()
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
        self._range = None
        self._range_sig = None

    def _thumb_params(self, arr) -> tuple[int, list[int]]:
        nz, ny, nx = int(arr.shape[2]), int(arr.shape[3]), int(arr.shape[4])
        stride = max(1, -(-max(ny, nx) // _THUMB_MAX))  # ceil
        n = max(1, min(_THUMB_MIP_PLANES, nz))
        planes = [int((i + 0.5) * nz / n) for i in range(n)]
        return stride, planes

    def _load_thumb(self, arr, ti: int, c: int) -> np.ndarray | None:
        """Decimated sparse-MIP read straight from the raw data (~1 ms)."""
        key = (ti, c)
        cached = self._thumb_cache.get(key)
        if cached is not None:
            self._thumb_cache.move_to_end(key)
            return cached
        stride, planes = self._thumb_params(arr)
        try:
            block = np.asarray(arr[ti, c, planes, ::stride, ::stride])
        except Exception:
            return None
        block = np.squeeze(block)
        if block.ndim == 3:
            thumb = block.max(axis=0)
        elif block.ndim == 2:
            thumb = block
        else:
            return None
        thumb = np.ascontiguousarray(thumb)
        self._thumb_cache[key] = thumb
        while len(self._thumb_cache) > _THUMB_CACHE_MAX:
            self._thumb_cache.popitem(last=False)
        return thumb

    def _contrast_range(self, arr, loaded: list[np.ndarray]) -> tuple[float, float]:
        if self._contrast_mode == _CONTRAST_DISPLAY:
            lo = float(getattr(arr, "_cached_vmin", 0.0) or 0.0)
            hi = float(getattr(arr, "_cached_vmax", 1000.0) or 1000.0)
            return lo, (hi if hi > lo else lo + 1.0)
        sig = (self._zblock, self._c_index, len(loaded))
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

    def _draw_grid(self, arr) -> None:
        g = self._grid
        ncols = max(1, len(g["cols"]))
        nrows = max(1, len(g["rows"]))
        placed = g["placed"].get(self._zblock, {})
        c = self._c_index

        loaded = [
            self._thumb_cache[(ti, c)]
            for (ti, _spc) in placed.values()
            if (ti, c) in self._thumb_cache
        ]
        lo, hi = self._contrast_range(arr, loaded)

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
                    thumb = self._thumb_cache.get((ti, c))
                    if thumb is None and budget > 0:
                        thumb = self._load_thumb(arr, ti, c)
                        budget -= 1
                    gpu = (
                        self._ensure_gpu((ti, c), thumb, lo, hi)
                        if thumb is not None else None
                    )
                    if gpu is not None:
                        draw_list.add_image(gpu.ref, pos, cmax)
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
