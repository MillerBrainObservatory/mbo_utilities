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

import re
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
_CONTRAST_MANUAL = 2
_CONTRAST_LABELS = ("Display", "Auto", "Manual")


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


def _index_centers(entries: list, idx_pos: int, stage_pos: int, n: int) -> list[float]:
    """Representative stage coordinate for each integer tile index.

    Averages the stage coordinate of every tile sharing an index; falls
    back to the index itself when no stage coordinate is available (keeps
    the z-block um readout working when digit indices drive placement).
    """
    centers = []
    for i in range(n):
        vals = [e[stage_pos] for e in entries if e[idx_pos] == i and e[stage_pos] is not None]
        centers.append(sum(vals) / len(vals) if vals else float(i))
    return centers


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
        self._manual_lo: float = 0.0
        self._manual_hi: float = 300.0

        # view mode: 0 = max-intensity projection (whole tile), 1 = single plane
        # read lazily per tile. ``_plane`` is the z index within each tile.
        self._view_mode: int = 0
        self._plane: int = 0
        self._nplanes: int = 1

        # content flips (axis "X"=horizontal, "Y"=vertical) + rotation (90deg
        # CCW steps, 0-3), keyed by (camera index, tile index) so each camera
        # keeps its own orientation. Set per tile from its right-click menu;
        # cameras are seeded with default orientation (see _seed_camera_defaults).
        self._tile_flips: dict[tuple, set] = {}
        self._tile_rot: dict[tuple, int] = {}
        self._flip_seeded: set = set()         # cameras whose defaults were applied
        self._tile_xyz: dict[int, tuple] = {}  # ti -> (tile_x, tile_y, tile_z)
        # "Rotated" acquisitions mount every camera 90deg on its side, so each
        # tile's content needs a default 90deg rotation to tile upright.
        self._is_rotated: bool = False
        # tile whose rotate/flip menu is open (drawn top-level after the grid
        # child so the popup isn't clipped by the child's scroll rect).
        self._orient_menu_ti: "int | None" = None
        self._orient_menu_open: bool = False

        # layout per (camera, z-block): {(ri, ci): ti}; ti->spc for labels.
        # Seeded from stage placement (mirrored for odd cameras); "Move tiles"
        # mutates it. Keyed by camera so each keeps its own arrangement.
        self._layout: dict[tuple, dict] = {}
        self._tile_spc: dict[int, int] = {}
        self._edit_layout: bool = False
        self._pick: "tuple | None" = None

        self._sig: str | None = None
        self._grid: dict | None = None
        self._channel_names: list[str] = []
        self._tile_labels: dict[int, str] = {}

        # source MIPs keyed by (ti, c, axis); display thumbs and GPU textures
        # keyed by (ti, c, flip_x, flip_y) so flipping a tile swaps to its own
        # cache slot instead of re-uploading every frame.
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
        self._layout = {}
        self._tile_spc = {}
        self._tile_flips = {}
        self._tile_rot = {}
        self._flip_seeded = set()
        self._tile_xyz = {}
        self._pick = None
        self._channel_names = list(getattr(arr, "channel_names", []) or [])
        self._is_rotated = str(
            (getattr(arr, "metadata", {}) or {}).get("camera_orientation", "")
        ).strip().lower() == "rotated"
        try:
            self._nplanes = max(1, int(arr.shape[2]))  # Z planes per tile
        except Exception:
            self._nplanes = 1
        self._plane = min(self._plane, self._nplanes - 1)

        tiles = arr.tile_metadata or {}
        entries = []
        # cell display label: the specimen_name grid token when present, else SPM##
        self._tile_labels = {
            int(ti): (
                str(t.get("specimen_name"))
                if t.get("specimen_name")
                else f"SPM{int(t.get('specimen', ti)):02d}"
            )
            for ti, t in tiles.items()
        }
        for ti, t in tiles.items():
            entries.append((
                int(ti), int(t.get("specimen", ti)),
                t.get("stage_x"), t.get("stage_y"), t.get("stage_z"),
                t.get("tile_x"), t.get("tile_y"), t.get("tile_z"),
            ))
        if not entries:
            self._grid = None
            return False
        self._tile_xyz = {e[0]: (e[5], e[6], e[7]) for e in entries}

        placed: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}
        # digit-encoded grid (specimen_name trailing XYZ) is authoritative
        # when present on every tile; else cluster stage coordinates.
        use_digits = all(
            e[5] is not None and e[6] is not None and e[7] is not None
            for e in entries
        )
        # Display axes match the verified BigStitcher layout (STEP 9): BDV-x =
        # -stage_y, BDV-y = +stage_x. So tile_x / stage_x runs top->bottom =
        # rows, and tile_y / stage_y runs right->LEFT = columns reversed
        # (higher tile_y is further left). Result: TL010 sits LEFT of TL000
        # (TL000 top-right) and TL100 sits underneath it. Per-tile move handles
        # datasets whose stage/camera sign differs.
        if use_digits:
            ncols = max(e[6] for e in entries) + 1  # tile_y -> columns
            nrows = max(e[5] for e in entries) + 1  # tile_x -> rows
            nz = max(e[7] for e in entries) + 1
            cols = _index_centers(entries, 6, 3, ncols)[::-1]  # tile_y, reversed
            rows = _index_centers(entries, 5, 2, nrows)  # tile_x / stage_x
            zblocks = _index_centers(entries, 7, 4, nz)
            for ti, spc, x, y, z, tx, ty, tz in entries:
                ci = ncols - 1 - int(ty)  # higher tile_y -> further left
                placed.setdefault(int(tz), {})[(int(tx), ci)] = (ti, spc)
        else:
            cols_asc = _cluster_axis([e[3] for e in entries])  # stage_y asc
            cols = cols_asc[::-1]  # reversed: higher stage_y -> further left
            rows = _cluster_axis([e[2] for e in entries])  # stage_x -> rows
            zblocks = _cluster_axis([e[4] for e in entries])
            ncols = len(cols_asc)
            for ti, spc, x, y, z, *_rest in entries:
                zi = _bin_index(zblocks, z)
                ri = _bin_index(rows, x)
                ci = ncols - 1 - _bin_index(cols_asc, y)
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

    def _mip_params(self, arr) -> tuple[int, int, float, float]:
        """``(xy_stride, z_stride, pix_xy_um, pix_z_um)`` for the MIP reads.

        ``z_stride`` is chosen so a decimated Z pixel spans about the same
        physical distance as a decimated Y/X pixel — that keeps xz/yz
        thumbnails close to square-pixelled. Falls back to no Z decimation
        and unit pixels when the voxel sizes aren't in the metadata.
        """
        ny, nx = int(arr.shape[3]), int(arr.shape[4])
        s = max(1, -(-max(ny, nx) // _THUMB_MAX))  # ceil
        dxy = float(arr.dx or 0.0)
        dz = float(arr.dz or 0.0)
        if dxy > 0 and dz > 0:
            sz = max(1, int(round(s * dxy / dz)))
            return s, sz, s * dxy, sz * dz
        return s, 1, 1.0, 1.0

    def _load_source_plane(self, arr, ti, c, z) -> np.ndarray | None:
        """Single decimated Z-plane for one tile, read lazily and cached per
        ``(ti, c, "plane", z)``. Only the requested plane is pulled from the
        lazy array, so scrubbing planes stays cheap even on raw stacks."""
        z = int(z)
        key = (ti, c, "plane", z)
        cached = self._mip_cache.get(key)
        if cached is not None:
            self._mip_cache.move_to_end(key)
            return cached
        s, _sz, _, _ = self._mip_params(arr)
        try:
            plane = np.asarray(arr[ti, c, z, ::s, ::s])
            plane = np.squeeze(plane)
            if plane.ndim != 2:
                return None
            self._mip_cache[key] = np.ascontiguousarray(plane)
        except Exception:
            return None
        while len(self._mip_cache) > _THUMB_CACHE_MAX:
            self._mip_cache.popitem(last=False)
        return self._mip_cache.get(key)

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

    def _contrast_range(
        self, arr, loaded: list[np.ndarray], plan_key: tuple
    ) -> tuple[float, float]:
        if self._contrast_mode == _CONTRAST_MANUAL:
            lo, hi = self._manual_lo, self._manual_hi
            return lo, (hi if hi > lo else lo + 1.0)
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

    def _suppress_fpl_right_click_menu(self) -> None:
        """Defer fastplotlib's standard right-click menu while the Tile Grid
        window has the mouse, so its right-click menu doesn't collide with ours.

        fpl's ``StandardRightClickMenu.update`` opens its popup whenever a
        right-click lands inside a subplot, ignoring whether imgui already wants
        the mouse. We wrap its ``get_subplot`` (which it gates the popup on) to
        return ``False`` when our window is open and imgui is capturing the
        mouse. Installed once; harmless when our window is closed.
        """
        fig = getattr(self.parent, "_figure", None)
        rcm = getattr(fig, "_right_click_menu", None)
        if rcm is None or getattr(rcm, "_mbo_guarded", False):
            return
        orig = rcm.get_subplot
        widget = self

        def _guarded_get_subplot():
            try:
                if widget._popup_open and imgui.get_io().want_capture_mouse:
                    return False
            except Exception:
                pass
            return orig()

        rcm.get_subplot = _guarded_get_subplot
        rcm._mbo_guarded = True

    def draw(self) -> None:
        arr = self._active_array()
        if arr is None or not self._ensure_grid(arr):
            return
        self._suppress_fpl_right_click_menu()
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
            if self._is_rotated:
                imgui.text_colored(_WHITE, "rotated: 90deg seeded")
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

        # Row 1 — source: Mode (MIP / single lazy plane), plane slider, z-block.
        imgui.set_next_item_width(100)
        m_changed, new_m = imgui.combo(
            "Mode##tilegrid", self._view_mode, ["MIP", "Plane"]
        )
        if m_changed:
            self._view_mode = int(new_m)
        if self._view_mode == 1 and self._nplanes > 1:
            imgui.same_line()
            imgui.set_next_item_width(260)
            p_changed, new_p = imgui.slider_int(
                "Plane##tilegrid", self._plane, 0, self._nplanes - 1,
                f"%d / {self._nplanes - 1}",
            )
            if p_changed:
                self._plane = int(new_p)
        if nz > 1:
            imgui.same_line()
            lo, hi = self._zblock_range(self._zblock)
            imgui.set_next_item_width(260)
            changed, new_z = imgui.slider_int(
                "Z-block##tilegrid", self._zblock, 0, nz - 1,
                f"%d/{nz - 1}  z {lo:.0f}-{hi:.0f}um",
            )
            if changed:
                self._zblock = int(new_z)

        # Row 2 — display: channel, colormap, contrast, manual min/max.
        if len(self._channel_names) > 1:
            imgui.set_next_item_width(140)
            c_changed, new_c = imgui.combo(
                "View##tilegrid", self._c_index, list(self._channel_names)
            )
            if c_changed:
                self._c_index = new_c
            imgui.same_line()
        imgui.set_next_item_width(120)
        cmap_changed, new_cmap = imgui.combo(
            "Cmap##tilegrid", self._cmap_idx, list(self._cmaps)
        )
        if cmap_changed:
            self._cmap_idx = new_cmap

        imgui.same_line()
        imgui.set_next_item_width(100)
        ctr_changed, new_ctr = imgui.combo(
            "Contrast##tilegrid", self._contrast_mode, list(_CONTRAST_LABELS)
        )
        if ctr_changed:
            self._contrast_mode = new_ctr

        if self._contrast_mode == _CONTRAST_MANUAL:
            imgui.same_line()
            imgui.set_next_item_width(90)
            lo_chg, lo_val = imgui.drag_float(
                "##tilemin", self._manual_lo, 1.0, 0.0, 65535.0, "min %.0f"
            )
            if lo_chg:
                self._manual_lo = float(lo_val)
            imgui.same_line()
            imgui.set_next_item_width(90)
            hi_chg, hi_val = imgui.drag_float(
                "##tilemax", self._manual_hi, 1.0, 0.0, 65535.0, "max %.0f"
            )
            if hi_chg:
                self._manual_hi = float(hi_val)
            imgui.same_line()
            if imgui.small_button("0-300##tilepreset"):
                self._manual_lo, self._manual_hi = 0.0, 300.0

        # Row 3 — layout: move tiles + resets + hint. Its own line so it can't
        # be pushed off the right edge. Right-click a tile to rotate / flip.
        _, self._edit_layout = imgui.checkbox("Move tiles", self._edit_layout)
        if not self._edit_layout:
            self._pick = None
        imgui.same_line()
        if imgui.small_button("Reset layout"):
            self._layout.pop((self._c_index, self._zblock), None)
            self._pick = None
        if imgui.is_item_hovered():
            imgui.set_tooltip("restore tile positions (this camera)")
        imgui.same_line()
        if imgui.small_button("Reset flips"):
            self._tile_flips.clear()
            self._tile_rot.clear()
            self._flip_seeded.clear()  # re-seed per-camera defaults next frame
        if imgui.is_item_hovered():
            imgui.set_tooltip("restore default tile flips/rotations")
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(0.2, 0.9, 1.0, 1.0),
            "right-click a tile to rotate / flip"
            + ("   |   click to pick/place" if self._edit_layout else ""),
        )

    def _camera_number(self, c: int):
        """Actual camera index (CM##) for view-combo index ``c``, or ``None``
        when the view isn't a single camera (fused pairs / non-camera views)."""
        try:
            name = self._channel_names[c]
        except (IndexError, TypeError):
            return None
        cams = re.findall(r"CM(\d+)", str(name))
        return int(cams[0]) if len(cams) == 1 else None

    def _camera_is_mirrored(self, c: int) -> bool:
        """Columns mirrored left-right for this camera (verified per-camera
        convention for the 4-camera IsoView). VW00 mirrors the opposing
        camera CM01; VW90 mirrors CM02 (not CM03). Non-single-camera views
        (fused) aren't mirrored.
        """
        return self._camera_number(c) in (1, 2)

    def _camera_default_hflip(self, c: int) -> bool:
        """VW90 cameras (CM02, CM03) default to H-flipping their tile_y==0
        tiles to make the beads tile (verified); VW00 gets no default flip."""
        return self._camera_number(c) in (2, 3)

    def _camera_default_rot(self, c: int) -> int:
        """Rotated datasets mount every camera 90deg on its side, so every
        tile's content needs a 90deg (CCW) rotation to tile upright. Verified
        on VW00 by bead overlap cross-correlation; VW90 follows the same mount.
        0 for Normal datasets."""
        return 1 if self._is_rotated else 0

    def _seed_camera_defaults(self, c: int) -> None:
        """Apply a camera's default per-tile orientation once: a 90deg rotation
        for Rotated datasets, plus the VW90 H-flips. Seeds into ``_tile_rot`` /
        ``_tile_flips`` so the user can still toggle them; re-applied after
        "Reset flips" (which clears the seeded marks)."""
        if c in self._flip_seeded:
            return
        self._flip_seeded.add(c)
        rot = self._camera_default_rot(c)
        if rot:
            for ti in self._tile_xyz:
                self._tile_rot[(c, ti)] = rot
        if not self._camera_default_hflip(c):
            return
        for ti, xyz in self._tile_xyz.items():
            if xyz[1] == 0:  # tile_y == 0
                self._tile_flips.setdefault((c, ti), set()).add("X")

    def _get_layout(self, g) -> dict:
        """Editable ``{(ri, ci): ti}`` for the current (camera, z-block).

        Lazily seeded from the stage-derived placement, mirrored left-right for
        odd (opposing) cameras. Keyed per camera so each camera keeps its own
        arrangement and manual moves don't leak across cameras.
        """
        c = self._c_index
        z = self._zblock
        key = (c, z)
        if key not in self._layout:
            src = g["placed"].get(z, {})
            ncols = max(1, len(g["cols"]))
            mirror = self._camera_is_mirrored(c)
            lay = {}
            for (ri, ci), (ti, spc) in src.items():
                cc = (ncols - 1 - ci) if mirror else ci
                lay[(ri, cc)] = ti
                self._tile_spc[ti] = spc
            self._layout[key] = lay
        return self._layout[key]

    def _swap_cells(self, layout: dict, a: tuple, b: tuple) -> None:
        """Swap the tiles in two cells (move into an empty cell if one side
        is empty)."""
        if a == b:
            return
        ta, tb = layout.get(a), layout.get(b)
        for cell, t in ((b, ta), (a, tb)):
            if t is None:
                layout.pop(cell, None)
            else:
                layout[cell] = t

    def _overlap_frac(self, arr, stride_key: str, centers: list, extent_um: float):
        """Fraction of a tile that overlaps its neighbor along one display axis.

        ``(extent − stride) / extent`` in µm. Stride from the declared
        ``tile_stride_*`` (cols=tile_y, rows=tile_x); falls back to the spacing
        of the grid's stage-coordinate centers. Returns ``None`` when there's
        no positive overlap or no usable stride.
        """
        meta = getattr(arr, "metadata", {}) or {}
        s = meta.get(stride_key)
        if not s and centers and len(centers) > 1:
            diffs = sorted(
                abs(centers[i + 1] - centers[i]) for i in range(len(centers) - 1)
            )
            d = diffs[len(diffs) // 2]
            s = d if d > 1.0 else None  # >1: real µm spacing, not index fallback
        if not s or extent_um <= 0 or float(s) >= extent_um:
            return None
        return (extent_um - float(s)) / extent_um

    def _draw_grid(self, arr) -> None:
        g = self._grid
        ncols = max(1, len(g["cols"]))
        nrows = max(1, len(g["rows"]))
        c = self._c_index
        self._seed_camera_defaults(c)  # apply per-camera default orientation once

        _s, _sz, pix_xy, pix_z = self._mip_params(arr)

        layout = self._get_layout(g)

        # auto-contrast samples cached thumbs for this channel + view mode/plane
        # (a single plane is much dimmer than a MIP, so they need separate
        # ranges). Per-tile flips/rotation don't change intensity.
        plane_key = self._plane if self._view_mode == 1 else -1
        loaded = [
            v for k, v in self._thumb_cache.items()
            if k[1] == c and k[2] == self._view_mode and k[3] == plane_key
        ]
        lo, hi = self._contrast_range(arr, loaded, (self._view_mode, plane_key))

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
                    clicked_r = imgui.is_item_clicked(imgui.MouseButton_.right)
                    entry = layout.get((ri, ci))
                    picked = self._pick == (ri, ci)
                    # move mode: click picks a tile, then click a cell to
                    # move/swap into it; outside move mode click jumps.
                    if clicked and self._edit_layout:
                        if self._pick is None:
                            if entry is not None:
                                self._pick = (ri, ci)
                        else:
                            self._swap_cells(layout, self._pick, (ri, ci))
                            self._pick = None
                    elif clicked and entry is not None:
                        self._jump_to_tile(entry)
                    if entry is None:
                        draw_list.add_rect(pos, cmax, yellow if picked else grey)
                        continue
                    ti = entry
                    spc = self._tile_spc.get(ti, ti)
                    # flips/rotation are per (camera, tile): each camera has its
                    # own physical orientation, so they don't carry across views.
                    okey = (c, ti)
                    tf = self._tile_flips.get(okey, set())
                    fx = "X" in tf  # horizontal mirror
                    fy = "Y" in tf  # vertical mirror
                    rot = self._tile_rot.get(okey, 0) % 4  # 90deg CCW steps
                    if clicked_r:
                        self._orient_menu_ti = ti  # opened after end_child
                        self._orient_menu_open = True
                    # source: single plane (lazy) or whole-tile MIP
                    plane_key = self._plane if self._view_mode == 1 else -1
                    gkey = (ti, c, self._view_mode, plane_key, rot, fx, fy)
                    thumb = self._thumb_cache.get(gkey)
                    if thumb is None and budget > 0:
                        if self._view_mode == 1:
                            src = self._load_source_plane(arr, ti, c, self._plane)
                        else:
                            src = self._load_source_mip(arr, ti, c, "xy")
                        if src is not None:
                            t = src
                            if rot:  # rotate first, then flips act on the view
                                t = np.rot90(t, rot)
                            if fy:  # Y flip = vertical mirror
                                t = t[::-1, :]
                            if fx:  # X flip = horizontal mirror
                                t = t[:, ::-1]
                            thumb = np.ascontiguousarray(t)
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
                        psx = psy = pix_xy  # xy MIP: both axes are lateral
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
                        # overlap guides: thin line at the expected seam with
                        # each present neighbor (extent − stride). Positions are
                        # for the un-rotated layout.
                        fh = self._overlap_frac(arr, "tile_stride_y", g["cols"], pw)
                        fv = self._overlap_frac(arr, "tile_stride_x", g["rows"], ph)
                        seam = imgui.color_convert_float4_to_u32(
                            imgui.ImVec4(0.2, 0.9, 1.0, 0.55)
                        )
                        if fh is not None:
                            if (ri, ci + 1) in layout:
                                x = ix + dw * (1.0 - fh)
                                draw_list.add_line(
                                    imgui.ImVec2(x, iy), imgui.ImVec2(x, iy + dh),
                                    seam, 1.0)
                            if (ri, ci - 1) in layout:
                                x = ix + dw * fh
                                draw_list.add_line(
                                    imgui.ImVec2(x, iy), imgui.ImVec2(x, iy + dh),
                                    seam, 1.0)
                        if fv is not None:
                            if (ri + 1, ci) in layout:
                                y = iy + dh * (1.0 - fv)
                                draw_list.add_line(
                                    imgui.ImVec2(ix, y), imgui.ImVec2(ix + dw, y),
                                    seam, 1.0)
                            if (ri - 1, ci) in layout:
                                y = iy + dh * fv
                                draw_list.add_line(
                                    imgui.ImVec2(ix, y), imgui.ImVec2(ix + dw, y),
                                    seam, 1.0)
                    else:
                        draw_list.add_rect_filled(pos, cmax, dark)
                    label = self._tile_labels.get(ti, f"SPM{spc:02d}")
                    if imgui.is_item_hovered():
                        tip = f"{label}  (tile {ti})"
                        tip += (
                            "\nclick: pick / place    right-click: rotate / flip"
                            if self._edit_layout else "\nright-click: rotate / flip"
                        )
                        imgui.set_tooltip(tip)
                    is_current = ti == cur_tile
                    cyan = imgui.color_convert_float4_to_u32(
                        imgui.ImVec4(0.2, 0.9, 1.0, 1.0))
                    border = cyan if picked else (yellow if is_current else grey)
                    draw_list.add_rect(
                        pos, cmax, border,
                        thickness=3.0 if (picked or is_current) else 1.0,
                    )
                    self._draw_cell_label(draw_list, pos, label)
                    marks = []
                    if rot:
                        marks.append(f"rot {rot * 90}")
                    if tf:
                        marks.append("flip " + "".join(
                            m for m, ax in (("H", "X"), ("V", "Y")) if ax in tf
                        ))
                    if marks:
                        self._draw_cell_label(
                            draw_list, imgui.ImVec2(pos.x, pos.y + 15.0),
                            "  ".join(marks),
                        )
        finally:
            imgui.pop_style_var(1)
            imgui.end_child()

        # rotate/flip menu, opened top-level (outside the scrolling child) so
        # imgui clamps it inside the window instead of clipping it on the right.
        if self._orient_menu_open:
            imgui.open_popup("tile_orient_menu")
            self._orient_menu_open = False
        ti = self._orient_menu_ti
        if ti is not None:
            if imgui.begin_popup("tile_orient_menu"):
                self._draw_tile_orient_menu(
                    ti, self._tile_labels.get(ti, f"tile {ti}")
                )
                imgui.end_popup()
            else:
                self._orient_menu_ti = None

    @staticmethod
    def _toggle_button(label: str, active: bool, w: float) -> bool:
        """Button that stays highlighted while ``active``; returns True on click."""
        if active:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.20, 0.45, 0.85, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.26, 0.52, 0.92, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.16, 0.38, 0.75, 1.0))
        clicked = imgui.button(label, imgui.ImVec2(w, 0))
        if active:
            imgui.pop_style_color(3)
        return clicked

    def _draw_tile_orient_menu(self, ti: int, label: str) -> None:
        """Rotate / flip menu for one tile (right-click popup body).

        Buttons are sized from measured text in a 2-column grid so the popup
        always fits its content (no clipping) and lines up cleanly.
        """
        imgui.text_disabled(f"Tile {label}  ({self._channel_names[self._c_index] if self._c_index < len(self._channel_names) else 'view'})")
        imgui.separator()
        style = imgui.get_style()
        sp = style.item_spacing.x
        bw = max(
            imgui.calc_text_size("Rotate CCW").x,
            imgui.calc_text_size("Rotate CW").x,
        ) + style.frame_padding.x * 2.0 + 8.0

        # per (camera, tile) so each camera keeps its own orientation
        okey = (self._c_index, ti)
        if imgui.button("Rotate CCW##ccw", imgui.ImVec2(bw, 0)):
            self._tile_rot[okey] = (self._tile_rot.get(okey, 0) + 1) % 4
        imgui.same_line()
        if imgui.button("Rotate CW##cw", imgui.ImVec2(bw, 0)):
            self._tile_rot[okey] = (self._tile_rot.get(okey, 0) - 1) % 4

        s = self._tile_flips.setdefault(okey, set())
        if self._toggle_button("Flip H##fh", "X" in s, bw):
            (s.discard if "X" in s else s.add)("X")
        imgui.same_line()
        if self._toggle_button("Flip V##fv", "Y" in s, bw):
            (s.discard if "Y" in s else s.add)("Y")

        if imgui.button("Reset##tileorient", imgui.ImVec2(bw * 2 + sp, 0)):
            self._tile_rot[okey] = 0
            s.clear()

    def _draw_cell_label(self, draw_list, pos, label: str) -> None:
        txt = str(label)
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
