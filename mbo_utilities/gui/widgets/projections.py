"""ProjectionsViewer - browse XY/XZ/YZ projection TIFFs from isoview pipeline outputs.

Activates whenever a loaded array originates from a Keller-lab isoview pipeline
(detected via metadata `stack_type == "isoview-*"` or a source path with
`*.raw.projections/`, `*.corrected.projections/`, or
`*.corrected/Results/MultiFused_geometric/` siblings).

Discovers all projection TIFFs lazily and presents a side-panel selector plus
a floating imgui popup that mirrors `summary_image.SummaryImageViewer`'s
toolbar / contrast / pan-zoom / save UX. Projections themselves are loaded on
demand (one tifffile read per selection); the GPU texture cache is bounded so
the 800+ projections in a typical dataset don't pin memory.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from imgui_bundle import imgui, portable_file_dialogs as pfd

from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.summary_image import (
    SIDE_PANEL_SECTION_COLOR,
    STATUS_ERROR_COLOR,
    STATUS_OK_COLOR,
    _CONTRAST_AUTO,
    _CONTRAST_MANUAL,
    _CONTRAST_MODES,
    _DEFAULT_COLORMAP,
    _DEFAULT_COLORMAPS,
    _GpuImage,
    _PIXEL_VALUES_MIN_ZOOM,
    _auto_range,
    _data_range,
    _format_value,
    _to_rgba,
    center_popup_on_open,
    draw_section_header,
)


_AXIS_LABELS = ("xy", "xz", "yz")
_AXIS_DISPLAY = {"xy": "XY (max-Z)", "xz": "XZ (max-Y)", "yz": "YZ (max-X)"}

_RE_FLAT = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)\.(xy|xz|yz)Projection\.tif$",
    re.IGNORECASE,
)
_RE_FUSED = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)_CM(\d+)_(?:VW|CHN)(\d+)\.(xy|xz|yz)Projection\.tif$",
    re.IGNORECASE,
)
_RE_VW_ONLY = re.compile(
    r"^SPM(\d+)_TM(\d+)_(?:VW|CHN)(\d+)\.(xy|xz|yz)Projection\.tif$",
    re.IGNORECASE,
)
_RE_TM = re.compile(r"^TM(\d{6})$", re.IGNORECASE)

# bound the GPU texture cache so scrubbing through 800+ projections doesn't
# blow VRAM. one projection is ~1-3 MiB; 16 textures is ~50 MiB worst case.
_MAX_GPU_CACHE = 16


def _isoview_root(arr) -> Path | None:
    """Locate the dataset root that holds projection trees for ``arr``.

    The root is the directory that contains either a `*.corrected/` child
    or sibling `*.{raw,corrected}.projections/` folders. Returns the first
    plausible candidate by walking up the array's source path.
    """
    src = getattr(arr, "source_path", None)
    if src is None:
        return None
    p = Path(src)
    if p.is_file():
        p = p.parent
    for cand in (p, *p.parents):
        if not cand.exists():
            continue
        if any(
            c.is_dir()
            and (c.name.endswith(".corrected") or c.name.endswith(".projections"))
            for c in cand.iterdir()
        ):
            return cand
        # the array's own path is already inside the .corrected tree
        if cand.name.endswith(".corrected"):
            return cand.parent
    return None


def _discover_projections(root: Path) -> dict[str, dict]:
    """Scan ``root`` for raw / corrected / fused projection trees.

    Returns a dict keyed by source group ("raw", "corrected", "fused"),
    each mapping to {"axes": [...], "views": [...], "files": {(axis, view, t): Path}}.
    Empty groups are omitted. Views are normalized strings (e.g. "CM00",
    "VW00") so the UI can list them without re-parsing.
    """
    groups: dict[str, dict] = {}

    def _add(group: str, axis: str, view: str, t: int, path: Path) -> None:
        g = groups.setdefault(
            group,
            {"axes": set(), "views": set(), "files": {}},
        )
        g["axes"].add(axis)
        g["views"].add(view)
        g["files"][(axis, view, t)] = path

    # flat siblings: <root>/<dataset>.{raw,corrected}.projections/SPM##_TM##_CM##.xyProjection.tif
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.endswith(".raw.projections"):
            label = "raw"
        elif child.name.endswith(".corrected.projections"):
            label = "corrected"
        else:
            continue
        for f in child.iterdir():
            m = _RE_FLAT.match(f.name)
            if not m:
                continue
            _, tm, cm, axis = m.groups()
            _add(label, axis.lower(), f"CM{int(cm):02d}", int(tm), f)

    # fused tree: <root>/<dataset>.corrected/Results/MultiFused_geometric/SPM##/TM######/...
    for child in root.iterdir():
        if not (child.is_dir() and child.name.endswith(".corrected")):
            continue
        fused_root = child / "Results" / "MultiFused_geometric"
        if not fused_root.is_dir():
            continue
        for spm_dir in fused_root.iterdir():
            if not spm_dir.is_dir() or not spm_dir.name.startswith("SPM"):
                continue
            for tm_dir in spm_dir.iterdir():
                if not tm_dir.is_dir() or not _RE_TM.match(tm_dir.name):
                    continue
                t = int(tm_dir.name[2:])
                for f in tm_dir.iterdir():
                    if not f.name.endswith(("Projection.tif", "Projection.TIF")):
                        continue
                    m = _RE_FUSED.match(f.name)
                    if m:
                        _, _, cm0, cm1, vw, axis = m.groups()
                        view = f"VW{int(vw):02d}"
                        _add("fused", axis.lower(), view, t, f)
                        continue
                    m = _RE_VW_ONLY.match(f.name)
                    if m:
                        _, _, vw, axis = m.groups()
                        view = f"VW{int(vw):02d}"
                        _add("fused", axis.lower(), view, t, f)

    # finalize: sort axes/views, drop empty groups
    finalized: dict[str, dict] = {}
    for label, g in groups.items():
        if not g["files"]:
            continue
        finalized[label] = {
            "axes": [a for a in _AXIS_LABELS if a in g["axes"]],
            "views": sorted(g["views"]),
            "files": g["files"],
        }
    return finalized


def _timepoints_for(group: dict, axis: str, view: str) -> list[int]:
    return sorted(
        {t for (a, v, t) in group["files"] if a == axis and v == view}
    )


class ProjectionsViewer(Widget):
    """Browse XY/XZ/YZ projections from isoview pipeline outputs."""

    name = "Projections"
    priority = 65  # right after SummaryImageViewer (60)

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._popup_open: bool = False
        self._needs_fit: bool = True
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._cmaps: list[str] = list(_DEFAULT_COLORMAPS)
        self._cmap_idx: int = self._cmaps.index(_DEFAULT_COLORMAP)
        self._cmap_synced_with_fpl: bool = False
        self._contrast_mode: int = _CONTRAST_AUTO
        self._show_pixel_values: bool = False

        # selection
        self._source: str | None = None
        self._axis: str | None = None
        self._view: str | None = None
        self._timepoint: int | None = None

        # discovery cache (lazy)
        self._projections: dict[str, dict] | None = None
        self._root: Path | None = None

        # per-image state
        self._array_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._gpu_cache: OrderedDict[tuple, _GpuImage] = OrderedDict()
        self._manual_lo: dict[tuple, float] = {}
        self._manual_hi: dict[tuple, float] = {}
        self._hist_cache: dict[tuple, np.ndarray] = {}

        # save dialog
        self._save_dialog: Any = None
        self._pending_save_key: tuple | None = None
        self._last_save_msg: str = ""

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        for arr in parent._get_data_arrays():
            md = getattr(arr, "metadata", None) or {}
            stack_type = str(md.get("stack_type", ""))
            if stack_type.startswith("isoview"):
                root = _isoview_root(arr)
                if root is not None:
                    return True
        return False

    def _backend(self):
        try:
            return self.parent._figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _active_array(self):
        for arr in self.parent._get_data_arrays():
            md = getattr(arr, "metadata", None) or {}
            if str(md.get("stack_type", "")).startswith("isoview"):
                return arr
        return None

    def _ensure_projections(self) -> bool:
        if self._projections is not None:
            return bool(self._projections)
        arr = self._active_array()
        if arr is None:
            self._projections = {}
            return False
        root = _isoview_root(arr)
        if root is None:
            self._projections = {}
            return False
        self._root = root
        self._projections = _discover_projections(root)
        if not self._projections:
            return False
        # initialize selection to the first plausible entry
        self._source = next(iter(self._projections))
        g = self._projections[self._source]
        self._axis = g["axes"][0] if g["axes"] else None
        self._view = g["views"][0] if g["views"] else None
        if self._axis and self._view:
            tps = _timepoints_for(g, self._axis, self._view)
            self._timepoint = tps[0] if tps else None
        return True

    def _current_path(self) -> Path | None:
        if not (self._projections and self._source and self._axis and self._view):
            return None
        if self._timepoint is None:
            return None
        return self._projections[self._source]["files"].get(
            (self._axis, self._view, self._timepoint)
        )

    def _selection_key(self) -> tuple:
        return (self._source, self._axis, self._view, self._timepoint)

    def _load_array(self, path: Path) -> np.ndarray | None:
        key = (str(path),)
        cached = self._array_cache.get(key)
        if cached is not None:
            self._array_cache.move_to_end(key)
            return cached
        try:
            arr = np.asarray(tifffile.imread(str(path)))
        except Exception:
            return None
        if arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                return None
        self._array_cache[key] = arr
        # bound the in-memory cache to the GPU cache size
        while len(self._array_cache) > _MAX_GPU_CACHE:
            self._array_cache.popitem(last=False)
        return arr

    def _sync_cmap_with_fpl(self) -> None:
        if self._cmap_synced_with_fpl:
            return
        iw = getattr(self.parent, "image_widget", None)
        if iw is None:
            return
        try:
            graphics = list(iw.graphics)
        except Exception:
            return
        if not graphics:
            return
        try:
            cmap_name = str(graphics[0].cmap)
        except Exception:
            return
        if not cmap_name:
            return
        if cmap_name not in self._cmaps:
            self._cmaps = [cmap_name] + list(_DEFAULT_COLORMAPS)
        self._cmap_idx = self._cmaps.index(cmap_name)
        self._cmap_synced_with_fpl = True

    def _reset_view(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._needs_fit = True

    def _get_range(self, key: tuple, arr: np.ndarray) -> tuple[float, float]:
        if self._contrast_mode == _CONTRAST_AUTO:
            return _auto_range(arr)
        if self._contrast_mode == _CONTRAST_MANUAL:
            lo = self._manual_lo.get(key)
            hi = self._manual_hi.get(key)
            if lo is None or hi is None:
                lo, hi = _auto_range(arr)
                self._manual_lo[key] = lo
                self._manual_hi[key] = hi
            if hi <= lo:
                hi = lo + 1e-6
            return lo, hi
        return _data_range(arr)

    def _ensure_gpu(self, key: tuple, arr: np.ndarray) -> _GpuImage | None:
        backend = self._backend()
        if backend is None:
            return None
        cmap = self._cmaps[self._cmap_idx]
        lo, hi = self._get_range(key, arr)
        gpu = self._gpu_cache.get(key)
        if gpu is None or gpu.arr is not arr:
            if gpu is not None:
                gpu.destroy()
                self._gpu_cache.pop(key, None)
            gpu = _GpuImage(backend, arr, cmap, lo, hi)
            self._gpu_cache[key] = gpu
            # evict oldest if over the cap
            while len(self._gpu_cache) > _MAX_GPU_CACHE:
                _, evicted = self._gpu_cache.popitem(last=False)
                evicted.destroy()
        else:
            gpu.reupload_if_changed(cmap, lo, hi)
            self._gpu_cache.move_to_end(key)
        return gpu

    def _get_histogram(self, key: tuple, arr: np.ndarray) -> np.ndarray:
        h = self._hist_cache.get(key)
        if h is not None:
            return h
        a = np.asarray(arr, dtype=np.float32)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            h = np.zeros(128, dtype=np.float32)
        else:
            counts, _ = np.histogram(finite, bins=128)
            h = counts.astype(np.float32)
        self._hist_cache[key] = h
        return h

    def _on_selection_changed(self) -> None:
        self._reset_view()

    def _draw_selectors(
        self,
        prefix: str = "side",
        include_axis: bool = True,
        include_timepoint: bool = True,
    ) -> None:
        """Render Source / Axis / View / Timepoint selectors.

        Source and View always render. Axis and Timepoint are gated by
        flags so the side panel can show a slim version (just Source and
        View) while the popup shows the full set. ``prefix`` disambiguates
        imgui ids so the two instances don't fight.
        """
        sources = list(self._projections.keys())
        if not sources:
            imgui.text_colored(STATUS_ERROR_COLOR, "no projections found")
            return

        if self._source not in sources:
            self._source = sources[0]
            self._on_selection_changed()

        imgui.set_next_item_width(110)
        idx = sources.index(self._source)
        changed, new_idx = imgui.combo(
            f"Source##{prefix}_src", idx, sources
        )
        if changed:
            self._source = sources[new_idx]
            g = self._projections[self._source]
            self._axis = g["axes"][0] if g["axes"] else None
            self._view = g["views"][0] if g["views"] else None
            tps = _timepoints_for(g, self._axis, self._view) if self._axis and self._view else []
            self._timepoint = tps[0] if tps else None
            self._on_selection_changed()

        g = self._projections[self._source]
        axes = g["axes"]
        if not axes:
            return
        if self._axis not in axes:
            self._axis = axes[0]
            self._on_selection_changed()

        if include_axis:
            imgui.same_line()
            imgui.set_next_item_width(110)
            labels = [_AXIS_DISPLAY.get(a, a) for a in axes]
            idx = axes.index(self._axis)
            changed, new_idx = imgui.combo(f"Axis##{prefix}_axis", idx, labels)
            if changed:
                self._axis = axes[new_idx]
                self._on_selection_changed()

        views = g["views"]
        if not views:
            return
        if self._view not in views:
            self._view = views[0]
            self._on_selection_changed()
        imgui.same_line()
        imgui.set_next_item_width(95)
        idx = views.index(self._view)
        changed, new_idx = imgui.combo(f"View##{prefix}_view", idx, views)
        if changed:
            self._view = views[new_idx]
            self._on_selection_changed()

        tps = _timepoints_for(g, self._axis, self._view)
        if not tps:
            imgui.text_colored(
                STATUS_ERROR_COLOR,
                f"no {self._axis} for {self._view} in {self._source}",
            )
            return
        if self._timepoint not in tps:
            self._timepoint = tps[0]
            self._on_selection_changed()

        if include_timepoint:
            imgui.set_next_item_width(220)
            # use slider_int over an integer index into tps so the slider step
            # is always 1 even when the timepoint values themselves skip
            cur_pos = tps.index(self._timepoint)
            changed, new_pos = imgui.slider_int(
                f"T##{prefix}_t",
                cur_pos,
                0,
                len(tps) - 1,
                f"%d / {len(tps) - 1}  (TM{tps[0]:06d}-{tps[-1]:06d})",
            )
            if changed:
                self._timepoint = tps[int(new_pos)]
                self._on_selection_changed()

    def draw(self) -> None:
        if not self._ensure_projections():
            return

        self._sync_cmap_with_fpl()
        draw_section_header("Projections")

        # side panel: slim — Source and View only. Axis and Timepoint live
        # in the popup so the Preview Tab stays uncluttered (no point in
        # scrubbing T or flipping axes here when there's no preview image
        # to react to it).
        self._draw_selectors(
            prefix="side", include_axis=False, include_timepoint=False
        )

        if imgui.button("Open viewer##projections_open"):
            self._popup_open = True
            self._reset_view()

        self._poll_save_dialog()

        if self._popup_open:
            self._draw_popup()

    def _draw_toolbar(self) -> None:
        # popup: full selector set including axis + timepoint slider.
        self._draw_selectors(
            prefix="popup", include_axis=True, include_timepoint=True
        )

        imgui.set_next_item_width(110)
        cmap_changed, new_cmap = imgui.combo(
            "Cmap##projections", self._cmap_idx, list(self._cmaps)
        )
        if cmap_changed:
            self._cmap_idx = new_cmap

        imgui.same_line()
        imgui.set_next_item_width(110)
        ctr_changed, new_ctr = imgui.combo(
            "Contrast##projections", self._contrast_mode, list(_CONTRAST_MODES)
        )
        if ctr_changed:
            self._contrast_mode = new_ctr

        imgui.same_line()
        if imgui.button("Reset##projections"):
            self._reset_view()

        imgui.same_line()
        if imgui.button("Save...##projections"):
            self._open_save_dialog()
        set_tooltip("Save the colormapped projection as a PNG (native resolution).")

        _, self._show_pixel_values = imgui.checkbox(
            "Pixel values##projections", self._show_pixel_values
        )
        set_tooltip(
            f"Draw the numeric value of each pixel.\n"
            f"Visible only when zoomed past {int(_PIXEL_VALUES_MIN_ZOOM)}x."
        )

        if self._last_save_msg:
            color = (
                STATUS_ERROR_COLOR
                if self._last_save_msg.startswith("Save failed")
                else STATUS_OK_COLOR
            )
            imgui.text_colored(color, self._last_save_msg)

    def _draw_contrast_panel(self, key: tuple, arr: np.ndarray) -> None:
        if self._contrast_mode != _CONTRAST_MANUAL:
            return
        data_lo, data_hi = _data_range(arr)
        lo = self._manual_lo.get(key)
        hi = self._manual_hi.get(key)
        if lo is None or hi is None:
            lo, hi = _auto_range(arr)
            self._manual_lo[key] = lo
            self._manual_hi[key] = hi

        bins = self._get_histogram(key, arr)
        if imgui.begin_child(
            "##projections_levels",
            imgui.ImVec2(0, 32),
            child_flags=imgui.ChildFlags_.borders,
        ):
            avail_w = max(imgui.get_content_region_avail().x, 100.0)
            hist_w = avail_w * 0.32
            slider_w = (avail_w - hist_w - 28) * 0.5

            imgui.plot_histogram(
                "##projections_hist", bins, graph_size=imgui.ImVec2(hist_w, 22)
            )
            imgui.same_line()
            imgui.set_next_item_width(slider_w)
            ch_lo, new_lo = imgui.slider_float(
                "min##projections", lo, data_lo, data_hi, "%.4g"
            )
            imgui.same_line()
            imgui.set_next_item_width(slider_w)
            ch_hi, new_hi = imgui.slider_float(
                "max##projections", hi, data_lo, data_hi, "%.4g"
            )
            if ch_lo:
                self._manual_lo[key] = min(new_lo, hi - 1e-6)
            if ch_hi:
                self._manual_hi[key] = max(new_hi, lo + 1e-6)
        imgui.end_child()

    def _draw_pixel_values(
        self,
        draw_list,
        arr: np.ndarray,
        canvas_pos,
        canvas_size,
        gpu: _GpuImage,
    ) -> None:
        if self._zoom < _PIXEL_VALUES_MIN_ZOOM:
            return
        x0_img = max(0, int(np.floor(-self._pan_x / self._zoom)))
        y0_img = max(0, int(np.floor(-self._pan_y / self._zoom)))
        x1_img = min(gpu.w, int(np.ceil((canvas_size.x - self._pan_x) / self._zoom)) + 1)
        y1_img = min(gpu.h, int(np.ceil((canvas_size.y - self._pan_y) / self._zoom)) + 1)
        if x1_img <= x0_img or y1_img <= y0_img:
            return
        n = (x1_img - x0_img) * (y1_img - y0_img)
        if n > 10_000:
            return

        rgba = gpu.rgba
        luma = (
            0.299 * rgba[..., 0].astype(np.float32)
            + 0.587 * rgba[..., 1].astype(np.float32)
            + 0.114 * rgba[..., 2].astype(np.float32)
        )

        white = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        black = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.0, 0.0, 1.0))
        z = self._zoom
        for y in range(y0_img, y1_img):
            sy = canvas_pos.y + self._pan_y + y * z + z * 0.5
            for x in range(x0_img, x1_img):
                sx = canvas_pos.x + self._pan_x + x * z + z * 0.5
                txt = _format_value(float(arr[y, x]), arr.dtype)
                tw = len(txt) * 6.0
                tx = sx - tw * 0.5
                ty = sy - 6.5
                color = black if luma[y, x] > 140 else white
                draw_list.add_text(imgui.ImVec2(tx, ty), color, txt)

    def _open_save_dialog(self) -> None:
        if self._save_dialog is not None:
            return
        path = self._current_path()
        if path is None:
            return
        cmap = self._cmaps[self._cmap_idx]
        default = f"{path.stem}_{cmap}.png"
        start_dir = str(self._root) if self._root else ""
        try:
            self._save_dialog = pfd.save_file(
                "Save projection as PNG",
                str(Path(start_dir) / default) if start_dir else default,
                ["PNG files", "*.png", "All files", "*"],
            )
        except Exception as e:
            self._last_save_msg = f"Save failed: {e}"
            self._save_dialog = None
        self._pending_save_key = self._selection_key()

    def _poll_save_dialog(self) -> None:
        if self._save_dialog is None:
            return
        try:
            ready = self._save_dialog.ready(0)
        except TypeError:
            ready = self._save_dialog.ready()
        if not ready:
            return
        try:
            target_str = self._save_dialog.result()
        except Exception:
            target_str = ""
        self._save_dialog = None
        if not target_str:
            return
        try:
            key = self._pending_save_key
            if key is None:
                return
            arr = self._array_cache.get(
                (str(self._projections[key[0]]["files"][(key[1], key[2], key[3])]),)
            )
            if arr is None:
                return
            cmap = self._cmaps[self._cmap_idx]
            lo, hi = self._get_range(key, arr)
            rgba = _to_rgba(arr, cmap, lo, hi)
            import imageio.v3 as iio

            target = Path(target_str)
            if target.suffix.lower() != ".png":
                target = target.with_suffix(".png")
            iio.imwrite(str(target), rgba)
            self._last_save_msg = f"Saved {target.name}"
        except Exception as e:
            self._last_save_msg = f"Save failed: {e}"

    def _draw_popup(self) -> None:
        path = self._current_path()
        title = path.name if path is not None else "no selection"
        center_popup_on_open(default_em=(52.0, 56.0), min_em=(38.0, 30.0))
        opened, self._popup_open = imgui.begin(
            f"Projections: {title}###projections_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            imgui.end()
            return

        self._draw_toolbar()

        path = self._current_path()
        if path is None:
            imgui.text_colored(STATUS_ERROR_COLOR, "no projection for current selection")
            imgui.end()
            return

        arr = self._load_array(path)
        if arr is None:
            imgui.text_colored(STATUS_ERROR_COLOR, f"failed to read {path.name}")
            imgui.end()
            return

        key = self._selection_key()
        self._draw_contrast_panel(key, arr)

        gpu = self._ensure_gpu(key, arr)
        if gpu is None:
            imgui.text_colored(
                STATUS_ERROR_COLOR,
                "GPU backend unavailable (figure not attached?)",
            )
            imgui.end()
            return

        h, w = gpu.h, gpu.w
        imgui.begin_child(
            "##projections_canvas",
            imgui.ImVec2(0, -28),
            child_flags=0,
            window_flags=imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse,
        )
        canvas_pos = imgui.get_cursor_screen_pos()
        canvas_size = imgui.get_content_region_avail()
        cw = max(canvas_size.x, 1.0)
        ch = max(canvas_size.y, 1.0)

        if self._needs_fit:
            self._zoom = float(min(cw / w, ch / h)) if w > 0 and h > 0 else 1.0
            self._pan_x = (cw - w * self._zoom) * 0.5
            self._pan_y = (ch - h * self._zoom) * 0.5
            self._needs_fit = False

        imgui.invisible_button("##projections_pan", imgui.ImVec2(cw, ch))
        io = imgui.get_io()

        if imgui.is_item_active():
            self._pan_x += io.mouse_delta.x
            self._pan_y += io.mouse_delta.y
        if imgui.is_item_hovered() and io.mouse_wheel != 0.0:
            mx = io.mouse_pos.x - canvas_pos.x
            my = io.mouse_pos.y - canvas_pos.y
            old = self._zoom
            self._zoom = float(np.clip(old * (1.1 ** io.mouse_wheel), 0.05, 64.0))
            scale = self._zoom / old
            self._pan_x = mx - (mx - self._pan_x) * scale
            self._pan_y = my - (my - self._pan_y) * scale

        img_min = imgui.ImVec2(canvas_pos.x + self._pan_x, canvas_pos.y + self._pan_y)
        img_max = imgui.ImVec2(img_min.x + w * self._zoom, img_min.y + h * self._zoom)
        clip_max = imgui.ImVec2(canvas_pos.x + cw, canvas_pos.y + ch)
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(canvas_pos, clip_max, True)
        draw_list.add_image(gpu.ref, img_min, img_max)

        if self._show_pixel_values:
            self._draw_pixel_values(draw_list, arr, canvas_pos, canvas_size, gpu)

        draw_list.pop_clip_rect()

        readout = ""
        if imgui.is_item_hovered():
            mx = io.mouse_pos.x - img_min.x
            my = io.mouse_pos.y - img_min.y
            px = int(mx / max(self._zoom, 1e-6))
            py = int(my / max(self._zoom, 1e-6))
            if 0 <= px < w and 0 <= py < h:
                try:
                    val = float(arr[py, px])
                    readout = f"px ({py}, {px}) = {val:.4g}"
                except Exception:
                    readout = ""
        imgui.end_child()

        amin, amax = _data_range(arr)
        footer = (
            f"{h}x{w}  {arr.dtype}  range [{amin:.4g}, {amax:.4g}]  "
            f"zoom {self._zoom:.2f}x"
        )
        if readout:
            footer = f"{readout}    |    {footer}"
        imgui.text(footer)

        imgui.end()

    def cleanup(self) -> None:
        for gpu in self._gpu_cache.values():
            gpu.destroy()
        self._gpu_cache.clear()
        self._array_cache.clear()
        self._hist_cache.clear()
