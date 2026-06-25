"""ProjectionsViewer - browse XY/XZ/YZ projection TIFFs for a loaded isoview stack.

The widget activates whenever the active array's ``projections()`` method
returns a non-empty dict (see :class:`mbo_utilities.arrays.isoview.IsoviewArray`).
Each stack subclass knows its own projection directory, so the widget no
longer scans the dataset root — it just consumes what the array reports.

Side panel: an "Open viewer" button plus a white-text overview of what the
stack has (axes, views, timepoint count). The popup keeps the full
contrast / cmap / pan-zoom / save UX.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from imgui_bundle import imgui, portable_file_dialogs as pfd

from mbo_utilities.gui._imgui_helpers import (
    button_width,
    draw_toolbar_row,
    set_tooltip,
)
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.summary_image import (
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


_AXIS_DISPLAY = {"xy": "XY (max-Z)", "xz": "XZ (max-Y)", "yz": "YZ (max-X)"}
_WHITE = (1.0, 1.0, 1.0, 1.0)

# bound the GPU texture cache so scrubbing through 800+ projections doesn't
# blow VRAM. one projection is ~1-3 MiB; 16 textures is ~50 MiB worst case.
_MAX_GPU_CACHE = 16


def _timepoints_for_axis(group: dict, axis: str) -> list[int]:
    """Union of tiles/timepoints across all views for ``axis``.

    Used for the slider so a tile acquired on only some cameras (e.g. a
    dropped CM0/CM1) still appears; the image area shows "no projection
    for current selection" when the active view lacks that tile.
    """
    return sorted({t for (a, v, t) in group["files"] if a == axis})


class ProjectionsViewer(Widget):
    """Browse XY/XZ/YZ projections produced for the loaded isoview stack."""

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
        self._axis: str | None = None
        self._view: str | None = None
        self._timepoint: int | None = None

        # discovery cache (lazy, sourced from arr.projections())
        self._projections: dict | None = None
        self._stack_type: str | None = None
        self._is_tiled: bool = False

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
            proj = getattr(arr, "projections", None)
            if callable(proj):
                try:
                    if proj():
                        return True
                except Exception:
                    continue
        return False

    def _backend(self):
        try:
            return self.parent._figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _active_array(self):
        for arr in self.parent._get_data_arrays():
            proj = getattr(arr, "projections", None)
            if callable(proj):
                try:
                    if proj():
                        return arr
                except Exception:
                    continue
        return None

    def _ensure_projections(self) -> bool:
        if self._projections is not None:
            return bool(self._projections)
        arr = self._active_array()
        if arr is None:
            self._projections = {}
            return False
        try:
            result = arr.projections()
        except Exception:
            result = None
        if not result:
            self._projections = {}
            return False
        self._projections = result
        self._stack_type = str((arr.metadata or {}).get("stack_type", ""))
        self._is_tiled = bool(getattr(arr, "is_tiled", False))
        # initialize selection to the first plausible entry
        axes = result.get("axes") or []
        views = result.get("views") or []
        self._axis = axes[0] if axes else None
        self._view = views[0] if views else None
        if self._axis:
            tps = _timepoints_for_axis(result, self._axis)
            self._timepoint = tps[0] if tps else None
        return True

    def _current_path(self) -> Path | None:
        if not (self._projections and self._axis and self._view):
            return None
        if self._timepoint is None:
            return None
        return self._projections["files"].get(
            (self._axis, self._view, self._timepoint)
        )

    def _selection_key(self) -> tuple:
        return (self._stack_type, self._axis, self._view, self._timepoint)

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

    def _draw_overview(self) -> None:
        """Compact white-text summary of what projections this stack has."""
        g = self._projections
        axes = g.get("axes") or []
        views = g.get("views") or []
        files = g.get("files") or {}
        if not files:
            imgui.text_colored(STATUS_ERROR_COLOR, "no projections")
            return

        tps_per_view = {
            v: len({t for (a, vv, t) in files if vv == v})
            for v in views
        }
        max_t = max(tps_per_view.values()) if tps_per_view else 0
        axes_str = ", ".join(a.upper() for a in axes)

        unit = "tiles" if self._is_tiled else "timepoints"
        imgui.text_colored(_WHITE, f"{len(files)} files")
        imgui.text_colored(_WHITE, f"axes: {axes_str}")
        imgui.text_colored(_WHITE, f"views: {', '.join(views)}")
        imgui.text_colored(_WHITE, f"{unit}: {max_t}")

    def _draw_popup_selectors(self) -> None:
        """Axis / View / Timepoint selectors inside the popup viewer."""
        g = self._projections
        axes = g.get("axes") or []
        views = g.get("views") or []
        if not axes or not views:
            return

        if self._axis not in axes:
            self._axis = axes[0]
            self._on_selection_changed()
        if self._view not in views:
            self._view = views[0]
            self._on_selection_changed()

        axis_labels = [_AXIS_DISPLAY.get(a, a) for a in axes]

        def _axis():
            ch, v = imgui.combo("##projaxis", axes.index(self._axis), axis_labels)
            if ch:
                self._axis = axes[v]
                self._on_selection_changed()

        def _view():
            ch, v = imgui.combo("##projview", views.index(self._view), views)
            if ch:
                self._view = views[v]
                self._on_selection_changed()

        items = [("Axis", 120.0, _axis), ("View", 100.0, _view)]

        # Union across views so a tile present on only some cameras is
        # still reachable; the image area reports when the active view
        # lacks the selected tile.
        tps = _timepoints_for_axis(g, self._axis)
        if tps:
            if self._timepoint not in tps:
                self._timepoint = tps[0]
                self._on_selection_changed()
            cur_pos = tps.index(self._timepoint)
            if self._is_tiled:
                label = "Tile"
                lo, hi = tps[0], tps[-1]
                if isinstance(lo, int):
                    value_fmt = f"%d / {len(tps) - 1}  (SPM{lo:02d}-SPM{hi:02d})"
                else:
                    value_fmt = f"%d / {len(tps) - 1}  ({lo}-{hi})"
            else:
                label = "T"
                value_fmt = f"%d / {len(tps) - 1}  (TM{tps[0]:06d}-{tps[-1]:06d})"

            def _tp():
                ch, v = imgui.slider_int(
                    "##projtp", cur_pos, 0, len(tps) - 1, value_fmt,
                )
                if ch:
                    self._timepoint = tps[int(v)]
                    self._on_selection_changed()
            items.append((label, 220.0, _tp))

        draw_toolbar_row(items)
        if not tps:
            imgui.text_colored(
                STATUS_ERROR_COLOR, f"no {self._axis} projections",
            )

    def draw(self) -> None:
        if not self._ensure_projections():
            return

        self._sync_cmap_with_fpl()
        draw_section_header("Projections")

        imgui.indent(8)
        try:
            if imgui.button("Open viewer##projections_open"):
                self._popup_open = True
                self._reset_view()

            self._draw_overview()
        finally:
            imgui.unindent(8)

        self._poll_save_dialog()

        if self._popup_open:
            self._draw_popup()

    def _draw_toolbar(self) -> None:
        self._draw_popup_selectors()

        def _cmap():
            ch, v = imgui.combo("##projcmap", self._cmap_idx, list(self._cmaps))
            if ch:
                self._cmap_idx = v

        def _contrast():
            ch, v = imgui.combo(
                "##projcontrast", self._contrast_mode, list(_CONTRAST_MODES)
            )
            if ch:
                self._contrast_mode = v

        def _reset():
            if imgui.button("Reset##projections"):
                self._reset_view()

        def _save():
            if imgui.button("Save...##projections"):
                self._open_save_dialog()
            set_tooltip(
                "Save the colormapped projection as a PNG (native resolution)."
            )

        draw_toolbar_row([
            ("Cmap", 110.0, _cmap),
            ("Contrast", 110.0, _contrast),
            (None, button_width("Reset"), _reset),
            (None, button_width("Save..."), _save),
        ])

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
        start_dir = str(path.parent) if path else ""
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
            path = self._projections["files"].get((key[1], key[2], key[3]))
            if path is None:
                return
            arr = self._array_cache.get((str(path),))
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
