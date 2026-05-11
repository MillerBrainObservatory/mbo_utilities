"""
SummaryImageViewer - browse 2D summary images stored in ops.npy.

Activates whenever a loaded array's metadata contains image-like 2D
ndarrays (typical case: BinArray that auto-loaded a sibling ops.npy).
Shows a selector in the side panel and a floating imgui popup window
with pan, zoom, colormap, contrast control, pixel-value overlay,
suite2p ROI overlay (when stat.npy is present), and PNG export.

Renders through fastplotlib's wgpu imgui backend (immvision can't be
used here - it requires OpenGL).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import wgpu
from cmap import Colormap
from imgui_bundle import imgui, hello_imgui, portable_file_dialogs as pfd

from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.widgets._base import Widget


SIDE_PANEL_SECTION_COLOR = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)
STATUS_OK_COLOR = imgui.ImVec4(0.3, 1.0, 0.3, 1.0)
STATUS_ERROR_COLOR = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)


def draw_section_header(title: str) -> None:
    imgui.spacing()
    imgui.separator()
    imgui.text_colored(SIDE_PANEL_SECTION_COLOR, title)
    imgui.spacing()


def center_popup_on_open(
    default_em: tuple[float, float] = (50.0, 55.0),
    min_em: tuple[float, float] = (35.0, 28.0),
    max_screen_frac: float = 0.92,
) -> None:
    viewport = imgui.get_main_viewport()
    em = hello_imgui.em_size(1.0)
    vw, vh = viewport.size.x, viewport.size.y
    w = min(default_em[0] * em, vw * max_screen_frac)
    h = min(default_em[1] * em, vh * max_screen_frac)
    imgui.set_next_window_size(imgui.ImVec2(w, h), imgui.Cond_.first_use_ever)
    imgui.set_next_window_pos(
        viewport.get_center(),
        imgui.Cond_.first_use_ever,
        pivot=imgui.ImVec2(0.5, 0.5),
    )
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(min_em[0] * em, min_em[1] * em),
        imgui.ImVec2(vw * max_screen_frac, vh * max_screen_frac),
    )


_PRIORITY_KEYS = (
    "meanImg",
    "meanImgE",
    "max_proj",
    "Vcorr",
    "refImg",
    "refImg0",
    "sdmov",
)

from mbo_utilities.gui._colormaps import (
    DEFAULT_COLORMAPS as _DEFAULT_COLORMAPS,
    DEFAULT_COLORMAP as _DEFAULT_COLORMAP,
)

_CONTRAST_MODES = ("Full", "Auto", "Manual")
_CONTRAST_AUTO = 1
_CONTRAST_MANUAL = 2

_PIXEL_VALUES_MIN_ZOOM = 16.0
_PIXEL_VALUES_MAX_CELLS = 10_000


def _is_image_like(v: Any) -> bool:
    return (
        isinstance(v, np.ndarray)
        and v.ndim == 2
        and v.size > 1
        and np.issubdtype(v.dtype, np.number)
    )


def _collect_keys(metadata: dict) -> list[str]:
    if not metadata:
        return []
    found = [k for k in _PRIORITY_KEYS if _is_image_like(metadata.get(k))]
    extras = sorted(
        k for k, v in metadata.items()
        if k not in _PRIORITY_KEYS and _is_image_like(v)
    )
    return found + extras


def _data_range(arr: np.ndarray) -> tuple[float, float]:
    a = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _auto_range(arr: np.ndarray) -> tuple[float, float]:
    a = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanpercentile(a, 1.0))
    hi = float(np.nanpercentile(a, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _to_rgba(arr: np.ndarray, cmap_name: str, lo: float, hi: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    span = max(hi - lo, 1e-12)
    n = np.clip((a - lo) / span, 0.0, 1.0)
    rgba = (Colormap(cmap_name)(n) * 255).astype(np.uint8)
    return np.ascontiguousarray(rgba)


def _format_value(v: float, dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return f"{int(v)}"
    av = abs(v)
    if av != 0 and (av < 0.01 or av >= 10000):
        return f"{v:.1e}"
    return f"{v:.2f}"


class _GpuImage:
    """Owns one wgpu texture + its imgui registration for a single image."""

    def __init__(self, backend, arr: np.ndarray, cmap: str, lo: float, hi: float):
        self.backend = backend
        self.arr = arr
        self.cmap = cmap
        self.lo = lo
        self.hi = hi
        self.h, self.w = arr.shape
        self._texture = None
        self._view = None
        self.ref = None
        self.rgba: np.ndarray | None = None
        self._upload()

    def _upload(self) -> None:
        self.rgba = _to_rgba(self.arr, self.cmap, self.lo, self.hi)
        device = self.backend._device
        self._texture = device.create_texture(
            size=(self.w, self.h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        device.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            self.rgba.tobytes(),
            {"offset": 0, "bytes_per_row": self.w * 4, "rows_per_image": self.h},
            (self.w, self.h, 1),
        )
        self._view = self._texture.create_view()
        self.ref = self.backend.register_texture(self._view)

    def reupload_if_changed(self, cmap: str, lo: float, hi: float) -> None:
        if cmap == self.cmap and lo == self.lo and hi == self.hi:
            return
        self.destroy()
        self.cmap = cmap
        self.lo = lo
        self.hi = hi
        self._upload()

    def destroy(self) -> None:
        if self.ref is not None:
            try:
                self.backend.unregister_texture(self.ref)
            except Exception:
                pass
            self.ref = None
        self._view = None
        if self._texture is not None:
            try:
                self._texture.destroy()
            except Exception:
                pass
            self._texture = None


def _array_data_path(arr) -> Path | None:
    """Best-effort lookup of the directory the array came from."""
    for attr in ("filenames", "filename"):
        v = getattr(arr, attr, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        try:
            p = Path(str(v))
        except (TypeError, ValueError):
            continue
        return p.parent if p.is_file() else p
    return None


def _load_rois(stat_dir: Path) -> tuple[list[np.ndarray], np.ndarray | None] | None:
    """Load suite2p stat.npy + iscell.npy and compute polygon contours.

    Returns (polygons, iscell_flags) or None if stat.npy not found.
    Each polygon is an (N, 2) float array of (x, y) image-pixel coords.
    """
    stat_path = stat_dir / "stat.npy"
    if not stat_path.exists():
        return None
    try:
        stat = np.load(stat_path, allow_pickle=True)
    except Exception:
        return None

    iscell_path = stat_dir / "iscell.npy"
    iscell = None
    if iscell_path.exists():
        try:
            iscell_arr = np.load(iscell_path, allow_pickle=True)
            iscell = iscell_arr[:, 0].astype(bool) if iscell_arr.ndim == 2 else iscell_arr.astype(bool)
        except Exception:
            iscell = None

    try:
        from skimage.measure import find_contours
    except ImportError:
        return None

    polygons: list[np.ndarray] = []
    for roi in stat:
        try:
            ypix = np.asarray(roi["ypix"], dtype=np.int32)
            xpix = np.asarray(roi["xpix"], dtype=np.int32)
        except (KeyError, TypeError):
            polygons.append(np.zeros((0, 2), dtype=np.float32))
            continue
        if ypix.size == 0:
            polygons.append(np.zeros((0, 2), dtype=np.float32))
            continue
        y0, x0 = int(ypix.min()), int(xpix.min())
        y1, x1 = int(ypix.max()), int(xpix.max())
        mask = np.zeros((y1 - y0 + 3, x1 - x0 + 3), dtype=np.uint8)
        mask[ypix - y0 + 1, xpix - x0 + 1] = 1
        contours = find_contours(mask, 0.5)
        if not contours:
            polygons.append(np.zeros((0, 2), dtype=np.float32))
            continue
        # take the longest contour (outer boundary of largest component)
        c = max(contours, key=len).astype(np.float32)
        # contour is (row, col) -> convert to (x, y) and shift back to image coords
        poly = np.empty_like(c)
        poly[:, 0] = c[:, 1] + x0 - 1
        poly[:, 1] = c[:, 0] + y0 - 1
        polygons.append(poly)
    return polygons, iscell


class SummaryImageViewer(Widget):
    """Browse 2D summary images stored in an array's metadata."""

    name = "Summary Images"
    priority = 60

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._selected: int = 0
        self._popup_open: bool = False
        self._cmaps: list[str] = list(_DEFAULT_COLORMAPS)
        self._cmap_idx: int = self._cmaps.index(_DEFAULT_COLORMAP)
        self._cmap_synced_with_fpl: bool = False
        self._contrast_mode: int = _CONTRAST_AUTO
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._needs_fit: bool = True
        self._show_pixel_values: bool = False
        self._show_rois: bool = False
        self._iscell_only: bool = True
        self._gpu: dict[str, _GpuImage] = {}
        self._manual_lo: dict[str, float] = {}
        self._manual_hi: dict[str, float] = {}
        self._hist_cache: dict[str, np.ndarray] = {}
        self._roi_cache: dict[Path, tuple[list[np.ndarray], np.ndarray | None] | None] = {}
        self._save_dialog: Any = None
        self._last_save_msg: str = ""

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        for arr in parent._get_data_arrays():
            md = getattr(arr, "metadata", None) or {}
            if _collect_keys(md):
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
            if _collect_keys(md):
                return arr
        return None

    def _active_metadata(self) -> dict:
        arr = self._active_array()
        return (getattr(arr, "metadata", None) or {}) if arr is not None else {}

    def _sync_cmap_with_fpl(self) -> None:
        """Adopt the parent ImageWidget's colormap as our default once.

        Done on first draw rather than __init__ so the parent's image_widget
        is reliably attached. If fpl is using a colormap we don't carry by
        default (any cmap string), prepend it so the user's choice stays
        available in the selector.
        """
        if self._cmap_synced_with_fpl:
            return
        iw = getattr(self.parent, "image_widget", None)
        if iw is None:
            return
        try:
            graphics = [nd.graphic for nd in iw.ndgraphics]
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

    def _get_range(self, key: str, arr: np.ndarray) -> tuple[float, float]:
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

    def _ensure_gpu(self, key: str, arr: np.ndarray) -> _GpuImage | None:
        backend = self._backend()
        if backend is None:
            return None
        cmap = self._cmaps[self._cmap_idx]
        lo, hi = self._get_range(key, arr)
        gpu = self._gpu.get(key)
        if gpu is None or gpu.arr is not arr:
            if gpu is not None:
                gpu.destroy()
            gpu = _GpuImage(backend, arr, cmap, lo, hi)
            self._gpu[key] = gpu
        else:
            gpu.reupload_if_changed(cmap, lo, hi)
        return gpu

    def _get_histogram(self, key: str, arr: np.ndarray) -> np.ndarray:
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

    def _get_rois(self):
        arr = self._active_array()
        if arr is None:
            return None
        data_dir = _array_data_path(arr)
        if data_dir is None:
            return None
        if data_dir not in self._roi_cache:
            self._roi_cache[data_dir] = _load_rois(data_dir)
        return self._roi_cache[data_dir]

    def draw(self) -> None:
        md = self._active_metadata()
        keys = _collect_keys(md)
        if not keys:
            return
        if self._selected >= len(keys):
            self._selected = 0

        self._sync_cmap_with_fpl()

        draw_section_header("Summary Images")

        imgui.set_next_item_width(140)
        changed, idx = imgui.combo("##sum_img_side", self._selected, list(keys))
        if changed:
            self._selected = idx
            self._reset_view()

        key = keys[self._selected]
        arr = md[key]
        imgui.text(f"{arr.shape[0]} x {arr.shape[1]}  {arr.dtype}")

        if imgui.button("Open viewer"):
            self._popup_open = True
            self._reset_view()

        # poll any pending save dialog regardless of popup state
        self._poll_save_dialog()

        if self._popup_open:
            self._draw_popup(keys, md)

    def _draw_toolbar(self, keys: list[str], md: dict) -> str:
        imgui.set_next_item_width(140)
        img_changed, new_idx = imgui.combo("Image", self._selected, list(keys))
        if img_changed:
            self._selected = new_idx
            self._reset_view()

        imgui.same_line()
        imgui.set_next_item_width(110)
        cmap_changed, new_cmap = imgui.combo("Cmap", self._cmap_idx, list(self._cmaps))
        if cmap_changed:
            self._cmap_idx = new_cmap

        imgui.same_line()
        imgui.set_next_item_width(110)
        ctr_changed, new_ctr = imgui.combo("Contrast", self._contrast_mode, list(_CONTRAST_MODES))
        if ctr_changed:
            self._contrast_mode = new_ctr

        imgui.same_line()
        if imgui.button("Reset"):
            self._reset_view()

        imgui.same_line()
        if imgui.button("Save..."):
            self._open_save_dialog(keys[self._selected], md[keys[self._selected]])
        set_tooltip("Save the colormapped image as a PNG (native resolution).")

        # row 2: overlays
        _, self._show_pixel_values = imgui.checkbox("Pixel values", self._show_pixel_values)
        set_tooltip(
            f"Draw the numeric value of each pixel on top of it.\n"
            f"Visible only when zoomed past {int(_PIXEL_VALUES_MIN_ZOOM)}x."
        )
        imgui.same_line()
        _, self._show_rois = imgui.checkbox("Show ROIs", self._show_rois)
        set_tooltip(
            "Overlay suite2p cell contours from stat.npy.\n"
            "Cyan = iscell, red = not-cell. Requires stat.npy in the data dir."
        )
        if self._show_rois:
            imgui.same_line()
            _, self._iscell_only = imgui.checkbox("iscell only", self._iscell_only)
            set_tooltip("Hide ROIs flagged as not-cell in iscell.npy.")

        if self._last_save_msg:
            color = STATUS_ERROR_COLOR if self._last_save_msg.startswith("Save failed") else STATUS_OK_COLOR
            imgui.text_colored(color, self._last_save_msg)

        return keys[self._selected]

    def _draw_contrast_panel(self, key: str, arr: np.ndarray) -> None:
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
        # tight bordered frame: 24-px histogram strip on the left, two
        # min/max sliders on the right side of the same row. Total height
        # ~30 px including child padding so it doesn't crowd the image.
        if imgui.begin_child(
            "##levels",
            imgui.ImVec2(0, 32),
            child_flags=imgui.ChildFlags_.borders,
        ):
            avail_w = max(imgui.get_content_region_avail().x, 100.0)
            hist_w = avail_w * 0.32
            slider_w = (avail_w - hist_w - 28) * 0.5

            imgui.plot_histogram(
                "##hist", bins,
                graph_size=imgui.ImVec2(hist_w, 22),
            )
            imgui.same_line()
            imgui.set_next_item_width(slider_w)
            ch_lo, new_lo = imgui.slider_float("min", lo, data_lo, data_hi, "%.4g")
            imgui.same_line()
            imgui.set_next_item_width(slider_w)
            ch_hi, new_hi = imgui.slider_float("max", hi, data_lo, data_hi, "%.4g")
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
        # visible image-pixel rect
        x0_img = max(0, int(np.floor(-self._pan_x / self._zoom)))
        y0_img = max(0, int(np.floor(-self._pan_y / self._zoom)))
        x1_img = min(gpu.w, int(np.ceil((canvas_size.x - self._pan_x) / self._zoom)) + 1)
        y1_img = min(gpu.h, int(np.ceil((canvas_size.y - self._pan_y) / self._zoom)) + 1)
        if x1_img <= x0_img or y1_img <= y0_img:
            return
        n = (x1_img - x0_img) * (y1_img - y0_img)
        if n > _PIXEL_VALUES_MAX_CELLS:
            return

        # luminance of colormapped pixels to pick text color
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
                # rough centering
                tw = len(txt) * 6.0
                tx = sx - tw * 0.5
                ty = sy - 6.5
                color = black if luma[y, x] > 140 else white
                draw_list.add_text(imgui.ImVec2(tx, ty), color, txt)

    def _draw_rois(
        self,
        draw_list,
        arr_shape: tuple[int, int],
        md: dict,
        canvas_pos,
        canvas_size,
    ) -> None:
        rois = self._get_rois()
        if rois is None:
            return
        polygons, iscell = rois
        if not polygons:
            return

        # offset for cropped images (Vcorr / max_proj are in xrange/yrange space)
        ly = md.get("Ly")
        lx = md.get("Lx")
        h_img, w_img = arr_shape
        ox, oy = 0.0, 0.0
        if ly and lx and (h_img != ly or w_img != lx):
            xrange = md.get("xrange")
            yrange = md.get("yrange")
            if xrange is not None and yrange is not None:
                ox = -float(xrange[0])
                oy = -float(yrange[0])
            else:
                return  # can't safely place ROIs

        z = self._zoom
        clip_x1 = canvas_pos.x + canvas_size.x
        clip_y1 = canvas_pos.y + canvas_size.y
        cyan = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.2, 0.95, 0.95, 0.9))
        red = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.95, 0.3, 0.3, 0.7))

        for i, poly in enumerate(polygons):
            if poly.shape[0] < 3:
                continue
            cell = True if iscell is None else bool(iscell[i]) if i < len(iscell) else True
            if self._iscell_only and not cell:
                continue
            color = cyan if cell else red
            pts = []
            for x_img, y_img in poly:
                sx = canvas_pos.x + self._pan_x + (x_img + ox) * z
                sy = canvas_pos.y + self._pan_y + (y_img + oy) * z
                if sx < canvas_pos.x - 8 or sy < canvas_pos.y - 8 or sx > clip_x1 + 8 or sy > clip_y1 + 8:
                    pts.clear()
                    break
                pts.append(imgui.ImVec2(sx, sy))
            if len(pts) >= 3:
                draw_list.add_polyline(pts, color, imgui.ImDrawFlags_.closed, 1.5)

    def _open_save_dialog(self, key: str, arr: np.ndarray) -> None:
        if self._save_dialog is not None:
            return
        cmap = self._cmaps[self._cmap_idx]
        default = f"{key}_{cmap}.png"
        start_dir = ""
        a = self._active_array()
        if a is not None:
            d = _array_data_path(a)
            if d is not None:
                start_dir = str(d)
        try:
            self._save_dialog = pfd.save_file(
                "Save summary image as PNG",
                str(Path(start_dir) / default) if start_dir else default,
                ["PNG files", "*.png", "All files", "*"],
            )
        except Exception as e:
            self._last_save_msg = f"Save failed: {e}"
            self._save_dialog = None
        self._pending_save_key = key

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
            path = self._save_dialog.result()
        except Exception:
            path = ""
        self._save_dialog = None
        if not path:
            return
        try:
            key = getattr(self, "_pending_save_key", None)
            if key is None:
                return
            md = self._active_metadata()
            arr = md.get(key)
            if arr is None:
                return
            cmap = self._cmaps[self._cmap_idx]
            lo, hi = self._get_range(key, arr)
            rgba = _to_rgba(arr, cmap, lo, hi)
            import imageio.v3 as iio
            target = Path(path)
            if target.suffix.lower() != ".png":
                target = target.with_suffix(".png")
            iio.imwrite(str(target), rgba)
            self._last_save_msg = f"Saved {target.name}"
        except Exception as e:
            self._last_save_msg = f"Save failed: {e}"

    def _draw_popup(self, keys: list[str], md: dict) -> None:
        key = keys[self._selected]
        arr = md[key]

        center_popup_on_open(default_em=(52.0, 56.0), min_em=(38.0, 30.0))
        opened, self._popup_open = imgui.begin(
            f"Summary Image: {key}###summary_image_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            imgui.end()
            return

        key = self._draw_toolbar(keys, md)
        arr = md[key]
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
            "##canvas",
            imgui.ImVec2(0, -28),
            child_flags=0,
            window_flags=imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse,
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

        imgui.invisible_button("##pan_capture", imgui.ImVec2(cw, ch))
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
        if self._show_rois:
            self._draw_rois(draw_list, arr.shape, md, canvas_pos, canvas_size)

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
        footer = f"{h}x{w}  {arr.dtype}  range [{amin:.4g}, {amax:.4g}]  zoom {self._zoom:.2f}x"
        if readout:
            footer = f"{readout}    |    {footer}"
        imgui.text(footer)

        imgui.end()

    def cleanup(self) -> None:
        for gpu in self._gpu.values():
            gpu.destroy()
        self._gpu.clear()
        self._hist_cache.clear()
        self._roi_cache.clear()
