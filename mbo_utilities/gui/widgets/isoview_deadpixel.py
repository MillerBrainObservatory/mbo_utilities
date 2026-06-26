"""Standalone Isoview dead-pixel correction tuning tool.

Floating ``imgui.begin`` window with sliders for ``background_percentile``
and median ``kernel_size`` plus a live overlay of pixels the dead-pixel
detector would flag on the XY max-projection per view.

The params live on the active :class:`IsoviewPipelineWidget` instance
(``_correct_*`` fields). The window reads + writes them in place so the
popup form and the preview stay in lockstep; Apply just closes, Cancel
restores the snapshot taken on open.

Approximation note: the real :func:`isoview.corrections.correct_dead_pixels`
computes std + mean projections along Z and runs a max-distance-from-linear
threshold on each. We only have XY max-projections cached on disk, so the
preview applies the same median-vs-projection deviation threshold to the
max-projection. The kernel size and threshold-determination math are
identical; the input statistic differs.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6 as fa
from scipy.ndimage import median_filter

from mbo_utilities.gui._imgui_helpers import button_width, draw_toolbar_row


_DEFAULT_CAMERA_VIEW_MAP = {0: 0, 1: 0, 2: 90, 3: 90}
_VIEW_COLORS = {0: (1.0, 0.35, 0.35, 1.0), 90: (1.0, 0.95, 0.4, 1.0)}
_CAMERA_COLORS = {
    0: (1.0, 0.35, 0.35, 1.0),
    1: (1.0, 0.85, 0.4, 1.0),
    2: (0.4, 0.9, 0.6, 1.0),
    3: (0.5, 0.75, 1.0, 1.0),
}
_TINT_RGB = (1.0, 0.35, 0.85)

_SLIDER_W = 220
_DISPLAY_DRAG_W = 110
_MAX_FILTER_CACHE = 8
_TARGET_PROJ_PX = 512       # preview math runs at <= this longest side


def _downsample_max(proj: np.ndarray, cap: int = _TARGET_PROJ_PX) -> tuple[np.ndarray, int]:
    """Max-pool downsample so the longest side is <= ``cap``.

    Max pooling keeps hot/dead pixels (the thing being detected) instead
    of averaging them away. Returns ``(downsampled float32, factor)``;
    factor 1 means no downsampling.
    """
    h, w = proj.shape
    longest = max(h, w)
    if longest <= cap:
        return np.ascontiguousarray(proj, dtype=np.float32), 1
    factor = int(np.ceil(longest / cap))
    hh = (h // factor) * factor
    ww = (w // factor) * factor
    blocks = proj[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor)
    ds = blocks.max(axis=(1, 3))
    return np.ascontiguousarray(ds, dtype=np.float32), factor


def _camera_view_map(arr: Any) -> dict[int, int]:
    meta = getattr(arr, "metadata", None) or {}
    m = meta.get("camera_view_map")
    if isinstance(m, dict):
        return {int(k): int(v) for k, v in m.items()}
    return dict(_DEFAULT_CAMERA_VIEW_MAP)


def _view_int_from_label(label: str) -> int | None:
    """Camera index for a projections-dict label.

    ``VW{angle}`` maps the view angle back to its camera (VW00->0,
    VW180->1, VW90->2, VW270->3); ``CM##`` and bare ints are the camera
    index directly.
    """
    if not isinstance(label, str):
        return None
    if label.startswith("VW") and label[2:].isdigit():
        from mbo_utilities.arrays.isoview.array import camera_from_view_label
        cam = camera_from_view_label(label)
        return cam if cam is not None else int(label[2:])
    if label.startswith("CM") and label[2:].isdigit():
        return int(label[2:])
    try:
        return int(label)
    except ValueError:
        return None


def _is_raw(arr: Any) -> bool:
    return str(getattr(arr, "kind", "") or "").lower() == "raw"


def _tp_label(slot) -> str:
    """Display label for a tiled projection slot: a specimen_name grid token
    (string) as-is, else SPM## for a legacy integer slot."""
    return f"SPM{slot:02d}" if isinstance(slot, int) else str(slot)


def _build_projection_index(
    arr: Any, projections: dict | None,
) -> dict[int, dict[int, Path]]:
    if not projections:
        return {}
    if "xy" not in (projections.get("axes") or []):
        return {}
    cv = _camera_view_map(arr)
    raw_mode = _is_raw(arr)

    per_view_by_label: dict[int, dict[str, dict[int, Path]]] = {}
    for (axis, label, t), path in (projections.get("files") or {}).items():
        if axis != "xy":
            continue
        raw_int = _view_int_from_label(label)
        if raw_int is None:
            continue
        if label.startswith("CM"):
            key = raw_int if raw_mode else cv.get(raw_int, raw_int)
        else:
            key = raw_int
        per_view_by_label.setdefault(key, {}).setdefault(label, {})[t] = Path(path)

    out: dict[int, dict[int, Path]] = {}
    for key, by_label in per_view_by_label.items():
        if not by_label:
            continue
        chosen = sorted(by_label.keys())[0]
        out[key] = dict(by_label[chosen])
    return out


def _common_timepoints(index: dict[int, dict[int, Path]]) -> list[int]:
    """Every tile/timepoint present in ANY camera (union).

    Union, not intersection: a tile acquired on only some cameras (e.g. a
    dropped CM0/CM1) must still appear so the user can work with the
    cameras that exist. Panels for cameras lacking that tile render blank.
    """
    if not index:
        return []
    tps: set[int] = set()
    for d in index.values():
        tps |= set(d.keys())
    return sorted(tps)


def _determine_threshold(
    sorted_array: np.ndarray, max_samples: int = 50000,
) -> float:
    """Mirror of :func:`isoview.corrections._determine_threshold`."""
    n = sorted_array.size
    if n == 0:
        return 0.0
    if n > max_samples:
        step = max(1, round(n / max_samples))
        sorted_array = sorted_array[::step]
        n = sorted_array.size
    x = np.arange(1, n + 1)
    p1 = np.array([1.0, float(sorted_array[0])])
    p2 = np.array([float(n), float(sorted_array[-1])])
    vec = p2 - p1
    norm2 = float(np.linalg.norm(vec) ** 2) or 1.0
    points = np.column_stack([x.astype(np.float64), sorted_array.astype(np.float64)])
    h = (points - p1) @ vec / norm2
    proj = p1 + h[:, np.newaxis] * vec
    distances = np.linalg.norm(points - proj, axis=1)
    return float(sorted_array[int(np.argmax(distances))])


def _detect_dead_pixels(
    proj: np.ndarray, background: float, kernel: int,
) -> tuple[np.ndarray, float, float]:
    """Approximate dead-pixel detection on a 2D projection.

    Returns ``(mask, dev_thresh, mean_thresh)``.

    Combines two checks the real algorithm does, here run on the same
    XY projection instead of std/mean along Z:

    - absolute deviation: |proj - median(proj)| > T_dev
    - relative deviation: |proj - median(proj)| / (median + ε) > T_rel,
      where median uses background-subtracted projection.
    """
    if kernel < 1:
        kernel = 1
    if kernel % 2 == 0:
        kernel += 1
    k = (int(kernel), int(kernel))

    f = proj.astype(np.float32)
    med_abs = median_filter(f, size=k, mode="reflect")
    dist_abs = np.abs(f - med_abs)
    t_abs = _determine_threshold(np.sort(dist_abs.ravel()))
    mask_abs = dist_abs > t_abs

    fb = f - float(background)
    med_rel = median_filter(fb, size=k, mode="reflect")
    dist_rel = np.abs((fb - med_rel) / (med_rel + 1e-10))
    t_rel = _determine_threshold(np.sort(dist_rel.ravel()))
    mask_rel = dist_rel > t_rel

    return (mask_abs | mask_rel), float(t_abs), float(t_rel)


def _compose_rgba(
    proj: np.ndarray, mask: np.ndarray, vmin: float, vmax: float,
) -> np.ndarray:
    denom = max(1e-6, float(vmax) - float(vmin))
    g = np.clip((proj.astype(np.float32) - float(vmin)) / denom, 0.0, 1.0)
    g8 = (g * 255).astype(np.uint8)

    rgba = np.empty(proj.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = g8
    rgba[..., 1] = g8
    rgba[..., 2] = g8
    rgba[..., 3] = 255

    if mask.any():
        r, gT, bT = _TINT_RGB
        rgba[mask, 0] = np.clip(r * 255, 0, 255).astype(np.uint8)
        rgba[mask, 1] = np.clip(gT * 255, 0, 255).astype(np.uint8)
        rgba[mask, 2] = np.clip(bT * 255, 0, 255).astype(np.uint8)
    return rgba


class _GpuRGBA:
    def __init__(self, backend, rgba: np.ndarray):
        import wgpu
        self._wgpu = wgpu
        self.backend = backend
        self.h, self.w = int(rgba.shape[0]), int(rgba.shape[1])
        self._texture = None
        self._view = None
        self.ref = None
        self._upload(rgba)

    def _write(self, rgba: np.ndarray) -> None:
        device = self.backend._device
        device.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(rgba).tobytes(),
            {"offset": 0, "bytes_per_row": self.w * 4, "rows_per_image": self.h},
            (self.w, self.h, 1),
        )

    def _upload(self, rgba: np.ndarray) -> None:
        wgpu = self._wgpu
        device = self.backend._device
        self._texture = device.create_texture(
            size=(self.w, self.h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._write(rgba)
        self._view = self._texture.create_view()
        self.ref = self.backend.register_texture(self._view)

    def reupload(self, rgba: np.ndarray) -> None:
        if rgba.shape[0] != self.h or rgba.shape[1] != self.w:
            self.destroy()
            self.h, self.w = int(rgba.shape[0]), int(rgba.shape[1])
            self._upload(rgba)
            return
        # Same size: write into the existing texture, keeping the view and
        # backend registration so we don't churn a GPU alloc + re-register
        # every frame a slider moves.
        self._write(rgba)

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


def _ensure_state(parent: Any) -> None:
    if not hasattr(parent, "_show_iso_dp_window"):
        parent._show_iso_dp_window = False
    if not hasattr(parent, "_iso_dp_window_open"):
        parent._iso_dp_window_open = False
    if not hasattr(parent, "_iso_dp_snapshot"):
        parent._iso_dp_snapshot: dict[str, float] | None = None
    if not hasattr(parent, "_iso_dp_cache_key"):
        parent._iso_dp_cache_key = None
    if not hasattr(parent, "_iso_dp_proj_index"):
        parent._iso_dp_proj_index: dict[int, dict[int, Path]] = {}
    if not hasattr(parent, "_iso_dp_timepoints"):
        parent._iso_dp_timepoints: list[int] = []
    if not hasattr(parent, "_iso_dp_current_tp"):
        parent._iso_dp_current_tp = 0
    if not hasattr(parent, "_iso_dp_proj_cache"):
        parent._iso_dp_proj_cache: dict[tuple[int, int], np.ndarray] = {}
    if not hasattr(parent, "_iso_dp_mask_cache"):
        parent._iso_dp_mask_cache: dict[tuple, tuple[np.ndarray, float, float]] = {}
        parent._iso_dp_mask_lru: list[tuple] = []
    if not hasattr(parent, "_iso_dp_gpu"):
        parent._iso_dp_gpu: dict[int, _GpuRGBA] = {}
        parent._iso_dp_compose_key: dict[int, tuple] = {}
    if not hasattr(parent, "_iso_dp_vmin"):
        parent._iso_dp_vmin = 0.0
    if not hasattr(parent, "_iso_dp_vmax"):
        parent._iso_dp_vmax = 1.0
    if not hasattr(parent, "_iso_dp_display_inited"):
        parent._iso_dp_display_inited = False
    if not hasattr(parent, "_iso_dp_dragging"):
        parent._iso_dp_dragging = False
    if not hasattr(parent, "_iso_dp_shown"):
        # view -> last computed {tp, bg, kernel, enabled, mask, t_abs,
        # t_rel}; held while a slider is dragged so the expensive median
        # filter defers to release.
        parent._iso_dp_shown: dict[int, dict] = {}


def _get_iso_array(parent: Any) -> Any | None:
    iw = getattr(parent, "image_widget", None)
    if iw is None or not iw.data:
        return None
    arr = iw.data[0]
    try:
        from mbo_utilities.gui.widgets.pipelines.isoview import _unwrap_array
        return _unwrap_array(arr)
    except Exception:
        return arr


def _get_iso_widget(parent: Any) -> Any | None:
    instances = getattr(parent, "_pipeline_instances", None) or {}
    return instances.get("Isoview")


def _get_backend(parent: Any) -> Any | None:
    iw = getattr(parent, "image_widget", None)
    if iw is None:
        return None
    try:
        return iw.figure.imgui_renderer.backend
    except AttributeError:
        return None


def _load_projection(path: Path) -> np.ndarray | None:
    try:
        import tifffile
    except ImportError:
        return None
    try:
        a = tifffile.imread(str(path))
    except Exception:
        return None
    if a.ndim == 3:
        a = a.max(axis=0)
    if a.ndim != 2:
        return None
    return np.asarray(a, dtype=np.float32)


def _destroy_gpu(parent: Any) -> None:
    for gpu in (getattr(parent, "_iso_dp_gpu", None) or {}).values():
        try:
            gpu.destroy()
        except Exception:
            pass
    parent._iso_dp_gpu = {}
    parent._iso_dp_compose_key = {}


def _initial_display_range(parent: Any) -> tuple[float, float]:
    return 0.0, 1000.0


def _get_working(parent: Any, view: int, tp: int):
    """Return ``(downsampled projection float32, factor)`` or ``None``.

    Max-pooled once to ``_TARGET_PROJ_PX`` so the per-frame median filter
    runs on the small working image instead of the full sensor.
    """
    key = (view, tp)
    cached = parent._iso_dp_proj_cache.get(key)
    if cached is not None:
        return cached
    path = (parent._iso_dp_proj_index.get(view) or {}).get(tp)
    if path is None:
        return None
    proj = _load_projection(path)
    if proj is None:
        return None
    work = _downsample_max(proj)
    parent._iso_dp_proj_cache[key] = work
    return work


def _get_projection(parent: Any, view: int, tp: int) -> np.ndarray | None:
    work = _get_working(parent, view, tp)
    return None if work is None else work[0]


def _get_mask(
    parent: Any, view: int, tp: int, background: float, kernel: int,
) -> tuple[np.ndarray, float, float] | None:
    work = _get_working(parent, view, tp)
    if work is None:
        return None
    proj, factor = work
    key = (int(view), tp, round(float(background), 3), int(kernel), factor)
    cache = parent._iso_dp_mask_cache
    if key in cache:
        try:
            parent._iso_dp_mask_lru.remove(key)
        except ValueError:
            pass
        parent._iso_dp_mask_lru.append(key)
        return cache[key]
    # When downsampling, floor the scaled window to 3 so the median stays
    # a real filter (a scaled kernel of 1 would be an identity and flag
    # nothing). At native resolution keep the user's kernel exactly.
    if factor > 1:
        k = max(3, int(round(int(kernel) / factor)))
    else:
        k = max(1, int(kernel))
    result = _detect_dead_pixels(proj, float(background), k)
    cache[key] = result
    parent._iso_dp_mask_lru.append(key)
    while len(parent._iso_dp_mask_lru) > _MAX_FILTER_CACHE:
        old = parent._iso_dp_mask_lru.pop(0)
        cache.pop(old, None)
    return result


def _ensure_loaded_for(parent: Any, arr: Any) -> None:
    key = (id(arr), getattr(arr, "scan_root", None))
    if parent._iso_dp_cache_key == key and parent._iso_dp_proj_index:
        return
    parent._iso_dp_cache_key = key
    parent._iso_dp_proj_index = {}
    parent._iso_dp_timepoints = []
    parent._iso_dp_current_tp = 0
    parent._iso_dp_proj_cache = {}
    parent._iso_dp_mask_cache = {}
    parent._iso_dp_mask_lru = []
    parent._iso_dp_shown = {}
    parent._iso_dp_display_inited = False
    _destroy_gpu(parent)

    try:
        projections = arr.projections() if hasattr(arr, "projections") else None
    except Exception:
        projections = None
    parent._iso_dp_proj_index = _build_projection_index(arr, projections)
    parent._iso_dp_timepoints = _common_timepoints(parent._iso_dp_proj_index)
    if parent._iso_dp_timepoints:
        parent._iso_dp_current_tp = parent._iso_dp_timepoints[0]

    vmin, vmax = _initial_display_range(parent)
    parent._iso_dp_vmin = vmin
    parent._iso_dp_vmax = vmax
    parent._iso_dp_display_inited = True


def open_window(parent: Any) -> None:
    _ensure_state(parent)
    iso = _get_iso_widget(parent)
    if iso is not None:
        parent._iso_dp_snapshot = {
            "background_percentile": float(iso._correct_background_percentile),
            "median_kernel_size": int(iso._correct_median_kernel_size),
            "median_kernel_enabled": bool(iso._correct_median_kernel_enabled),
        }
    parent._show_iso_dp_window = True


def _restore_snapshot(parent: Any) -> None:
    snap = getattr(parent, "_iso_dp_snapshot", None)
    if not snap:
        return
    iso = _get_iso_widget(parent)
    if iso is None:
        return
    iso._correct_background_percentile = float(snap["background_percentile"])
    iso._correct_median_kernel_size = int(snap["median_kernel_size"])
    iso._correct_median_kernel_enabled = bool(snap["median_kernel_enabled"])


@contextmanager
def _apply_button_style():
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.10, 0.45, 0.10, 1.0))
    try:
        yield
    finally:
        imgui.pop_style_color(3)


def draw_window(parent: Any) -> None:
    _ensure_state(parent)

    was_open = parent._iso_dp_window_open
    if parent._show_iso_dp_window:
        parent._iso_dp_window_open = True
        parent._show_iso_dp_window = False
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(
            viewport.get_center(),
            imgui.Cond_.appearing,
            pivot=imgui.ImVec2(0.5, 0.5),
        )
        imgui.set_next_window_size(
            imgui.ImVec2(960, 720),
            imgui.Cond_.appearing,
        )

    if not parent._iso_dp_window_open:
        if was_open:
            _destroy_gpu(parent)
            parent._iso_dp_cache_key = None
            parent._iso_dp_proj_index = {}
        return

    arr = _get_iso_array(parent)
    iso = _get_iso_widget(parent)
    title = f"{fa.ICON_FA_BUG}  Isoview Dead-Pixel##iso_dp_window"
    flags = (
        imgui.WindowFlags_.no_collapse
        | imgui.WindowFlags_.no_docking
        | imgui.WindowFlags_.no_saved_settings
    )
    expanded, parent._iso_dp_window_open = imgui.begin(
        title, p_open=parent._iso_dp_window_open, flags=flags,
    )
    try:
        if not expanded:
            return
        if arr is None or iso is None or not hasattr(arr, "scan_root"):
            imgui.text_colored(
                imgui.ImVec4(0.95, 0.7, 0.4, 1.0),
                "Open an isoview dataset and the Run tab to tune dead-pixel correction.",
            )
            return

        _ensure_loaded_for(parent, arr)

        imgui.text_colored(
            imgui.ImVec4(0.55, 0.75, 1.0, 1.0),
            f"{arr.scan_root}",
        )
        imgui.spacing()
        _draw_display_controls(parent)
        imgui.separator()
        _draw_param_controls(parent, iso)
        imgui.separator()
        imgui.spacing()
        _draw_view_previews(parent, iso)

        imgui.separator()
        if imgui.button("Reset to defaults##iso_dp_reset", imgui.ImVec2(160, 0)):
            iso._correct_background_percentile = 5.0
            iso._correct_median_kernel_size = 3
            iso._correct_median_kernel_enabled = True
        imgui.same_line()
        if imgui.button("Cancel##iso_dp_cancel", imgui.ImVec2(120, 0)):
            _restore_snapshot(parent)
            parent._iso_dp_window_open = False
        imgui.same_line()
        with _apply_button_style():
            if imgui.button("Apply##iso_dp_apply", imgui.ImVec2(160, 0)):
                parent._iso_dp_window_open = False
    finally:
        imgui.end()


def _draw_display_controls(parent: Any) -> None:
    speed = max(1.0, abs(parent._iso_dp_vmax - parent._iso_dp_vmin) / 200.0)

    def _vmin():
        _, parent._iso_dp_vmin = imgui.drag_float(
            "##iso_dp_vmin", float(parent._iso_dp_vmin), speed, 0.0, 0.0, "%.0f",
        )

    def _vmax():
        _, parent._iso_dp_vmax = imgui.drag_float(
            "##iso_dp_vmax", float(parent._iso_dp_vmax), speed, 0.0, 0.0, "%.0f",
        )

    def _auto():
        if imgui.button("Auto##iso_dp_auto"):
            lo, hi = _initial_display_range(parent)
            parent._iso_dp_vmin = lo
            parent._iso_dp_vmax = hi

    items = [
        ("Min", _DISPLAY_DRAG_W, _vmin),
        ("Max", _DISPLAY_DRAG_W, _vmax),
        (None, button_width("Auto"), _auto),
    ]

    tiled = bool(getattr(_get_iso_array(parent), "is_tiled", False))
    tps = parent._iso_dp_timepoints
    if tps:
        try:
            cur_idx = tps.index(parent._iso_dp_current_tp)
        except ValueError:
            cur_idx = 0
        label = "Tile" if tiled else "Timepoint"
        value_fmt = (
            f"{_tp_label(tps[cur_idx])}  ({cur_idx + 1}/{len(tps)})"
            if tiled
            else f"TM{tps[cur_idx]:06d}  ({cur_idx + 1}/{len(tps)})"
        )

        def _tp():
            ch, v = imgui.slider_int(
                "##iso_dp_tp", cur_idx, 0, max(0, len(tps) - 1), value_fmt,
            )
            if ch:
                parent._iso_dp_current_tp = tps[max(0, min(v, len(tps) - 1))]
        items.append((label, 220.0, _tp))

    draw_toolbar_row(items)
    if parent._iso_dp_vmax <= parent._iso_dp_vmin:
        parent._iso_dp_vmax = parent._iso_dp_vmin + 1.0
    if not tps:
        unit = "per-tile" if tiled else "per-timepoint"
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            f"(no {unit} projections — slider values still save)",
        )


def _draw_param_controls(parent: Any, iso: Any) -> None:
    """Sets ``_iso_dp_dragging`` so the per-view median-filter recompute
    defers to slider release.
    """
    imgui.text_colored(
        imgui.ImVec4(1.0, 0.85, 0.4, 1.0), "Dead-pixel params",
    )
    imgui.spacing()
    dragging = False

    _, iso._correct_median_kernel_enabled = imgui.checkbox(
        "Enable median filter##iso_dp",
        bool(iso._correct_median_kernel_enabled),
    )

    imgui.set_next_item_width(_SLIDER_W)
    _, iso._correct_median_kernel_size = imgui.slider_int(
        "kernel##iso_dp",
        int(iso._correct_median_kernel_size), 1, 31, "%d",
    )
    dragging |= imgui.is_item_active()
    imgui.same_line()
    imgui.text_colored(
        imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
        "  N×N median window (odd values preferred)",
    )

    imgui.set_next_item_width(_SLIDER_W)
    _, iso._correct_background_percentile = imgui.slider_float(
        "background pct##iso_dp",
        float(iso._correct_background_percentile), 0.0, 50.0, "%.2f",
    )
    dragging |= imgui.is_item_active()
    imgui.same_line()
    imgui.text_colored(
        imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
        "  baseline subtracted before relative-deviation test",
    )
    parent._iso_dp_dragging = dragging


def _draw_view_previews(parent: Any, iso: Any) -> None:
    if not parent._iso_dp_proj_index:
        imgui.text_colored(
            imgui.ImVec4(0.95, 0.7, 0.4, 1.0),
            "No XY projections available for this dataset — "
            "slider values still save but the preview is empty.",
        )
        return

    views = sorted(parent._iso_dp_proj_index.keys())
    if not views:
        return

    arr = _get_iso_array(parent)
    raw_mode = arr is not None and _is_raw(arr)

    n = len(views)
    ncols = 2 if n > 2 else n
    nrows = (n + ncols - 1) // ncols
    spacing = 12.0

    avail = imgui.get_content_region_avail()
    text_h = imgui.get_text_line_height_with_spacing()
    # per cell: label line above, image box, two info lines below
    cell_text_h = text_h * 3.5
    # leave room for the separator + button row drawn after the previews
    reserve_bottom = imgui.get_frame_height_with_spacing() + text_h
    cell_w_h = (avail.x - spacing * (ncols - 1)) / ncols
    cell_w_v = (avail.y - reserve_bottom - spacing * (nrows - 1)) / nrows - cell_text_h
    cell_w = max(160.0, min(cell_w_h, cell_w_v))

    for i, view in enumerate(views):
        if i % ncols != 0:
            imgui.same_line(0.0, spacing)
        imgui.begin_group()
        try:
            _draw_one_view(parent, iso, view, cell_w, raw_mode)
        finally:
            imgui.end_group()


def _draw_one_view(parent: Any, iso: Any, view: int, cell_w: float, raw_mode: bool = False) -> None:
    from mbo_utilities.arrays.isoview.array import camera_view_label
    color = _CAMERA_COLORS.get(view, _VIEW_COLORS.get(view, (0.6, 0.8, 1.0, 1.0)))
    label_text = camera_view_label(view)
    imgui.text_colored(imgui.ImVec4(*color), label_text)

    tp = parent._iso_dp_current_tp
    bg = float(iso._correct_background_percentile)
    kernel = int(iso._correct_median_kernel_size)
    enabled = bool(iso._correct_median_kernel_enabled)

    proj = _get_projection(parent, view, tp)
    if proj is None:
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            "(no XY projection at this timepoint)",
        )
        imgui.dummy(imgui.ImVec2(cell_w, cell_w * 0.6))
        return

    # While a param slider is dragged, reuse the last computed mask instead
    # of re-running the (slow) median filter every frame; recompute on
    # release. A timepoint or enable change always recomputes.
    dragging = bool(getattr(parent, "_iso_dp_dragging", False))
    shown = parent._iso_dp_shown.get(view)
    stale = (
        shown is None
        or shown.get("tp") != tp
        or shown.get("enabled") != enabled
        or (
            not dragging
            and (shown.get("bg") != bg or shown.get("kernel") != kernel)
        )
    )
    if stale:
        result = _get_mask(parent, view, tp, bg, kernel) if enabled else None
        if result is None:
            mask = np.zeros(proj.shape, dtype=bool)
            t_abs = t_rel = 0.0
        else:
            mask, t_abs, t_rel = result
        shown = {
            "tp": tp, "bg": bg, "kernel": kernel, "enabled": enabled,
            "mask": mask, "t_abs": t_abs, "t_rel": t_rel,
        }
        parent._iso_dp_shown[view] = shown

    mask = shown["mask"]
    t_abs, t_rel = shown["t_abs"], shown["t_rel"]
    flagged_pct = 100.0 * mask.sum() / mask.size if mask.size else 0.0

    compose_key = (
        view, shown["tp"], shown["bg"], shown["kernel"], shown["enabled"],
        round(float(parent._iso_dp_vmin), 4),
        round(float(parent._iso_dp_vmax), 4),
    )
    backend = _get_backend(parent)
    if backend is None:
        imgui.text_colored(
            imgui.ImVec4(0.95, 0.7, 0.4, 1.0),
            "(no GPU backend — preview unavailable)",
        )
        return

    if parent._iso_dp_compose_key.get(view) != compose_key:
        rgba = _compose_rgba(
            proj, mask,
            float(parent._iso_dp_vmin), float(parent._iso_dp_vmax),
        )
        gpu = parent._iso_dp_gpu.get(view)
        if gpu is None:
            try:
                gpu = _GpuRGBA(backend, rgba)
            except Exception:
                gpu = None
            if gpu is not None:
                parent._iso_dp_gpu[view] = gpu
        else:
            try:
                gpu.reupload(rgba)
            except Exception:
                pass
        parent._iso_dp_compose_key[view] = compose_key

    gpu = parent._iso_dp_gpu.get(view)
    if gpu is None or gpu.ref is None:
        return

    scale = min(cell_w / gpu.w, cell_w / gpu.h)
    disp_w = max(96.0, gpu.w * scale)
    disp_h = max(96.0, gpu.h * scale)
    p0 = imgui.get_cursor_screen_pos()
    p1 = imgui.ImVec2(p0.x + disp_w, p0.y + disp_h)
    imgui.get_window_draw_list().add_image(gpu.ref, p0, p1)
    imgui.dummy(imgui.ImVec2(disp_w, disp_h))

    imgui.text_colored(
        imgui.ImVec4(0.85, 0.85, 0.85, 1.0),
        f"T_abs={t_abs:.1f}  T_rel={t_rel:.3f}",
    )
    color_pct = (
        imgui.ImVec4(0.95, 0.5, 0.5, 1.0)
        if flagged_pct > 5.0
        else imgui.ImVec4(0.5, 0.85, 0.5, 1.0)
    )
    imgui.text_colored(color_pct, f"flagged {flagged_pct:.2f}% of pixels")
