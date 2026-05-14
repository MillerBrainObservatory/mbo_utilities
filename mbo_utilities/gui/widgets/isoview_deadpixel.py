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


_DEFAULT_CAMERA_VIEW_MAP = {0: 0, 1: 0, 2: 90, 3: 90}
_VIEW_COLORS = {0: (1.0, 0.35, 0.35, 1.0), 90: (1.0, 0.95, 0.4, 1.0)}
_TINT_RGB = (1.0, 0.35, 0.85)

_SLIDER_W = 220
_DISPLAY_DRAG_W = 110
_MAX_FILTER_CACHE = 8
_SUBSAMPLE_FACTOR_HINT = 100


def _camera_view_map(arr: Any) -> dict[int, int]:
    meta = getattr(arr, "metadata", None) or {}
    m = meta.get("camera_view_map")
    if isinstance(m, dict):
        return {int(k): int(v) for k, v in m.items()}
    return dict(_DEFAULT_CAMERA_VIEW_MAP)


def _view_int_from_label(label: str) -> int | None:
    if not isinstance(label, str):
        return None
    if label.startswith("VW") and label[2:].isdigit():
        return int(label[2:])
    if label.startswith("CM") and label[2:].isdigit():
        return int(label[2:])
    try:
        return int(label)
    except ValueError:
        return None


def _build_projection_index(
    arr: Any, projections: dict | None,
) -> dict[int, dict[int, Path]]:
    if not projections:
        return {}
    if "xy" not in (projections.get("axes") or []):
        return {}
    cv = _camera_view_map(arr)

    per_view_by_label: dict[int, dict[str, dict[int, Path]]] = {}
    for (axis, label, t), path in (projections.get("files") or {}).items():
        if axis != "xy":
            continue
        raw_int = _view_int_from_label(label)
        if raw_int is None:
            continue
        view_int = (
            cv.get(raw_int, raw_int) if label.startswith("CM") else raw_int
        )
        per_view_by_label.setdefault(view_int, {}).setdefault(label, {})[
            int(t)
        ] = Path(path)

    out: dict[int, dict[int, Path]] = {}
    for view_int, by_label in per_view_by_label.items():
        if not by_label:
            continue
        chosen = sorted(by_label.keys())[0]
        out[view_int] = dict(by_label[chosen])
    return out


def _common_timepoints(index: dict[int, dict[int, Path]]) -> list[int]:
    if not index:
        return []
    sets = [set(d.keys()) for d in index.values()]
    if not sets:
        return []
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return sorted(common)


def _percentile_interp(data: np.ndarray, percentile: float) -> float:
    if data.size == 0:
        return 0.0
    sorted_data = np.sort(data)
    n = sorted_data.size
    p_rank = 100.0 * (np.arange(n) + 0.5) / n
    return float(np.interp(
        percentile, p_rank, sorted_data,
        left=sorted_data[0], right=sorted_data[-1],
    ))


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

    f = proj.astype(np.float64)
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

    def _upload(self, rgba: np.ndarray) -> None:
        wgpu = self._wgpu
        device = self.backend._device
        self._texture = device.create_texture(
            size=(self.w, self.h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        device.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(rgba).tobytes(),
            {"offset": 0, "bytes_per_row": self.w * 4, "rows_per_image": self.h},
            (self.w, self.h, 1),
        )
        self._view = self._texture.create_view()
        self.ref = self.backend.register_texture(self._view)

    def reupload(self, rgba: np.ndarray) -> None:
        if rgba.shape[0] != self.h or rgba.shape[1] != self.w:
            self.destroy()
            self.h, self.w = int(rgba.shape[0]), int(rgba.shape[1])
            self._upload(rgba)
            return
        self.destroy()
        self._upload(rgba)

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


def _get_projection(parent: Any, view: int, tp: int) -> np.ndarray | None:
    key = (view, tp)
    proj = parent._iso_dp_proj_cache.get(key)
    if proj is not None:
        return proj
    path = (parent._iso_dp_proj_index.get(view) or {}).get(tp)
    if path is None:
        return None
    proj = _load_projection(path)
    if proj is None:
        return None
    parent._iso_dp_proj_cache[key] = proj
    return proj


def _get_mask(
    parent: Any, view: int, tp: int, background: float, kernel: int,
) -> tuple[np.ndarray, float, float] | None:
    key = (int(view), int(tp), round(float(background), 3), int(kernel))
    cache = parent._iso_dp_mask_cache
    if key in cache:
        try:
            parent._iso_dp_mask_lru.remove(key)
        except ValueError:
            pass
        parent._iso_dp_mask_lru.append(key)
        return cache[key]
    proj = _get_projection(parent, view, tp)
    if proj is None:
        return None
    result = _detect_dead_pixels(proj, float(background), int(kernel))
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


def close_window(parent: Any) -> None:
    _ensure_state(parent)
    parent._iso_dp_window_open = False
    _destroy_gpu(parent)
    parent._iso_dp_cache_key = None
    parent._iso_dp_proj_index = {}


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
            f"Source: {arr.scan_root}",
        )
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
                "Approximation: the production algorithm uses std + mean "
                "projections along Z. This preview uses the cached XY "
                "max-projection — kernel + threshold response is real, "
                "absolute pixel set may differ.",
            )
        finally:
            imgui.pop_text_wrap_pos()
        imgui.spacing()
        _draw_display_controls(parent)
        imgui.separator()
        _draw_param_controls(iso)
        imgui.separator()
        imgui.spacing()
        _draw_view_previews(parent, iso)

        imgui.separator()
        if imgui.button("Reset to defaults##iso_dp_reset", imgui.ImVec2(160, 0)):
            iso._correct_background_percentile = 3.0
            iso._correct_median_kernel_size = 5
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
    imgui.text_colored(imgui.ImVec4(0.85, 0.85, 0.85, 1.0), "Display:")
    imgui.same_line()

    imgui.set_next_item_width(_DISPLAY_DRAG_W)
    speed = max(1.0, abs(parent._iso_dp_vmax - parent._iso_dp_vmin) / 200.0)
    _, parent._iso_dp_vmin = imgui.drag_float(
        "vmin##iso_dp", float(parent._iso_dp_vmin), speed, 0.0, 0.0, "%.0f",
    )
    imgui.same_line()
    imgui.set_next_item_width(_DISPLAY_DRAG_W)
    _, parent._iso_dp_vmax = imgui.drag_float(
        "vmax##iso_dp", float(parent._iso_dp_vmax), speed, 0.0, 0.0, "%.0f",
    )
    if parent._iso_dp_vmax <= parent._iso_dp_vmin:
        parent._iso_dp_vmax = parent._iso_dp_vmin + 1.0

    imgui.same_line()
    if imgui.button("Auto##iso_dp_auto"):
        lo, hi = _initial_display_range(parent)
        parent._iso_dp_vmin = lo
        parent._iso_dp_vmax = hi

    tps = parent._iso_dp_timepoints
    if not tps:
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            "(no per-timepoint projections — slider values still save)",
        )
        return

    try:
        cur_idx = tps.index(int(parent._iso_dp_current_tp))
    except ValueError:
        cur_idx = 0
    imgui.text_colored(imgui.ImVec4(0.85, 0.85, 0.85, 1.0), "Timepoint:")
    imgui.same_line()
    imgui.set_next_item_width(220)
    changed, new_idx = imgui.slider_int(
        "##iso_dp_tp", cur_idx, 0, max(0, len(tps) - 1),
        f"TM{tps[cur_idx]:06d}  ({cur_idx + 1}/{len(tps)})",
    )
    if changed:
        parent._iso_dp_current_tp = tps[max(0, min(new_idx, len(tps) - 1))]


def _draw_param_controls(iso: Any) -> None:
    imgui.text_colored(
        imgui.ImVec4(1.0, 0.85, 0.4, 1.0), "Dead-pixel params",
    )
    imgui.spacing()

    _, iso._correct_median_kernel_enabled = imgui.checkbox(
        "Enable median filter##iso_dp",
        bool(iso._correct_median_kernel_enabled),
    )

    imgui.set_next_item_width(_SLIDER_W)
    _, iso._correct_median_kernel_size = imgui.slider_int(
        "kernel##iso_dp",
        int(iso._correct_median_kernel_size), 1, 31, "%d",
    )
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
    imgui.same_line()
    imgui.text_colored(
        imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
        "  baseline subtracted before relative-deviation test",
    )


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

    avail_x = imgui.get_content_region_avail().x
    n = len(views)
    spacing = 12.0
    cell_w = max(280.0, (avail_x - spacing * (n - 1)) / n)

    for i, view in enumerate(views):
        if i > 0:
            imgui.same_line(0.0, spacing)
        imgui.begin_group()
        try:
            _draw_one_view(parent, iso, view, cell_w)
        finally:
            imgui.end_group()


def _draw_one_view(parent: Any, iso: Any, view: int, cell_w: float) -> None:
    color = _VIEW_COLORS.get(view, (0.6, 0.8, 1.0, 1.0))
    imgui.text_colored(imgui.ImVec4(*color), f"VW{view:02d}")

    tp = int(parent._iso_dp_current_tp)
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

    if enabled:
        result = _get_mask(parent, view, tp, bg, kernel)
    else:
        result = None
    if result is None:
        mask = np.zeros(proj.shape, dtype=bool)
        t_abs = 0.0
        t_rel = 0.0
    else:
        mask, t_abs, t_rel = result
    flagged_pct = 100.0 * mask.sum() / mask.size if mask.size else 0.0

    compose_key = (
        view, tp, bg, kernel, enabled,
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
