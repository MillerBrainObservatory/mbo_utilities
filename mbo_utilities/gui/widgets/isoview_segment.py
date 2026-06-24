"""Standalone Isoview segmentation tuning tool.

Floating ``imgui.begin`` window with sliders for ``segment_threshold``,
``mask_percentile``, ``gauss_sigma``, ``gauss_kernel`` and a live mask
overlay on the XY max-projection for each view.

The four params live on the active :class:`IsoviewPipelineWidget`
instance (``_correct_*`` fields). The window reads + writes them in
place so the popup form and the preview stay in lockstep; Apply just
closes, Cancel restores the snapshot taken on open.

Preview math mirrors :func:`isoview.segmentation.segment_foreground`
reduced to 2D — separable XY gaussian + adaptive level — so a value
that looks good here tracks what the pipeline will compute over the
full 3D volume.

Cache key for the convolution result is
``(view, timepoint, sigma, kernel)``. Threshold and mask_percentile
changes reuse the cached filter; gauss_sigma / gauss_kernel changes
re-run the convolution.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6 as fa
from scipy.ndimage import convolve1d

from mbo_utilities.gui._imgui_helpers import button_width, draw_toolbar_row


_DEFAULT_CAMERA_VIEW_MAP = {0: 0, 1: 0, 2: 90, 3: 90}
_VIEW_COLORS = {0: (1.0, 0.35, 0.35, 1.0), 90: (1.0, 0.95, 0.4, 1.0)}
_CAMERA_COLORS = {
    0: (1.0, 0.35, 0.35, 1.0),
    1: (1.0, 0.85, 0.4, 1.0),
    2: (0.4, 0.9, 0.6, 1.0),
    3: (0.5, 0.75, 1.0, 1.0),
}
_TINT_RGB = (0.2, 0.95, 0.6)  # foreground overlay color (cyan-green)

_SLIDER_W = 220
_DISPLAY_DRAG_W = 110
_MAX_FILTER_CACHE = 8       # (view, tp, sigma, kernel) entries
_MAX_GPU_TEXTURES = 4       # one per view typically, headroom for switches
_SUBSAMPLE_FACTOR = 100     # matches isoview's percentile subsample
_TARGET_PROJ_PX = 512       # preview math runs at <= this longest side


def _camera_view_map(arr: Any) -> dict[int, int]:
    meta = getattr(arr, "metadata", None) or {}
    m = meta.get("camera_view_map")
    if isinstance(m, dict):
        return {int(k): int(v) for k, v in m.items()}
    return dict(_DEFAULT_CAMERA_VIEW_MAP)


def _channel_to_view(arr: Any) -> dict[int, int]:
    cv = _camera_view_map(arr)
    out: dict[int, int] = {}
    for ci, vk in enumerate(getattr(arr, "views", []) or []):
        if isinstance(vk, tuple) and len(vk) == 3:
            out[ci] = int(vk[2])
        elif isinstance(vk, tuple) and len(vk) >= 1:
            out[ci] = cv.get(int(vk[0]), int(vk[0]))
        else:
            out[ci] = cv.get(int(vk), int(vk))
    return out


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


def _is_raw(arr: Any) -> bool:
    return str(getattr(arr, "kind", "") or "").lower() == "raw"


def _tp_label(slot) -> str:
    """Display label for a tiled projection slot: a specimen_name grid token
    (string) as-is, else SPM## for a legacy integer slot."""
    return f"SPM{slot:02d}" if isinstance(slot, int) else str(slot)


def _build_projection_index(
    arr: Any, projections: dict | None,
) -> dict[int, dict[int, Path]]:
    """``{key: {timepoint: xy_projection_path}}``.

    For raw arrays, key is the camera int (CM00→0, CM01→1, ...) so each
    camera renders its own preview. For corrected/fused arrays the key
    is the view int (VW00→0, VW90→90).
    """
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
    """Every tile/timepoint present in ANY view (union).

    Union, not intersection: a tile acquired on only some cameras (e.g. a
    dropped CM0/CM1) must still appear so the user can work with the
    cameras that exist. Panels for views lacking that tile render blank.
    """
    if not index:
        return []
    tps: set[int] = set()
    for d in index.values():
        tps |= set(d.keys())
    return sorted(tps)


def _percentile_interp(data: np.ndarray, percentile: float) -> float:
    """Mirror of :func:`isoview.corrections.percentile_interp`."""
    if data.size == 0:
        return 0.0
    sorted_data = np.sort(data)
    n = sorted_data.size
    p_rank = 100.0 * (np.arange(n) + 0.5) / n
    return float(np.interp(
        percentile, p_rank, sorted_data,
        left=sorted_data[0], right=sorted_data[-1],
    ))


def _make_gauss_kernel(sigma: float, size: int) -> np.ndarray:
    half = int(np.ceil(size / 2))
    x = np.arange(-half, half + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _gauss_2d(proj: np.ndarray, sigma: float, size: int) -> np.ndarray:
    """Separable 2D gaussian filter — XY only, matches
    :func:`segment_foreground` reduced to a single Z-slice.
    """
    k = _make_gauss_kernel(sigma, size).astype(np.float32)
    f = convolve1d(proj.astype(np.float32), k, axis=0, mode="nearest")
    f = convolve1d(f, k, axis=1, mode="nearest")
    return np.round(f).astype(np.uint16)


def _downsample_mean(proj: np.ndarray, cap: int = _TARGET_PROJ_PX) -> tuple[np.ndarray, int]:
    """Box-mean downsample so the longest side is <= ``cap``.

    Mean pooling is itself a low-pass, so it is consistent with the
    gaussian-smoothed field the segmentation thresholds on. Returns
    ``(downsampled float32, factor)``; factor 1 means no downsampling.
    """
    h, w = proj.shape
    longest = max(h, w)
    if longest <= cap:
        return np.ascontiguousarray(proj, dtype=np.float32), 1
    factor = int(np.ceil(longest / cap))
    hh = (h // factor) * factor
    ww = (w // factor) * factor
    blocks = proj[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor)
    ds = blocks.mean(axis=(1, 3), dtype=np.float32)
    return np.ascontiguousarray(ds, dtype=np.float32), factor


def _adaptive_level(
    filtered: np.ndarray, mask_percentile: float, threshold: float,
) -> tuple[float, float, float]:
    """Return ``(min_intensity, mean_intensity, level)`` for the mask."""
    sub = filtered.ravel()[::_SUBSAMPLE_FACTOR]
    min_i = _percentile_interp(sub, mask_percentile)
    above = filtered[filtered > min_i]
    mean_i = float(above.mean()) if above.size else float(min_i)
    level = float(min_i) + (mean_i - float(min_i)) * float(threshold)
    return float(min_i), mean_i, level


def _compose_rgba(
    proj: np.ndarray, mask: np.ndarray, vmin: float, vmax: float,
) -> np.ndarray:
    """Grayscale projection with foreground tinted by :data:`_TINT_RGB`."""
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
        base = g8[mask].astype(np.float32) * 0.55
        rgba[mask, 0] = np.clip(base + r * 0.45 * 255, 0, 255).astype(np.uint8)
        rgba[mask, 1] = np.clip(base + gT * 0.45 * 255, 0, 255).astype(np.uint8)
        rgba[mask, 2] = np.clip(base + bT * 0.45 * 255, 0, 255).astype(np.uint8)
    return rgba


class _GpuRGBA:
    """Owns one wgpu texture seeded from an RGBA uint8 buffer.

    Mirrors :class:`summary_image._GpuImage` but consumes a pre-built
    RGBA buffer (we compose the mask overlay in numpy, then upload).
    """

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
            # Source resolution changed (different view); rebuild.
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
    if not hasattr(parent, "_show_iso_seg_window"):
        parent._show_iso_seg_window = False
    if not hasattr(parent, "_iso_seg_window_open"):
        parent._iso_seg_window_open = False
    if not hasattr(parent, "_iso_seg_snapshot"):
        parent._iso_seg_snapshot: dict[str, float] | None = None
    if not hasattr(parent, "_iso_seg_cache_key"):
        parent._iso_seg_cache_key = None
    if not hasattr(parent, "_iso_seg_proj_index"):
        parent._iso_seg_proj_index: dict[int, dict[int, Path]] = {}
    if not hasattr(parent, "_iso_seg_timepoints"):
        parent._iso_seg_timepoints: list[int] = []
    if not hasattr(parent, "_iso_seg_current_tp"):
        parent._iso_seg_current_tp = 0
    if not hasattr(parent, "_iso_seg_proj_cache"):
        # (view, tp) -> proj float32 numpy
        parent._iso_seg_proj_cache: dict[tuple[int, int], np.ndarray] = {}
    if not hasattr(parent, "_iso_seg_filtered_cache"):
        # (view, tp, sigma, kernel) -> filtered uint16
        parent._iso_seg_filtered_cache: dict[tuple, np.ndarray] = {}
        parent._iso_seg_filtered_lru: list[tuple] = []
    if not hasattr(parent, "_iso_seg_gpu"):
        # view -> _GpuRGBA (one per view, reuploaded on compose)
        parent._iso_seg_gpu: dict[int, _GpuRGBA] = {}
        # compose key per view, to skip work when nothing changed
        parent._iso_seg_compose_key: dict[int, tuple] = {}
    if not hasattr(parent, "_iso_seg_vmin"):
        parent._iso_seg_vmin = 0.0
    if not hasattr(parent, "_iso_seg_vmax"):
        parent._iso_seg_vmax = 1.0
    if not hasattr(parent, "_iso_seg_display_inited"):
        parent._iso_seg_display_inited = False
    if not hasattr(parent, "_iso_seg_dragging"):
        parent._iso_seg_dragging = False
    if not hasattr(parent, "_iso_seg_shown"):
        # view -> last computed {tp, sigma, kernel, threshold, mask_pct,
        # min_i, mean_i, level, mask}; held while a slider is dragged so
        # the expensive gaussian/level recompute defers to release.
        parent._iso_seg_shown: dict[int, dict] = {}


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
    for gpu in (getattr(parent, "_iso_seg_gpu", None) or {}).values():
        try:
            gpu.destroy()
        except Exception:
            pass
    parent._iso_seg_gpu = {}
    parent._iso_seg_compose_key = {}


def _initial_display_range(parent: Any) -> tuple[float, float]:
    """Default to (0, 1000) — matches the host viewer's isoview defaults."""
    return 0.0, 1000.0


def _get_working(parent: Any, view: int, tp: int):
    """Return ``(downsampled projection float32, factor)`` or ``None``.

    The projection is mean-pooled once to ``_TARGET_PROJ_PX`` so every
    per-frame op (gaussian, level, compose, upload) runs on the small
    working image instead of the full sensor resolution.
    """
    key = (view, tp)
    cached = parent._iso_seg_proj_cache.get(key)
    if cached is not None:
        return cached
    path = (parent._iso_seg_proj_index.get(view) or {}).get(tp)
    if path is None:
        return None
    proj = _load_projection(path)
    if proj is None:
        return None
    work = _downsample_mean(proj)
    parent._iso_seg_proj_cache[key] = work
    return work


def _get_projection(parent: Any, view: int, tp: int) -> np.ndarray | None:
    work = _get_working(parent, view, tp)
    return None if work is None else work[0]


def _get_filtered(
    parent: Any, view: int, tp: int, sigma: float, kernel: int,
) -> np.ndarray | None:
    """Cached 2D gaussian. Re-runs when sigma or kernel changes.

    sigma/kernel are scaled by the projection's downsample factor so the
    smoothing still matches the full-resolution pipeline footprint.
    """
    work = _get_working(parent, view, tp)
    if work is None:
        return None
    proj, factor = work
    key = (int(view), tp, round(float(sigma), 3), int(kernel), factor)
    cache = parent._iso_seg_filtered_cache
    if key in cache:
        try:
            parent._iso_seg_filtered_lru.remove(key)
        except ValueError:
            pass
        parent._iso_seg_filtered_lru.append(key)
        return cache[key]
    s = max(0.3, float(sigma) / factor)
    k = max(1, int(round(int(kernel) / factor)))
    filtered = _gauss_2d(proj, s, k)
    cache[key] = filtered
    parent._iso_seg_filtered_lru.append(key)
    while len(parent._iso_seg_filtered_lru) > _MAX_FILTER_CACHE:
        old = parent._iso_seg_filtered_lru.pop(0)
        cache.pop(old, None)
    return filtered


def _ensure_loaded_for(parent: Any, arr: Any) -> None:
    """(Re)load projection index when the dataset changes."""
    key = (id(arr), getattr(arr, "scan_root", None))
    if parent._iso_seg_cache_key == key and parent._iso_seg_proj_index:
        return
    parent._iso_seg_cache_key = key
    parent._iso_seg_proj_index = {}
    parent._iso_seg_timepoints = []
    parent._iso_seg_current_tp = 0
    parent._iso_seg_proj_cache = {}
    parent._iso_seg_filtered_cache = {}
    parent._iso_seg_filtered_lru = []
    parent._iso_seg_shown = {}
    parent._iso_seg_display_inited = False
    _destroy_gpu(parent)

    try:
        projections = arr.projections() if hasattr(arr, "projections") else None
    except Exception:
        projections = None
    parent._iso_seg_proj_index = _build_projection_index(arr, projections)
    parent._iso_seg_timepoints = _common_timepoints(parent._iso_seg_proj_index)
    if parent._iso_seg_timepoints:
        parent._iso_seg_current_tp = parent._iso_seg_timepoints[0]

    vmin, vmax = _initial_display_range(parent)
    parent._iso_seg_vmin = vmin
    parent._iso_seg_vmax = vmax
    parent._iso_seg_display_inited = True


def open_window(parent: Any) -> None:
    """Trigger the seg preview window. Snapshots current params for Cancel."""
    _ensure_state(parent)
    iso = _get_iso_widget(parent)
    if iso is not None:
        parent._iso_seg_snapshot = {
            "segment_threshold": float(iso._correct_segment_threshold),
            "mask_percentile": float(iso._correct_mask_percentile),
            "gauss_sigma": float(iso._correct_gauss_sigma),
            "gauss_kernel": int(iso._correct_gauss_kernel),
        }
    parent._show_iso_seg_window = True


def close_window(parent: Any) -> None:
    _ensure_state(parent)
    parent._iso_seg_window_open = False
    _destroy_gpu(parent)
    parent._iso_seg_cache_key = None
    parent._iso_seg_proj_index = {}


def _restore_snapshot(parent: Any) -> None:
    snap = getattr(parent, "_iso_seg_snapshot", None)
    if not snap:
        return
    iso = _get_iso_widget(parent)
    if iso is None:
        return
    iso._correct_segment_threshold = float(snap["segment_threshold"])
    iso._correct_mask_percentile = float(snap["mask_percentile"])
    iso._correct_gauss_sigma = float(snap["gauss_sigma"])
    iso._correct_gauss_kernel = int(snap["gauss_kernel"])


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
    """Render the segmentation preview. Call once per frame alongside
    other floating windows.
    """
    _ensure_state(parent)

    was_open = parent._iso_seg_window_open
    if parent._show_iso_seg_window:
        parent._iso_seg_window_open = True
        parent._show_iso_seg_window = False
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

    if not parent._iso_seg_window_open:
        if was_open:
            _destroy_gpu(parent)
            parent._iso_seg_cache_key = None
            parent._iso_seg_proj_index = {}
        return

    arr = _get_iso_array(parent)
    iso = _get_iso_widget(parent)
    title = f"{fa.ICON_FA_WAND_MAGIC_SPARKLES}  Isoview Segmentation##iso_seg_window"
    flags = (
        imgui.WindowFlags_.no_collapse
        | imgui.WindowFlags_.no_docking
        | imgui.WindowFlags_.no_saved_settings
    )
    expanded, parent._iso_seg_window_open = imgui.begin(
        title, p_open=parent._iso_seg_window_open, flags=flags,
    )
    try:
        if not expanded:
            return
        if arr is None or iso is None or not hasattr(arr, "scan_root"):
            imgui.text_colored(
                imgui.ImVec4(0.95, 0.7, 0.4, 1.0),
                "Open an isoview dataset and the Run tab to tune segmentation.",
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
        if imgui.button("Reset to defaults", imgui.ImVec2(160, 0)):
            iso._correct_segment_threshold = 0.4
            iso._correct_mask_percentile = 1.0
            iso._correct_gauss_sigma = 2.0
            iso._correct_gauss_kernel = 5
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
            _restore_snapshot(parent)
            parent._iso_seg_window_open = False
        imgui.same_line()
        with _apply_button_style():
            if imgui.button("Apply", imgui.ImVec2(160, 0)):
                parent._iso_seg_window_open = False
    finally:
        imgui.end()


def _draw_display_controls(parent: Any) -> None:
    """vmin/vmax + timepoint scrubber, same affordance as the crop window."""
    speed = max(1.0, abs(parent._iso_seg_vmax - parent._iso_seg_vmin) / 200.0)

    def _vmin():
        _, parent._iso_seg_vmin = imgui.drag_float(
            "##iso_seg_vmin", float(parent._iso_seg_vmin), speed, 0.0, 0.0, "%.0f",
        )

    def _vmax():
        _, parent._iso_seg_vmax = imgui.drag_float(
            "##iso_seg_vmax", float(parent._iso_seg_vmax), speed, 0.0, 0.0, "%.0f",
        )

    def _auto():
        if imgui.button("Auto##iso_seg_auto"):
            lo, hi = _initial_display_range(parent)
            parent._iso_seg_vmin = lo
            parent._iso_seg_vmax = hi

    items = [
        ("Min", _DISPLAY_DRAG_W, _vmin),
        ("Max", _DISPLAY_DRAG_W, _vmax),
        (None, button_width("Auto"), _auto),
    ]

    tiled = bool(getattr(_get_iso_array(parent), "is_tiled", False))
    tps = parent._iso_seg_timepoints
    if tps:
        try:
            cur_idx = tps.index(parent._iso_seg_current_tp)
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
                "##iso_seg_tp", cur_idx, 0, max(0, len(tps) - 1), value_fmt,
            )
            if ch:
                parent._iso_seg_current_tp = tps[max(0, min(v, len(tps) - 1))]
        items.append((label, 220.0, _tp))

    draw_toolbar_row(items)
    if parent._iso_seg_vmax <= parent._iso_seg_vmin:
        parent._iso_seg_vmax = parent._iso_seg_vmin + 1.0
    if not tps:
        unit = "per-tile" if tiled else "per-timepoint"
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            f"(no {unit} projections — slider values still save)",
        )


def _draw_param_controls(parent: Any, iso: Any) -> None:
    """Four sliders that write directly to the iso widget's _correct_*
    fields, so the popup form stays in sync. Sets ``_iso_seg_dragging``
    so the per-view recompute can defer to slider release.
    """
    imgui.text_colored(imgui.ImVec4(1.0, 0.85, 0.4, 1.0), "Segmentation params")
    imgui.spacing()
    dragging = False

    imgui.set_next_item_width(_SLIDER_W)
    _, iso._correct_segment_threshold = imgui.slider_float(
        "threshold##iso_seg",
        float(iso._correct_segment_threshold), 0.0, 1.0, "%.3f",
    )
    dragging |= imgui.is_item_active()
    imgui.same_line()
    imgui.text_colored(
        imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
        "  level = min + (mean − min) * threshold",
    )

    imgui.set_next_item_width(_SLIDER_W)
    _, iso._correct_mask_percentile = imgui.slider_float(
        "mask percentile##iso_seg",
        float(iso._correct_mask_percentile), 0.0, 100.0, "%.2f",
    )
    dragging |= imgui.is_item_active()
    imgui.same_line()
    imgui.text_colored(
        imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
        "  percentile on filtered values for the min baseline",
    )

    imgui.set_next_item_width(_SLIDER_W)
    _, iso._correct_gauss_sigma = imgui.slider_float(
        "gauss sigma##iso_seg",
        float(iso._correct_gauss_sigma), 0.5, 10.0, "%.2f",
    )
    dragging |= imgui.is_item_active()
    imgui.same_line()
    imgui.text_colored(
        imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
        "  XY gaussian sigma (z derives from scaling)",
    )

    imgui.set_next_item_width(_SLIDER_W)
    _, iso._correct_gauss_kernel = imgui.slider_int(
        "gauss kernel##iso_seg",
        int(iso._correct_gauss_kernel), 1, 31, "%d",
    )
    dragging |= imgui.is_item_active()
    imgui.same_line()
    imgui.text_colored(
        imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
        "  XY gaussian kernel size (odd values preferred)",
    )
    parent._iso_seg_dragging = dragging


def _draw_view_previews(parent: Any, iso: Any) -> None:
    if not parent._iso_seg_proj_index:
        imgui.text_colored(
            imgui.ImVec4(0.95, 0.7, 0.4, 1.0),
            "No XY projections available for this dataset — "
            "slider values still save but the preview is empty.",
        )
        return

    views = sorted(parent._iso_seg_proj_index.keys())
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
    if raw_mode:
        from mbo_utilities.arrays.isoview.array import camera_view_label
        color = _CAMERA_COLORS.get(view, (0.6, 0.8, 1.0, 1.0))
        label_text = camera_view_label(view)
    else:
        color = _VIEW_COLORS.get(view, (0.6, 0.8, 1.0, 1.0))
        label_text = f"VW{view:02d}"
    imgui.text_colored(imgui.ImVec4(*color), label_text)

    tp = parent._iso_seg_current_tp
    sigma = float(iso._correct_gauss_sigma)
    kernel = int(iso._correct_gauss_kernel)
    threshold = float(iso._correct_segment_threshold)
    mask_pct = float(iso._correct_mask_percentile)

    proj = _get_projection(parent, view, tp)
    if proj is None:
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            "(no XY projection at this timepoint)",
        )
        imgui.dummy(imgui.ImVec2(cell_w, cell_w * 0.6))
        return

    # While a param slider is dragged, reuse the last computed mask/level
    # instead of re-running the gaussian + level every frame; recompute on
    # release. A timepoint change always recomputes (different image).
    dragging = bool(getattr(parent, "_iso_seg_dragging", False))
    shown = parent._iso_seg_shown.get(view)
    stale = (
        shown is None
        or shown.get("tp") != tp
        or (
            not dragging
            and (
                shown.get("sigma") != sigma
                or shown.get("kernel") != kernel
                or shown.get("threshold") != threshold
                or shown.get("mask_pct") != mask_pct
            )
        )
    )
    if stale:
        filtered = _get_filtered(parent, view, tp, sigma, kernel)
        if filtered is None:
            return
        min_i, mean_i, level = _adaptive_level(filtered, mask_pct, threshold)
        mask = filtered > level
        shown = {
            "tp": tp, "sigma": sigma, "kernel": kernel,
            "threshold": threshold, "mask_pct": mask_pct,
            "min_i": min_i, "mean_i": mean_i, "level": level, "mask": mask,
        }
        parent._iso_seg_shown[view] = shown

    mask = shown["mask"]
    min_i, mean_i, level = shown["min_i"], shown["mean_i"], shown["level"]
    kept_pct = 100.0 * mask.sum() / mask.size if mask.size else 0.0

    compose_key = (
        view, shown["tp"], shown["sigma"], shown["kernel"],
        round(shown["threshold"], 5), round(shown["mask_pct"], 4),
        round(float(parent._iso_seg_vmin), 4),
        round(float(parent._iso_seg_vmax), 4),
    )
    backend = _get_backend(parent)
    if backend is None:
        imgui.text_colored(
            imgui.ImVec4(0.95, 0.7, 0.4, 1.0),
            "(no GPU backend — preview unavailable)",
        )
        return

    if parent._iso_seg_compose_key.get(view) != compose_key:
        rgba = _compose_rgba(
            proj, mask,
            float(parent._iso_seg_vmin), float(parent._iso_seg_vmax),
        )
        gpu = parent._iso_seg_gpu.get(view)
        if gpu is None:
            try:
                gpu = _GpuRGBA(backend, rgba)
            except Exception:
                gpu = None
            if gpu is not None:
                parent._iso_seg_gpu[view] = gpu
        else:
            try:
                gpu.reupload(rgba)
            except Exception:
                pass
        parent._iso_seg_compose_key[view] = compose_key

    gpu = parent._iso_seg_gpu.get(view)
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
        f"min={min_i:.1f}  mean={mean_i:.1f}  level={level:.1f}",
    )
    imgui.text_colored(
        imgui.ImVec4(0.5, 0.85, 0.5, 1.0),
        f"kept {kept_pct:.1f}% foreground",
    )
