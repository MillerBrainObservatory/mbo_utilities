"""Standalone Isoview crop tool.

Floating ``imgui.begin`` window (not a popup — popups flicker in tight
host windows, see ``_options_popup``) with per-view crop drags and a
masked-projection preview.

The preview consumes the pre-computed XY projection TIFFs from
``IsoviewArray.projections()`` — one per (view, timepoint). Projections
are uploaded once to wgpu via :class:`_GpuImage` and reused across
frames; only the overlay rectangle moves as the user drags. Switching
timepoints triggers a fresh upload (bounded LRU cache so VRAM doesn't
blow up scrubbing through 800 TMs). The shared ``vmin`` / ``vmax``
controls re-upload through ``_GpuImage.reupload_if_changed``.

State lives in :mod:`mbo_utilities.gui._isoview_crop_state` so the same
crops are visible to the Run-tab readout and the submit-time builders.
The store is keyed by the raw acquisition path, so crops set against a
raw IsoviewArray persist when the user later reopens the corrected or
fused output.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as fa

from mbo_utilities.gui import _isoview_crop_state as crop_state


_DEFAULT_CAMERA_VIEW_MAP = {0: 0, 1: 0, 2: 90, 3: 90}
# View colors mirror the napari widget so the two tools feel related.
_VIEW_COLORS = {0: (1.0, 0.35, 0.35, 1.0), 90: (1.0, 0.95, 0.4, 1.0)}

# Compact drag widths — replaces the wide slider_int from the v1 layout.
_DRAG_W = 78
_DISPLAY_DRAG_W = 110

# LRU cap on (view, timepoint) GPU textures. Matches summary_image's
# _MAX_GPU_CACHE — each 16-bit max-projection rounds to ~2–4 MiB of
# VRAM after the RGBA expansion, so 16 is ~50 MiB worst case.
_MAX_GPU_TEXTURES = 16


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


def _representative_channel_per_view(arr: Any) -> dict[int, int]:
    mapping = _channel_to_view(arr)
    out: dict[int, int] = {}
    for ci, view in mapping.items():
        if view not in out:
            out[view] = ci
    return out


def _view_shape(arr: Any) -> tuple[int, int, int] | None:
    shape = getattr(arr, "shape", None)
    if not shape or len(shape) != 5:
        return None
    _, _, nz, ny, nx = shape
    return int(nz), int(ny), int(nx)


def _view_int_from_label(label: str) -> int | None:
    """Extract the view int from a projections-dict label."""
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
    """Return ``{view_int: {timepoint: xy_projection_path}}``.

    For each view, picks the lowest-numbered camera that maps to it (or
    the VW## directly for fused) and harvests every timepoint that
    camera produced. The timepoint slider walks the intersection of
    these per-view timepoint sets so every view stays in sync.
    """
    if not projections:
        return {}
    if "xy" not in (projections.get("axes") or []):
        return {}
    cv = _camera_view_map(arr)

    # First pass: collect {view_int: {source_label: {t: path}}} so we
    # can deterministically pick one source per view below.
    per_view_by_label: dict[int, dict[str, dict[int, Path]]] = {}
    for (axis, label, t), path in (projections.get("files") or {}).items():
        if axis != "xy":
            continue
        raw_int = _view_int_from_label(label)
        if raw_int is None:
            continue
        view_int = cv.get(raw_int, raw_int) if label.startswith("CM") else raw_int
        per_view_by_label.setdefault(view_int, {}).setdefault(label, {})[int(t)] = Path(path)

    # Pick the source whose label sorts first (CM00 over CM01, etc.).
    out: dict[int, dict[int, Path]] = {}
    for view_int, by_label in per_view_by_label.items():
        if not by_label:
            continue
        chosen = sorted(by_label.keys())[0]
        out[view_int] = dict(by_label[chosen])
    return out


def _common_timepoints(index: dict[int, dict[int, Path]]) -> list[int]:
    """Sorted list of timepoints present for every view."""
    if not index:
        return []
    sets = [set(d.keys()) for d in index.values()]
    if not sets:
        return []
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return sorted(common)


def _ensure_window_state(parent: Any) -> None:
    if not hasattr(parent, "_show_iso_crop_window"):
        parent._show_iso_crop_window = False
    if not hasattr(parent, "_iso_crop_window_open"):
        parent._iso_crop_window_open = False
    if not hasattr(parent, "_iso_crop_cache_key"):
        parent._iso_crop_cache_key = None
    if not hasattr(parent, "_iso_crop_shapes"):
        parent._iso_crop_shapes: dict[int, tuple[int, int, int]] = {}
    # Widget-local edits — only committed to crop_state on Apply.
    # Keyed by view int; values are dicts with z/y/x tuples.
    if not hasattr(parent, "_iso_crop_pending"):
        parent._iso_crop_pending: dict[int, dict[str, tuple[int, int]]] = {}
    if not hasattr(parent, "_iso_crop_proj_index"):
        parent._iso_crop_proj_index: dict[int, dict[int, Path]] = {}
    if not hasattr(parent, "_iso_crop_timepoints"):
        parent._iso_crop_timepoints: list[int] = []
    if not hasattr(parent, "_iso_crop_current_tp"):
        parent._iso_crop_current_tp = 0
    if not hasattr(parent, "_iso_crop_gpu"):
        # keyed by (view, timepoint)
        parent._iso_crop_gpu: dict[tuple[int, int], Any] = {}
    if not hasattr(parent, "_iso_crop_gpu_lru"):
        parent._iso_crop_gpu_lru: list[tuple[int, int]] = []
    if not hasattr(parent, "_iso_crop_vmin"):
        parent._iso_crop_vmin = 0.0
    if not hasattr(parent, "_iso_crop_vmax"):
        parent._iso_crop_vmax = 1.0
    if not hasattr(parent, "_iso_crop_display_inited"):
        parent._iso_crop_display_inited = False


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


def _destroy_gpu_images(parent: Any) -> None:
    cache = getattr(parent, "_iso_crop_gpu", None) or {}
    for gpu in cache.values():
        try:
            gpu.destroy()
        except Exception:
            pass
    parent._iso_crop_gpu = {}
    parent._iso_crop_gpu_lru = []


def _evict_lru(parent: Any) -> None:
    while len(parent._iso_crop_gpu_lru) > _MAX_GPU_TEXTURES:
        old_key = parent._iso_crop_gpu_lru.pop(0)
        gpu = parent._iso_crop_gpu.pop(old_key, None)
        if gpu is not None:
            try:
                gpu.destroy()
            except Exception:
                pass


def _get_or_upload(parent: Any, view: int, timepoint: int) -> Any | None:
    """Return the ``_GpuImage`` for (view, timepoint), uploading on miss.

    LRU evicts past :data:`_MAX_GPU_TEXTURES`. Re-uploads when the
    shared vmin/vmax changes (handled inline via
    :meth:`_GpuImage.reupload_if_changed`).
    """
    key = (view, timepoint)
    gpu = parent._iso_crop_gpu.get(key)
    if gpu is not None:
        # Refresh LRU position
        try:
            parent._iso_crop_gpu_lru.remove(key)
        except ValueError:
            pass
        parent._iso_crop_gpu_lru.append(key)
        gpu.reupload_if_changed(
            "gray", float(parent._iso_crop_vmin), float(parent._iso_crop_vmax),
        )
        return gpu

    index = parent._iso_crop_proj_index.get(view) or {}
    path = index.get(timepoint)
    if path is None:
        return None
    proj = _load_projection(path)
    if proj is None:
        return None
    backend = _get_backend(parent)
    if backend is None:
        return None
    from mbo_utilities.gui.widgets.summary_image import _GpuImage
    try:
        gpu = _GpuImage(
            backend, proj, "gray",
            float(parent._iso_crop_vmin), float(parent._iso_crop_vmax),
        )
    except Exception:
        return None
    parent._iso_crop_gpu[key] = gpu
    parent._iso_crop_gpu_lru.append(key)
    _evict_lru(parent)
    return gpu


def _initial_display_range(parent: Any, arr: Any) -> tuple[float, float]:
    """Default to (0, 1000) — matches the host viewer's isoview defaults
    so contrast stays consistent across the main canvas and this window.
    """
    return 0.0, 1000.0


def _ensure_loaded_for(parent: Any, arr: Any) -> None:
    """(Re)resolve per-view extents + projection index when the dataset
    changes. Seeds full-extent crops, picks a sensible initial display
    range, and clears any cached textures from a prior dataset.
    """
    key = (id(arr), getattr(arr, "scan_root", None))
    if parent._iso_crop_cache_key == key and parent._iso_crop_shapes:
        return
    parent._iso_crop_cache_key = key
    parent._iso_crop_shapes = {}
    parent._iso_crop_proj_index = {}
    parent._iso_crop_timepoints = []
    parent._iso_crop_current_tp = 0
    parent._iso_crop_display_inited = False
    _destroy_gpu_images(parent)

    shp = _view_shape(arr)
    if shp is None:
        return
    rep = _representative_channel_per_view(arr)
    parent._iso_crop_pending = {}
    nz, ny, nx = shp
    for view in rep.keys():
        parent._iso_crop_shapes[view] = shp
        existing = crop_state.get_crops(arr).get(view)
        if existing is not None and existing.get("shape") == shp:
            parent._iso_crop_pending[view] = {
                "z": tuple(existing["z"]),
                "y": tuple(existing["y"]),
                "x": tuple(existing["x"]),
            }
        else:
            parent._iso_crop_pending[view] = {
                "z": (0, nz), "y": (0, ny), "x": (0, nx),
            }

    try:
        projections = arr.projections() if hasattr(arr, "projections") else None
    except Exception:
        projections = None
    parent._iso_crop_proj_index = _build_projection_index(arr, projections)
    parent._iso_crop_timepoints = _common_timepoints(parent._iso_crop_proj_index)
    if parent._iso_crop_timepoints:
        parent._iso_crop_current_tp = parent._iso_crop_timepoints[0]

    vmin, vmax = _initial_display_range(parent, arr)
    parent._iso_crop_vmin = vmin
    parent._iso_crop_vmax = vmax
    parent._iso_crop_display_inited = True


def open_window(parent: Any) -> None:
    """Trigger the crop editor on next frame. Idempotent.

    Invalidates the cache so the window re-seeds pending bounds from
    crop_state — keeps the floating window reactive to manual edits
    made in the Parameters popup between opens.
    """
    _ensure_window_state(parent)
    parent._show_iso_crop_window = True
    parent._iso_crop_cache_key = None


def close_window(parent: Any) -> None:
    """Hide the editor and release GPU textures."""
    _ensure_window_state(parent)
    parent._iso_crop_window_open = False
    _destroy_gpu_images(parent)
    parent._iso_crop_cache_key = None
    parent._iso_crop_shapes = {}


def draw_window(parent: Any) -> None:
    """Render the crop editor. Call once per frame from the main
    preview-widget render loop (alongside the other floating windows).
    """
    _ensure_window_state(parent)

    was_open = parent._iso_crop_window_open
    if parent._show_iso_crop_window:
        parent._iso_crop_window_open = True
        parent._show_iso_crop_window = False
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(
            viewport.get_center(),
            imgui.Cond_.appearing,
            pivot=imgui.ImVec2(0.5, 0.5),
        )
        imgui.set_next_window_size(
            imgui.ImVec2(900, 760),
            imgui.Cond_.appearing,
        )

    if not parent._iso_crop_window_open:
        if was_open:
            _destroy_gpu_images(parent)
            parent._iso_crop_cache_key = None
            parent._iso_crop_shapes = {}
        return

    arr = _get_iso_array(parent)
    title = f"{fa.ICON_FA_CROP}  Isoview Crop##iso_crop_window"
    flags = (
        imgui.WindowFlags_.no_collapse
        | imgui.WindowFlags_.no_docking
        | imgui.WindowFlags_.no_saved_settings
    )
    expanded, parent._iso_crop_window_open = imgui.begin(
        title, p_open=parent._iso_crop_window_open, flags=flags,
    )
    try:
        if not expanded:
            return
        if arr is None or not hasattr(arr, "scan_root"):
            imgui.text_colored(
                imgui.ImVec4(0.95, 0.7, 0.4, 1.0),
                "No isoview array loaded.",
            )
            return

        _ensure_loaded_for(parent, arr)

        if not parent._iso_crop_shapes:
            imgui.text_colored(
                imgui.ImVec4(0.95, 0.7, 0.4, 1.0),
                "Could not resolve view shapes for this dataset.",
            )
            return

        imgui.text_colored(
            imgui.ImVec4(0.55, 0.75, 1.0, 1.0),
            f"{arr.scan_root}",
        )
        imgui.spacing()
        _draw_display_controls(parent, arr)
        imgui.separator()

        sorted_views = sorted(parent._iso_crop_shapes)
        for i, view in enumerate(sorted_views):
            _draw_view_section(parent, arr, view, is_first=(i == 0))
            imgui.spacing()

        imgui.separator()
        if imgui.button("Reset all", imgui.ImVec2(120, 0)):
            for view, shp in parent._iso_crop_shapes.items():
                nz, ny, nx = shp
                parent._iso_crop_pending[view] = {
                    "z": (0, nz), "y": (0, ny), "x": (0, nx),
                }
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
            # Drop unsaved drags by discarding pending state. The next
            # open seeds from crop_state again.
            parent._iso_crop_window_open = False
        imgui.same_line()
        # Apply: commit every pending view bound to the store + close.
        # The Run-tab submits read from crop_state, so this is the
        # explicit "make this run's crop" gesture.
        with _apply_button_style():
            if imgui.button("Apply", imgui.ImVec2(160, 0)):
                for view, shp in parent._iso_crop_shapes.items():
                    pending = parent._iso_crop_pending.get(view)
                    if pending is None:
                        continue
                    nz, ny, nx = shp
                    crop_state.set_view_bounds(
                        arr, view,
                        z=pending["z"], y=pending["y"], x=pending["x"],
                        shape=(nz, ny, nx),
                    )
                parent._iso_crop_window_open = False
    finally:
        imgui.end()


def _draw_display_controls(parent: Any, arr: Any) -> None:
    """Shared vmin/vmax + timepoint scrubber. Both apply to every view."""
    imgui.text_colored(
        imgui.ImVec4(0.85, 0.85, 0.85, 1.0),
        "Display:",
    )
    imgui.same_line()

    # Loose drag bounds — keep some headroom over the current dataset
    # so users can over-pull (e.g. to wash out bright fiducials).
    imgui.set_next_item_width(_DISPLAY_DRAG_W)
    speed = max(1.0, abs(parent._iso_crop_vmax - parent._iso_crop_vmin) / 200.0)
    _, parent._iso_crop_vmin = imgui.drag_float(
        "vmin##iso_crop", float(parent._iso_crop_vmin), speed, 0.0, 0.0, "%.0f",
    )
    imgui.same_line()
    imgui.set_next_item_width(_DISPLAY_DRAG_W)
    _, parent._iso_crop_vmax = imgui.drag_float(
        "vmax##iso_crop", float(parent._iso_crop_vmax), speed, 0.0, 0.0, "%.0f",
    )
    if parent._iso_crop_vmax <= parent._iso_crop_vmin:
        parent._iso_crop_vmax = parent._iso_crop_vmin + 1.0

    imgui.same_line()
    if imgui.button("Auto##iso_crop_auto"):
        lo, hi = _initial_display_range(parent, arr)
        parent._iso_crop_vmin = lo
        parent._iso_crop_vmax = hi

    tps = parent._iso_crop_timepoints
    if not tps:
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            "(no per-timepoint projections — preview will not animate)",
        )
        return

    # Timepoint scrubber — use the index into the common-timepoint list
    # so it's always 0..N-1 regardless of TM numbering gaps.
    try:
        cur_idx = tps.index(int(parent._iso_crop_current_tp))
    except ValueError:
        cur_idx = 0
    imgui.text_colored(
        imgui.ImVec4(0.85, 0.85, 0.85, 1.0),
        "Timepoint:",
    )
    imgui.same_line()
    imgui.set_next_item_width(220)
    changed, new_idx = imgui.slider_int(
        f"##iso_crop_tp", cur_idx, 0, max(0, len(tps) - 1),
        f"TM{tps[cur_idx]:06d}  ({cur_idx + 1}/{len(tps)})",
    )
    if changed:
        parent._iso_crop_current_tp = tps[max(0, min(new_idx, len(tps) - 1))]


def _draw_view_section(parent: Any, arr: Any, view: int, *, is_first: bool = False) -> None:
    nz, ny, nx = parent._iso_crop_shapes[view]

    # All edits go through pending state so the user can Cancel cleanly.
    pending = parent._iso_crop_pending.setdefault(view, {
        "z": (0, nz), "y": (0, ny), "x": (0, nx),
    })
    z0, z1 = pending["z"]
    y0, y1 = pending["y"]
    x0, x1 = pending["x"]

    color = _VIEW_COLORS.get(view, (0.6, 0.8, 1.0, 1.0))
    imgui.text_colored(imgui.ImVec4(*color), f"VW{view:02d}   shape=(Z={nz}, Y={ny}, X={nx})")

    imgui.push_id(f"iso_crop_vw_{view}")
    try:
        avail_x = imgui.get_content_region_avail().x
        # Compact drag column on the left; preview eats whatever's left.
        controls_w = 2 * _DRAG_W + 56  # 2 drags + label + spacing per row
        preview_w = max(220.0, avail_x - controls_w - 16.0)

        imgui.begin_group()
        try:
            z0, z1 = _drag_pair("z", z0, z1, 0, nz)
            y0, y1 = _drag_pair("y", y0, y1, 0, ny)
            x0, x1 = _drag_pair("x", x0, x1, 0, nx)

            # clamp lows below highs (matches the napari widget)
            z0 = min(z0, z1 - 1)
            y0 = min(y0, y1 - 1)
            x0 = min(x0, x1 - 1)

            imgui.text_colored(
                imgui.ImVec4(0.7, 0.85, 0.7, 1.0),
                f"kept ({z1 - z0}, {y1 - y0}, {x1 - x0})",
            )
            if is_first and len(parent._iso_crop_shapes) > 1:
                if imgui.button("Apply to all", imgui.ImVec2(120, 0)):
                    for other_view, shp in parent._iso_crop_shapes.items():
                        if other_view == view:
                            continue
                        o_nz, o_ny, o_nx = shp
                        parent._iso_crop_pending[other_view] = {
                            "z": (min(z0, o_nz - 1), min(z1, o_nz)),
                            "y": (min(y0, o_ny - 1), min(y1, o_ny)),
                            "x": (min(x0, o_nx - 1), min(x1, o_nx)),
                        }
        finally:
            imgui.end_group()

        imgui.same_line()
        imgui.begin_group()
        try:
            _draw_view_preview(
                parent, view, preview_w,
                ny=ny, nx=nx,
                y0=y0, y1=y1, x0=x0, x1=x1,
                color=color,
            )
        finally:
            imgui.end_group()

        parent._iso_crop_pending[view] = {
            "z": (z0, z1), "y": (y0, y1), "x": (x0, x1),
        }
    finally:
        imgui.pop_id()


@contextmanager
def _apply_button_style():
    """Same green-button styling the Run-tab uses for its primary
    action. Keeps the "this is the commit" affordance consistent.
    """
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.10, 0.45, 0.10, 1.0))
    try:
        yield
    finally:
        imgui.pop_style_color(3)


def _drag_pair(axis: str, lo: int, hi: int, vmin: int, vmax: int) -> tuple[int, int]:
    """Render two compact ``slider_int`` widgets for the (lo, hi) pair.
    Ctrl-click any slider to type a value directly.
    """
    imgui.text(axis)
    imgui.same_line()
    imgui.set_next_item_width(_DRAG_W)
    _, lo = imgui.slider_int(
        f"##iso_crop_{axis}_lo", int(lo), int(vmin), max(int(vmin), int(vmax) - 1),
        "%d",
    )
    imgui.same_line()
    imgui.set_next_item_width(_DRAG_W)
    _, hi = imgui.slider_int(
        f"##iso_crop_{axis}_hi", int(hi), int(vmin) + 1, int(vmax),
        "%d",
    )
    return int(lo), int(hi)


def _draw_view_preview(
    parent: Any, view: int, preview_w: float,
    *,
    ny: int, nx: int,
    y0: int, y1: int, x0: int, x1: int,
    color: tuple[float, float, float, float],
) -> None:
    """Draw the XY-projection texture for ``view`` at the current
    timepoint, with the crop-box overlaid.
    """
    timepoint = int(parent._iso_crop_current_tp)
    gpu = _get_or_upload(parent, view, timepoint)
    if gpu is None:
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            "(no XY projection at this timepoint)",
        )
        imgui.dummy(imgui.ImVec2(preview_w, preview_w * 0.6))
        return

    src_h, src_w = gpu.h, gpu.w
    if src_h <= 0 or src_w <= 0:
        return
    scale = min(preview_w / src_w, preview_w / src_h)
    disp_w = max(96.0, src_w * scale)
    disp_h = max(96.0, src_h * scale)

    p0 = imgui.get_cursor_screen_pos()
    p1 = imgui.ImVec2(p0.x + disp_w, p0.y + disp_h)
    draw_list = imgui.get_window_draw_list()
    draw_list.add_image(gpu.ref, p0, p1)

    sx = disp_w / float(nx) if nx else disp_w / src_w
    sy = disp_h / float(ny) if ny else disp_h / src_h
    rect_p0 = imgui.ImVec2(p0.x + x0 * sx, p0.y + y0 * sy)
    rect_p1 = imgui.ImVec2(p0.x + x1 * sx, p0.y + y1 * sy)
    col = imgui.get_color_u32(imgui.ImVec4(*color))
    draw_list.add_rect(rect_p0, rect_p1, col, 0.0, 0, 2.0)

    imgui.dummy(imgui.ImVec2(disp_w, disp_h))
