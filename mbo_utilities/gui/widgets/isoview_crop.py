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
from mbo_utilities.gui._imgui_helpers import button_width, draw_toolbar_row


_DEFAULT_CAMERA_VIEW_MAP = {0: 0, 1: 0, 2: 90, 3: 90}
# Per-camera colors (match the segmentation / dead-pixel widgets).
_CAMERA_COLORS = {
    0: (1.0, 0.35, 0.35, 1.0),
    1: (1.0, 0.85, 0.4, 1.0),
    2: (0.4, 0.9, 0.6, 1.0),
    3: (0.5, 0.75, 1.0, 1.0),
}

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


def cameras_for_crop(arr: Any) -> list[int]:
    """Camera ints (CM##) present in the loaded array.

    Uses the array's actual ``views`` (a corrected tree's C axis is the
    cameras), since the XML-synthesized ``camera_view_map`` is often
    incomplete (e.g. only ``{0, 1}`` when stack_direction lists one arm).
    Falls back to the map, then to 4 cameras.
    """
    out: list[int] = []
    for v in (getattr(arr, "views", None) or []):
        if isinstance(v, tuple) and v:
            out.append(int(v[0]))
        else:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                pass
    if out:
        return sorted(set(out))
    cv = _camera_view_map(arr)
    return sorted(cv.keys()) if cv else [0, 1, 2, 3]


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


def _tp_label(slot) -> str:
    """Display label for a tiled projection slot: a specimen_name grid token
    (string) as-is, else SPM## for a legacy integer slot."""
    return f"SPM{slot:02d}" if isinstance(slot, int) else str(slot)


def _build_projection_index(
    arr: Any, projections: dict | None,
) -> dict[int, dict[int, Path]]:
    """Return ``{camera_int: {timepoint: xy_projection_path}}``.

    Keyed by the camera (CM##) directly — no view collapsing — so each
    of the cameras gets its own preview. The timepoint slider walks the
    union of the per-camera timepoint sets, so a tile present on only some
    cameras still appears.
    """
    if not projections:
        return {}
    if "xy" not in (projections.get("axes") or []):
        return {}

    per_cam_by_label: dict[int, dict[str, dict[int, Path]]] = {}
    for (axis, label, t), path in (projections.get("files") or {}).items():
        if axis != "xy":
            continue
        cam_int = _view_int_from_label(label)
        if cam_int is None:
            continue
        per_cam_by_label.setdefault(cam_int, {}).setdefault(label, {})[t] = Path(path)

    out: dict[int, dict[int, Path]] = {}
    for cam_int, by_label in per_cam_by_label.items():
        if not by_label:
            continue
        chosen = sorted(by_label.keys())[0]
        out[cam_int] = dict(by_label[chosen])
    return out


def _common_timepoints(index: dict[int, dict[int, Path]]) -> list[int]:
    """Every tile/timepoint present in ANY camera (union).

    Union, not intersection: a tile acquired on only some cameras (e.g. a
    dropped CM0/CM1) must still appear so the user can crop the cameras
    that exist. Panels for cameras lacking that tile render blank.
    """
    if not index:
        return []
    tps: set[int] = set()
    for d in index.values():
        tps |= set(d.keys())
    return sorted(tps)


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
    # Keyed by (tile, camera); tile is None when non-tiled, else the SPM int.
    # Values are dicts with z/y/x tuples.
    if not hasattr(parent, "_iso_crop_pending"):
        parent._iso_crop_pending: dict[tuple, dict[str, tuple[int, int]]] = {}
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


def _seed_bounds(existing: dict | None, shp: tuple[int, int, int]) -> dict:
    """Bounds dict seeded from a stored crop, else full extent."""
    nz, ny, nx = shp
    if existing is not None and tuple(existing.get("shape", ())) == tuple(shp):
        return {
            "z": tuple(existing["z"]),
            "y": tuple(existing["y"]),
            "x": tuple(existing["x"]),
        }
    return {"z": (0, nz), "y": (0, ny), "x": (0, nx)}


def _pending_tile(parent: Any, arr: Any):
    """Active tile key for pending crops: current tile token when tiled,
    else None."""
    if bool(getattr(arr, "is_tiled", False)):
        return parent._iso_crop_current_tp
    return None


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
    cameras = cameras_for_crop(arr)
    for cam in cameras:
        parent._iso_crop_shapes[cam] = shp

    try:
        projections = arr.projections() if hasattr(arr, "projections") else None
    except Exception:
        projections = None
    parent._iso_crop_proj_index = _build_projection_index(arr, projections)
    parent._iso_crop_timepoints = _common_timepoints(parent._iso_crop_proj_index)
    if parent._iso_crop_timepoints:
        parent._iso_crop_current_tp = parent._iso_crop_timepoints[0]

    # Seed pending bounds, keyed by (tile, camera). tile is None for non-tiled
    # datasets; for tiled it is each SPM that has a projection, so every tile
    # carries its own per-camera crop.
    parent._iso_crop_pending = {}
    if bool(getattr(arr, "is_tiled", False)):
        tile_crops = crop_state.get_tile_crops(arr)
        tiles = parent._iso_crop_timepoints or [parent._iso_crop_current_tp]
        for tile in tiles:
            existing_cams = tile_crops.get(tile, {})
            for cam in cameras:
                parent._iso_crop_pending[(tile, cam)] = _seed_bounds(
                    existing_cams.get(cam), shp,
                )
    else:
        existing = crop_state.get_crops(arr)
        for cam in cameras:
            parent._iso_crop_pending[(None, cam)] = _seed_bounds(
                existing.get(cam), shp,
            )

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

        sorted_cams = sorted(parent._iso_crop_shapes)
        # Scroll region for the camera sections so they never spill past the
        # window; size it to leave room for the bottom button row, and cap
        # each preview's height so all cameras fit without scrolling.
        reserve_bottom = (
            imgui.get_frame_height_with_spacing()
            + imgui.get_text_line_height_with_spacing()
        )
        region_h = max(140.0, imgui.get_content_region_avail().y - reserve_bottom)
        per_cam_h = region_h / max(1, len(sorted_cams))
        if imgui.begin_child(
            "##iso_crop_sections", imgui.ImVec2(0, region_h),
        ):
            for i, cam in enumerate(sorted_cams):
                _draw_camera_section(
                    parent, arr, cam, is_first=(i == 0), max_h=per_cam_h,
                )
                imgui.spacing()
        imgui.end_child()

        imgui.separator()
        tiled = bool(getattr(arr, "is_tiled", False))
        if tiled and len(parent._iso_crop_timepoints) > 1:
            cur = parent._iso_crop_current_tp
            if imgui.button(f"Copy {_tp_label(cur)} crop to all tiles", imgui.ImVec2(0, 0)):
                for cam in parent._iso_crop_shapes:
                    src = parent._iso_crop_pending.get((cur, cam))
                    if src is None:
                        continue
                    for tile in parent._iso_crop_timepoints:
                        parent._iso_crop_pending[(tile, cam)] = dict(src)

        if imgui.button("Reset all", imgui.ImVec2(120, 0)):
            for (tile, cam), _ in list(parent._iso_crop_pending.items()):
                nz, ny, nx = parent._iso_crop_shapes.get(cam, (0, 0, 0))
                parent._iso_crop_pending[(tile, cam)] = {
                    "z": (0, nz), "y": (0, ny), "x": (0, nx),
                }
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
            # Drop unsaved drags by discarding pending state. The next
            # open seeds from crop_state again.
            parent._iso_crop_window_open = False
        imgui.same_line()
        # Apply: commit every pending bound to the store + close. The Run-tab
        # submits read from crop_state, so this is the explicit "make this
        # run's crop" gesture. Tiled datasets commit per (tile, camera).
        with _apply_button_style():
            if imgui.button("Apply", imgui.ImVec2(160, 0)):
                for (tile, cam), pending in parent._iso_crop_pending.items():
                    shp = parent._iso_crop_shapes.get(cam)
                    if shp is None:
                        continue
                    nz, ny, nx = shp
                    if tile is None:
                        crop_state.set_camera_bounds(
                            arr, cam,
                            z=pending["z"], y=pending["y"], x=pending["x"],
                            shape=(nz, ny, nx),
                        )
                    else:
                        crop_state.set_tile_camera_bounds(
                            arr, tile, cam,
                            z=pending["z"], y=pending["y"], x=pending["x"],
                            shape=(nz, ny, nx),
                        )
                parent._iso_crop_window_open = False
    finally:
        imgui.end()


def _draw_display_controls(parent: Any, arr: Any) -> None:
    """Shared vmin/vmax + timepoint scrubber. Both apply to every view."""
    # Loose drag bounds — keep some headroom over the current dataset
    # so users can over-pull (e.g. to wash out bright fiducials).
    speed = max(1.0, abs(parent._iso_crop_vmax - parent._iso_crop_vmin) / 200.0)

    def _vmin():
        _, parent._iso_crop_vmin = imgui.drag_float(
            "##iso_crop_vmin", float(parent._iso_crop_vmin), speed, 0.0, 0.0, "%.0f",
        )

    def _vmax():
        _, parent._iso_crop_vmax = imgui.drag_float(
            "##iso_crop_vmax", float(parent._iso_crop_vmax), speed, 0.0, 0.0, "%.0f",
        )

    def _auto():
        if imgui.button("Auto##iso_crop_auto"):
            lo, hi = _initial_display_range(parent, arr)
            parent._iso_crop_vmin = lo
            parent._iso_crop_vmax = hi

    items = [
        ("Min", _DISPLAY_DRAG_W, _vmin),
        ("Max", _DISPLAY_DRAG_W, _vmax),
        (None, button_width("Auto"), _auto),
    ]

    tiled = bool(getattr(arr, "is_tiled", False))
    tps = parent._iso_crop_timepoints
    if tps:
        # Scrubber — use the index into the common-tile/timepoint list so it's
        # always 0..N-1 regardless of SPM/TM numbering gaps.
        try:
            cur_idx = tps.index(parent._iso_crop_current_tp)
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
                "##iso_crop_tp", cur_idx, 0, max(0, len(tps) - 1), value_fmt,
            )
            if ch:
                parent._iso_crop_current_tp = tps[max(0, min(v, len(tps) - 1))]
        items.append((label, 220.0, _tp))

    draw_toolbar_row(items)
    if parent._iso_crop_vmax <= parent._iso_crop_vmin:
        parent._iso_crop_vmax = parent._iso_crop_vmin + 1.0
    if not tps:
        unit = "per-tile" if tiled else "per-timepoint"
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            f"(no {unit} projections — preview will not animate)",
        )


def _draw_camera_section(
    parent: Any, arr: Any, camera: int, *,
    is_first: bool = False, max_h: float | None = None,
) -> None:
    nz, ny, nx = parent._iso_crop_shapes[camera]
    tile = _pending_tile(parent, arr)
    key = (tile, camera)

    # All edits go through pending state so the user can Cancel cleanly.
    pending = parent._iso_crop_pending.setdefault(key, {
        "z": (0, nz), "y": (0, ny), "x": (0, nx),
    })
    z0, z1 = pending["z"]
    y0, y1 = pending["y"]
    x0, x1 = pending["x"]

    from mbo_utilities.arrays.isoview.array import camera_view_label
    color = _CAMERA_COLORS.get(camera, (0.6, 0.8, 1.0, 1.0))
    spm = f"{_tp_label(tile)}  " if tile is not None else ""
    imgui.text_colored(
        imgui.ImVec4(*color),
        f"{spm}{camera_view_label(camera)}   shape=(Z={nz}, Y={ny}, X={nx})",
    )

    imgui.push_id(f"iso_crop_cm_{camera}")
    try:
        avail_x = imgui.get_content_region_avail().x
        # Compact drag column on the left; preview eats whatever's left.
        controls_w = 2 * _DRAG_W + 56  # 2 drags + label + spacing per row
        preview_w = max(220.0, avail_x - controls_w - 16.0)
        # Cap preview height so every camera section fits the scroll region;
        # without this the preview scales to its full width and one camera
        # can be ~600px tall, pushing the rest off-window.
        header_h = imgui.get_text_line_height_with_spacing()
        preview_h = (
            preview_w if max_h is None
            else max(96.0, max_h - header_h - imgui.get_style().item_spacing.y)
        )

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
                if imgui.button("Apply to all cameras", imgui.ImVec2(0, 0)):
                    for other_cam, shp in parent._iso_crop_shapes.items():
                        if other_cam == camera:
                            continue
                        o_nz, o_ny, o_nx = shp
                        parent._iso_crop_pending[(tile, other_cam)] = {
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
                parent, camera, preview_w,
                ny=ny, nx=nx,
                y0=y0, y1=y1, x0=x0, x1=x1,
                color=color, preview_h=preview_h,
            )
        finally:
            imgui.end_group()

        parent._iso_crop_pending[key] = {
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
    preview_h: float | None = None,
) -> None:
    """Draw the XY-projection texture for ``view`` at the current
    timepoint, with the crop-box overlaid. ``preview_h`` caps the
    rendered height; when ``None`` it falls back to ``preview_w``.
    """
    box_h = preview_w if preview_h is None else preview_h
    timepoint = parent._iso_crop_current_tp
    gpu = _get_or_upload(parent, view, timepoint)
    if gpu is None:
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.65, 1.0),
            "(no XY projection at this timepoint)",
        )
        imgui.dummy(imgui.ImVec2(preview_w, min(preview_w * 0.6, box_h)))
        return

    src_h, src_w = gpu.h, gpu.w
    if src_h <= 0 or src_w <= 0:
        return
    scale = min(preview_w / src_w, box_h / src_h)
    disp_w = max(64.0, src_w * scale)
    disp_h = max(64.0, src_h * scale)

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
