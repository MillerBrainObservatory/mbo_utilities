"""Shared Options popup (GPU adapter + debug logging).

Used by both the launch FileDialog and the main PreviewDataWidget's File menu
so the same widget configures both. Styled to match the Help / Keybinds /
Pipeline Settings popups (PopupAutoSize + begin_popup_modal).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as fa

from mbo_utilities import log as _mbo_log
from mbo_utilities.preferences import (
    get_gpu_index,
    set_gpu_index,
    get_debug_logging,
    set_debug_logging,
    get_compute_gpu,
    set_compute_gpu,
)
from mbo_utilities.gui._imgui_helpers import PopupAutoSize


_COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)
_COL_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)
_COL_WARN = imgui.ImVec4(0.90, 0.65, 0.20, 1.0)


def _live_render_adapter(parent: Any) -> Any | None:
    """The adapter the live figure renders with (ground truth, no enumeration)."""
    try:
        iw = getattr(parent, "image_widget", None)
        if iw is not None and getattr(iw, "figure", None) is not None:
            return iw.figure.renderer.device.adapter
    except Exception:
        pass
    return None


def _refresh_gpu_panel(parent: Any) -> None:
    """Snapshot render GPU + compute GPU + device memory.

    Subprocess / wgpu calls — run on popup open and on Refresh, never per
    frame. Render adapter comes from the live figure if present, else the
    cached adapter list (NEVER enumerate here — it clobbers the GL context).
    """
    from mbo_utilities import gpu as _gpu

    # render GPU
    live = _live_render_adapter(parent)
    if live is not None:
        parent._options_render_gpu = _gpu.render_gpu(live_adapter=live)
    else:
        adapters = getattr(parent, "_options_gpu_adapters", None) or []
        sel = parent._options_gpu_idx
        if 0 <= sel < len(adapters):
            idx, source = sel, "preference"
        else:  # auto -> resolve to the adapter wgpu actually picks
            idx, source = getattr(parent, "_options_gpu_default_idx", -1), "auto"
        if 0 <= idx < len(adapters):
            parent._options_render_gpu = {
                "summary": _gpu._adapter_summary(adapters[idx]),
                "source": source, "index": idx,
            }
        else:
            parent._options_render_gpu = {
                "summary": "wgpu default", "source": "auto", "index": -1,
            }

    # compute GPU + device memory
    try:
        parent._options_compute_info = _gpu.compute_gpu()
        parent._options_gpu_devices = _gpu.gpu_devices()
    except Exception:
        parent._options_compute_info = {"enabled": False, "name": "?"}
        parent._options_gpu_devices = []

    # compute-device selector options: Auto, CPU (off), then one per device
    opts = ["auto", "cpu"]
    labels = ["Auto (use GPU)", "Off (CPU only)"]
    for d in parent._options_gpu_devices:
        opts.append(str(d["index"]))
        labels.append(f"GPU {d['index']}: {d['name']}")
    parent._options_compute_opts = opts
    parent._options_compute_labels = labels


def _ensure_gpu_list(parent: Any) -> None:
    """Read the pre-warmed adapter cache. NEVER call
    ``fpl.enumerate_adapters()`` from here — it initializes wgpu, which
    clobbers GLFW's WGL current-context (Glfw Error 65544) and makes
    the host window flicker. The cache is primed in ``_run_gui_impl``
    before ``immapp.run`` takes over the GL context.
    """
    if getattr(parent, "_options_gpu_adapters", None) is not None:
        return
    from mbo_utilities.gui._gpu_cache import get_adapters, get_default_index
    parent._options_gpu_adapters = list(get_adapters())
    parent._options_gpu_default_idx = get_default_index()

    def _adapter_name(a) -> str:
        info = getattr(a, "info", {}) or {}
        return info.get("device", info.get("description", "?"))

    # name the actual GPU behind "auto" instead of a generic label
    default_idx = parent._options_gpu_default_idx
    if 0 <= default_idx < len(parent._options_gpu_adapters):
        labels = [f"auto ({_adapter_name(parent._options_gpu_adapters[default_idx])})"]
    else:
        labels = ["auto (default)"]
    for i, a in enumerate(parent._options_gpu_adapters):
        info = getattr(a, "info", {}) or {}
        type_ = info.get("adapter_type", info.get("device_type", "?"))
        labels.append(f"{i}: {_adapter_name(a)} [{type_}]")
    parent._options_gpu_labels = labels


def draw_options_popup(parent: Any) -> None:
    """Draw the Options popup. Open with ``parent._show_options_popup = True``.

    Persists every change to ``~/.mbo/settings/preferences.json`` immediately.
    """
    if not hasattr(parent, "_show_options_popup"):
        parent._show_options_popup = False
    if not hasattr(parent, "_options_sizer"):
        parent._options_sizer = PopupAutoSize(
            "Options##options_popup", anchor="center"
        )
    if not hasattr(parent, "_options_gpu_idx"):
        parent._options_gpu_idx = get_gpu_index()
    if not hasattr(parent, "_options_debug"):
        parent._options_debug = get_debug_logging()
    if not hasattr(parent, "_options_compute_gpu"):
        parent._options_compute_gpu = get_compute_gpu()

    if parent._show_options_popup:
        # re-sync from prefs each open so changes from the CLI or another
        # window aren't shadowed by a stale snapshot.
        parent._options_gpu_idx = get_gpu_index()
        parent._options_debug = get_debug_logging()
        parent._options_compute_gpu = get_compute_gpu()
        parent._options_gpu_dirty = True  # refresh in body, after _ensure_gpu_list
        parent._options_sizer.before_open()
        imgui.open_popup("Options##options_popup")
        parent._show_options_popup = False

    flags = parent._options_sizer.flags(imgui.WindowFlags_.no_saved_settings)
    opened, visible = imgui.begin_popup_modal(
        "Options##options_popup", p_open=True, flags=flags,
    )
    if not opened:
        return
    try:
        if not visible:
            imgui.close_current_popup()
            return

        _ensure_gpu_list(parent)
        # snapshot on open + throttled (~1.5s) for live utilization, without
        # hammering nvidia-smi every frame.
        _now = time.monotonic()
        if getattr(parent, "_options_gpu_dirty", True) or (
            _now - getattr(parent, "_options_gpu_last_refresh", 0.0) >= 1.5
        ):
            _refresh_gpu_panel(parent)
            parent._options_gpu_dirty = False
            parent._options_gpu_last_refresh = _now

        imgui.text_colored(_COL_ACCENT, f"{fa.ICON_FA_GEARS}  Options")
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 4))

        imgui.text_colored(_COL_DIM, "GPU adapter (render)")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Pick which GPU to render with. 'auto' lets wgpu choose "
                "(usually the DiscreteGPU). Takes effect on next launch."
            )
        imgui.set_next_item_width(hello_imgui.em_size(20))
        ui_idx = parent._options_gpu_idx + 1  # 0 == "auto"
        changed, new_ui_idx = imgui.combo(
            "##gpu_adapter", ui_idx, parent._options_gpu_labels
        )
        if changed:
            parent._options_gpu_idx = new_ui_idx - 1
            set_gpu_index(parent._options_gpu_idx)
            _refresh_gpu_panel(parent)

        # what fastplotlib actually renders with right now
        rg = getattr(parent, "_options_render_gpu", None)
        if rg:
            note = {"live": "", "preference": "  (selected)",
                    "auto": "  (auto)"}.get(rg.get("source"), "")
            imgui.text_colored(_COL_DIM, f"  using: {rg['summary']}{note}")

        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 4))

        # compute GPU (CUDA: suite2p / cupy / cellpose) — separate from the
        # wgpu render adapter above.
        imgui.text_colored(_COL_DIM, "Compute GPU (suite2p / cellpose / registration)")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Whether CUDA is used for processing. 'Off' forces CPU for "
                "suite2p, registration, and cellpose. Applies to tasks "
                "started after this change."
            )
        opts = getattr(parent, "_options_compute_opts", ["auto", "cpu"])
        labels = getattr(parent, "_options_compute_labels",
                         ["Auto (use GPU)", "Off (CPU only)"])
        cur = parent._options_compute_gpu
        cur_idx = opts.index(cur) if cur in opts else 0
        imgui.set_next_item_width(hello_imgui.em_size(20))
        changed, new_idx = imgui.combo("##compute_gpu", cur_idx, labels)
        if changed:
            parent._options_compute_gpu = opts[new_idx]
            set_compute_gpu(opts[new_idx])
            # apply now so subprocesses spawned this session inherit it
            from mbo_utilities.gpu import apply_gpu_policy
            apply_gpu_policy(opts[new_idx])
            _refresh_gpu_panel(parent)

        imgui.same_line()
        if imgui.small_button("Refresh"):
            _refresh_gpu_panel(parent)

        # what compute would use right now
        ci = getattr(parent, "_options_compute_info", None)
        if ci:
            if ci.get("enabled"):
                imgui.text_colored(
                    _COL_DIM,
                    f"  using: {ci['name']}  (cuda:{ci.get('torch_index', 0)})",
                )
            else:
                imgui.text_colored(_COL_WARN, f"  using: {ci.get('name', 'CPU')}")

        for d in getattr(parent, "_options_gpu_devices", []):
            total = d["total_mb"] or 0
            used = d["used_mb"] or 0
            free = d["free_mb"] or 0
            pct = (used / total * 100) if total else 0
            util = d["util_pct"]
            temp = d["temp_c"]
            tail = []
            if temp is not None:
                tail.append(f"{temp:.0f}C")
            tail_s = ("  " + " ".join(tail)) if tail else ""
            util_s = f"{util:.0f}%" if util is not None else "?"
            imgui.text_colored(
                _COL_DIM,
                f"  GPU {d['index']}: usage {util_s}, mem {used:.0f}/{total:.0f} MB "
                f"({pct:.0f}%), {free:.0f} MB free{tail_s}",
            )

        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 4))

        changed, new_debug = imgui.checkbox("Debug logging", parent._options_debug)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Verbose console logs (per-read timings, zarr chunk shapes, "
                "ms timings). Same effect as launching with MBO_DEBUG=1."
            )
        if changed:
            parent._options_debug = new_debug
            set_debug_logging(new_debug)
            _mbo_log.set_global_level(
                logging.DEBUG if new_debug else logging.INFO
            )

        imgui.dummy(imgui.ImVec2(0, 8))
        btn_w = hello_imgui.em_size(6)
        imgui.set_cursor_pos_x((imgui.get_window_width() - btn_w) * 0.5)
        if imgui.button("Close", imgui.ImVec2(btn_w, 0)):
            imgui.close_current_popup()
    finally:
        imgui.end_popup()
