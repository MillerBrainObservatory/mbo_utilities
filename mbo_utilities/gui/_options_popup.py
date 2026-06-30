"""Shared Options popup (render GPU adapter + debug logging).

Used by both the launch FileDialog and the main PreviewDataWidget's File
menu so the same widget configures both. Styled to match the Help /
Keybinds / Pipeline Settings popups (PopupAutoSize + begin_popup_modal).

Live GPU usage lives in the Process Console's System panel; the compute
(suite2p / cellpose) device lives in the suite2p settings.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as fa

from mbo_utilities import log as _mbo_log
from mbo_utilities.preferences import (
    get_gpu_index,
    set_gpu_index,
    get_compute_gpu,
    set_compute_gpu,
    get_debug_logging,
    set_debug_logging,
    get_mem_monitor,
    set_mem_monitor,
    get_mem_monitor_interval,
    set_mem_monitor_interval,
)
from mbo_utilities.gui._imgui_helpers import PopupAutoSize


_COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)
_COL_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)


def compute_gpu_devices() -> list:
    """nvidia-smi compute devices (cached per call site). Empty on no GPU."""
    try:
        from mbo_utilities.gpu import gpu_devices
        return gpu_devices()
    except Exception:
        return []


def compute_gpu_options(devices: list) -> tuple[list[str], list[str]]:
    """(values, labels) for the compute-GPU combo.

    Values are the persisted `compute_gpu` tokens — "auto", "cpu", or a device
    index string (which `run_gui` pins via CUDA_VISIBLE_DEVICES). One entry per
    physical GPU governs cellpose + suite2p compute uniformly.
    """
    values = ["auto", "cpu"]
    labels = ["auto", "cpu"]
    for i, d in enumerate(devices):
        idx = int(d.get("index", i))
        values.append(str(idx))
        labels.append(f"{idx}: {d.get('name', '?')}")
    return values, labels


def compute_gpu_current_index(values: list[str]) -> int:
    """Index into `values` for the persisted compute-GPU choice.

    A digit pref is a device index (run_gui pins it via CUDA_VISIBLE_DEVICES),
    so it resolves to that device's option — not "auto".
    """
    cur = get_compute_gpu().strip().lower()
    if cur in ("cpu", "off", "false", "no", "none"):
        return 1
    if cur in ("", "auto"):
        return 0
    try:
        return values.index(cur)
    except ValueError:
        return 0


def apply_compute_gpu(value: str) -> None:
    """Persist a compute-GPU choice and apply it live for newly spawned jobs.

    Workers copy os.environ at spawn, so set CUDA_VISIBLE_DEVICES now: a device
    index pins it, "cpu" forces CPU, "auto" restores full visibility.
    """
    set_compute_gpu(value)
    if value == "auto":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif value == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # the index came from the nvidia-smi-ordered list; pin that ordering
        # (CUDA defaults to FASTEST_FIRST) so the CUDA device the worker uses
        # matches the GPU label the user picked. setdefault respects a
        # user-set order. Mirrors gpu.apply_gpu_policy.
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = value


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
    """Snapshot the render adapter fastplotlib uses.

    Comes from the live figure if present, else the cached adapter list
    (NEVER enumerate here — it clobbers the GL context).
    """
    from mbo_utilities import gpu as _gpu

    live = _live_render_adapter(parent)
    if live is not None:
        parent._options_render_gpu = _gpu.render_gpu(live_adapter=live)
        return
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
    if not hasattr(parent, "_options_mem"):
        parent._options_mem = get_mem_monitor()
    if not hasattr(parent, "_options_mem_interval"):
        parent._options_mem_interval = get_mem_monitor_interval()
    if not hasattr(parent, "_options_compute_devices"):
        parent._options_compute_devices = []

    if parent._show_options_popup:
        # re-sync from prefs each open so changes from the CLI or another
        # window aren't shadowed by a stale snapshot.
        parent._options_gpu_idx = get_gpu_index()
        parent._options_debug = get_debug_logging()
        parent._options_mem = get_mem_monitor()
        parent._options_mem_interval = get_mem_monitor_interval()
        # nvidia-smi is a subprocess; refresh the compute-device list once per
        # open, not per frame.
        parent._options_compute_devices = compute_gpu_devices()
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
        _refresh_gpu_panel(parent)

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

        imgui.dummy(imgui.ImVec2(0, 8))

        # Compute GPU: governs suite2p + cellpose (CUDA_VISIBLE_DEVICES).
        imgui.text_colored(_COL_DIM, "Compute GPU (suite2p / cellpose)")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Which GPU suite2p and cellpose run on. 'auto' uses all "
                "visible GPUs, 'cpu' forces CPU. Applies to newly started "
                "jobs; the suite2p Torch Device can still override per run."
            )
        devices = getattr(parent, "_options_compute_devices", []) or []
        values, labels = compute_gpu_options(devices)
        sel = compute_gpu_current_index(values)
        imgui.set_next_item_width(hello_imgui.em_size(20))
        changed, new_sel = imgui.combo("##compute_gpu", sel, labels)
        if changed and 0 <= new_sel < len(values):
            apply_compute_gpu(values[new_sel])

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
            # workers copy os.environ at spawn (ProcessManager.spawn), so set
            # MBO_DEBUG here to carry the toggle into the next isoview run —
            # the worker's isoview log bridge reads it to forward DEBUG.
            os.environ["MBO_DEBUG"] = "1" if new_debug else "0"
            _mbo_log.set_global_level(
                logging.DEBUG if new_debug else logging.INFO
            )

        changed, new_mem = imgui.checkbox("Log memory usage", parent._options_mem)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Sample RAM use to logs/mem_<id>.csv each tick.")
        if changed:
            parent._options_mem = new_mem
            set_mem_monitor(new_mem)
        imgui.same_line()
        imgui.begin_disabled(not parent._options_mem)
        imgui.set_next_item_width(hello_imgui.em_size(5))
        changed, new_iv = imgui.input_float(
            "tick (s)##mem_interval", parent._options_mem_interval, 0.0, 0.0, "%.1f"
        )
        imgui.end_disabled()
        if changed:
            parent._options_mem_interval = max(0.25, new_iv)
            set_mem_monitor_interval(parent._options_mem_interval)

        imgui.dummy(imgui.ImVec2(0, 8))
        btn_w = hello_imgui.em_size(6)
        imgui.set_cursor_pos_x((imgui.get_window_width() - btn_w) * 0.5)
        if imgui.button("Close", imgui.ImVec2(btn_w, 0)):
            imgui.close_current_popup()
    finally:
        imgui.end_popup()
