"""Shared Options popup (GPU adapter + debug logging).

Used by both the launch FileDialog and the main PreviewDataWidget's File menu
so the same widget configures both. Styled to match the Help / Keybinds /
Pipeline Settings popups (PopupAutoSize + begin_popup_modal).
"""
from __future__ import annotations

import logging
from typing import Any

from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as fa

from mbo_utilities import log as _mbo_log
from mbo_utilities.preferences import (
    get_gpu_index,
    set_gpu_index,
    get_debug_logging,
    set_debug_logging,
)
from mbo_utilities.gui._imgui_helpers import PopupAutoSize


_COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)
_COL_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)


def _ensure_gpu_list(parent: Any) -> None:
    """Read the pre-warmed adapter cache. NEVER call
    ``fpl.enumerate_adapters()`` from here — it initializes wgpu, which
    clobbers GLFW's WGL current-context (Glfw Error 65544) and makes
    the host window flicker. The cache is primed in ``_run_gui_impl``
    before ``immapp.run`` takes over the GL context.
    """
    if getattr(parent, "_options_gpu_adapters", None) is not None:
        return
    from mbo_utilities.gui._gpu_cache import get_adapters
    parent._options_gpu_adapters = list(get_adapters())
    labels = ["auto (default)"]
    for i, a in enumerate(parent._options_gpu_adapters):
        info = getattr(a, "info", {}) or {}
        name = info.get("device", info.get("description", f"adapter {i}"))
        type_ = info.get("adapter_type", info.get("device_type", "?"))
        labels.append(f"{i}: {name} [{type_}]")
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

    if parent._show_options_popup:
        # re-sync from prefs each open so changes from the CLI or another
        # window aren't shadowed by a stale snapshot.
        parent._options_gpu_idx = get_gpu_index()
        parent._options_debug = get_debug_logging()
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

        imgui.text_colored(_COL_ACCENT, f"{fa.ICON_FA_GEARS}  Options")
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 4))

        imgui.text_colored(_COL_DIM, "GPU adapter")
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
