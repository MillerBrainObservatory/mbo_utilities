"""Scope inspector helper used by debug popups."""

from __future__ import annotations

import inspect

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui._imgui_helpers import fmt_value

__all__ = ["draw_scope"]

_NAME_COLOR = imgui.ImVec4(0.95, 0.80, 0.30, 1.0)
_VALUE_COLOR = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)


def draw_scope():
    """Draw a scope inspector showing local variables from the calling frame."""
    with imgui_ctx.begin_child("Scope Inspector"):
        frame = inspect.currentframe().f_back
        vars_all = {**frame.f_locals}
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4))
        try:
            for name, val in sorted(vars_all.items()):
                if (
                    inspect.ismodule(val)
                    or (name.startswith("_") or name.endswith("_"))
                    or callable(val)
                ):
                    continue
                imgui.text_colored(_NAME_COLOR, name)
                imgui.same_line(spacing=16)
                imgui.text_colored(_VALUE_COLOR, fmt_value(val))
        finally:
            imgui.pop_style_var()
