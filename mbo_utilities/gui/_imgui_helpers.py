"""
Generic ImGui helper utilities.

These are pure UI helpers with no domain-specific logic.
"""

from imgui_bundle import imgui

__all__ = [
    "checkbox_with_tooltip",
    "compact_header",
    "draw_checkbox_grid",
    "fmt_multivalue",
    "fmt_value",
    "set_tooltip",
    "settings_row_with_popup",
]


def checkbox_with_tooltip(label: str, value: bool, tooltip: str) -> bool:
    """Draw a checkbox with a (?) tooltip."""
    _changed, value = imgui.checkbox(label, value)
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(tooltip)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
    return value


def set_tooltip(tooltip: str, show_mark: bool = True) -> None:
    """Set a tooltip on the previous item, optionally with a (?) marker."""
    if show_mark:
        imgui.same_line()
        imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(tooltip)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def draw_checkbox_grid(
    items: list[tuple[str, bool]],
    id_prefix: str,
    on_change: callable,
    item_width: float | None = None,
    min_columns: int = 1,
    max_columns: int = 6,
) -> None:
    """
    Draw a grid of checkboxes that adapts column count to available width.

    Parameters
    ----------
    items : list[tuple[str, bool]]
        List of (label, checked) tuples for each checkbox.
    id_prefix : str
        Unique prefix for imgui IDs.
    on_change : callable
        Callback(index, new_value) called when a checkbox changes.
    item_width : float, optional
        Width per item in pixels. If None, calculated from longest label.
    min_columns : int
        Minimum columns to show (default 1).
    max_columns : int
        Maximum columns to allow (default 6).
    """
    if not items:
        return

    if item_width is None:
        checkbox_width = 20
        padding = 16
        longest_label = max(len(label) for label, _ in items)
        char_width = imgui.get_font_size() * 0.5
        item_width = checkbox_width + (longest_label * char_width) + padding

    available_width = imgui.get_content_region_avail().x
    num_columns = max(min_columns, min(max_columns, int(available_width / item_width)))

    if imgui.begin_table(f"##{id_prefix}_grid", num_columns, imgui.TableFlags_.none):
        for i, (label, checked) in enumerate(items):
            col = i % num_columns
            if col == 0:
                imgui.table_next_row()
            imgui.table_next_column()

            changed, new_checked = imgui.checkbox(f"{label}##{id_prefix}{i}", checked)
            if changed:
                on_change(i, new_checked)

        imgui.end_table()


def compact_header(label: str, default_open: bool = False) -> bool:
    """
    Draw a compact collapsing header with reduced padding.

    Returns True if the header is open, False if collapsed.
    """
    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 2))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 2))

    flags = imgui.TreeNodeFlags_.default_open if default_open else 0
    is_open = imgui.collapsing_header(label, flags)
    if isinstance(is_open, tuple):
        is_open = is_open[0]

    imgui.pop_style_var(2)
    return is_open


# Track popup states globally by popup_id
_popup_states: dict[str, bool] = {}


def settings_row_with_popup(
    popup_id: str,
    label: str,
    enabled: bool,
    draw_settings_content: callable,
    tooltip: str = "",
    checkbox_tooltip: str = "",
    popup_width: float = 400,
    popup_height: float = 0,
) -> tuple[bool, bool]:
    """
    Draw a compact settings row: [checkbox] Label [Settings button] -> popup.

    Parameters
    ----------
    popup_id : str
        Unique identifier for the popup.
    label : str
        Label shown next to checkbox and as popup title.
    enabled : bool
        Current enabled state for the checkbox.
    draw_settings_content : callable
        Function to draw the popup content (no arguments).
    tooltip : str, optional
        Tooltip for the Settings button.
    checkbox_tooltip : str, optional
        Tooltip for the checkbox.
    popup_width : float, optional
        Width of the popup window (default 400).
    popup_height : float, optional
        Height of the popup window (0 = auto-size).

    Returns
    -------
    tuple[bool, bool]
        (enabled_changed, new_enabled_value)
    """
    global _popup_states

    if popup_id not in _popup_states:
        _popup_states[popup_id] = False

    changed, new_enabled = imgui.checkbox(f"##{popup_id}_checkbox", enabled)

    if checkbox_tooltip and imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(checkbox_tooltip)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()

    imgui.same_line()
    imgui.text(label)

    imgui.same_line()
    if imgui.button(f"Settings##{popup_id}"):
        _popup_states[popup_id] = True
        imgui.open_popup(f"{label} Settings##{popup_id}")

    if tooltip and imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(tooltip)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()

    if popup_height > 0:
        imgui.set_next_window_size(
            imgui.ImVec2(popup_width, popup_height), imgui.Cond_.first_use_ever
        )
    else:
        imgui.set_next_window_size(
            imgui.ImVec2(popup_width, 0), imgui.Cond_.first_use_ever
        )

    opened, visible = imgui.begin_popup_modal(
        f"{label} Settings##{popup_id}",
        p_open=True if _popup_states[popup_id] else None,
        flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
    )

    if opened:
        if not visible:
            _popup_states[popup_id] = False
            imgui.close_current_popup()
        else:
            draw_settings_content()
            imgui.spacing()
            imgui.separator()
            if imgui.button("Close", imgui.ImVec2(80, 0)):
                _popup_states[popup_id] = False
                imgui.close_current_popup()

        imgui.end_popup()

    return changed, new_enabled


def fmt_value(x) -> str:
    """Format a value for display."""
    if x is None:
        return "—"
    if isinstance(x, (str, bool, int, float)):
        return repr(x)
    if isinstance(x, (bytes, bytearray)):
        return f"<{len(x)} bytes>"
    if isinstance(x, (tuple, list)):
        if len(x) <= 8:
            return repr(x)
        return f"[len={len(x)}]"
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        try:
            if x.size <= 8:
                return repr(x.tolist())
            return f"<shape={tuple(x.shape)}, dtype={x.dtype}>"
        except Exception:
            return f"<array dtype={x.dtype}>"
    return f"<{type(x).__name__}>"


def fmt_multivalue(value, max_items: int = 8) -> str:
    """Format a value that may be a list of per-camera values."""
    if isinstance(value, (list, tuple)):
        if len(value) <= max_items:
            formatted = []
            for v in value:
                if isinstance(v, float):
                    if v == int(v):
                        formatted.append(str(int(v)))
                    else:
                        formatted.append(f"{v:.4g}")
                else:
                    formatted.append(str(v))
            return "[" + ", ".join(formatted) + "]"
        formatted = []
        for v in value[:max_items]:
            if isinstance(v, float):
                formatted.append(f"{v:.4g}")
            else:
                formatted.append(str(v))
        return "[" + ", ".join(formatted) + f", +{len(value)-max_items}...]"
    return fmt_value(value)
