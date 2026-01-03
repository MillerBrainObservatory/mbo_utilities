"""
Generic ImGui/ImPlot helper utilities.

This module contains:
- ImPlot styling presets (seaborn light/dark)
- Popup sizing helpers
- Checkbox and tooltip helpers
- Value formatting utilities
"""

from imgui_bundle import imgui, hello_imgui, implot, ImVec4, ImVec2

__all__ = [
    "begin_popup_size",
    "checkbox_with_tooltip",
    "compact_header",
    "draw_checkbox_grid",
    "fmt_multivalue",
    "fmt_value",
    "set_tooltip",
    "settings_row_with_popup",
    "style_seaborn",
    "style_seaborn_dark",
]


# =============================================================================
# Popup sizing
# =============================================================================


def begin_popup_size():
    """Calculate popup size based on window dimensions."""
    width_em = hello_imgui.em_size(1.0)  # 1em in pixels
    win_w = imgui.get_window_width()
    win_h = imgui.get_window_height()

    # 75% of window size in ems
    w = win_w * 0.75 / width_em
    h = win_h * 0.75 / width_em

    # Clamp in em units (roughly 300-800 px if 1em ≈ 15px)
    w = min(max(w, 20), 60)
    h = min(max(h, 20), 60)

    return hello_imgui.em_to_vec2(w, h)


# =============================================================================
# ImPlot styling
# =============================================================================


def style_seaborn():
    """Apply seaborn light theme to ImPlot."""
    style = implot.get_style()
    style.set_color_(implot.Col_.line.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.fill.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_outline.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_fill.value, implot.AUTO_COL)

    style.set_color_(implot.Col_.error_bar.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.frame_bg.value, ImVec4(1.00, 1.00, 1.00, 1.00))
    style.set_color_(implot.Col_.plot_bg.value, ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.plot_border.value, ImVec4(0.00, 0.00, 0.00, 0.00))
    style.set_color_(implot.Col_.legend_bg.value, ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.legend_border.value, ImVec4(0.80, 0.81, 0.85, 1.00))
    style.set_color_(implot.Col_.legend_text.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.title_text.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.inlay_text.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.axis_text.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.axis_grid.value, ImVec4(1.00, 1.00, 1.00, 1.00))
    style.set_color_(implot.Col_.axis_bg_hovered.value, ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.axis_bg_active.value, ImVec4(0.92, 0.92, 0.95, 0.75))
    style.set_color_(implot.Col_.selection.value, ImVec4(1.00, 0.65, 0.00, 1.00))
    style.set_color_(implot.Col_.crosshairs.value, ImVec4(0.23, 0.10, 0.64, 0.50))

    style.line_weight = 1.5
    style.marker = implot.Marker_.none.value
    style.marker_size = 4
    style.marker_weight = 1
    style.fill_alpha = 1.0
    style.error_bar_size = 5
    style.error_bar_weight = 1.5
    style.digital_bit_height = 8
    style.digital_bit_gap = 4
    style.plot_border_size = 0
    style.minor_alpha = 1.0
    style.major_tick_len = ImVec2(0, 0)
    style.minor_tick_len = ImVec2(0, 0)
    style.major_tick_size = ImVec2(0, 0)
    style.minor_tick_size = ImVec2(0, 0)
    style.major_grid_size = ImVec2(1.2, 1.2)
    style.minor_grid_size = ImVec2(1.2, 1.2)
    style.plot_padding = ImVec2(12, 12)
    style.label_padding = ImVec2(5, 5)
    style.legend_padding = ImVec2(5, 5)
    style.mouse_pos_padding = ImVec2(5, 5)
    style.plot_min_size = ImVec2(300, 225)


def style_seaborn_dark():
    """Apply seaborn dark theme to ImPlot."""
    style = implot.get_style()

    # Auto colors for lines and markers
    style.set_color_(implot.Col_.line.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.fill.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_outline.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_fill.value, implot.AUTO_COL)

    # Backgrounds and axes
    style.set_color_(implot.Col_.frame_bg.value, ImVec4(0.15, 0.17, 0.2, 1.00))
    style.set_color_(implot.Col_.plot_bg.value, ImVec4(0.13, 0.15, 0.18, 1.00))
    style.set_color_(implot.Col_.plot_border.value, ImVec4(0.00, 0.00, 0.00, 0.00))
    style.set_color_(implot.Col_.axis_grid.value, ImVec4(0.35, 0.40, 0.45, 0.5))
    style.set_color_(implot.Col_.axis_text.value, ImVec4(0.9, 0.9, 0.9, 1.0))
    style.set_color_(implot.Col_.axis_bg_hovered.value, ImVec4(0.25, 0.27, 0.3, 1.00))
    style.set_color_(implot.Col_.axis_bg_active.value, ImVec4(0.25, 0.27, 0.3, 0.75))

    # Legends and labels
    style.set_color_(implot.Col_.legend_bg.value, ImVec4(0.13, 0.15, 0.18, 1.00))
    style.set_color_(implot.Col_.legend_border.value, ImVec4(0.4, 0.4, 0.4, 1.00))
    style.set_color_(implot.Col_.legend_text.value, ImVec4(0.9, 0.9, 0.9, 1.00))
    style.set_color_(implot.Col_.title_text.value, ImVec4(1.0, 1.0, 1.0, 1.00))
    style.set_color_(implot.Col_.inlay_text.value, ImVec4(0.9, 0.9, 0.9, 1.00))

    # Misc
    style.set_color_(implot.Col_.error_bar.value, ImVec4(0.9, 0.9, 0.9, 1.00))
    style.set_color_(implot.Col_.selection.value, ImVec4(1.00, 0.65, 0.00, 1.00))
    style.set_color_(implot.Col_.crosshairs.value, ImVec4(0.8, 0.8, 0.8, 0.5))

    # Sizes
    style.line_weight = 1.5
    style.marker = implot.Marker_.none.value
    style.marker_size = 4
    style.marker_weight = 1
    style.fill_alpha = 1.0
    style.error_bar_size = 5
    style.error_bar_weight = 1.5
    style.digital_bit_height = 8
    style.digital_bit_gap = 4
    style.plot_border_size = 0
    style.minor_alpha = 0.3
    style.major_tick_len = ImVec2(0, 0)
    style.minor_tick_len = ImVec2(0, 0)
    style.major_tick_size = ImVec2(0, 0)
    style.minor_tick_size = ImVec2(0, 0)
    style.major_grid_size = ImVec2(1.2, 1.2)
    style.minor_grid_size = ImVec2(1.2, 1.2)
    style.plot_padding = ImVec2(12, 12)
    style.label_padding = ImVec2(5, 5)
    style.legend_padding = ImVec2(5, 5)
    style.mouse_pos_padding = ImVec2(5, 5)
    style.plot_min_size = ImVec2(300, 225)


# =============================================================================
# Checkbox and tooltip helpers
# =============================================================================


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
