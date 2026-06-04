"""
Generic ImGui/ImPlot helper utilities.

This module contains:
- ImPlot styling presets (seaborn light/dark)
- Popup sizing helpers
- Checkbox and tooltip helpers
- Value formatting utilities
"""

from contextlib import contextmanager

from imgui_bundle import imgui, hello_imgui, implot, ImVec4, ImVec2

__all__ = [
    "PopupAutoSize",
    "begin_popup_size",
    "checkbox_with_tooltip",
    "draw_boxed_label",
    "draw_checkbox_grid",
    "fmt_multivalue",
    "fmt_value",
    "set_tooltip",
    "settings_row_with_popup",
    "style_imgui_opaque",
    "style_seaborn_dark",
    "text_wrapped_cell",
    "tooltip_marks_right",
]


def draw_boxed_label(
    text: str,
    *,
    font=None,
    pad_x: float = 4.0,
    pad_y: float = 2.0,
    rounding: float = 3.0,
    thickness: float = 1.0,
) -> None:
    """Render `text` as a label surrounded by a thin rectangle.

    Border and text use the current `Col_.text` color, so any pushed text
    color (e.g. a "modified value" tint) flows through to both. Layout
    space is claimed via `imgui.dummy`, so a following `same_line()` /
    `set_tooltip` (?) marker positions correctly.

    If `font` is provided it's pushed for the size measurement, dummy
    sizing, and text rendering — pass a bold font here for "bold + box"
    emphasis.
    """
    if font is not None:
        imgui.push_font(font, font.legacy_size)
    try:
        text_size = imgui.calc_text_size(text)
        origin = imgui.get_cursor_screen_pos()
        box_w = text_size.x + 2 * pad_x
        box_h = text_size.y + 2 * pad_y
        imgui.dummy(ImVec2(box_w, box_h))
        draw_list = imgui.get_window_draw_list()
        col = imgui.get_color_u32(imgui.Col_.text)
        draw_list.add_rect(
            origin,
            ImVec2(origin.x + box_w, origin.y + box_h),
            col,
            rounding,
            0,
            thickness,
        )
        draw_list.add_text(
            ImVec2(origin.x + pad_x, origin.y + pad_y), col, text
        )
    finally:
        if font is not None:
            imgui.pop_font()

_IMGUI_OPAQUE_APPLIED = False


# stack of override modes for set_tooltip's mark alignment.
# when populated, the top of the stack overrides the per-call `align` arg.
# entered via the `tooltip_marks_right()` context manager — used by the
# unified pipeline-settings popup to right-align (?) markers inside each
# column-child without affecting set_tooltip's default behavior elsewhere.
_TOOLTIP_ALIGN_STACK: list[str] = []


@contextmanager
def tooltip_marks_right():
    """Right-align the (?) marker for any set_tooltip call inside this block.

    Use sparingly — pairs only with set_tooltip calls that happen inside a
    bounded child window (column / panel) so the right edge is meaningful.
    """
    _TOOLTIP_ALIGN_STACK.append("right")
    try:
        yield
    finally:
        _TOOLTIP_ALIGN_STACK.pop()


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


class PopupAutoSize:
    """Make a modal popup track its content size on every frame.

    Wraps ``WindowFlags_.always_auto_resize`` so the popup always fits
    its content — expanding a collapsing header inside the popup grows
    the window, collapsing it shrinks it back. No caching, no snap,
    no scrollbars from layout drift.

    Usage:

        sizer = PopupAutoSize("My Popup##id")     # construct once
        ...
        sizer.before_open()                       # before open_popup()
        imgui.open_popup("My Popup##id")
        ...
        opened, visible = imgui.begin_popup_modal(
            "My Popup##id",
            p_open=True,
            flags=sizer.flags(imgui.WindowFlags_.no_saved_settings),
        )

    Manual user-driven resizing is disabled by ``always_auto_resize``;
    that's a deliberate trade — the goal is "popup fits content", so
    handing resizing back to imgui removes both the scrollbar-on-grow
    failure mode and any need to double-click an edge.
    """

    def __init__(
        self,
        popup_id: str,
        *,
        anchor: str = "top",
        top_offset: float = 4.0,
        auto_resize: bool = True,
    ) -> None:
        """``anchor`` is one of ``"top"`` (horizontally centered, hanging
        from just below the title bar — recommended for popups that may
        grow tall as the user expands collapsing sections) or ``"center"``
        (viewport-centered). ``top_offset`` is the gap in pixels between
        the title bar and the popup's top edge when ``anchor="top"``.

        Set ``auto_resize=False`` to keep imgui's ``always_auto_resize``
        flag off — useful for popups whose body contains a ``begin_child``
        with negative-fill height (e.g. a scrollable list), which would
        collapse to zero size under auto-resize. The position policy
        still applies.
        """
        self.popup_id = popup_id
        self.anchor = anchor
        self.top_offset = top_offset
        self.auto_resize = auto_resize

    def before_open(self) -> None:
        """Call just before ``open_popup`` so the popup positions on appear."""
        viewport = imgui.get_main_viewport()
        if self.anchor == "center":
            imgui.set_next_window_pos(
                viewport.get_center(),
                imgui.Cond_.appearing,
                pivot=ImVec2(0.5, 0.5),
            )
            return
        # default "top": horizontally centered in the work area, top edge
        # ``top_offset`` px below the viewport's usable top (work_pos.y
        # already excludes any host menu bar imgui draws).
        work_pos = viewport.work_pos
        work_size = viewport.work_size
        imgui.set_next_window_pos(
            ImVec2(work_pos.x + work_size.x * 0.5, work_pos.y + self.top_offset),
            imgui.Cond_.appearing,
            pivot=ImVec2(0.5, 0.0),
        )

    def flags(self, extra: int = 0) -> int:
        """Return the flag mask for ``begin_popup_modal``.

        Adds ``always_auto_resize`` unless ``auto_resize=False`` was
        passed at construction.
        """
        if not self.auto_resize:
            return int(extra)
        return int(extra) | int(imgui.WindowFlags_.always_auto_resize)


# =============================================================================
# ImGui global styling
# =============================================================================


def style_imgui_opaque():
    """Force fully opaque popups, modals, child windows, and frames.

    idempotent — safe to call from any per-frame entry point. only mutates
    the live imgui context the first time it's called.
    """
    global _IMGUI_OPAQUE_APPLIED
    if _IMGUI_OPAQUE_APPLIED:
        return
    if imgui.get_current_context() is None:
        return
    _IMGUI_OPAQUE_APPLIED = True

    style = imgui.get_style()

    # global alpha — full opacity for everything except disabled widgets
    style.alpha = 1.0
    style.disabled_alpha = 0.6

    # window/popup/child backgrounds — fully opaque
    style.set_color_(imgui.Col_.window_bg.value, ImVec4(0.07, 0.08, 0.10, 1.00))
    style.set_color_(imgui.Col_.child_bg.value, ImVec4(0.07, 0.08, 0.10, 1.00))
    style.set_color_(imgui.Col_.popup_bg.value, ImVec4(0.07, 0.08, 0.10, 1.00))
    style.set_color_(imgui.Col_.menu_bar_bg.value, ImVec4(0.10, 0.11, 0.13, 1.00))

    # frame backgrounds (inputs, sliders, checkboxes) — opaque so they
    # don't bleed through the popup they sit in
    style.set_color_(imgui.Col_.frame_bg.value, ImVec4(0.13, 0.15, 0.18, 1.00))
    style.set_color_(imgui.Col_.frame_bg_hovered.value, ImVec4(0.18, 0.20, 0.24, 1.00))
    style.set_color_(imgui.Col_.frame_bg_active.value, ImVec4(0.22, 0.25, 0.30, 1.00))

    # title bars
    style.set_color_(imgui.Col_.title_bg.value, ImVec4(0.05, 0.06, 0.08, 1.00))
    style.set_color_(imgui.Col_.title_bg_active.value, ImVec4(0.10, 0.12, 0.16, 1.00))
    style.set_color_(imgui.Col_.title_bg_collapsed.value, ImVec4(0.05, 0.06, 0.08, 1.00))

    # borders — solid
    style.set_color_(imgui.Col_.border.value, ImVec4(0.30, 0.32, 0.36, 1.00))
    style.set_color_(imgui.Col_.border_shadow.value, ImVec4(0.00, 0.00, 0.00, 0.00))

    # dim overlay drawn behind a modal — strong dark to hide app contents
    style.set_color_(imgui.Col_.modal_window_dim_bg.value, ImVec4(0.0, 0.0, 0.0, 0.85))
    style.set_color_(imgui.Col_.nav_windowing_dim_bg.value, ImVec4(0.0, 0.0, 0.0, 0.85))


# =============================================================================
# ImPlot styling
# =============================================================================


def style_seaborn_dark():
    """
    Apply seaborn dark theme to ImPlot.
    """
    style = implot.get_style()

    def _set(attr_name, color):
        col = getattr(implot.Col_, attr_name, None)
        if col is not None:
            style.set_color_(col.value, color)

    # auto colors for lines and markers
    for attr in ("line", "fill", "marker_outline", "marker_fill"):
        _set(attr, implot.AUTO_COL)

    # backgrounds and axes
    _set("frame_bg", ImVec4(0.15, 0.17, 0.2, 1.00))
    _set("plot_bg", ImVec4(0.13, 0.15, 0.18, 1.00))
    _set("plot_border", ImVec4(0.00, 0.00, 0.00, 0.00))
    _set("axis_grid", ImVec4(0.35, 0.40, 0.45, 0.5))
    _set("axis_text", ImVec4(0.9, 0.9, 0.9, 1.0))
    _set("axis_bg_hovered", ImVec4(0.25, 0.27, 0.3, 1.00))
    _set("axis_bg_active", ImVec4(0.25, 0.27, 0.3, 0.75))

    # legends and labels
    _set("legend_bg", ImVec4(0.13, 0.15, 0.18, 1.00))
    _set("legend_border", ImVec4(0.4, 0.4, 0.4, 1.00))
    _set("legend_text", ImVec4(0.9, 0.9, 0.9, 1.00))
    _set("title_text", ImVec4(1.0, 1.0, 1.0, 1.00))
    _set("inlay_text", ImVec4(0.9, 0.9, 0.9, 1.00))

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


def set_tooltip(
    tooltip: str,
    show_mark: bool = True,
    align: str = "left",
    mark_dimmed: bool = True,
) -> None:
    """Set a tooltip on the previous item, optionally with a (?) marker.

    `align` controls (?) placement: "left" (default, immediately after the
    previous item) or "right" (snapped to the right edge of the current
    window/column). The active `tooltip_marks_right()` context manager, if
    any, overrides `align`.

    `mark_dimmed` controls (?) color: True (default) uses the disabled-text
    color so the marker reads as a soft hint; False renders it in normal
    text color when the marker should stand out.
    """
    if show_mark:
        imgui.same_line()
        effective_align = (
            _TOOLTIP_ALIGN_STACK[-1] if _TOOLTIP_ALIGN_STACK else align
        )
        if effective_align == "right":
            avail = imgui.get_content_region_avail().x
            qm = imgui.calc_text_size("(?)").x
            if avail > qm + 4:
                imgui.set_cursor_pos_x(
                    imgui.get_cursor_pos_x() + avail - qm - 4
                )
        if mark_dimmed:
            imgui.text_disabled("(?)")
        else:
            imgui.text("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(tooltip)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def text_wrapped_cell(text: str, color: "ImVec4 | None" = None) -> None:
    """Wrapped text that wraps at the current table cell's right edge.

    Inside a table, push_text_wrap_pos(0.0) wraps at the window edge (past
    the cell), so long cell values clip. Wrap at the cell width instead.
    """
    wrap_x = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
    imgui.push_text_wrap_pos(wrap_x)
    try:
        if color is None:
            imgui.text_unformatted(text)
        else:
            imgui.text_colored(color, text)
    finally:
        imgui.pop_text_wrap_pos()


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
            # only inline tiny arrays of plain scalar dtypes. object arrays
            # (suite2p Vmap is shape=(5,), dtype=object, each element a
            # different 2D float array per pyramid level) would otherwise
            # take the size<=8 fast path and dump every inner array via
            # repr(tolist()) — pages of numbers in the metadata viewer
            # instead of a one-line summary.
            scalar_kinds = "biufcSU"  # bool / int / uint / float / complex / str
            if x.size <= 8 and getattr(x.dtype, "kind", "") in scalar_kinds:
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
