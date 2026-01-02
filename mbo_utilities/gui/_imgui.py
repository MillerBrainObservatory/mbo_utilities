"""
ImGui/ImPlot styling and utilities.

This module contains:
- ImPlot styling presets (seaborn light/dark)
- Popup sizing helpers
"""

from imgui_bundle import imgui, hello_imgui, implot, ImVec4, ImVec2


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
