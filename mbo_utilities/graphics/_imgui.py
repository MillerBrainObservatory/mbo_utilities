import shutil

import numpy as np
from icecream import ic
from imgui_bundle import imgui, hello_imgui, implot, ImVec4, ImVec2
from pathlib import Path

from mbo_utilities.file_io import (
    _get_mbo_project_root,
    get_mbo_dirs,
)


def setup_imgui():
    project_assets: Path = _get_mbo_project_root().joinpath("assets")
    mbo_dirs = get_mbo_dirs()

    imgui_path = mbo_dirs["base"].joinpath("imgui")
    imgui_path.mkdir(exist_ok=True)

    imgui_ini_path = imgui_path.joinpath("imgui.ini")
    imgui_ini_path.parent.mkdir(exist_ok=True)
    imgui.create_context()
    imgui.get_io().set_ini_filename("/home/flynn/lbm_data/mbo_filesettings.ini")

    if not project_assets.is_dir():
        ic("Assets folder not found.")
        return

    assets_path = imgui_path.joinpath("assets")
    assets_path.mkdir(exist_ok=True)

    shutil.copytree(project_assets, assets_path, dirs_exist_ok=True)
    hello_imgui.set_assets_folder(str(project_assets))

    font_path = (
        assets_path / "fonts" / "JetBrainsMono" / "JetBrainsMonoNerdFont-Bold.ttf"
    )
    try:
        if font_path.is_file():
            imgui.get_io().fonts.clear()
            font_path = Path(font_path).expanduser().resolve(strict=True)
            imgui.get_io().fonts.add_font_from_file_ttf(str(font_path), 16.0)
        else:
            ic("Font not found:", font_path)
    except Exception as e:
        ic("Error loading font:", e)


def begin_popup_size():
    width_em = hello_imgui.em_size(1.0)  # 1em in pixels
    win_w = imgui.get_window_width()
    win_h = imgui.get_window_height()

    # 75% of window size in ems
    w = win_w * 0.75 / width_em
    h = win_h * 0.75 / width_em  # same em size applies for height in most UIs

    # Clamp in em units
    w = min(max(w, 20), 60)  # roughly 300–800 px if 1em ≈ 15px
    h = min(max(h, 20), 60)

    return hello_imgui.em_to_vec2(w, h)


def ndim_to_frame(arr, t=0, z=0) -> np.ndarray:
    if arr.ndim == 4:  # TZXY
        return arr[t, z]
    if arr.ndim == 3:  # TXY
        return arr[t]
    if arr.ndim == 2:  # XY
        return arr
    raise ValueError(f"Unsupported data shape: {arr.shape}")


def style_seaborn():
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
    style = implot.get_style()

    # Auto colors for lines and markers
    style.set_color_(implot.Col_.line.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.fill.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_outline.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_fill.value, implot.AUTO_COL)

    # Backgrounds and axes
    style.set_color_(
        implot.Col_.frame_bg.value, ImVec4(0.15, 0.17, 0.2, 1.00)
    )  # dark gray
    style.set_color_(
        implot.Col_.plot_bg.value, ImVec4(0.13, 0.15, 0.18, 1.00)
    )  # darker gray
    style.set_color_(implot.Col_.plot_border.value, ImVec4(0.00, 0.00, 0.00, 0.00))
    style.set_color_(
        implot.Col_.axis_grid.value, ImVec4(0.35, 0.40, 0.45, 0.5)
    )  # light grid
    style.set_color_(
        implot.Col_.axis_text.value, ImVec4(0.9, 0.9, 0.9, 1.0)
    )  # light text
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
