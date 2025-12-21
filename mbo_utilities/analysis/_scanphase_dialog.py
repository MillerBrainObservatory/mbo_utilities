"""
File dialog for scan-phase analysis command.

This dialog provides file selection with context about the scan-phase
analysis tool and how to interpret its outputs.
"""

from pathlib import Path

import mbo_utilities as mbo
from imgui_bundle import (
    hello_imgui,
    imgui,
    imgui_ctx,
    portable_file_dialogs as pfd,
)
from mbo_utilities.gui._widgets import set_tooltip
from mbo_utilities.preferences import (
    get_default_open_dir,
    get_last_dir,
    set_last_dir,
    add_recent_file,
)


# Colors - matching main file dialog
COL_BG = imgui.ImVec4(0.11, 0.11, 0.12, 1.0)
COL_BG_CARD = imgui.ImVec4(0.16, 0.16, 0.17, 1.0)
COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)
COL_ACCENT_HOVER = imgui.ImVec4(0.25, 0.55, 0.90, 1.0)
COL_ACCENT_ACTIVE = imgui.ImVec4(0.15, 0.45, 0.80, 1.0)
COL_TEXT = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
COL_TEXT_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)
COL_BORDER = imgui.ImVec4(0.35, 0.35, 0.37, 0.7)
COL_SECONDARY = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
COL_SECONDARY_HOVER = imgui.ImVec4(0.42, 0.42, 0.44, 1.0)
COL_SECONDARY_ACTIVE = imgui.ImVec4(0.28, 0.28, 0.30, 1.0)
COL_SUCCESS = imgui.ImVec4(0.2, 0.7, 0.3, 1.0)
COL_WARNING = imgui.ImVec4(0.9, 0.7, 0.2, 1.0)


def push_button_style(primary=True):
    if primary:
        imgui.push_style_color(imgui.Col_.button, COL_ACCENT)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_ACCENT_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_ACCENT_ACTIVE)
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
    else:
        imgui.push_style_color(imgui.Col_.button, COL_SECONDARY)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_SECONDARY_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_SECONDARY_ACTIVE)
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)


def pop_button_style():
    imgui.pop_style_var(2)
    imgui.pop_style_color(4)


class ScanPhaseFileDialog:
    """File dialog with scan-phase analysis context."""

    def __init__(self):
        self.selected_path = None
        self._open_file = None
        self._select_folder = None
        self._default_dir = str(get_default_open_dir())

    def render(self):
        # Global style
        imgui.push_style_color(imgui.Col_.window_bg, COL_BG)
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
        imgui.push_style_color(imgui.Col_.border, COL_BORDER)
        imgui.push_style_color(imgui.Col_.separator, imgui.ImVec4(0.35, 0.35, 0.37, 0.6))
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.22, 0.22, 0.23, 1.0))
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.28, 0.28, 0.29, 1.0))
        imgui.push_style_color(imgui.Col_.check_mark, COL_ACCENT)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(1.0, 0.8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, hello_imgui.em_to_vec2(0.6, 0.4))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, hello_imgui.em_to_vec2(0.6, 0.4))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)

        win_w = imgui.get_window_width()
        win_h = imgui.get_window_height()

        with imgui_ctx.begin_child("##main", size=imgui.ImVec2(0, 0)):
            imgui.push_id("scanphase_dialog")

            # Header
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            title = "Scan-Phase Analysis"
            title_sz = imgui.calc_text_size(title)
            imgui.set_cursor_pos_x((win_w - title_sz.x) * 0.5)
            imgui.text_colored(COL_ACCENT, title)

            subtitle = "Bidirectional Scanning Phase Correction Analysis"
            sub_sz = imgui.calc_text_size(subtitle)
            imgui.set_cursor_pos_x((win_w - sub_sz.x) * 0.5)
            imgui.text_colored(COL_TEXT_DIM, subtitle)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            # Action buttons
            btn_w = hello_imgui.em_size(16)
            btn_h = hello_imgui.em_size(1.8)
            btn_x = (win_w - btn_w) * 0.5

            imgui.set_cursor_pos_x(btn_x)
            push_button_style(primary=True)
            if imgui.button("Select File", imgui.ImVec2(btn_w, btn_h)):
                self._open_file = pfd.open_file(
                    "Select imaging data for scan-phase analysis",
                    self._default_dir,
                    ["Image Files", "*.tif *.tiff *.zarr *.npy *.bin",
                     "All Files", "*"],
                )
            pop_button_style()
            set_tooltip("Select a ScanImage TIFF or other imaging data file")

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            imgui.set_cursor_pos_x(btn_x)
            push_button_style(primary=True)
            if imgui.button("Select Folder", imgui.ImVec2(btn_w, btn_h)):
                self._select_folder = pfd.select_folder(
                    "Select folder with imaging data",
                    self._default_dir
                )
            pop_button_style()
            set_tooltip("Select folder containing TIFF files")

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))

            # Info cards
            card_w = min(hello_imgui.em_size(32), win_w - hello_imgui.em_size(2))
            card_x = (win_w - card_w) * 0.5
            imgui.set_cursor_pos_x(card_x)

            imgui.push_style_color(imgui.Col_.child_bg, COL_BG_CARD)
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 6.0)

            # What this tool does
            with imgui_ctx.begin_child(
                "##what_it_does",
                size=imgui.ImVec2(card_w, hello_imgui.em_size(8)),
                child_flags=imgui.ChildFlags_.borders
            ):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.indent(hello_imgui.em_size(0.4))

                imgui.text_colored(COL_ACCENT, "What This Tool Does")
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))

                imgui.text_wrapped(
                    "Analyzes phase offset variation across your data to help "
                    "you choose optimal correction parameters:"
                )
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))
                imgui.bullet_text("Temporal: How offset changes over frames")
                imgui.bullet_text("Z-plane: Variation between imaging planes")
                imgui.bullet_text("Spatial: Uniformity across X and Y")
                imgui.bullet_text("Window size: Effect of frame averaging")
                imgui.bullet_text("FFT vs Integer: Method comparison")

                imgui.unindent(hello_imgui.em_size(0.4))

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            # How to interpret outputs
            imgui.set_cursor_pos_x(card_x)
            with imgui_ctx.begin_child(
                "##interpret",
                size=imgui.ImVec2(card_w, hello_imgui.em_size(9)),
                child_flags=imgui.ChildFlags_.borders
            ):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.indent(hello_imgui.em_size(0.4))

                imgui.text_colored(COL_ACCENT, "Interpreting Outputs")
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))

                imgui.text_colored(COL_SUCCESS, "offset_std < 0.1:")
                imgui.same_line()
                imgui.text("Very stable, use method='mean'")

                imgui.text_colored(COL_WARNING, "offset_std > 0.3:")
                imgui.same_line()
                imgui.text("Variable, consider method='frame'")

                imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))

                imgui.text_colored(COL_SUCCESS, "spatial_std < 0.2:")
                imgui.same_line()
                imgui.text("Uniform, global correction works")

                imgui.text_colored(COL_WARNING, "spatial_std > 0.5:")
                imgui.same_line()
                imgui.text("Non-uniform, check scanner calibration")

                imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))

                imgui.text("If offset has fractional component > 0.2:")
                imgui.text_colored(COL_ACCENT, "  -> use_fft=True for subpixel precision")

                imgui.unindent(hello_imgui.em_size(0.4))

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            # Parameter adjustment guide
            imgui.set_cursor_pos_x(card_x)
            with imgui_ctx.begin_child(
                "##params",
                size=imgui.ImVec2(card_w, hello_imgui.em_size(8)),
                child_flags=imgui.ChildFlags_.borders
            ):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.indent(hello_imgui.em_size(0.4))

                imgui.text_colored(COL_ACCENT, "Adjusting Parameters")
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))

                imgui.text("Based on results, adjust MboRawArray:")
                imgui.text_colored(COL_TEXT_DIM, "  arr = imread(path, fix_phase=True)")
                imgui.text_colored(COL_TEXT_DIM, "  arr.use_fft = True")
                imgui.text_colored(COL_TEXT_DIM, "  arr.phasecorr_method = 'mean'")
                imgui.text_colored(COL_TEXT_DIM, "  arr.upsample = 5")
                imgui.text_colored(COL_TEXT_DIM, "  arr.border = 4")
                imgui.text_colored(COL_TEXT_DIM, "  arr.max_offset = 4")

                imgui.unindent(hello_imgui.em_size(0.4))

            imgui.pop_style_var()
            imgui.pop_style_color()

            # File completion handlers
            if self._open_file and self._open_file.ready():
                result = self._open_file.result()
                if result:
                    self.selected_path = result[0] if isinstance(result, list) else result
                    add_recent_file(self.selected_path, file_type="file")
                    set_last_dir("open_file", self.selected_path)
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._open_file = None

            if self._select_folder and self._select_folder.ready():
                self.selected_path = self._select_folder.result()
                if self.selected_path:
                    add_recent_file(self.selected_path, file_type="folder")
                    set_last_dir("open_folder", self.selected_path)
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._select_folder = None

            # Quit button
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            qsz = imgui.ImVec2(hello_imgui.em_size(5), hello_imgui.em_size(1.5))
            imgui.set_cursor_pos_x(win_w - qsz.x - hello_imgui.em_size(1.0))
            push_button_style(primary=False)
            if imgui.button("Cancel", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True
            pop_button_style()

            imgui.pop_id()

        imgui.pop_style_var(4)
        imgui.pop_style_color(8)


def select_scanphase_file() -> str | None:
    """
    Show file selection dialog for scan-phase analysis.

    Returns
    -------
    str or None
        Selected file path, or None if cancelled.
    """
    from mbo_utilities.gui import _setup  # triggers setup on import
    from mbo_utilities.gui._setup import get_default_ini_path
    from imgui_bundle import immapp, hello_imgui

    dlg = ScanPhaseFileDialog()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Scan-Phase Analysis"
    params.app_window_params.window_geometry.size = (500, 720)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = True
    params.ini_filename = get_default_ini_path("scanphase_dialog")
    params.callbacks.show_gui = dlg.render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)

    return dlg.selected_path
