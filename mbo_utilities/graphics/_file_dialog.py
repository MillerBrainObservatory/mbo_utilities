import os, shutil
import urllib.request
from pathlib import Path
import imgui_bundle
import mbo_utilities as mbo
from mbo_utilities.graphics._widgets import set_tooltip
from imgui_bundle import (
    imgui,
    imgui_md,
    hello_imgui,
    imgui_ctx,
    portable_file_dialogs as pfd,
)


def setup_imgui():
    assets = Path(mbo.get_mbo_dirs()["base"]) / "imgui" / "assets"
    fonts_dst = assets / "fonts"
    fonts_dst.mkdir(parents=True, exist_ok=True)
    (assets / "static").mkdir(parents=True, exist_ok=True)

    fonts_src = Path(imgui_bundle.__file__).parent / "assets" / "fonts"
    for p in fonts_src.rglob("*"):
        if p.is_file():
            d = fonts_dst / p.relative_to(fonts_src)
            if not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)

    roboto_dir = fonts_dst / "Roboto"
    roboto_dir.mkdir(parents=True, exist_ok=True)
    required = [
        roboto_dir / "Roboto-Regular.ttf",
        roboto_dir / "Roboto-Bold.ttf",
        roboto_dir / "Roboto-RegularItalic.ttf",
        fonts_dst / "fontawesome-webfont.ttf",
    ]
    fallback = next((t for t in roboto_dir.glob("*.ttf")), None)
    for need in required:
        if not need.exists() and fallback and fallback.exists():
            need.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback, need)

    hello_imgui.set_assets_folder(str(assets))


# setup_imgui()


# colors - high contrast dark theme for better visibility
COL_BG = imgui.ImVec4(0.11, 0.11, 0.12, 1.0)
COL_BG_CARD = imgui.ImVec4(0.16, 0.16, 0.17, 1.0)
COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)  # Darker blue for better text contrast
COL_ACCENT_HOVER = imgui.ImVec4(0.25, 0.55, 0.90, 1.0)
COL_ACCENT_ACTIVE = imgui.ImVec4(0.15, 0.45, 0.80, 1.0)
COL_TEXT = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)  # Pure white for maximum visibility
COL_TEXT_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)  # Lighter dim text
COL_BORDER = imgui.ImVec4(0.35, 0.35, 0.37, 0.7)
COL_SECONDARY = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)  # Lighter secondary buttons
COL_SECONDARY_HOVER = imgui.ImVec4(0.42, 0.42, 0.44, 1.0)
COL_SECONDARY_ACTIVE = imgui.ImVec4(0.28, 0.28, 0.30, 1.0)


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


class FileDialog:
    def __init__(self):
        self.selected_path = None
        self._open_multi = None
        self._select_folder = None
        self._widget_enabled = True
        self.metadata_only = False
        self.split_rois = False

    @property
    def widget_enabled(self):
        return self._widget_enabled

    @widget_enabled.setter
    def widget_enabled(self, value):
        self._widget_enabled = value

    def render(self):
        # global style - high contrast for visibility
        imgui.push_style_color(imgui.Col_.window_bg, COL_BG)
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
        imgui.push_style_color(imgui.Col_.border, COL_BORDER)
        imgui.push_style_color(imgui.Col_.separator, imgui.ImVec4(0.35, 0.35, 0.37, 0.6))
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.22, 0.22, 0.23, 1.0))
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.28, 0.28, 0.29, 1.0))
        imgui.push_style_color(imgui.Col_.check_mark, COL_ACCENT)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(3.0, 3.0))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, hello_imgui.em_to_vec2(1.2, 0.8))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, hello_imgui.em_to_vec2(1.2, 1.0))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)

        win_w = imgui.get_window_width()

        # disable scrollbars - content is fixed size
        child_flags = imgui.ChildFlags_.none
        window_flags = imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
        with imgui_ctx.begin_child("##main", size=imgui.ImVec2(0, 0), child_flags=child_flags, window_flags=window_flags):
            imgui.push_id("pfd")

            # header with larger text
            imgui.dummy(hello_imgui.em_to_vec2(0, 1.5))

            # Use larger font for title
            imgui.push_font(imgui.get_io().fonts.fonts[0])  # Default font but we'll scale it
            old_scale = imgui.get_font().scale
            imgui.get_font().scale = 1.8
            imgui.push_font(imgui.get_font())

            title = "Miller Brain Observatory"
            title_sz = imgui.calc_text_size(title)
            imgui.set_cursor_pos_x((win_w - title_sz.x) * 0.5)
            imgui.text_colored(COL_ACCENT, title)

            imgui.pop_font()
            imgui.get_font().scale = old_scale

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.5))

            # Larger subtitle
            old_scale = imgui.get_font().scale
            imgui.get_font().scale = 1.2
            imgui.push_font(imgui.get_font())

            subtitle = "Data Preview & Utilities"
            sub_sz = imgui.calc_text_size(subtitle)
            imgui.set_cursor_pos_x((win_w - sub_sz.x) * 0.5)
            imgui.text_colored(COL_TEXT_DIM, subtitle)

            imgui.pop_font()
            imgui.get_font().scale = old_scale

            imgui.dummy(hello_imgui.em_to_vec2(0, 1.5))
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 1.5))

            # action buttons - larger and more visible
            btn_w = hello_imgui.em_size(24)
            btn_h = hello_imgui.em_size(3.5)
            btn_x = (win_w - btn_w) * 0.5

            # Scale up button text
            old_scale = imgui.get_font().scale
            imgui.get_font().scale = 1.3
            imgui.push_font(imgui.get_font())

            # open files
            imgui.set_cursor_pos_x(btn_x)
            push_button_style(primary=True)
            if imgui.button("Open File(s)", imgui.ImVec2(btn_w, btn_h)):
                self._open_multi = pfd.open_file(
                    "Select files",
                    "",
                    ["Image Files", "*.tif *.tiff *.zarr *.npy *.bin",
                     "All Files", "*"],
                    pfd.opt.multiselect
                )
            pop_button_style()
            set_tooltip("Select one or more image files")

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.8))

            # select folder
            imgui.set_cursor_pos_x(btn_x)
            push_button_style(primary=True)
            if imgui.button("Select Folder", imgui.ImVec2(btn_w, btn_h)):
                self._select_folder = pfd.select_folder("Select folder")
            pop_button_style()
            set_tooltip("Select folder with image data")

            imgui.pop_font()
            imgui.get_font().scale = old_scale

            imgui.dummy(hello_imgui.em_to_vec2(0, 2.0))

            # features section - larger text for better visibility
            feat_w = hello_imgui.em_size(36)
            feat_x = (win_w - feat_w) * 0.5
            imgui.set_cursor_pos_x(feat_x)

            imgui.push_style_color(imgui.Col_.child_bg, COL_BG_CARD)
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 8.0)
            feat_child_flags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.borders
            with imgui_ctx.begin_child("##features", size=imgui.ImVec2(feat_w, 0), child_flags=feat_child_flags):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.7))

                # Scale up feature text
                old_scale = imgui.get_font().scale
                imgui.get_font().scale = 1.15
                imgui.push_font(imgui.get_font())

                # Supported Formats
                imgui.indent(hello_imgui.em_size(1.2))
                imgui.text_colored(COL_ACCENT, "Supported Formats")
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))
                imgui.text_colored(COL_TEXT_DIM, "• ScanImage multi-ROI TIFFs")
                imgui.text_colored(COL_TEXT_DIM, "• TIFF, Zarr, HDF5, NumPy")
                imgui.text_colored(COL_TEXT_DIM, "• Suite2p binary output")

                imgui.dummy(hello_imgui.em_to_vec2(0, 0.6))
                imgui.separator()
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.6))

                # Features
                imgui.text_colored(COL_ACCENT, "Features")
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))
                imgui.text_colored(COL_TEXT_DIM, "• Interactive 3D/4D visualization")
                imgui.text_colored(COL_TEXT_DIM, "• ROI stitching & separation")
                imgui.text_colored(COL_TEXT_DIM, "• Scan-phase correction")
                imgui.text_colored(COL_TEXT_DIM, "• Format conversion")

                imgui.dummy(hello_imgui.em_to_vec2(0, 0.6))
                imgui.separator()
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.6))

                # Options
                imgui.text_colored(COL_ACCENT, "Options")
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))

                _, self._widget_enabled = imgui.checkbox("Enable preview widget", self._widget_enabled)
                _, self.split_rois = imgui.checkbox("Separate multi-ROIs", self.split_rois)
                _, self.metadata_only = imgui.checkbox("Metadata only", self.metadata_only)

                imgui.pop_font()
                imgui.get_font().scale = old_scale

                imgui.unindent(hello_imgui.em_size(1.2))
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.7))

            imgui.pop_style_var()
            imgui.pop_style_color()

            imgui.dummy(hello_imgui.em_to_vec2(0, 1.5))

            # file/folder completion
            if self._open_multi and self._open_multi.ready():
                self.selected_path = self._open_multi.result()
                if self.selected_path:
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._open_multi = None
            if self._select_folder and self._select_folder.ready():
                self.selected_path = self._select_folder.result()
                if self.selected_path:
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._select_folder = None

            # quit button - larger and more visible
            imgui.dummy(hello_imgui.em_to_vec2(0, 2.0))

            old_scale = imgui.get_font().scale
            imgui.get_font().scale = 1.2
            imgui.push_font(imgui.get_font())

            qsz = imgui.ImVec2(hello_imgui.em_size(10), hello_imgui.em_size(2.5))
            imgui.set_cursor_pos_x(win_w - qsz.x - hello_imgui.em_size(2.5))
            push_button_style(primary=False)
            if imgui.button("Quit", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True
            pop_button_style()

            imgui.pop_font()
            imgui.get_font().scale = old_scale

            imgui.pop_id()

        imgui.pop_style_var(4)
        imgui.pop_style_color(8)


if __name__ == "__main__":
    pass
