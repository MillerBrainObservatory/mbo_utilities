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


# colors
COL_BG_DARK = imgui.ImVec4(0.12, 0.12, 0.14, 1.0)
COL_BG_CARD = imgui.ImVec4(0.18, 0.18, 0.22, 1.0)
COL_ACCENT = imgui.ImVec4(0.35, 0.55, 0.85, 1.0)
COL_ACCENT_HOVER = imgui.ImVec4(0.45, 0.65, 0.95, 1.0)
COL_ACCENT_ACTIVE = imgui.ImVec4(0.25, 0.45, 0.75, 1.0)
COL_TEXT = imgui.ImVec4(0.92, 0.92, 0.94, 1.0)
COL_TEXT_DIM = imgui.ImVec4(0.6, 0.6, 0.65, 1.0)
COL_BORDER = imgui.ImVec4(0.3, 0.3, 0.35, 0.5)
COL_SUCCESS = imgui.ImVec4(0.3, 0.7, 0.4, 1.0)
COL_SECONDARY = imgui.ImVec4(0.4, 0.4, 0.45, 1.0)
COL_SECONDARY_HOVER = imgui.ImVec4(0.5, 0.5, 0.55, 1.0)


def push_button_style(primary=True):
    if primary:
        imgui.push_style_color(imgui.Col_.button, COL_ACCENT)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_ACCENT_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_ACCENT_ACTIVE)
    else:
        imgui.push_style_color(imgui.Col_.button, COL_SECONDARY)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_SECONDARY_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_SECONDARY)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)


def pop_button_style():
    imgui.pop_style_var(2)
    imgui.pop_style_color(3)


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
        # global style
        imgui.push_style_color(imgui.Col_.window_bg, COL_BG_DARK)
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
        imgui.push_style_color(imgui.Col_.border, COL_BORDER)
        imgui.push_style_color(imgui.Col_.separator, imgui.ImVec4(0.3, 0.3, 0.35, 0.3))
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.15, 0.15, 0.18, 1.0))
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.2, 0.2, 0.24, 1.0))
        imgui.push_style_color(imgui.Col_.check_mark, COL_ACCENT)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(2.5, 2.5))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, hello_imgui.em_to_vec2(0.8, 0.5))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, hello_imgui.em_to_vec2(1.0, 0.8))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 4.0)

        win_w = imgui.get_window_width()

        with imgui_ctx.begin_child("##main", size=imgui.ImVec2(0, 0)):
            imgui.push_id("pfd")

            # header
            imgui.dummy(hello_imgui.em_to_vec2(0, 1.0))
            title = "Miller Brain Observatory"
            title_sz = imgui.calc_text_size(title)
            imgui.set_cursor_pos_x((win_w - title_sz.x) * 0.5)
            imgui.text_colored(COL_ACCENT, title)

            subtitle = "Data Preview & Utilities"
            sub_sz = imgui.calc_text_size(subtitle)
            imgui.set_cursor_pos_x((win_w - sub_sz.x) * 0.5)
            imgui.text_colored(COL_TEXT_DIM, subtitle)

            imgui.dummy(hello_imgui.em_to_vec2(0, 1.5))
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 1.0))

            # description
            desc = "Preview raw ScanImage TIFFs, 3D/4D TIFF/Zarr stacks, and Suite2p outputs."
            desc_sz = imgui.calc_text_size(desc)
            imgui.set_cursor_pos_x((win_w - desc_sz.x) * 0.5)
            imgui.text_colored(COL_TEXT_DIM, desc)

            imgui.dummy(hello_imgui.em_to_vec2(0, 2.0))

            # action buttons
            btn_w = hello_imgui.em_size(20)
            btn_h = hello_imgui.em_size(2.8)
            btn_x = (win_w - btn_w) * 0.5

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
            set_tooltip("Open one or multiple supported files")

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.8))

            # select folder
            imgui.set_cursor_pos_x(btn_x)
            push_button_style(primary=True)
            if imgui.button("Select Folder", imgui.ImVec2(btn_w, btn_h)):
                self._select_folder = pfd.select_folder("Select folder")
            pop_button_style()
            set_tooltip("Select a folder containing image data")

            imgui.dummy(hello_imgui.em_to_vec2(0, 2.5))

            # options section
            opts_w = hello_imgui.em_size(28)
            opts_x = (win_w - opts_w) * 0.5
            imgui.set_cursor_pos_x(opts_x)

            imgui.push_style_color(imgui.Col_.child_bg, COL_BG_CARD)
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 8.0)
            with imgui_ctx.begin_child("##options", size=imgui.ImVec2(opts_w, hello_imgui.em_size(9)), child_flags=imgui.ChildFlags_.borders):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
                imgui.text_colored(COL_TEXT_DIM, "  Options")
                imgui.separator()
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

                imgui.indent(hello_imgui.em_size(0.8))

                _, self._widget_enabled = imgui.checkbox("Enable data preview widget", self._widget_enabled)
                set_tooltip("Enable or disable the interactive visualization widget")

                _, self.split_rois = imgui.checkbox("Separate ScanImage mROIs", self.split_rois)
                set_tooltip("Display each ScanImage mROI separately (raw TIFFs only)")

                _, self.metadata_only = imgui.checkbox("Metadata preview only", self.metadata_only)
                set_tooltip("Load only metadata for selected files")

                imgui.unindent(hello_imgui.em_size(0.8))

            imgui.pop_style_var()
            imgui.pop_style_color()

            imgui.dummy(hello_imgui.em_to_vec2(0, 2.0))

            # links
            links_w = hello_imgui.em_size(24)
            links_x = (win_w - links_w) * 0.5
            imgui.set_cursor_pos_x(links_x)

            link_btn_w = hello_imgui.em_size(7)
            link_btn_h = hello_imgui.em_size(2.0)

            push_button_style(primary=False)
            if imgui.button("Docs", imgui.ImVec2(link_btn_w, link_btn_h)):
                import webbrowser
                webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/")
            imgui.same_line()
            if imgui.button("Assembly", imgui.ImVec2(link_btn_w, link_btn_h)):
                import webbrowser
                webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/assembly.html")
            imgui.same_line()
            if imgui.button("Examples", imgui.ImVec2(link_btn_w, link_btn_h)):
                import webbrowser
                webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/api/usage.html")
            pop_button_style()

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

            # quit button - use spacing to push to bottom
            imgui.dummy(hello_imgui.em_to_vec2(0, 2.0))
            qsz = imgui.ImVec2(hello_imgui.em_size(8), hello_imgui.em_size(2.0))
            imgui.set_cursor_pos_x(win_w - qsz.x - hello_imgui.em_size(2))
            push_button_style(primary=False)
            if imgui.button("Quit", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True
            pop_button_style()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.5))

            imgui.pop_id()

        imgui.pop_style_var(4)
        imgui.pop_style_color(8)


if __name__ == "__main__":
    pass
