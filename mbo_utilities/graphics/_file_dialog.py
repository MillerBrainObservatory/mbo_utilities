import os, shutil
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
        pad = hello_imgui.em_to_vec2(3, 1)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(2, 2))
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.9, 0.9, 0.9, 1.0))
        imgui.push_style_color(imgui.Col_.separator, imgui.ImVec4(0.7, 0.7, 0.7, 0.25))

        with imgui_ctx.begin_child("#outer", size=imgui.ImVec2(-pad.x * 2, 0)):
            with imgui_ctx.begin_child("#fd", size=imgui.ImVec2(-pad.x, 0)):
                imgui.push_id("pfd")

                # header --------------------------------------------------
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.8))
                imgui.separator()
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.8))

                imgui_md.render_unindented("""
                # General Python and shell utilities developed for the Miller Brain Observatory (MBO) workflows.

                ## Preview raw ScanImage TIFFs, 3D (planar)/4D (volumetric) TIFF/Zarr stacks, and Suite2p raw/registered outputs.

                Load a directory of raw ScanImage files to run the data-preview widget, which allows visualization of projections, mean-subtraction, and preview scan-phase correction.

                [Docs Overview](https://millerbrainobservatory.github.io/mbo_utilities/) |
                [Assembly Guide](https://millerbrainobservatory.github.io/mbo_utilities/assembly.html) |
                [Function Examples](https://millerbrainobservatory.github.io/mbo_utilities/api/usage.html)
                """)

                imgui.dummy(hello_imgui.em_to_vec2(0, 4))

                # prompt --------------------------------------------------
                txt = "Select a file, multiple files, or a folder to preview:"
                imgui.set_cursor_pos_x((imgui.get_window_width() - imgui.calc_text_size(txt).x) * 0.5)
                imgui.text_colored(imgui.ImVec4(1.0, 0.85, 0.3, 1.0), txt)
                imgui.dummy(hello_imgui.em_to_vec2(0, 1))

                # open files button --------------------------------------
                bsz_file = hello_imgui.em_to_vec2(18, 2.4)
                x_file = (imgui.get_window_width() - bsz_file.x) * 0.5
                imgui.set_cursor_pos_x(x_file)
                if imgui.button("Open File(s)", bsz_file):
                    self._open_multi = pfd.open_file("Select files", options=pfd.opt.multiselect)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Open one or multiple supported files.")

                imgui.dummy(hello_imgui.em_to_vec2(0, 1.5))

                # select folder button -----------------------------------
                bsz_folder = hello_imgui.em_to_vec2(12, 2.0)
                x_folder = (imgui.get_window_width() - bsz_folder.x) * 0.5
                imgui.set_cursor_pos_x(x_folder)
                if imgui.button("Select Folder", bsz_folder):
                    self._select_folder = pfd.select_folder("Select folder")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Select a folder containing image data.")

                # load options -------------------------------------------
                imgui.dummy(hello_imgui.em_to_vec2(0, 2.0))
                imgui.text_colored(imgui.ImVec4(0.9, 0.75, 0.2, 1), "Load Options")
                imgui.separator()
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.6))

                imgui.begin_group()
                _, self.split_rois = imgui.checkbox(
                    "(Raw ScanImage tiffs only) Separate ScanImage mROIs", self.split_rois
                )
                set_tooltip(
                    "Display each ScanImage mROI separately in the preview widget. "
                    "Does not affect files on disk."
                )

                _, self.widget_enabled = imgui.checkbox(
                    "Enable 'Data Preview' widget", self._widget_enabled
                )
                set_tooltip(
                    "Enable or disable the interactive 'Data Preview' visualization widget."
                )

                _, self.metadata_only = imgui.checkbox(
                    "Metadata Preview Only", self.metadata_only
                )
                set_tooltip(
                    "Load only metadata for selected files (experimental)."
                )
                imgui.end_group()

                # file/folder completion ---------------------------------
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

                # quit button --------------------------------------------
                qsz = hello_imgui.em_to_vec2(10, 1.8)
                imgui.set_cursor_pos(
                    imgui.ImVec2(
                        imgui.get_window_width() - qsz.x - hello_imgui.em_size(1.5),
                        imgui.get_window_height() - qsz.y - hello_imgui.em_size(1.5),
                    )
                )
                if imgui.button("Quit", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                    self.selected_path = None
                    hello_imgui.get_runner_params().app_shall_exit = True

                imgui.pop_id()

        imgui.pop_style_color(2)
        imgui.pop_style_var()

if __name__ == "__main__":
    pass