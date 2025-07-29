from pathlib import Path
import os
from imgui_bundle import (
    imgui,
    imgui_md,
    hello_imgui,
    imgui_ctx,
)
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities import get_mbo_dirs
from mbo_utilities.graphics._widgets import set_tooltip

MBO_THREADING_ENABLED = bool(
    int(os.getenv("MBO_THREADING_ENABLED", "1"))
)  # export MBO_DEV=1 to enable

MBO_ASSETS_PATH = get_mbo_dirs()["assets"]
MBO_LOGO_PATH = Path(MBO_ASSETS_PATH).joinpath("static", "logo_utilities.png")

HAS_LOGO = MBO_LOGO_PATH.exists() and MBO_LOGO_PATH.is_file()

if not HAS_LOGO:
    print("Warning: MBO logo not found at", MBO_LOGO_PATH)


class FileDialog:
    def __init__(self):
        self._assets_path = MBO_ASSETS_PATH
        self._logo_path = MBO_LOGO_PATH
        self.selected_path = None
        self._open_multi = None
        self._select_folder = None
        self._threading_enabled = MBO_THREADING_ENABLED
        self._widget_enabled = True
        self.metadata_only = False

    @property
    def widget_enabled(self):
        return self._widget_enabled

    @widget_enabled.setter
    def widget_enabled(self, value):
        self._widget_enabled = value

    @property
    def threading_enabled(self):
        return self._threading_enabled

    @threading_enabled.setter
    def threading_enabled(self, value):
        self._threading_enabled = value
        # print a warning because
        # this is generally done to fix a bug
        if not value:
            print(
                "Threading disabled,"
                " please report issues at https://github.com/MillerBrainObservatory/mbo_utilities/issues"
            )
            os.environ["MBO_THREADING_ENABLED"] = "0"
        if value:
            os.environ["MBO_THREADING_ENABLED"] = "1"

    def render(self):
        with imgui_ctx.begin_child("#fd"):
            imgui.push_id("pfd")

            # header --------------------------------------------------
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.5))

            # if HAS_LOGO:
            #     hello_imgui.image_from_asset(str(self._logo_path.expanduser().resolve().absolute()))

            imgui_md.render_unindented("""
            # General Python and shell utilities developed for the Miller Brain Observatory (MBO) workflows.

            ## Preview raw TIFFs, TIFF stacks, ScanImage files, or numpy memmaps. Load a directory of raw ScanImage files to run the data-preview widget, which allows visualization of projections, mean-subtraction, and preview scan-phase correction.

            [Docs Overview](https://millerbrainobservatory.github.io/mbo_utilities/) |
            [Assembly Guide](https://millerbrainobservatory.github.io/mbo_utilities/assembly.html) |
            [Function Examples](https://millerbrainobservatory.github.io/mbo_utilities/api/usage.html)
            """)

            imgui.dummy(hello_imgui.em_to_vec2(0, 5))

            # centre prompt ------------------------------------------
            txt = "Select a file, multiple files, or a folder to preview:"
            imgui.set_cursor_pos_x(
                (imgui.get_window_width() - imgui.calc_text_size(txt).x) * 0.5
            )
            imgui.text_colored(imgui.ImVec4(1, 0.85, 0.3, 1), txt)
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.5))

            # Centered “Open File(s)” button
            bsz_file = hello_imgui.em_to_vec2(18, 2.4)
            x_file = (imgui.get_window_width() - bsz_file.x) * 0.5
            imgui.set_cursor_pos_x(x_file)
            if imgui.button("Open File(s)", bsz_file):
                self._open_multi = pfd.open_file(
                    "Select files", options=pfd.opt.multiselect
                )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Open one or more TIFF / BIN files.")

            # small vertical gap
            imgui.dummy(hello_imgui.em_to_vec2(0, 1.0))

            # Centered, smaller “Select Folder” button underneath
            bsz_folder = hello_imgui.em_to_vec2(12, 2.0)
            x_folder = (imgui.get_window_width() - bsz_folder.x) * 0.5
            imgui.set_cursor_pos_x(x_folder)
            if imgui.button("Select Folder", bsz_folder):
                self._select_folder = pfd.select_folder("Select folder")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Pick a directory of TIFFs.")

            # load options -------------------------------------------
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.7))
            imgui.text_colored(imgui.ImVec4(1, 0.85, 0.3, 1), "Load Options")

            imgui.begin_group()
            _, self.threading_enabled = imgui.checkbox(
                "Enable Threading", self.threading_enabled
            )
            set_tooltip(
                "Enable/disable threading for the data preview widget. "
                "Useful to turn this off if you experience issues with the widget or for debugging."
                "For issues, please report here: "
                "https://github.com/MillerBrainObservatory/mbo_utilities/issues/new"
            )
            _, self.widget_enabled = imgui.checkbox(
                "Enable 'Data Preview' widget", self._widget_enabled
            )
            set_tooltip(
                "Enable/disable the 'Data Preview' widget. "
                "This widget allows you to visualize projections,"
                " mean-subtraction,"
                " and preview scan-phase correction."
            )

            _, self.metadata_only = imgui.checkbox(
                "Metadata Preview Only", self.metadata_only
            )
            set_tooltip(
                "Enable/disable the 'Data Preview' widget. "
                "This widget allows you to visualize projections,"
                " mean-subtraction,"
                " and preview scan-phase correction."
            )
            imgui.end_group()

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

            # quit button bottom-right -------------------------------
            qsz = hello_imgui.em_to_vec2(10, 1.8)
            imgui.set_cursor_pos(
                imgui.ImVec2(
                    imgui.get_window_width() - qsz.x - hello_imgui.em_size(1),
                    imgui.get_window_height() - qsz.y - hello_imgui.em_size(1),
                )
            )
            if imgui.button("Quit", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True

            imgui.pop_id()
