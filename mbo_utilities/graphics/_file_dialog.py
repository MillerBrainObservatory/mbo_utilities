import os
from imgui_bundle import (
    imgui,
    imgui_md,
    hello_imgui,
    imgui_ctx,
)
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities.graphics._widgets import set_tooltip

MBO_THREADING_ENABLED = bool(
    int(os.getenv("MBO_THREADING_ENABLED", "1"))
)  # export MBO_DEV=1 to enable


class FileDialog:
    def __init__(self):
        self.selected_path = None
        self._open_multi = None
        self._select_folder = None
        self._threading_enabled = MBO_THREADING_ENABLED

    @property
    def threading_enabled(self):
        return self._threading_enabled

    @threading_enabled.setter
    def threading_enabled(self, value):
        self._threading_enabled = value
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

            imgui_md.render_unindented("""
            # MBO Utilities

            General Python and shell utilities developed for the Miller Brain Observatory (MBO) workflows.

            [![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/mbo_utilities/)

            Preview raw TIFFs, TIFF stacks, ScanImage files, or numpy memmaps. Load a directory of raw ScanImage files to run the data-preview widget, which allows visualization of projections, mean-subtraction, and preview scan-phase correction.

            [Docs Overview](https://millerbrainobservatory.github.io/mbo_utilities/) |
            [Assembly Guide](https://millerbrainobservatory.github.io/mbo_utilities/assembly.html) |
            [Function Examples](https://millerbrainobservatory.github.io/mbo_utilities/api/usage.html)
            """)

            imgui.dummy(hello_imgui.em_to_vec2(0, 5))

            # centre prompt ------------------------------------------
            txt = "Select a file, multiple files, or a folder to preview:"
            imgui.set_cursor_pos_x((imgui.get_window_width() - imgui.calc_text_size(txt).x) * 0.5)
            imgui.text_colored(imgui.ImVec4(1, 0.85, 0.3, 1), txt)
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.5))

            # centred file buttons -----------------------------------
            bsz = hello_imgui.em_to_vec2(18, 2.4)
            gap = hello_imgui.em_size(1.2)
            tot = bsz.x * 2 + gap
            imgui.set_cursor_pos_x((imgui.get_window_width() - tot) * 0.5)

            if imgui.button("Open File(s)", bsz):
                self._open_multi = pfd.open_file(
                    "Select files", options=pfd.opt.multiselect
                )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Open one or more TIFF / NPY files.")

            imgui.same_line(spacing=gap)
            if imgui.button("Select Folder", bsz):
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
            _, self.save_phase_png = imgui.checkbox(
                "Save Phase PNG", getattr(self, "save_phase_png", False)
            )
            set_tooltip(
                "If enabled, the scan-phase will be corrected and a PNG of the phase will be saved in the same directory as the data file."
            )
            imgui.end_group()

            # OS-dialog results --------------------------------------
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
