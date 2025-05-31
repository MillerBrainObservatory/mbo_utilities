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
            imgui.set_cursor_pos_x(
                (imgui.get_window_width() - imgui.calc_text_size(txt).x) * 0.5
            )
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
            _, self.threading_enabled = imgui.checkbox(
                "Enable Threading", self.threading_enabled
            )
            _, self.fix_phase = imgui.checkbox(
                "Fix Scan-Phase", getattr(self, "fix_phase", True)
            )
            _, self.save_phase_png = imgui.checkbox(
                "Save Phase PNG", getattr(self, "save_phase_png", False)
            )

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

    def render2(self):
        with imgui_ctx.begin_child("#fd"):
            imgui.push_id("pfd")

            # Add top padding
            imgui.dummy(hello_imgui.em_to_vec2(1.5, 1.5))

            # Main Header
            imgui_md.render_unindented("""
            # MBO Utilities

            General Python and shell utilities developed for the Miller Brain Observatory (MBO) workflows.

            [![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/mbo_utilities/)

            Preview raw TIFFs, TIFF stacks, ScanImage files, or numpy memmaps. Load a directory of raw ScanImage files to run the data-preview widget, which allows visualization of projections, mean-subtraction, and preview scan-phase correction.

            [Docs Overview](https://millerbrainobservatory.github.io/mbo_utilities/) |
            [Assembly Guide](https://millerbrainobservatory.github.io/mbo_utilities/assembly.html) |
            [Function Examples](https://millerbrainobservatory.github.io/mbo_utilities/api/usage.html)
            """)

            imgui.separator()
            imgui.new_line()

            # Centered descriptive instruction
            desc = "Select a file, multiple files, or a folder to preview:"
            window_width = imgui.get_window_width()
            desc_width = imgui.calc_text_size(desc).x
            imgui.set_cursor_pos_x((window_width - desc_width) * 0.5)
            imgui.text_colored(imgui.ImVec4(1.0, 0.85, 0.3, 1.0), desc)

            imgui.new_line()

            button_size = hello_imgui.em_to_vec2(20, 2.5)
            spacing = hello_imgui.em_size(2)

            total_w = button_size.x * 2 + spacing
            start_x = (imgui.get_window_width() - total_w) * 0.5
            imgui.set_cursor_pos_x(start_x)

            if imgui.button("Open File(s)", button_size):
                self._open_multi = pfd.open_file(
                    "Select files", options=pfd.opt.multiselect
                )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Open one or more TIFF/NPY files.")

            imgui.same_line(spacing=spacing)
            if imgui.button("Select Folder", button_size):
                self._select_folder = pfd.select_folder("Select folder")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Pick a directory of TIFFs.")

            imgui.new_line()
            cancel_x = (imgui.get_window_width() - button_size.x) * 0.5
            imgui.set_cursor_pos_x(cancel_x)
            if imgui.button(
                "Quit", hello_imgui.em_to_vec2(20, 1.5)
            ) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None

            # results
            if self._open_multi and self._open_multi.ready():
                res = self._open_multi.result()
                self.selected_path = res
                if res:
                    self._open_multi = None
                    hello_imgui.get_runner_params().app_shall_exit = True

            if self._select_folder and self._select_folder.ready():
                res = self._select_folder.result()
                self.selected_path = res
                if res:
                    self._select_folder = None
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._select_folder = None

            imgui.dummy(hello_imgui.em_to_vec2(20, 20))
            imgui.separator()
            # load options, threading, and debug info
            imgui.push_id("load_options")

            imgui.text_colored(imgui.ImVec4(1.0, 0.85, 0.3, 1.0), "Load Options")
            set_tooltip(
                "Run separate threads, disable for debugging or if you exprerience issues and file"
                "an issue at here: https://github.com/MillerBrainObservatory/mbo_utilities/issues."
            )
            _, self.threading_enabled = imgui.checkbox(
                "Enable Threading", self.threading_enabled
            )

            imgui.pop_id()
            imgui.pop_id()
