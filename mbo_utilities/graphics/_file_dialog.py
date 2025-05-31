from imgui_bundle import (
    imgui,
    imgui_md,
    hello_imgui,
    imgui_ctx,
)
from imgui_bundle import portable_file_dialogs as pfd


class FileDialog:
    def __init__(self):
        self.selected_path = None
        self._open_multi = None
        self._select_folder = None

    def render(self):
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
                self._open_multi = pfd.open_file("Select files", options=pfd.opt.multiselect)
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
            if imgui.button("Quit", hello_imgui.em_to_vec2(20, 1.5)) or imgui.is_key_pressed(imgui.Key.escape):
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

            imgui.pop_id()

