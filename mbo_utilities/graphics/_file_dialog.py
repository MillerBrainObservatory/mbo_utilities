from imgui_bundle import (
    imgui,
    imgui_md,
    hello_imgui,
)
from imgui_bundle import portable_file_dialogs as pfd


class FileDialog:
    def __init__(self):
        self.selected_path = None
        self._open_multi = None
        self._select_folder = None

    def render(self):
        imgui.push_id("pfd")

        imgui_md.render_unindented("""
        # Miller Brain Observatory Utilities
        [Miller Brain Observatory](mbo.rockefeller.edu) 
        [mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities/tree/subpixel-phasecorr) 
        [docs](https://millerbrainobservatory.github.io/mbo_utilities/) 
        """)
        if imgui.button("Open file (multiselect)"):
            self._open_multi = pfd.open_file("Select files", options=pfd.opt.multiselect)
        if self._open_multi and self._open_multi.ready():
            self.selected_path = self._open_multi.result()
            self._open_multi = None
            hello_imgui.get_runner_params().app_shall_exit = True

        imgui.same_line()
        if imgui.button("Select folder"):
            self._select_folder = pfd.select_folder("Select folder")
        if self._select_folder and self._select_folder.ready():
            self.selected_path = self._select_folder.result()
            self._select_folder = None
            hello_imgui.get_runner_params().app_shall_exit = True

        imgui.pop_id()