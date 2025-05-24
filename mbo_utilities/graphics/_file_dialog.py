import shutil
from pathlib import Path
from typing import List

from icecream import ic
from imgui_bundle import (
    imgui,
    imgui_md,
    hello_imgui,
)
from imgui_bundle import portable_file_dialogs as pfd
import os


class FileDialog:
    def __init__(self):
        self.open_file_dialog = None
        self.open_file_multiselect = None
        self.save_file_dialog = None
        self.select_folder_dialog = None
        self.last_file_selection = ""
        self.selected_path = None
        self.last_file_selection = ""
        self.icon_type = pfd.icon.info
        self.message_dialog = None
        self.message_choice_type = pfd.choice.ok

    def render(self):
        imgui.push_id("pfd")

        imgui_md.render_unindented("""
        # Miller Brain Observatory Utilities
        [Miller Brain Observatory](mbo.rockefeller.edu) 
        [mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities/tree/subpixel-phasecorr) 
        [docs](https://millerbrainobservatory.github.io/mbo_utilities/) 
        """)

        imgui.text("      ---   File dialogs   ---")
        if imgui.button("Open file"):
            self.open_file_dialog = pfd.open_file("Select file")
        if self.open_file_dialog and self.open_file_dialog.ready():
            self.selected_path = self.open_file_dialog.result()
            self.open_file_dialog = None
            # continue other processing here

        imgui.same_line()
        if imgui.button("Open file (multiselect)"):
            self.open_file_multiselect = pfd.open_file(
                "Select file", options=pfd.opt.multiselect
            )
        if self.open_file_multiselect and self.open_file_multiselect.ready():
            self.selected_path = self.open_file_multiselect.result()
            self.open_file_multiselect = None

        imgui.same_line()
        if imgui.button("Select folder"):
            self.select_folder_dialog = pfd.select_folder("Select folder")
        if self.select_folder_dialog and self.select_folder_dialog.ready():
            self.selected_path = self.select_folder_dialog.result()
            self.select_folder_dialog = None

        if self.selected_path:
            imgui.text(self.last_file_selection)
            hello_imgui.get_runner_params().app_shall_exit = True

        imgui.pop_id()


fd = FileDialog()


def render_file_dialog():
    global fd
    fd.render()


if __name__ == "__main__":

    def setup_imgui():
        from mbo_utilities import get_mbo_project_root, mbo_paths

        # Assets
        project_assets: Path = get_mbo_project_root().joinpath("assets")

        if not project_assets.is_dir():
            ic("Assets folder not found.")
            return

        imgui_path = mbo_paths["base"].joinpath("imgui")
        imgui_path.mkdir(exist_ok=True)

        assets_path = imgui_path.joinpath("assets")
        assets_path.mkdir(exist_ok=True)

        shutil.copytree(project_assets, assets_path, dirs_exist_ok=True)
        hello_imgui.set_assets_folder(str(project_assets))

    setup_imgui()
    from imgui_bundle.demos_python import demo_utils

    demo_utils.set_hello_imgui_demo_assets_folder()

    from imgui_bundle import immapp
    from imgui_bundle import hello_imgui

    imgui_ini_path = os.path.expanduser("~/.mbo/imgui.ini")
    os.makedirs(os.path.dirname(imgui_ini_path), exist_ok=True)

    runner_params = hello_imgui.RunnerParams()
    loc = hello_imgui.ini_settings_location(hello_imgui.IniFolderType.home_folder)
    runner_params.ini_filename = imgui_ini_path
    runner_params.callbacks.show_gui = render_file_dialog

    hello_imgui.run(runner_params)

    immapp.run(render_file_dialog, with_markdown=True, window_size=(500, 500))  # type: ignore
    if fd.selected_path:
        print(fd.selected_path)
