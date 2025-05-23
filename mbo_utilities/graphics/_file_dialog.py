from typing import List
from imgui_bundle import (
    imgui,
    imgui_md,
    immapp,
)
from imgui_bundle import portable_file_dialogs as pfd

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
            self.open_file_multiselect = pfd.open_file("Select file", options=pfd.opt.multiselect)
        if self.open_file_multiselect and self.open_file_multiselect.ready():
            self.selected_path = self.open_file_multiselect.result()
            self.open_file_multiselect = None

        imgui.same_line()
        if imgui.button("Select folder"):
            self.select_folder_dialog = pfd.select_folder("Select folder")
        if self.select_folder_dialog and self.select_folder_dialog.ready():
            self.selected_path=self.select_folder_dialog.result()
            self.select_folder_dialog = None

        if self.last_file_selection:
            imgui.text(self.last_file_selection)

        # imgui.same_line()
        # if imgui.button("Add message"):
        #     self.message_dialog = pfd.message("Message title", "This is an example message", self.message_choice_type, self.icon_type)
        #
        # if self.message_dialog and self.message_dialog.ready():
        #     print("msg ready: " + str(self.message_dialog.result()))
        #     self.message_dialog = None

        imgui.same_line()
        for choice in (
            pfd.choice.ok, pfd.choice.yes_no, pfd.choice.yes_no_cancel,
            pfd.choice.retry_cancel, pfd.choice.abort_retry_ignore
        ):
            if imgui.radio_button(choice.name, self.message_choice_type == choice):
                self.message_choice_type = choice
            imgui.same_line()

        imgui.new_line()
        imgui.pop_id()


fd = FileDialog()

def demo_gui():
    fd.render()

if __name__ == "__main__":
    from imgui_bundle.demos_python import demo_utils
    demo_utils.set_hello_imgui_demo_assets_folder()
    from imgui_bundle import immapp

    immapp.run(demo_gui, with_markdown=True, window_size=(1000, 1000))  # type: ignore
    if fd.selected_path:
        print(fd.selected_path)
        print(fd.selected_path)
        print(fd.selected_path)
