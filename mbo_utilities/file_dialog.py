try:
    from imgui_bundle import immapp, imgui, portable_file_dialogs as pfd
except ImportError:
    print("imgui-bundle is not installed. Install with `pip install imgui-bundle`.")
import icecream as ic

_selected_result = None


@immapp.static(
    open_file=None, open_file_multi=None, select_folder=None, last_selection=""
)
def file_dialog_base():
    static = file_dialog_base
    global _selected_result

    def handle_result(result):
        global _selected_result
        _selected_result = result
        close = True
        return _selected_result

    if imgui.button("Open File"):
        static.open_file = pfd.open_file("Choose file")
    if static.open_file and static.open_file.ready():
        handle_result(static.open_file.result())
        static.open_file = None

    imgui.same_line()
    if imgui.button("Open Multiple Files"):
        static.open_file_multi = pfd.open_file(
            "Choose files", options=pfd.opt.multiselect
        )
    if static.open_file_multi and static.open_file_multi.ready():
        handle_result(static.open_file_multi.result())
        static.open_file_multi = None

    imgui.same_line()
    if imgui.button("Select Folder"):
        static.select_folder = pfd.select_folder("Choose folder")
    if static.select_folder and static.select_folder.ready():
        handle_result(static.select_folder.result())
        static.select_folder = None

    if _selected_result:
        static.last_selection = (
            "\n".join(_selected_result)
            if isinstance(_selected_result, list)
            else _selected_result
        )
        imgui.separator()
        imgui.text_wrapped("Selected:\n" + static.last_selection)


def open_file_or_folder_dialog():
    close = False
    global _selected_result
    global close
    _selected_result = None
    while not close:
        immapp.run(file_dialog_base, "Select File(s) or Folder")
    return _selected_result
