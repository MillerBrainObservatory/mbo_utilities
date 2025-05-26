import numpy as np
from mbo_utilities.file_io import get_files, read_scan, to_lazy_array, _is_arraylike
from mbo_utilities.graphics.imgui import PreviewDataWidget
from imgui_bundle import imgui, portable_file_dialogs as pfd, hello_imgui, immapp
import fastplotlib as fpl

import os


selection_store = {"result": None}

@immapp.static(file_result=None, files_result=None, folder_result=None)
def open_file_menu():
    static = open_file_menu
    global selection_store

    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):
            clicked, _ = imgui.menu_item("Open File...", "Ctrl+O", False, True)
            if clicked:
                static.file_result = pfd.open_file("Choose file").result()[0]
                selection_store["result"] = static.file_result
                hello_imgui.get_runner_params().app_shall_exit = True

            clicked, _ = imgui.menu_item("Open Multiple Files...", "Ctrl+Shift+O", False, True)
            if clicked:
                static.files_result = pfd.open_file("Choose files", options=pfd.opt.multiselect).result()
                selection_store["result"] = static.files_result
                hello_imgui.get_runner_params().app_shall_exit = True

            clicked, _ = imgui.menu_item("Open Folder...", "Ctrl+F", False, True)
            if clicked:
                static.folder_result = pfd.select_folder("Choose folder").result()
                selection_store["result"] = get_files(static.folder_result)
                hello_imgui.get_runner_params().app_shall_exit = True

            imgui.end_menu()
        imgui.end_main_menu_bar()

def gui_app():
    open_file_menu()


def make_image_widget(data, fpath=None, gui=False):
    sample = data[0] if isinstance(data, (list, tuple)) else data
    assert _is_arraylike(sample), f"Input should be array-like, not {type(sample)}"
    assert sample.ndim >= 2, f"Expected 2D sample, not: {sample.ndim} dims"

    nx, ny = sample.shape[-2:]
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={"size": (nx, ny)},
        graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
        window_funcs={"t": (np.mean, 1)},
    )
    if gui:
        ui = PreviewDataWidget(iw=iw, fpath=fpath)
        iw.figure.add_gui(ui)
    return iw


if __name__ == "__main__":
    from mbo_utilities.graphics._file_dialog import FileDialog

    file_dialog = FileDialog()
    def render_file_dialog():
        file_dialog.render()

    immapp.run(render_file_dialog, with_markdown=True)
    selected = file_dialog.selected_path
    # data = tifffile.memmap("/home/flynn/lbm_data/assembled_default/plane_11.tif")
    fpath = "D:/tests/data"
    data = read_scan(fpath)
    sample = data[0] if isinstance(data, (list, tuple)) else data
    assert _is_arraylike(sample), f"Input should be array-like, not {type(sample)}"
    assert sample.ndim >= 2, f"Expected 2D sample, not: {sample.ndim} dims"

    nx, ny = sample.shape[-2:]
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={"size": (nx*2, ny*2)},
        graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
        window_funcs={"t": (np.mean, 1)},
    )
    edge_gui = PreviewDataWidget(iw=iw, fpath=fpath)
    iw.figure.add_gui(edge_gui)
    iw.show()
    fpl.loop.run()