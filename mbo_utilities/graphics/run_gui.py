import click
import numpy as np

from ..util import is_imgui_installed
from .imgui import PreviewDataWidget
from ..file_io import (
    to_lazy_array,
    get_files,
    read_scan,
    _is_arraylike,
)

if is_imgui_installed():
    import fastplotlib as fpl
    from imgui_bundle import portable_file_dialogs as pfd, immapp, imgui


@immapp.static(file_result="", files_result="", folder_result="")
def select_input_popup():
    result = None
    if imgui.begin_popup("Select Input"):
        if imgui.menu_item("File", "Ctrl+O")[0]:
            result = pfd.open_file("Choose file").result()
            imgui.close_current_popup()
        if imgui.menu_item("Files", "Ctrl+Shift+F")[0]:
            result = pfd.open_file("Choose files", options=pfd.opt.multiselect).result()
            imgui.close_current_popup()
        if imgui.menu_item("Folder", "Ctrl+F")[0]:
            folder = pfd.select_folder("Choose folder").result()
            result = get_files(folder)
            imgui.close_current_popup()
        imgui.end_popup()
    return result

def gui_app():
    if imgui.button("Select…"):
        imgui.open_popup("Select Input")
    return select_input_popup() or []


@immapp.static(file_result="", files_result="", folder_result="")
def select_any_input_popup():
    result = None
    if imgui.begin_popup("Select Input Type"):
        clicked, _ = imgui.menu_item("Select File", "Ctrl+O", False, True)
        if clicked:
            static.file_result = pfd.open_file("Choose file").result()[0]
            result = static.file_result
        clicked, _ = imgui.menu_item("Select Multiple Files", "Ctrl+Shift+F", False, True)
        if clicked:
            static.files_result = pfd.open_file("Choose files", options=pfd.opt.multiselect).result()
            result = static.files_result
        clicked, _ = imgui.menu_item("Select Folder", "Ctrl+F", False, True)
        if clicked:
            static.folder_result = pfd.select_folder("Choose folder").result()
            result = get_files(static.folder_result, str_contains="", max_depth=0)
        imgui.end_popup()
    return result

@click.command()
@click.option('--roi', '-r', type=click.IntRange(1, 10), default=None)
@click.argument('data_in', required=False)
def run_gui(data_in=None, **kwargs):
    """Open a GUI to preview data of any supported type."""
    if data_in is None:
        fpath = str(gui_app())
        files = get_files(fpath)
        data = read_scan(files)
    elif _is_arraylike(data_in):
        data = data_in
    else:
        data = to_lazy_array(data_in)

    if isinstance(data, list):
        sample = data[0]
    else:
        sample = data

    if sample.ndim < 2:
        raise ValueError(f"Invalid input shape: expected >=2D, got {sample.shape}")

    nx, ny = sample.shape[-2:]
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={"size": (nx, ny)},
        graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
        window_funcs={"t": (np.mean, 0)},
    )

    if kwargs.get("gui"):
        gui = PreviewDataWidget(iw=iw)
        iw.figure.add_gui(gui)

    iw.show()
    fpl.loop.run()

