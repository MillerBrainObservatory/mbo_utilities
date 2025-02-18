import sys

from pathlib import Path
import numpy as np

from qtpy.QtWidgets import QMainWindow, QFileDialog, QApplication
from qtpy import QtGui, QtCore
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
import dask.array as da
from tqdm import tqdm

from mbo_utilities import get_files, stack_from_files, read_scan, is_raw_scanimage
from mbo_utilities.file_io import ScanMultiROIReordered
from mbo_utilities.util import is_running_jupyter

try:
    from imgui_bundle import imgui, icons_fontawesome_6 as fa
except ImportError:
    raise ImportError("Please install imgui via `conda install -c conda-forge imgui-bundle`")




def load_dialog_folder(parent=None, directory=None):
    if directory is None:
        directory = str(Path().home())
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open folder with raw data OR with assembled z-planes (none session per folder).",
        "directory": directory,
    }
    path = QFileDialog.getExistingDirectory(**dlg_kwargs)
    return load_data_path(path)


def load_data_path(path: str | Path):
    """
    Returns a scan or volume assembled from z-plane files.
    The output is fed directly into a fastplotlib ImageWidget.
    """
    save_folder = Path(path)
    if not save_folder.exists():
        print("Folder does not exist")
        return
    plane_folders = get_files(save_folder, ".tif", 2)
    if plane_folders:
        if is_raw_scanimage(plane_folders[0]):
            # load raw scanimage tiffs
            return read_scan(plane_folders, join_contiguous=True)
        else:
            # load assembled z-planes
            return stack_from_files(plane_folders)
    else:
        print("No processed z-plane folders in folder")
        return


class LBMMainWindow(QMainWindow):
    def __init__(self, data):
        super(LBMMainWindow, self).__init__()

        print('Setting up main window')
        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("LBM-CaImAn-Python Pipeline")

        app_icon = QtGui.QIcon()
        icon_path = str(Path().home() / ".lbm" / "icons" / "icon_caiman_python.svg")
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")
        self.window_funcs = {}
        self.image_widget = fpl.ImageWidget(
            data=data,
            histogram_widget=False,
            graphic_kwargs={"vmin": -350, "vmax": 13000},
        )
        # if self.image_widget.figure.canvas.__class__.__name__ == "QRenderCanvas":
        self.resize(1000, 800)
        qwidget = self.image_widget.show()
        self.setCentralWidget(qwidget)


class SummaryDataWidget(EdgeWindow):
    def __init__(self, image_widget, size, location):
        flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        super().__init__(figure=image_widget.figure, size=size, location=location, title="Preview Data",
                         window_flags=flags)
        self.image_widget = image_widget

    def update(self):

        something_changed = False

        # int entries for gaussian filter order
        if imgui.button("Mean Z"):
            self.mean_z()
        if imgui.is_item_hovered(0):
            imgui.set_tooltip("Calculate the mean image for each z-plane")

        # calculate stats and display in a text widget
        imgui.new_line()
        imgui.separator()
        imgui.text("Statistics")
        if imgui.button("Calculate Noise"):
            self.calculate_noise()
            # display loading bar
            imgui.text("Calculating noise...")
        imgui.new_line()
        imgui.separator()

        if something_changed:
            print("something changed")

    def mean_z(self):
        data = self.image_widget.data[0]
        lazy_arr = da.empty_like(data)
        for i in tqdm(range(data.shape[1]), desc="Calculating Mean Image for each z"):
            lazy_arr[:, i, ...] = data[:, i, ...].mean(axis=1)

        print("Showing mean z")
        mean_z_widget = fpl.ImageWidget(data=lazy_arr, histogram_widget=False, figure_kwargs={"size": (1000, 800)})
        mean_z_widget.show()


def parse_data_path(fpath):
    data_path = Path(fpath).expanduser().resolve()
    print(f"Reading data from '{data_path}'")
    if not data_path.exists():
        raise FileNotFoundError(f"Path '{data_path}' does not exist as a file or directory.")
    if data_path.is_dir():
        return load_data_path(data_path)
    else:
        raise FileNotFoundError(f"Path '{data_path}' is not a directory.")


def run_gui(data_in: None | str | Path | ScanMultiROIReordered | np.ndarray = None) -> None | fpl.ImageWidget:
    # parse data
    if data_in is None:
        print("No data provided, opening file dialog")
        data = load_dialog_folder(parent=None, directory=None)
    elif isinstance(data_in, ScanMultiROIReordered):
        # if data is a ScanMultiROIReordered object, use it directly
        print("Using ScanMultiROIReordered object")
        data = data_in
    elif isinstance(data_in, (str, Path)):
        # if data is a string or Path, load it from the path
        print("Loading data from path")
        data = load_data_path(data_in)
    else:
        raise TypeError(f"Unsupported data type: {type(data_in)}")

    if is_running_jupyter():
        print("Running in Jupyter")
        # if running in jupyter, return the image widget to show in the notebook
        print("Creating image widget")
        image_widget = fpl.ImageWidget(
            data=data,
            histogram_widget=False,
            graphic_kwargs={"vmin": -350, "vmax": 13000},
        )
        print("Running in Jupyter, returning image widget")
        return image_widget
    else:
        # if running in a standalone script, set up the main window
        app = QApplication(sys.argv)
        main_window = LBMMainWindow(data)
        main_window.show()
        main_window.resize(1000, 800)
        app.exec()


if __name__ == "__main__":
    run_gui()
