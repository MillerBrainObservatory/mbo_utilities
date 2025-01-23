import webbrowser
import sys

from pathlib import Path
from qtpy.QtWidgets import QMainWindow, QFileDialog, QApplication
from qtpy import QtGui, QtCore
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
import dask.array as da
from tqdm import tqdm

from mbo_utilities import get_files_ext, stack_from_files, read_scan

try:
    from imgui_bundle import imgui, icons_fontawesome_6 as fa
except ImportError:
    raise ImportError("Please install imgui via `conda install -c conda-forge imgui-bundle`")


def load_dialog_folder(parent=None):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open folder with raw data or with z-planes.",
        "directory": str(Path().home()),
    }
    path = QFileDialog.getExistingDirectory(**dlg_kwargs)
    return load_folder(path)


def load_folder(path: str | Path):
    save_folder = Path(path)
    if not save_folder.exists():
        print("Folder does not exist")
        return
    plane_folders = get_files_ext(save_folder, ".tif", 2)
    if plane_folders:
        try:
            return stack_from_files(plane_folders)
        except Exception as e:
            return read_scan(plane_folders, join_contiguous=True)
    else:
        print("No processed planeX folders in folder")
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
        )
        qwidget = self.image_widget.show()
        self.setCentralWidget(qwidget)
        self.resize(self.image_widget.data[0].shape[-2], self.image_widget.data[0].shape[-1])


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


def run_gui():
    app = QApplication(sys.argv)
    data = load_dialog_folder()
    main_window = LBMMainWindow(data)
    main_window.show()
    app.exec()


if __name__ == "__main__":
    run_gui()
