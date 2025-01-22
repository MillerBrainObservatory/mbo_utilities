import webbrowser
import sys

from pathlib import Path
from qtpy.QtWidgets import QMainWindow, QFileDialog, QApplication
from qtpy import QtGui, QtCore
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

from mbo_utilities import get_files_ext, stack_from_files

try:
    from imgui_bundle import imgui, icons_fontawesome_6 as fa
except ImportError:
    raise ImportError("Please install imgui via `conda install -c conda-forge imgui-bundle`")


def load_dialog_folder():
    dlg_kwargs = {
        "parent": None,
        "caption": "Open folder with z-planes",
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
        return stack_from_files(plane_folders)
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
        self.menu_widget = MenuWidget(self.image_widget, size=50)
        self.summary_widget = SummaryDataWidget(self.image_widget, size=50)

        # gui = MenuWidget(self.image_widget, size=50)
        gui = SummaryDataWidget(self.image_widget, size=50)

        self.image_widget.figure.add_gui(gui)
        qwidget = self.image_widget.show()
        self.setCentralWidget(qwidget)
        self.resize(self.image_widget.data[0].shape[-2], self.image_widget.data[0].shape[-1])
        print('Done!')


class SummaryDataWidget(EdgeWindow):
    def __init__(self, image_widget, size):
        flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        super().__init__(figure=image_widget.figure, size=size, location="right", title="Preview Data",
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
        mean_z_widget = fpl.ImageWidget(data=self.image_widget.data[0].mean(axis=1))
        mean_z_widget.show()

    def calculate_noise(self):
        pass
        # current_z = self.image_widget.current_index["z"]
        # lazy_arr = self.image_widget.data[0]
        # arr_z = lazy_arr[:, current_z, ...].T
        # sn, _ = get_noise_fft(arr_z.compute())
        # # setattr(self.image_widget.figure[0, 0], f"snr_{i}", sn)
        # self.snr = sn
        # self.image_widget.figure[0, 0].add_text(f"SNR: {sn:.2f}")


class MenuWidget(EdgeWindow):
    def __init__(self, image_widget, size):
        flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        super().__init__(
            figure=image_widget.figure,
            size=size,
            location="top",
            title="Toolbar",
            window_flags=flags,
        )
        self.image_widget = image_widget

    def update(self):

        if imgui.button("Documentation"):
            webbrowser.open(
                "https://millerbrainobservatory.github.io/LBM-CaImAn-Python/"
            )

        imgui.same_line()

        imgui.push_font(self._fa_icons)
        if imgui.button(label=fa.ICON_FA_FOLDER_OPEN):
            print("Opening file dialog")
            load_dialog_folder(self.image_widget)

        imgui.pop_font()
        if imgui.is_item_hovered(0):
            imgui.set_tooltip("Open a file dialog to load data")

        imgui.same_line()


def run():
    app = QApplication(sys.argv)
    data = load_dialog_folder()
    main_window = LBMMainWindow(data)
    main_window.show()
    app.exec()
    # fpl.loop.run()


if __name__ == "__main__":
    run()
    # fpl.loop.run()
