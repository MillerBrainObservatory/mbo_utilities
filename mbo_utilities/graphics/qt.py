import os
import sys
from pathlib import Path
import fastplotlib as fpl

from ..file_io import to_lazy_array

try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
    from PyQt5 import QtGui, QtCore
except ImportError:
    raise ImportError(f"Failed to import Qt from {Path(__file__).name}."
                      f" Please install Qt from https://pypi.org/project/qt")


def load_dialog_folder(directory=None):
    if directory is None:
        directory = str(Path.home())
    path = QFileDialog.getExistingDirectory(parent=None, caption="Open folder with raw data OR assembled z-planes", directory=directory)
    return to_lazy_array(path)

def render_qt_widget(data=None):
    app = QApplication(sys.argv)

    if data is None:
        data = load_dialog_folder(directory=None)

    main_window = LBMMainWindow()
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=True,
    )
    # start the widget playing in a loop
    iw._image_widget_sliders._loop = True  # noqa
    qwidget = iw.show()  # need to display before playing

    main_window.setCentralWidget(qwidget)  # noqa
    main_window.resize(data.shape[-1], data.shape[-2])
    main_window.show()
    app.exec_()
    # return app


class LBMMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("MBO Widget")
        icon_path = str(Path.home() / ".lbm" / "icons" / "icon_suite2p_python.svg")
        app_icon = QtGui.QIcon()
        for size in (16, 24, 32, 48, 64, 256):
            app_icon.addFile(icon_path, QtCore.QSize(size, size))
        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.resize(1000, 800)
