import os
import sys
from pathlib import Path
import fastplotlib as fpl
from icecream import ic

from ..file_io import to_lazy_array

try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
    from PyQt5 import QtGui, QtCore
except ImportError:
    raise ImportError(
        f"Failed to import Qt from {Path(__file__).name}."
        f" Please install Qt from https://pypi.org/project/qt"
    )


def load_dialog_folder(directory=None):
    if directory is None:
        directory = str(Path.home())
    path = QFileDialog.getExistingDirectory(
        parent=None,
        caption="Open folder with raw data OR assembled z-planes",
        directory=directory,
    )
    return to_lazy_array(path)


def render_qt_widget(data=None):
    if data is None:
        data = load_dialog_folder(directory=None)
        ic(data)

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
    fpl.loop.run()
    # app.exec_()

import glfw
import os
from pathlib import Path
from OpenGL.GL import *

def create_glfw_window(width=1500, height=800, title="MBO Widget"):
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    return window