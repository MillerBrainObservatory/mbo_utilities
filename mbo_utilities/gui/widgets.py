import sys

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

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

import matplotlib
# Important: before importing pyplot, set the renderer to Tk,
# so that the figure is not displayed on the screen before we can capture it.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imgui_bundle import immapp, imgui, imgui_fig, imgui_ctx
import numpy as np
from numpy.typing import NDArray


class AnimatedFigure:
    """A class that encapsulates a Matplotlib figure, and provides a method to animate it."""
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    amplitude: float = 1.0
    plotted_curve: matplotlib.lines.Line2D
    phase: float
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes

    def __init__(self):
        # Data for plotting
        self.phase = 0.0
        self.x = np.arange(0.0, 2.0, 0.01)
        self.y = 1 + np.sin(2 * np.pi * self.x + self.phase) * self.amplitude

        # Create a figure and a set of subplots
        self.fig, self.ax = plt.subplots()

        # Plot the data
        self.plotted_curve, = self.ax.plot(self.x, self.y)

        # Add labels and title
        self.ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='Simple Plot: Voltage vs. Time')

        # Add a grid
        self.ax.grid()

    def animate(self):
        self.phase += 0.1
        self.y = 1 + np.sin(2 * np.pi * self.x + self.phase) * self.amplitude
        self.plotted_curve.set_ydata(self.y)


def main():
    animated_figure = AnimatedFigure()

    # Create a static figure
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = np.sin(x) * np.exp(-x ** 2 / 20)
    static_fig, static_ax = plt.subplots()
    static_ax.plot(x, y)

    def gui():
        # Show an animated figure
        with imgui_ctx.begin_group():
            animated_figure.animate()
            imgui_fig.fig("Animated figure", animated_figure.fig, refresh_image=True, show_options_button=False)
            imgui.set_next_item_width(immapp.em_size(20))
            _, animated_figure.amplitude = imgui.slider_float("amplitude", animated_figure.amplitude, 0.1, 2.0)

        imgui.same_line()

        # Show a static figure
        imgui_fig.fig("Temp", static_fig)

    runner_params = immapp.RunnerParams()
    runner_params.fps_idling.fps_idle = 0  # disable idling so things are speedy
    runner_params.app_window_params.window_geometry.size = (1400, 600)
    runner_params.app_window_params.window_title = "LBM Main Window"
    runner_params.callbacks.show_gui = gui
    immapp.run(runner_params)


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


def update_colocalization(shift_x=None, shift_y=None, image_a=None, image_b=None):
    from scipy.ndimage import shift
    image_b_shifted = shift(image_b, shift=(shift_y, shift_x), mode='nearest')
    shape = image_a.shape

    # Normalize intensity to ensure even scaling
    image_a = image_a / np.max(image_a)
    image_b_shifted = image_b_shifted / np.max(image_b_shifted)

    colocalization = np.zeros((*shape, 3))
    colocalization[..., 1] = image_a  # Green channel
    colocalization[..., 0] = image_b_shifted  # Red channel

    # Boost overlap only where both channels are significant (> 0.3 threshold to remove weak static)
    mask = (image_a > 0.3) & (image_b_shifted > 0.3)
    colocalization[..., 2] = np.where(mask, np.minimum(image_a, image_b_shifted), 0)

    return colocalization


def plot_colocalization_hist(max_proj1, max_proj2_shifted, bins=100):
    # Flatten the images
    x = max_proj1.flatten()
    y = max_proj2_shifted.flatten()

    # Plot 2D histogram
    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=bins, cmap='inferno', density=True)
    plt.colorbar(label='Density')
    plt.xlabel('Max Projection 1 (Green)')
    plt.ylabel('Max Projection 2 (Red)')
    plt.title('2D Histogram of Colocalization')
    plt.show()


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
