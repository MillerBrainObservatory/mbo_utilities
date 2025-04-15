import sys
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import fastplotlib as fpl
import dask.array as da
from mbo_utilities import get_files, zstack_from_files, read_scan, is_raw_scanimage
from mbo_utilities.file_io import ScanMultiROIReordered
from mbo_utilities.util import is_running_jupyter

try:
    from qtpy.QtWidgets import QMainWindow, QFileDialog, QApplication
    from qtpy import QtGui, QtCore
    HAS_QT = True
except ImportError:
    HAS_QT = False

try:
    from imgui_bundle import imgui_fig, imgui_ctx, imgui, icons_fontawesome_6 as fa
    from fastplotlib.ui import EdgeWindow
    HAS_IMGUI = True
except ImportError:
    HAS_IMGUI = False


def load_dialog_folder(parent=None, directory=None):
    if directory is None:
        directory = str(Path.home())
    path = QFileDialog.getExistingDirectory(parent=parent, caption="Open folder with raw data OR assembled z-planes", directory=directory)
    return load_data_path(path)

def run_gui(data_in: None | str | Path | ScanMultiROIReordered | np.ndarray = None) -> None | fpl.ImageWidget:
    if is_running_jupyter():
        if data_in is None:
            raise ValueError("Running in Jupyter: please provide a data path, file dialogs are not supported.")
        if isinstance(data_in, ScanMultiROIReordered):
            data = data_in
        else:
            data = load_data_path(data_in)
        image_widget = fpl.ImageWidget(data=data, histogram_widget=True, graphic_kwargs={"vmin": -350, "vmax": 13000})
        image_widget.show()
        return image_widget
    else:
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        if data_in is None:
            data = load_dialog_folder(parent=None, directory=None)
        else:
            data = load_data_path(data_in)
        main_window = LBMMainWindow(data)
        main_window.show()
        main_window.resize(1000, 800)
        app.exec()

def load_data_path(path: str | Path):
    save_folder = Path(path)
    if not save_folder.exists():
        print("Folder does not exist")
        return
    plane_folders = get_files(save_folder, ".tif", 2)
    if plane_folders:
        if is_raw_scanimage(plane_folders[0]):
            return read_scan(plane_folders)
        else:
            return zstack_from_files(plane_folders)
    print("No processed z-plane folders in folder")
    return


class AnimatedFigure:
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    amplitude: float = 1.0
    plotted_curve: matplotlib.lines.Line2D
    phase: float
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes

    def __init__(self):
        self.phase = 0.0
        self.x = np.arange(0.0, 2.0, 0.01)
        self.y = 1 + np.sin(2 * np.pi * self.x + self.phase) * self.amplitude
        self.fig, self.ax = plt.subplots()
        self.plotted_curve, = self.ax.plot(self.x, self.y)
        self.ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='Simple Plot: Voltage vs. Time')
        self.ax.grid()

    def animate(self):
        self.phase += 0.1
        self.y = 1 + np.sin(2 * np.pi * self.x + self.phase) * self.amplitude
        self.plotted_curve.set_ydata(self.y)


def main():
    animated_figure = AnimatedFigure()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = np.sin(x) * np.exp(-x ** 2 / 20)
    static_fig, static_ax = plt.subplots()
    static_ax.plot(x, y)

    def gui():
        with imgui_ctx.begin_group():
            animated_figure.animate()
            imgui_fig.fig("Animated figure", animated_figure.fig, refresh_image=True, show_options_button=False)
            imgui.set_next_item_width(immapp.em_size(20))
            _, animated_figure.amplitude = imgui.slider_float("amplitude", animated_figure.amplitude, 0.1, 2.0)
        imgui.same_line()
        imgui_fig.fig("Temp", static_fig)

    runner_params = immapp.RunnerParams()
    runner_params.fps_idling.fps_idle = 0
    runner_params.app_window_params.window_geometry.size = (1400, 600)
    runner_params.app_window_params.window_title = "LBM Main Window"
    runner_params.callbacks.show_gui = gui
    immapp.run(runner_params)


class LBMMainWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("LBM-CaImAn-Python Pipeline")
        icon_path = str(Path.home() / ".lbm" / "icons" / "icon_caiman_python.svg")
        app_icon = QtGui.QIcon()
        for size in (16, 24, 32, 48, 64, 256):
            app_icon.addFile(icon_path, QtCore.QSize(size, size))
        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.image_widget = fpl.ImageWidget(data=data, histogram_widget=False, graphic_kwargs={"vmin": -350, "vmax": 13000})
        self.resize(1000, 800)
        self.setCentralWidget(self.image_widget.show())


class SummaryDataWidget(EdgeWindow):
    def __init__(self, image_widget, size, location):
        flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        super().__init__(figure=image_widget.figure, size=size, location=location, title="Preview Data", window_flags=flags)
        self.image_widget = image_widget

    def update(self):
        something_changed = False
        if imgui.button("Mean Z"):
            self.mean_z()
        if imgui.is_item_hovered(0):
            imgui.set_tooltip("Calculate the mean image for each z-plane")
        imgui.new_line()
        imgui.separator()
        imgui.text("Statistics")
        if imgui.button("Calculate Noise"):
            self.calculate_noise()
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
    raise FileNotFoundError(f"Path '{data_path}' is not a directory.")


def update_colocalization(shift_x=None, shift_y=None, image_a=None, image_b=None):
    from scipy.ndimage import shift
    image_b_shifted = shift(image_b, shift=(shift_y, shift_x), mode='nearest')
    image_a = image_a / np.max(image_a)
    image_b_shifted = image_b_shifted / np.max(image_b_shifted)
    shape = image_a.shape
    colocalization = np.zeros((*shape, 3))
    colocalization[..., 1] = image_a
    colocalization[..., 0] = image_b_shifted
    mask = (image_a > 0.3) & (image_b_shifted > 0.3)
    colocalization[..., 2] = np.where(mask, np.minimum(image_a, image_b_shifted), 0)
    return colocalization


def plot_colocalization_hist(max_proj1, max_proj2_shifted, bins=100):
    x = max_proj1.flatten()
    y = max_proj2_shifted.flatten()
    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=bins, cmap='inferno', density=True)
    plt.colorbar(label='Density')
    plt.xlabel('Max Projection 1 (Green)')
    plt.ylabel('Max Projection 2 (Red)')
    plt.title('2D Histogram of Colocalization')
    plt.show()


def run_gui(data_in: None | str | Path | ScanMultiROIReordered | np.ndarray = None) -> None | fpl.ImageWidget:
    if is_running_jupyter():
        if data_in is None:
            raise ValueError("Running in Jupyter: please provide a data path, file dialogs are not supported.")
        data = load_data_path(data_in)

        image_widget = fpl.ImageWidget(data=data, histogram_widget=True, graphic_kwargs={"vmin": -350, "vmax": 13000})
        image_widget.show()
        return image_widget
    else:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        data = load_dialog_folder(parent=None, directory=None) if data_in is None else load_data_path(data_in)
        main_window = LBMMainWindow(data)
        main_window.show()
        main_window.resize(1000, 800)
        app.exec()


if __name__ == "__main__":
    run_gui()
