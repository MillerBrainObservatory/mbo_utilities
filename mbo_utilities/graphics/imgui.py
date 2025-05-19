from pathlib import Path
from typing import Literal

from dask import array as da
from scipy.ndimage import gaussian_filter
import numpy as np
import h5py
from tqdm import tqdm

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

from imgui_bundle import imgui, implot
from imgui_bundle import portable_file_dialogs as pfd

from .. import mbo_home
from ..util import norm_minmax, norm_percentile


def imgui_dynamic_table(
    table_id: str, data_lists: list, titles: list = None, selected_index: int = None
):
    """
    Draw a dynamic table using ImGui with customizable column titles and highlighted row selection.

    Parameters
    ----------
    table_id : str
        Unique identifier for the table.
    data_lists : list of lists
        A list of columns, where each inner list represents a column's data.
        All columns must have the same length.
    titles : list of str, optional
        Column titles corresponding to `data_lists`. If None, default names are assigned as "Column N".
    selected_index : int, optional
        Index of the row to highlight. Default is None (no highlighting).

    Raises
    ------
    ValueError
        If `data_lists` is empty or columns have inconsistent lengths.
        If `titles` is provided but does not match the number of columns.
    """

    if not data_lists or any(len(col) != len(data_lists[0]) for col in data_lists):
        raise ValueError(
            "data_lists columns must have consistent lengths and cannot be empty."
        )

    num_columns = len(data_lists)
    if titles is None:
        titles = [f"Column {i + 1}" for i in range(num_columns)]
    elif len(titles) != num_columns:
        raise ValueError(
            "Number of titles must match the number of columns in data_lists."
        )

    if imgui.begin_table(
        table_id,
        num_columns,
        flags=imgui.TableFlags_.borders | imgui.TableFlags_.resizable,
    ):  # noqa
        for title in titles:
            imgui.table_setup_column(title, imgui.TableColumnFlags_.width_stretch)  # noqa

        for title in titles:
            imgui.table_next_column()
            imgui.text(title)

        for i, row_values in enumerate(zip(*data_lists)):
            imgui.table_next_row()
            for value in row_values:
                imgui.table_next_column()
                if i == selected_index:
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0, 250, 35, 1))  # noqa

                imgui.set_cursor_pos_x(
                    imgui.get_cursor_pos_x()
                    + imgui.get_column_width()
                    - imgui.calc_text_size(f"{int(value)}").x
                    - imgui.get_style().item_spacing.x
                )
                imgui.text(f"{int(value)}")

                if i == selected_index:
                    imgui.pop_style_color()

        imgui.end_table()


def implot_pollen(pollen_offsets, offset_store, zstack):
    imgui.begin_child("Z-Stack Analysis")

    z_planes = np.arange(len(pollen_offsets)) + 1

    # Offset Comparison Plot
    if implot.begin_plot("Offset Comparison Across Z-Planes"):
        implot.plot_bars(
            "Calibration Offsets",
            z_planes,
            np.array([p[0] for p in pollen_offsets]),
            bar_size=0.4,
        )
        implot.plot_bars(  # noqa
            "Selected Offsets",
            z_planes,
            np.array([s[0] for s in offset_store]),
            bar_size=0.4,
            shift=0.2,
        )
        implot.end_plot()

    # Mean Intensity Plot
    if implot.begin_plot("Mean Intensity Across Z-Planes"):
        mean_intensities = np.array(
            [np.mean(zstack[z]) for z in range(zstack.shape[0])]
        )
        implot.plot_line("Mean Intensity", z_planes, mean_intensities)
        implot.end_plot()

    # Cumulative Offset Drift Plot
    if implot.begin_plot("Cumulative Offset Drift"):
        cumulative_shifts = np.cumsum(np.array([s[0] for s in offset_store]))
        implot.plot_line("Cumulative Shift", z_planes, cumulative_shifts)
        implot.end_plot()

    imgui.end_child()

def apply_phase_offset(frame: np.ndarray, offset: float) -> np.ndarray:
    from scipy.ndimage import fourier_shift
    result = frame.copy()
    rows = result[1::2, :]
    f = np.fft.fftn(rows)
    fshift = fourier_shift(f, (0, offset))
    result[1::2, :] = np.fft.ifftn(fshift).real
    return result

def compute_phase_offset(frame: np.ndarray, upsample: int = 10, exclude_center_px: int = 4) -> float:
    from skimage.registration import phase_cross_correlation
    if frame.ndim == 3:
        frame = np.mean(frame, axis=0)
    _, w = frame.shape
    cx = w // 2
    keep_left = slice(None, cx - exclude_center_px)
    keep_right = slice(cx + exclude_center_px, None)

    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    pre_crop = np.concatenate([pre[:m, keep_left], pre[:m, keep_right]], axis=1)
    post_crop = np.concatenate([post[:m, keep_left], post[:m, keep_right]], axis=1)

    shift, _, _ = phase_cross_correlation(pre_crop, post_crop, upsample_factor=upsample)
    return float(shift[1])


class PreviewDataWidget(EdgeWindow):
    def __init__(
        self,
        iw: fpl.ImageWidget,
        fpath: str | None = None,
        size: int=350,
        location: Literal["top", "bottom", "left", "right"]="right",
        title: str="Preview Data",
    ):
        super().__init__(figure=iw.figure, size=size, location=location, title=title)
        if fpath is None:
            self.fpath = Path(mbo_home)
        else:
            self.fpath = Path(fpath)

        self.h5name = None

        self.figure = iw.figure
        self.image_widget = iw
        self.shape = self.image_widget.data[0].shape

        self.nz = self.shape[0]

        self.offset_store = np.zeros(shape=self.nz)
        self._current_offset = 0

        self.proj = "mean"
        self.image_widget.add_event_handler(self.track_slider, "current_index")

    @property
    def current_offset(self):
        return self._current_offset

    @current_offset.setter
    def current_offset(self, value):
        self._current_offset = value
        self.apply_offset()

    def update(self):

        button_size = imgui.ImVec2(140, 20)
        if imgui.button("Calculate Offset", button_size):
            self.current_offset = self.calculate_offset()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Automatically calculates the best offset for the selected Z-plane."
            )

        # Here, I want a widget displaying the calculated offset with an "apply" button that does nothing for now
        # the offset is subpixel

    def calculate_offset(self):
        ind = self.image_widget.current_index["t"]
        frame = self.image_widget.data[0][ind].copy()
        return compute_phase_offset(frame, upsample=20)

    def apply_offset(self):
        ind = self.image_widget.current_index["t"]
        frame = self.image_widget.data[0][ind].copy()
        offset = self.current_offset
        corrected = apply_phase_offset(frame, offset)
        self.image_widget.figure[0, 0].graphics[0].data[:] = corrected

    def preview_before_after(self):
        ind = self.image_widget.current_index["t"]
        before = self.image_widget.data[0][ind].copy()
        offset = compute_phase_offset(before, upsample=20)
        after = apply_phase_offset(before, offset)
        return before, after, offset
        # Assuming GUI can handle dual previews:

    # def calculate_offset(self):
    #     ind = self.image_widget.current_index["t"]
    #     frame = self.image_widget.data[0][ind].copy()
    #     return NotImplementedError
    #
    # def apply_offset(self):
    #     ind = self.image_widget.current_index["t"]
    #     frame = self.image_widget.data[0][ind].copy()
    #     frame[0::2, :] = np.roll(
    #         self.image_widget.data[0][ind][0::2, :], shift=-self.current_offset, axis=1
    #     )
    #     self.image_widget.figure[0, 0].graphics[0].data[:] = frame

    def track_slider(self, ev):
        """events to emit when z-plane changes"""
        t_index = ev["t"]
        return
        # self.current_offset = int(self.offset_store[t_index][0])
        # self.apply_offset()

    def save_to_file(self):
        if not self.h5name.is_file():
            print(f"Error: File {self.h5name} does not exist.")
            return
        try:
            with h5py.File(self.h5name.resolve(), "r+") as f:
                if "scan_corrections" in f:
                    del f["scan_corrections"]
                f.create_dataset("scan_corrections", data=np.array(self.offset_store))
                print(f"Offsets successfully saved to {self.h5name}")

            imgui.open_popup("Save Successful")

        except Exception as e:
            print(f"Failed to save offsets: {e}")

    def blend(self):
        nz = self.image_widget.data[0].shape[0]
        c_index = self.image_widget.current_index["t"]
        if c_index < nz:
            frame = self.image_widget.data[0][c_index]
            frame_n = self.image_widget.data[0][c_index + 1]
            tmp = norm_percentile(frame * frame_n)
            self.image_widget.data[0][c_index] = norm_minmax(tmp)

    def load_pollen_offsets(
        self,
    ):
        with h5py.File(self.fpath[0], "r") as f1:
            dx = np.array(f1["x_shifts"])
            dy = np.array(f1["y_shifts"])
            ofs_volume = np.array(f1["scan_corrections"])
            self.h5name = Path(self.fpath[0])
        return ofs_volume

    @staticmethod
    def open_file_dialog(self):
        file_dialog = pfd.open_file(
            title="Select a pollen calibration file",
            filters=["*.tiff", "*.tif", "*.h5", "*.hdf5"],
            options=pfd.opt.none,
        )
        return file_dialog.result()


class SummaryDataWidget(EdgeWindow):
    def __init__(self, image_widget, size, location):
        flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        super().__init__(
            figure=image_widget.figure,
            size=size,
            location=location,
            title="Preview Data",
            window_flags=flags,
        )
        self.image_widget = image_widget

        self.gaussian_sigma = 0.0

    def update(self):
        something_changed = False

        imgui.text("Gaussian Filter")
        changed, value = imgui.slider_float(
            "Sigma", v=self.gaussian_sigma, v_min=0.0, v_max=20.0
        )
        if changed:
            self.gaussian_sigma = value
            something_changed = True

        imgui.separator()
        imgui.text("Image Processing")

        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply Gaussian smoothing to current frame")

        if imgui.button("Compute Temporal Mean"):
            self.temporal_mean()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Compute mean image across time dimension")

        if imgui.button("Compute Temporal StdDev"):
            self.temporal_std()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Compute std-dev image across time dimension")

        if imgui.button("Blend Adjacent Z-Planes"):
            self.blend_adjacent()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Blend current z-plane with adjacent planes")

        imgui.separator()
        imgui.text("Suite2p Previews")

        if imgui.button("Bandpass Filter"):
            self.apply_bandpass()

        if imgui.button("Median Projection"):
            self.median_projection()

        if imgui.button("Variance Map"):
            self.variance_map()

        if imgui.button("Edge Detection"):
            self.edge_detection()

        if imgui.button("High-Pass Filter"):
            self.highpass_filter()

        if imgui.button("Denoised Mean"):
            self.denoised_mean()
        imgui.separator()
        imgui.text("Statistics")
        if something_changed:
            self.apply_gaussian()

    def temporal_mean(self):
        """Apply a temporal mean projection around the current frame without changing dimensions."""

        z_idx = self.image_widget.current_index.get("z", 0)
        t_idx = self.image_widget.current_index.get("t", 0)

        data = self.image_widget.data[0]

        # window around current t
        window_size = 5
        half_window = window_size // 2
        t_min = max(0, t_idx - half_window)
        t_max = min(data.shape[1], t_idx + half_window + 1)

        # average only across small t window, keeping z, x, y shape
        averaged = data[:, t_min:t_max, ...].mean(axis=1)  # shape (z, x, y)

        # only show z_idx slice
        frame = averaged[z_idx]

        # update current view without changing the underlying array
        self.image_widget.figure[0, 0].graphics[0].data[:] = frame

    def temporal_std(self):
        """Standard deviation across time"""
        z_idx = self.image_widget.current_index.get("t", 0)
        frame = self.image_widget.data[0][:, z_idx, ...].std(axis=0)
        self.image_widget.figure[0, 0].graphics[0].data[:] = frame

    def blend_adjacent(self):
        """Blend current z with previous and next (if they exist)"""
        t_idx = self.image_widget.current_index.get("t", 0)
        data = self.image_widget.data[0]
        nz = data.shape[0]

        frames = [data[t_idx]]
        if t_idx > 0:
            frames.append(data[t_idx - 1])
        if t_idx < nz - 1:
            frames.append(data[t_idx + 1])

        blended = np.mean(frames, axis=0)
        self.image_widget.figure[0, 0].graphics[0].data[:] = blended

    def apply_bandpass(self):
        from scipy.ndimage import gaussian_filter

        frame = self.image_widget.managed_graphics[0].data.value.copy()
        lowpass = gaussian_filter(frame, sigma=3)
        highpass = frame - gaussian_filter(frame, sigma=20)
        bandpassed = frame - lowpass + highpass
        self.image_widget.figure[0, 0].graphics[0].data[:] = bandpassed

    def median_projection(self):
        data = self.image_widget.data[0]
        med_proj = np.median(data, axis=1)  # median across time
        t_idx = self.image_widget.current_index.get("t", 0)
        self.image_widget.figure[0, 0].graphics[0].data[:] = med_proj[t_idx]

    def variance_map(self):
        data = self.image_widget.data[0]
        var_proj = np.var(data, axis=1)
        t_idx = self.image_widget.current_index.get("t", 0)
        self.image_widget.figure[0, 0].graphics[0].data[:] = var_proj[t_idx]

    def edge_detection(self):
        from scipy.ndimage import sobel

        frame = self.image_widget.managed_graphics[0].data.value.copy()
        edge_x = sobel(frame, axis=0)
        edge_y = sobel(frame, axis=1)
        edges = np.hypot(edge_x, edge_y)
        self.image_widget.figure[0, 0].graphics[0].data[:] = edges

    def highpass_filter(self):
        from scipy.ndimage import gaussian_filter

        frame = self.image_widget.managed_graphics[0].data.value.copy()
        low = gaussian_filter(frame, sigma=10)
        highpass = frame - low
        self.image_widget.figure[0, 0].graphics[0].data[:] = highpass

    def denoised_mean(self):
        data = self.image_widget.data[0]
        t_idx = self.image_widget.current_index.get("t", 0)
        window = data[:, t_idx - 5 : t_idx + 5].mean(axis=1)
        self.image_widget.figure[0, 0].graphics[0].data[:] = window[t_idx]

    def apply_gaussian(self):
        self.image_widget.frame_apply = {
            0: lambda image_data: gaussian_filter(image_data, sigma=self.gaussian_sigma)
        }

    def calculate_noise(self):
        pass
