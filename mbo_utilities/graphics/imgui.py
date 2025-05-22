import webbrowser
from pathlib import Path
from typing import Literal
import threading

from imgui_bundle import hello_imgui
from imgui_bundle import imgui_ctx

from ..assembly import save_as

from scipy.ndimage import gaussian_filter, fourier_shift
from skimage.registration import phase_cross_correlation

import numpy as np
import h5py
from icecream import ic

from ..file_io import Scan_MBO, SAVE_AS_TYPES

try:
    import cupy as cp  # noqa
    from cusignal import register_translation  # GPU version of phase_cross_correlation # noqa

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

from imgui_bundle import imgui, implot
from imgui_bundle import portable_file_dialogs as pfd

from .. import mbo_home, read_scan
from ..util import norm_minmax, norm_percentile


def imgui_dynamic_table(
    table_id: str, data_lists: list, titles: list = None, selected_index: int = None
):
    """
    imgui dynamic table helper

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
        flags=imgui.TableFlags_.borders | imgui.TableFlags_.resizable,  # noqa
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

    result = frame.copy()
    rows = result[1::2, :]
    f = np.fft.fftn(rows)
    fshift = fourier_shift(f, (0, offset))
    result[1::2, :] = np.fft.ifftn(fshift).real
    return result


def compute_phase_offset(
    frame: np.ndarray, upsample: int = 10, exclude_center_px: int = 0
) -> float:

    if frame.ndim == 3:
        frame = np.mean(frame, axis=0)
    _, w = frame.shape

    frame = frame.astype(np.float32)

    cx = w // 2
    keep_left = slice(None, cx - exclude_center_px)
    keep_right = slice(cx + exclude_center_px, None)

    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    pre_crop = np.concatenate([pre[:m, keep_left], pre[:m, keep_right]], axis=1)
    post_crop = np.concatenate([post[:m, keep_left], post[:m, keep_right]], axis=1)

    ic(pre_crop)
    ic(post_crop)

    # transfer to GPU
    if HAS_CUPY:
        pre_gpu = cp.asarray(pre_crop)
        post_gpu = cp.asarray(post_crop)
        shift, _, _ = register_translation(
            cp.asarray(pre_gpu), cp.asarray(post_gpu), upsample_factor=upsample
        )
    else:
        shift, _, _ = phase_cross_correlation(
            pre_crop, post_crop, upsample_factor=upsample  # noqa
        )

    return float(shift[1])

def _save_as(path, **kwargs):
    """
    read scan from path for threading
    there must be a better way to do this
    """
    scan = read_scan(path)
    save_as(scan, **kwargs)

def _apply_offset(frame, offset):
    result = frame.copy()
    rows = frame[1::2, :]
    f = np.fft.fftn(rows)
    fshift = fourier_shift(f, (0, offset))
    result[1::2, :] = np.fft.ifftn(fshift).real
    return result

class PreviewDataWidget(EdgeWindow):
    def __init__(
        self,
        iw: fpl.ImageWidget,
        fpath: str | None = None,
        size: int = 350,
        location: Literal["bottom", "right"] = "right",
        title: str = "Data Preview",
    ):
        super().__init__(figure=iw.figure, size=size, location=location, title=title)
        # whether to set iw managed_graphics data when setting calculate offset
        self._open_save_popup = None
        self._show_debug_panel = None
        self._log_buffer = []

        self.max_offset = 8
        self.fpath = Path(fpath) if fpath else Path(mbo_home)
        self.h5name = None
        self.image_widget = iw
        self.shape = self.image_widget.data[0].shape
        self.nz = self.shape[0]
        self.offset_store = np.zeros(self.nz)

        self._gaussian_sigma = 0
        self._current_offset = 0.0
        self._window_size = 1

        self._phase_upsample = 20
        self._auto_update = False
        self._proj = "mean"
        self._current_saving_plane = 0

        self.fpath = Path(fpath) if fpath else Path(mbo_home)

        self._save_dir = str(getattr(self, "_save_dir", ""))
        self._planes_str = str(getattr(self, "_planes_str", ""))

        self._ext = str(getattr(self, "_ext", ".tiff"))
        self._ext_idx = SAVE_AS_TYPES.index(".tiff")

        self._selected_planes = set()

        self._overwrite = True
        self._fix_phase = True
        self._debug = False
        self._progress_value = 0.0

        if isinstance(self.image_widget.data[0], Scan_MBO):
            self.is_mbo_scan = True
        else:
            self.is_mbo_scan = False

        for subplot in self.image_widget.figure:
            subplot.toolbar = False

        self.image_widget.add_event_handler(self.track_slider, "current_index")
        self.image_widget._image_widget_sliders._loop = True  # noqa

    @property
    def gaussian_sigma(self):
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value):
        if value > 0:
            self._gaussian_sigma = value
            self.image_widget.frame_apply = {0: self._combined_frame_apply}

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, value):
        if value != self._proj:
            self.image_widget.window_funcs["t"].func = getattr(np, value)
            self._proj = value

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        self.image_widget.window_funcs["t"].window_size = value
        self._window_size = value

    @property
    def current_offset(self):
        return self._current_offset

    @current_offset.setter
    def current_offset(self, value):
        if value:
            self._current_offset = value
            self.image_widget.frame_apply = {0: self._combined_frame_apply}

    @property
    def phase_upsample(self):
        return self._phase_upsample

    @phase_upsample.setter
    def phase_upsample(self, value):
        if value > 0:
            self._phase_upsample = value
            self.image_widget.frame_apply = {0: self._combined_frame_apply}

    @property
    def auto_update(self):
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        self._auto_update = value
        if value:
            self.image_widget.frame_apply = {0: self._combined_frame_apply}

    def update(self):
        with imgui_ctx.begin_child("##", window_flags=imgui.WindowFlags_.menu_bar):  # noqa
            # Top Menu Bar
            if imgui.begin_menu_bar():
                if imgui.begin_menu("File", True):
                    if imgui.menu_item("Save as", "Ctrl+S", p_selected=False, enabled=self.is_mbo_scan)[0]:
                        self._open_save_popup = True
                    imgui.end_menu()
                if imgui.begin_menu("Docs", True):
                    if imgui.menu_item("Open Docs", "Ctrl+I", p_selected=False, enabled=True)[0]:
                        webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/")
                    imgui.end_menu()

                if imgui.begin_menu("Settings", True):
                    if imgui.menu_item("Style", "Ctrl+Shift+S", p_selected=False, enabled=self.is_mbo_scan)[0]:
                        with imgui_ctx.begin("Settings"):
                            imgui.show_font_selector('Font Selector')
                    if imgui.menu_item("Debug Panel", "", p_selected=False, enabled=True)[0]:
                        self._show_debug_panel = not getattr(self, "_show_debug_panel", False)
                    imgui.end_menu()

                imgui.end_menu_bar()

            # Save As Popup Dialog
            if getattr(self, "_open_save_popup", False):
                imgui.open_popup("Save As")
                self._open_save_popup = False

            if imgui.begin_popup_modal("Save As")[0]:
                # Directory + Ext
                imgui.set_next_item_width(hello_imgui.em_size(20))
                # TODO: make _save_dir a property to expand ~
                _, self._save_dir = imgui.input_text("Save Dir", str(Path(self._save_dir).expanduser().resolve()), 256)
                imgui.same_line()
                if imgui.button("Browse"):
                    home = Path().home()
                    res = pfd.select_folder(str(home))
                    if res:
                        self._save_dir = res.result()

                imgui.set_next_item_width(hello_imgui.em_size(20))
                _, self._ext_idx = imgui.combo("Ext", self._ext_idx, SAVE_AS_TYPES)
                self._ext = SAVE_AS_TYPES[self._ext_idx]

                # Options Section
                imgui.separator()
                imgui.text("Options")

                def checkbox_with_tooltip(_label, _value, _tooltip):
                    _, _value = imgui.checkbox(_label, _value)
                    imgui.same_line()
                    imgui.text_disabled("(?)")
                    if imgui.is_item_hovered():
                        imgui.begin_tooltip()
                        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                        imgui.text_unformatted(_tooltip)
                        imgui.pop_text_wrap_pos()
                        imgui.end_tooltip()
                    return _value

                self._overwrite = checkbox_with_tooltip("Overwrite", self._overwrite,
                                                        "Replace any existing output files.")
                self._fix_phase = checkbox_with_tooltip("Fix Phase", self._fix_phase,
                                                        "Apply scan-phase correction to interleaved lines.")
                self._debug = checkbox_with_tooltip("Debug", self._debug,
                                                    "Enable debug outputs such as intermediate images.")

                # Z-plane selection
                imgui.separator()
                imgui.text("Select Planes")

                try:
                    num_planes = self.image_widget.data[0].num_channels  # noqa
                except Exception as e:
                    num_planes = 1
                    hello_imgui.log(hello_imgui.LogLevel.error, f"Could not read number of planes: {e}")

                if imgui.button("All"):
                    self._selected_planes = set(range(num_planes))
                imgui.same_line()
                if imgui.button("None"):
                    self._selected_planes = set()

                imgui.columns(2, borders=False)
                for i in range(num_planes):
                    imgui.push_id(i)
                    selected = i in self._selected_planes
                    _, selected = imgui.checkbox(f"Plane {i}", selected)
                    if selected:
                        self._selected_planes.add(i)
                    else:
                        self._selected_planes.discard(i)
                    imgui.pop_id()
                    imgui.next_column()
                imgui.columns(1)

                # Buttons
                imgui.separator()
                if imgui.button("Save", imgui.ImVec2(100, 0)):
                    if not self._save_dir:
                        self._save_dir = mbo_home
                    try:
                        save_planes = [p + 1 for p in self._selected_planes]
                        save_kwargs = {
                            "path": self.fpath,
                            "savedir": self._save_dir,
                            "planes": save_planes,
                            "overwrite": self._overwrite,
                            "fix_phase": self._fix_phase,
                            "debug": self._debug,
                            "ext": self._ext,
                            "save_phase_png": False,
                            "target_chunk_mb": 20,
                            "progress_callback": self.gui_progress_callback,
                        }
                        threading.Thread(target=_save_as, kwargs=save_kwargs).start()
                        imgui.close_current_popup()
                    except Exception as e:
                        hello_imgui.log(hello_imgui.LogLevel.error, f"Save failed: {e}")
                imgui.same_line()
                if imgui.button("Cancel"):
                    imgui.close_current_popup()

                imgui.end_popup()

            # Section: Window Functions
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Window Functions")
            imgui.separator()

            imgui.begin_group()

            projection_options = ["mean", "max", "std"]
            current_idx = projection_options.index(self.proj)
            proj_changed, new_idx = imgui.combo("Projection", current_idx, projection_options)
            if proj_changed:
                self.proj = projection_options[new_idx]
                self._log_buffer.append(f"Set projection: {self.proj}")
                self._log_buffer = self._log_buffer[-100:]

            winsize_changed, new_winsize = imgui.input_int("Window Size", self.window_size, step=1, step_fast=2)
            if winsize_changed and new_winsize > 0:
                self.window_size = new_winsize
                self._log_buffer.append(f"New Window Size: {new_winsize}")
                self._log_buffer = self._log_buffer[-100:]


            gaussian_changed, new_gaussian_sigma = imgui.slider_float(label="sigma", v=self.gaussian_sigma, v_min=0.0, v_max=20.0, )
            if gaussian_changed:
                self.gaussian_sigma = new_gaussian_sigma
                self._log_buffer.append(f"New gaussian sigma: {new_gaussian_sigma}")
                self._log_buffer = self._log_buffer[-100:]

            imgui.end_group()

            # Section: Scan-phase Correction
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.6, 1.0, 1.0), "Scan-Phase Correction")
            imgui.separator()
            imgui.begin_group()

            imgui.text("Offset")
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(1.0, 0.5, 0.0, 1.0), f"{self.current_offset:.3f}")

            upsample_changed, upsample_val = imgui.input_int("Upsample", self._phase_upsample, step=1, step_fast=2)
            if upsample_changed:
                self.phase_upsample = max(1, upsample_val)
                self._log_buffer.append(f"New upsample: {upsample_val}")
                self._log_buffer = self._log_buffer[-100:]  # limit to last 100 lines

            max_offset_changed, max_offset = imgui.input_int("max-offset", self.max_offset, step=1, step_fast=2)
            if max_offset_changed:
                self.max_offset = max(1, max_offset)
                self._log_buffer.append(f"New max-offset: {max_offset}")
                self._log_buffer = self._log_buffer[-100:]

            auto_changed, new_auto_update = imgui.checkbox("Auto Update", self.auto_update)
            if auto_changed:
                self.auto_update = new_auto_update
                if new_auto_update:
                    self.calculate_offset()
                self._log_buffer.append(f"Auto-update changed: {self.auto_update}")
                self._log_buffer = self._log_buffer[-100:]

            if imgui.button("Calculate", imgui.ImVec2(0, 20)):
                self._log_buffer.append(f"Calculating offset")
                self._log_buffer = self._log_buffer[-100:]
                self.calculate_offset()
            imgui.same_line()
            if imgui.button("Reset", imgui.ImVec2(0, 20)):
                self._log_buffer.append(f"Reset offset")
                self._log_buffer = self._log_buffer[-100:]
                self.current_offset = 0

            imgui.end_group()

            # Section: Mean Subtraction
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.6, 1.0, 1.0), "Mean-Subtraction")
            imgui.separator()
            imgui.begin_group()

            with imgui_ctx.begin_child("##progress"):
                # Always show progress bar when active
                if 0.0 < self._progress_value < 1.0:
                    imgui.spacing()
                    progress_bar_height = hello_imgui.em_size(1.4)
                    imgui.set_cursor_pos_y(
                        imgui.get_window_height() - progress_bar_height - hello_imgui.em_size(1.5)
                    )

                    label = f"Saving plane {self._current_saving_plane}..."  # Update this value externally
                    text_color = imgui.ImVec4(1, 1, 1, 1)
                    bar_color = imgui.ImVec4(0.2, 0.7, 0.2, 1) if self._progress_value >= 1.0 else imgui.ImVec4(0.3, 0.3,
                                                                                                                0.3, 1)
                    imgui.push_style_color(imgui.Col_.plot_histogram, bar_color)  # noqa
                    imgui.push_style_color(imgui.Col_.text, text_color)  # noqa
                    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(6, 4))  # noqa

                    imgui.progress_bar(
                        self._progress_value,
                        imgui.ImVec2(-1, progress_bar_height),
                        label,
                    )

                    imgui.pop_style_var()
                    imgui.pop_style_color(2)

        if getattr(self, "_show_debug_panel", False):
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "Debug Output")
            with imgui_ctx.begin("##debug_log"):
                for log_line in getattr(self, "_log_buffer", []):
                    imgui.text_wrapped(log_line)

    def get_raw_frame(self):
        return self.image_widget.data[0][
                   self.image_widget.current_index["t"],
                   self.image_widget.current_index["z"],
                   :, :]

    def get_managed_frame(self):
        return self.image_widget.managed_graphics[0].data.value

    def gui_progress_callback(self, fraction, current_plane):
        self._progress_value = fraction
        self._current_saving_plane = current_plane

    def _combined_frame_apply(self, frame: np.ndarray) -> np.ndarray:
        if self._current_offset:
            frame = apply_phase_offset(frame, self.current_offset)
        if self._gaussian_sigma > 0:
            frame = gaussian_filter(frame, sigma=self.gaussian_sigma)
        return frame

    def calculate_offset(self):
        """Get the current frame, calculate the offset"""
        frame = self.get_raw_frame()
        self._log_buffer.append(f"Old offset: {self.current_offset}")
        ofs = compute_phase_offset(frame, upsample=self._phase_upsample)
        self.current_offset = ofs
        self._log_buffer.append(f"New offset: {self.current_offset}")
        self._log_buffer = self._log_buffer[-100:]
        # force update
        # self.image_widget.current_index = self.image_widget.current_index

    def track_slider(self, ev=None):
        """called whenever a frame of the movie changes.
        Note frame_apply is called before this code is reached.
        """
        if self.auto_update:
             self.calculate_offset()

    def edge_detection(self):
        from scipy.ndimage import sobel

        frame = self.image_widget.managed_graphics[0].data.value.copy()
        edge_x = sobel(frame, axis=0)
        edge_y = sobel(frame, axis=1)
        edges = np.hypot(edge_x, edge_y)
        self.image_widget.managed_graphics[0].data[:] = edges

    def highpass_filter(self):
        from scipy.ndimage import gaussian_filter

        frame = self.image_widget.managed_graphics[0].data[:]
        low = gaussian_filter(frame, sigma=self.gaussian_sigma)
        highpass = frame - low
        self.image_widget.managed_graphics[0].data[:] = highpass

    def denoised_mean(self):
        data = self.image_widget.data[0]
        t_idx = self.image_widget.current_index.get("t", 0)
        window = data[:, t_idx - 5 : t_idx + 5].mean(axis=1)
        self.image_widget.managed_graphics[0].data[:] = window[t_idx]

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
