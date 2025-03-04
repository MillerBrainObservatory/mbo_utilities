from pathlib import Path
import numpy as np
import h5py
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
import imgui_bundle
from imgui_bundle import imgui, implot
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities import (
    return_scan_offset,
    get_files,
    norm_minmax,
    norm_percentile
)


def implot_pollen(pollen_offsets, offset_store, zstack):
    imgui.begin_child("Z-Stack Analysis")

    z_planes = np.arange(len(pollen_offsets)) + 1

    # Offset Comparison Plot
    if implot.begin_plot("Offset Comparison Across Z-Planes"):
        implot.plot_bars("Calibration Offsets", z_planes, np.array([p[0] for p in pollen_offsets]), bar_size=0.4)
        implot.plot_bars("Selected Offsets", z_planes, np.array([s[0] for s in offset_store]), bar_size=0.4, shift=0.2)
        implot.end_plot()

    # Mean Intensity Plot
    if implot.begin_plot("Mean Intensity Across Z-Planes"):
        mean_intensities = np.array([np.mean(zstack[z]) for z in range(zstack.shape[0])])
        implot.plot_line("Mean Intensity", z_planes, mean_intensities)
        implot.end_plot()

    # Cumulative Offset Drift Plot
    if implot.begin_plot("Cumulative Offset Drift"):
        cumulative_shifts = np.cumsum(np.array([s[0] for s in offset_store]))
        implot.plot_line("Cumulative Shift", z_planes, cumulative_shifts)
        implot.end_plot()

    imgui.end_child()


class PollenCalibration(EdgeWindow):
    def __init__(self, fpath=None, iw=None, size=350, location="right", title="Pollen Calibration", depth=5):
        super().__init__(figure=iw.figure, size=size, location=location, title=title, )

        self.fpath = fpath
        self.pollen_offsets = self.load_pollen_offsets(depth=depth)
        self.pollen_offsets_original = self.pollen_offsets.copy()
        self.figure = iw.figure
        self.data_store = iw.data[:].copy()
        self.image_widget = iw
        self.original_data = iw.data[:]
        self._current_offset = 0
        self.offset_store = self.pollen_offsets.copy()
        self.proj = 'mean'

        self.image_widget.add_event_handler(self.track_slider, "current_index")

    @property
    def current_offset(self):
        return self._current_offset

    @current_offset.setter
    def current_offset(self, value):
        self._current_offset = value
        self.apply_offset()

    def update(self):
        something_changed = False
        button_size = imgui.ImVec2(140, 20)

        offset_changed, value = imgui.input_int(label="offset", v=self.current_offset, step=1, step_fast=2, )
        if offset_changed:
            self.current_offset = value

        # Calculate Offset
        if imgui.button("Calculate Offset", button_size):
            self.current_offset = self.calculate_offset()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Automatically calculates the best offset for the selected Z-plane.")

        if imgui.button("Open File", imgui.ImVec2(140, 20)):
            selected_file = self.open_file_dialog()
            if selected_file:
                print("Selected File:", selected_file)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Open File")

        # switch max <--> mean
        if imgui.button("Switch Projection", button_size):
            self.switch()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Switch to max/mean projection, whichever is currently not selected.")

        # Store offset in table
        if imgui.button("Store Offset", button_size):
            ind = self.image_widget.current_index["t"]
            self.offset_store[ind] = self.current_offset
            self.apply_offset()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Set the current offset as the selected value.")

        # Save offset store
        if imgui.button("Save Selected", button_size):
            self.save_to_file()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Overwrite scan-phase values with current selection.")
        if imgui.begin_popup("Save Successful"):  # popup if success
            imgui.text("Offsets successfully saved!")
            if imgui.button("OK"):
                imgui.close_current_popup()
            imgui.end_popup()

        if imgui.begin_table("table", 3, flags=imgui.TableFlags_.borders | imgui.TableFlags_.resizable):
            imgui.table_setup_column("Z-Plane", imgui.TableColumnFlags_.width_fixed, 50)
            imgui.table_setup_column("Calibration Value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column("Selected Value", imgui.TableColumnFlags_.width_stretch)

            imgui.table_next_column()
            imgui.text("Z-Plane")
            imgui.table_next_column()
            imgui.text("pollen")
            imgui.table_next_column()
            imgui.text("sel")

            imgui.table_next_row()
            selected_index = self.image_widget.current_index["t"]

            for i, (offset_pollen, offset_selection) in enumerate(zip(self.pollen_offsets, self.offset_store)):
                imgui.table_next_row()
                imgui.table_next_column()

                if i == selected_index:
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0, 250, 35, 1))  # Highlight selected row
                imgui.text(f"Plane {i + 1}")

                # right align columns
                imgui.table_next_column()
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + imgui.get_column_width() - imgui.calc_text_size(
                    f"{offset_pollen[0]}").x - imgui.get_style().item_spacing.x)
                imgui.text(f"{offset_pollen[0]}")

                imgui.table_next_column()
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + imgui.get_column_width() - imgui.calc_text_size(
                    f"{offset_selection[0]}").x - imgui.get_style().item_spacing.x)
                imgui.text(f"{offset_selection[0]}")

                if i == selected_index:
                    imgui.pop_style_color()

            imgui.end_table()

        if something_changed:
            self.apply_offset()

    def calculate_offset(self):
        ind = self.image_widget.current_index["t"]
        frame = self.image_widget.data[0][ind].copy()
        return return_scan_offset(frame)

    def apply_offset(self):
        ind = self.image_widget.current_index["t"]
        frame = self.image_widget.data[0][ind].copy()
        frame[0::2, :] = np.roll(self.image_widget.data[0][ind][0::2, :], shift=-self.current_offset, axis=1)
        self.image_widget.figure[0, 0].graphics[0].data[:] = frame

    def track_slider(self, ev):
        t_index = ev["t"]
        self.current_offset = int(self.offset_store[t_index][0])
        self.apply_offset()

    def switch(self):
        pass
        # ind = self.image_widget.current_index["t"]
        # if self.proj == 'mean':
        #     print('mean')
        #     self.image_widget.set_data(zstack)
        #     self.image_widget.current_index["t"] = ind
        #     self.proj = "max"
        # else:
        #     print('max')
        #     self.image_widget.set_data(zstack_mean)
        #     self.image_widget.figure[0, 0].graphics[0].data[ind, ...] = zstack_mean[ind, ...]
        #     self.image_widget.current_index["t"] = ind
        #     self.proj = "mean"

    def save_to_file(self):
        if not self.h5name.is_file():
            print(f"Error: File {self.h5name} does not exist.")
            return
        try:
            fpath = self.h5name.resolve()
            with h5py.File(fpath, 'r+') as f:
                if "scan_corrections" in f:
                    del f["scan_corrections"]  # Remove old dataset

                f.create_dataset("scan_corrections", data=np.array(self.offset_store))
                print(f"Offsets successfully saved to {fpath}")

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

    def load_pollen_offsets(self, depth=5):
        fpath = Path(self.fpath)
        if fpath.is_dir():
            fpath = get_files(fpath, '.h5', depth)
            print(fpath)
            if not fpath:
                raise Exception("No file found")
        elif fpath.is_file():
            fpath = [fpath]
        with h5py.File(fpath[0], 'r') as f1:
            # dx = np.array(f1['x_shifts'])
            # dy = np.array(f1['y_shifts'])
            ofs_volume = np.array(f1['scan_corrections'])
            self.h5name = Path(fpath[0])
        return ofs_volume

    def open_file_dialog(self):
        """Opens a file selection dialog using portable_file_dialogs."""
        file_dialog = pfd.open_file(
            title="Select a File",
            filters=["*.tiff", "*.tif", "*.h5", "*.hdf5"],
            options=pfd.opt.none  # No multi-selection
        )

        if file_dialog.ready():  # Wait for the result
            selected_files = file_dialog.result()
            if selected_files:
                print(selected_files)
                return selected_files[0]  # Get the first selected file
        return None
