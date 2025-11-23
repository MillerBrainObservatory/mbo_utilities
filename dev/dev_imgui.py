"""
Example Data Loader
===================

Use portable file dialogs to open a file.

"""

# test_example = false
# sphinx_gallery_pygfx_docs = 'screenshot'

from pathlib import Path
import mbo_utilities as mbo
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui, portable_file_dialogs as pfd

# Initial data path
data_path = r"E:\tests\lbm\mbo_utilities\big_raw"
data = mbo.imread(data_path)
print(f"Initial data shape: {data.shape}")

# Create ImageWidget
iw = fpl.ImageWidget(
    [data],
    names=["My Custom Filetype"],
    slider_dim_names=("t", "z"),
)

class DataLoaderWidget(EdgeWindow):
    """ImGui widget for loading new data into the ImageWidget."""

    def __init__(self, image_widget, initial_path: str):
        super().__init__(
            figure=image_widget.figure,
            size=280,
            location="right",
            title="Data Loader",
        )
        self.iw = image_widget
        self.current_path = initial_path
        self.status_msg = ""
        self.status_color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
        self._folder_dialog = None
        self._file_dialog = None
        self._current_data_shape = data.shape

    def update(self):
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 6))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(6, 4))

        # Header
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.4, 0.8, 1.0, 1.0), "Load New Dataset")
        imgui.separator()
        imgui.spacing()

        # Current data info
        imgui.text("Current Data:")
        imgui.indent()
        imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), f"Shape: {self._current_data_shape}")
        imgui.unindent()
        imgui.spacing()

        # Path section
        imgui.text("Data Path:")
        avail_width = imgui.get_content_region_avail().x
        imgui.set_next_item_width(avail_width)
        changed, new_path = imgui.input_text("##path", self.current_path)
        if changed:
            self.current_path = new_path

        imgui.spacing()

        # Open File / Open Folder buttons
        button_width = (avail_width - 8) / 2

        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.3, 0.5, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.4, 0.6, 1.0))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.2, 0.4, 1.0))

        if imgui.button("Open File", imgui.ImVec2(button_width, 0)):
            # Where the file-explorer opens by default
            # you can set this to a directory / server share you regularly open
            start_dir = str(Path(self.current_path).parent) if Path(self.current_path).exists() else str(Path.home())
            self._file_dialog = pfd.open_file(
                "Select Data File",
                start_dir,
                ["TIFF Files", "*.tif *.tiff", "Raw Files", "*.raw", "All Files", "*.*"]
            )

        imgui.same_line()

        if imgui.button("Open Folder", imgui.ImVec2(button_width, 0)):
            start_dir = self.current_path if Path(self.current_path).exists() else str(Path.home())
            self._folder_dialog = pfd.select_folder("Select Data Folder", start_dir)

        imgui.pop_style_color(3)

        imgui.spacing()

        # Load button (full width, green)
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.7, 0.3, 1.0))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.4, 0.1, 1.0))
        if imgui.button("Load Data", imgui.ImVec2(avail_width, 0)):
            self._load_data()
        imgui.pop_style_color(3)

        # Check if file dialog has result
        if self._file_dialog is not None and self._file_dialog.ready():
            result = self._file_dialog.result()
            if result and len(result) > 0:
                self.current_path = result[0]
            self._file_dialog = None

        # Check if folder dialog has result
        if self._folder_dialog is not None and self._folder_dialog.ready():
            result = self._folder_dialog.result()
            if result:
                self.current_path = result
            self._folder_dialog = None

        # Status message
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if self.status_msg:
            # Wrap text for long messages
            imgui.push_text_wrap_pos(imgui.get_content_region_avail().x)
            imgui.text_colored(self.status_color, self.status_msg)
            imgui.pop_text_wrap_pos()

        imgui.pop_style_var(2)

    def _load_data(self):
        """Load new data from the current path."""
        if not self.current_path:
            self.status_msg = "Error: No path specified"
            self.status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
            return

        path = Path(self.current_path)
        if not path.exists():
            self.status_msg = f"Error: Path does not exist"
            self.status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
            return

        try:
            self.status_msg = "Loading..."
            self.status_color = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)

            new_data = mbo.imread(self.current_path)

            # Update ImageWidget data using iw-array API
            self.iw.data[0] = new_data

            # Reset indices
            self.iw.indices["t"] = 0
            if new_data.ndim >= 4:
                self.iw.indices["z"] = 0

            self._current_data_shape = new_data.shape
            self.status_msg = f"Loaded successfully!\nShape: {new_data.shape}"
            self.status_color = imgui.ImVec4(0.3, 1.0, 0.3, 1.0)
            print(f"Loaded: {self.current_path}, shape: {new_data.shape}")

        except Exception as e:
            self.status_msg = f"Error: {str(e)}"
            self.status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
            print(f"Error loading data: {e}")


iw.show()

# Add the data loader widget to the figure
loader = DataLoaderWidget(iw, data_path)
iw.figure.add_gui(loader)

fpl.loop.run()
