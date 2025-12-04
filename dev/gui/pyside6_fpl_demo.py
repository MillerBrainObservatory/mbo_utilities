"""
Demo: Embedding fastplotlib ImageWidget in a PySide6 Qt application.

This shows how to combine fastplotlib's GPU-accelerated rendering
with native Qt widgets (like suite2p uses).

The key insight is that fastplotlib's ImageWidget has a .canvas attribute
which is a rendercanvas QRenderWidget that can be embedded in any Qt layout.
"""

import sys
import numpy as np

# Import PySide6 FIRST before any rendercanvas imports
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QSlider

# Fix suite2p PySide6 compatibility issue (suite2p uses PyQt5-style enum access)
# suite2p's RangeSlider.paintEvent uses self.NoTicks which doesn't exist in PySide6
QSlider.NoTicks = QSlider.TickPosition.NoTicks

# Now import fastplotlib (it will use PySide6)
import fastplotlib as fpl


class MainWindow(QtWidgets.QMainWindow):
    """Main window combining fastplotlib with Qt widgets."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fastplotlib + PySide6 Demo")
        self.setGeometry(100, 100, 1200, 800)

        # Keep reference to suite2p window so it doesn't get garbage collected
        self.suite2p_window = None
        self._last_ichosen = None  # Track last selected cell to detect changes

        # Timer to poll suite2p for selection changes (suite2p doesn't emit signals)
        self.suite2p_poll_timer = QtCore.QTimer()
        self.suite2p_poll_timer.timeout.connect(self._poll_suite2p_selection)
        self.suite2p_poll_timer.start(100)  # Poll every 100ms

        # Create central widget with layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Left panel: controls (like suite2p has)
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(250)

        # Add some controls
        left_layout.addWidget(QtWidgets.QLabel("Controls"))

        # Suite2p GUI button
        self.suite2p_btn = QtWidgets.QPushButton("Open Suite2p GUI")
        self.suite2p_btn.clicked.connect(self.open_suite2p_gui)
        left_layout.addWidget(self.suite2p_btn)

        left_layout.addWidget(self._create_separator())

        # Frame slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 99)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        left_layout.addWidget(QtWidgets.QLabel("Frame:"))
        left_layout.addWidget(self.frame_slider)

        # Colormap selector
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["viridis", "gray", "plasma", "magma", "inferno"])
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_change)
        left_layout.addWidget(QtWidgets.QLabel("Colormap:"))
        left_layout.addWidget(self.cmap_combo)

        # ROI info (like suite2p)
        self.roi_label = QtWidgets.QLabel("Selected ROI: None")
        left_layout.addWidget(self.roi_label)

        # Add stretch to push controls to top
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # Right panel: fastplotlib visualization
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # Create sample data (100 frames, 512x512)
        self.data = self._generate_sample_data()

        # Create fastplotlib ImageWidget
        # Note: We pass the Qt widget as parent implicitly by adding to layout later
        self.iw = fpl.ImageWidget(
            data=self.data,
            cmap="viridis",
            histogram_widget=True,
        )

        # Get the Qt canvas widget from fastplotlib
        # The ImageWidget wraps a Figure which has a canvas
        canvas_widget = self.iw.figure.canvas

        # Add the canvas to our layout
        right_layout.addWidget(canvas_widget)

        main_layout.addWidget(right_panel, stretch=1)

        # Start the animation loop
        self.iw.show()

    def _create_separator(self):
        """Create a horizontal line separator."""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    def open_suite2p_gui(self, statfile=None):
        """Open the suite2p GUI in a separate window.

        Positions both windows side-by-side, each taking half the screen.

        Parameters
        ----------
        statfile : str | Path, optional
            Path to stat.npy file to load. If None, opens empty GUI.
        """
        # Qt signal passes bool (checked state) - ignore it
        if isinstance(statfile, bool):
            statfile = None

        try:
            from suite2p.gui.gui2p import MainWindow as Suite2pMainWindow

            # Create suite2p window
            self.suite2p_window = Suite2pMainWindow(statfile=statfile)

            # Get screen geometry for side-by-side layout
            screen = QtWidgets.QApplication.primaryScreen()
            if screen:
                screen_geom = screen.availableGeometry()
                screen_x = screen_geom.x()
                screen_y = screen_geom.y()
                screen_w = screen_geom.width()
                screen_h = screen_geom.height()

                # Split screen in half
                half_width = screen_w // 2

                # Position main window on LEFT half
                self.setGeometry(screen_x, screen_y, half_width, screen_h)

                # Position suite2p on RIGHT half
                self.suite2p_window.setGeometry(
                    screen_x + half_width, screen_y, half_width, screen_h
                )

                # Allow suite2p to be resized smaller
                self.suite2p_window.setMinimumSize(400, 300)

            self.suite2p_window.show()

        except ImportError as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Suite2p Not Available",
                f"Could not import suite2p GUI:\n{e}\n\n"
                "Install with: pip install suite2p"
            )

    def _generate_sample_data(self):
        """Generate sample time-series data like calcium imaging."""
        n_frames = 100
        height = 256
        width = 256

        # Create base noise
        data = np.random.randn(n_frames, height, width).astype(np.float32) * 10 + 100

        # Add some "ROIs" - circular regions with time-varying intensity
        n_rois = 10
        np.random.seed(42)
        for _ in range(n_rois):
            cy, cx = np.random.randint(30, height - 30), np.random.randint(30, width - 30)
            radius = np.random.randint(8, 20)

            # Create mask
            y, x = np.ogrid[:height, :width]
            mask = ((y - cy) ** 2 + (x - cx) ** 2) <= radius**2

            # Create time-varying signal (calcium transients)
            signal = np.zeros(n_frames)
            for t in range(0, n_frames, np.random.randint(15, 30)):
                if t < n_frames:
                    # Exponential decay transient
                    decay = np.exp(-np.arange(min(20, n_frames - t)) / 5)
                    signal[t : t + len(decay)] += decay * np.random.uniform(50, 150)

            # Apply signal to ROI pixels
            for t in range(n_frames):
                data[t][mask] += signal[t]

        return data

    def on_frame_change(self, frame):
        """Handle frame slider change."""
        # Update the ImageWidget's time index
        if hasattr(self.iw, "set_data"):
            # For ImageWidget, we update the slider
            self.iw.sliders["t"].set_value(frame)

    def on_cmap_change(self, cmap_name):
        """Handle colormap change."""
        try:
            # Update colormap for all graphics in the ImageWidget
            for graphic in self.iw.figure[0, 0]:
                if hasattr(graphic, "cmap"):
                    graphic.cmap = cmap_name
        except Exception as e:
            print(f"Error changing colormap: {e}")

    def _poll_suite2p_selection(self):
        """Poll suite2p window for selection changes."""
        if self.suite2p_window is None:
            return
        if not hasattr(self.suite2p_window, 'loaded') or not self.suite2p_window.loaded:
            return

        # Get current selection
        ichosen = getattr(self.suite2p_window, 'ichosen', None)
        imerge = getattr(self.suite2p_window, 'imerge', [])

        # Check if selection changed
        if ichosen != self._last_ichosen:
            self._last_ichosen = ichosen
            self.on_suite2p_cell_selected(ichosen, imerge)

    def on_suite2p_cell_selected(self, cell_idx, selected_cells):
        """Called when a cell is selected in suite2p.

        Parameters
        ----------
        cell_idx : int
            Index of the primarily selected cell (ichosen)
        selected_cells : list of int
            List of all selected cells (imerge), for multi-select with Shift/Ctrl
        """
        # Update the ROI label
        if len(selected_cells) > 1:
            self.roi_label.setText(f"Selected ROIs: {selected_cells}")
        else:
            self.roi_label.setText(f"Selected ROI: {cell_idx}")

        # Access suite2p data for the selected cell
        s2p = self.suite2p_window
        if hasattr(s2p, 'stat') and cell_idx is not None:
            stat = s2p.stat[cell_idx]
            # stat contains: 'ypix', 'xpix', 'lam', 'med', 'npix', 'radius', etc.
            print(f"Cell {cell_idx}: center=({stat.get('med', 'N/A')}), "
                  f"npix={stat.get('npix', 'N/A')}, "
                  f"radius={stat.get('radius', 'N/A'):.2f}")

        # Access fluorescence traces if available
        if hasattr(s2p, 'Fcell') and cell_idx is not None:
            trace = s2p.Fcell[cell_idx, :]
            print(f"  Trace shape: {trace.shape}, mean={trace.mean():.2f}")

        # You can add custom visualization here, e.g.:
        # - Highlight the selected cell in your fastplotlib view
        # - Plot the fluorescence trace
        # - Show cell statistics


def main():
    # Create Qt application
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
