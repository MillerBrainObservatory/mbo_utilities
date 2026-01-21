"""
Integrated classifier window combining Suite2p viewer with diagnostics.

Provides bidirectional synchronization between mask visualization
and filter-based cell classification.
"""

import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QSlider, QGroupBox, QCheckBox,
    QFileDialog, QMessageBox, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QAction
import pyqtgraph as pg

from .shared_model import SharedDataModel
from .suite2p_embedded import Suite2pEmbedded


class DiagnosticsPanel(QWidget):
    """Qt-based diagnostics panel with filter sliders.

    Connects to SharedDataModel for reactive updates.

    Parameters
    ----------
    model : SharedDataModel
        Shared data model.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._updating_from_model = False

        # filter state
        self._snr_min = 0.0
        self._snr_max = 100.0
        self._shot_noise_min = 0.0
        self._shot_noise_max = 100.0
        self._skew_min = -10.0
        self._skew_max = 10.0
        self._activity_min = 0.0
        self._activity_max = 1.0

        # current thresholds
        self._filter_snr_min = 0.0
        self._filter_shot_noise_max = 100.0
        self._filter_skew_min = -10.0
        self._filter_activity_min = 0.0

        self._setup_ui()
        self._connect_model()

    def _setup_ui(self):
        """Create the UI."""
        layout = QVBoxLayout(self)

        # info section
        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout(info_group)

        self.path_label = QLabel("Path: -")
        self.path_label.setWordWrap(True)
        info_layout.addWidget(self.path_label)

        self.counts_label = QLabel("ROIs: 0 | Cells: 0")
        self.counts_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        info_layout.addWidget(self.counts_label)

        layout.addWidget(info_group)

        # roi navigation
        nav_group = QGroupBox("ROI Navigation")
        nav_layout = QVBoxLayout(nav_group)

        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI:"))
        self.roi_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_slider.setMinimum(0)
        self.roi_slider.setMaximum(0)
        self.roi_slider.valueChanged.connect(self._on_roi_slider_changed)
        roi_row.addWidget(self.roi_slider)
        self.roi_label = QLabel("0")
        self.roi_label.setMinimumWidth(50)
        roi_row.addWidget(self.roi_label)
        nav_layout.addLayout(roi_row)

        self.show_cells_only = QCheckBox("Show only cells")
        self.show_cells_only.setChecked(True)
        self.show_cells_only.stateChanged.connect(self._update_roi_slider_range)
        nav_layout.addWidget(self.show_cells_only)

        layout.addWidget(nav_group)

        # filter sliders
        filter_group = QGroupBox("Filter Thresholds")
        filter_layout = QVBoxLayout(filter_group)

        # snr
        filter_layout.addWidget(QLabel("SNR (min):"))
        self.snr_slider = QSlider(Qt.Orientation.Horizontal)
        self.snr_slider.setMinimum(0)
        self.snr_slider.setMaximum(1000)
        self.snr_slider.valueChanged.connect(self._on_snr_changed)
        filter_layout.addWidget(self.snr_slider)
        self.snr_label = QLabel("0.00")
        filter_layout.addWidget(self.snr_label)

        # shot noise
        filter_layout.addWidget(QLabel("Shot Noise (max):"))
        self.shot_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.shot_noise_slider.setMinimum(0)
        self.shot_noise_slider.setMaximum(1000)
        self.shot_noise_slider.setValue(1000)
        self.shot_noise_slider.valueChanged.connect(self._on_shot_noise_changed)
        filter_layout.addWidget(self.shot_noise_slider)
        self.shot_noise_label = QLabel("100.00")
        filter_layout.addWidget(self.shot_noise_label)

        # skewness
        filter_layout.addWidget(QLabel("Skewness (min):"))
        self.skew_slider = QSlider(Qt.Orientation.Horizontal)
        self.skew_slider.setMinimum(0)
        self.skew_slider.setMaximum(1000)
        self.skew_slider.valueChanged.connect(self._on_skew_changed)
        filter_layout.addWidget(self.skew_slider)
        self.skew_label = QLabel("0.00")
        filter_layout.addWidget(self.skew_label)

        # activity
        filter_layout.addWidget(QLabel("Activity (min):"))
        self.activity_slider = QSlider(Qt.Orientation.Horizontal)
        self.activity_slider.setMinimum(0)
        self.activity_slider.setMaximum(1000)
        self.activity_slider.valueChanged.connect(self._on_activity_changed)
        filter_layout.addWidget(self.activity_slider)
        self.activity_label = QLabel("0.0%")
        filter_layout.addWidget(self.activity_label)

        # reset button
        self.reset_btn = QPushButton("Reset Filters")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        filter_layout.addWidget(self.reset_btn)

        layout.addWidget(filter_group)

        # stats section
        stats_group = QGroupBox("Current ROI Stats")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_labels = {}
        for name in ["SNR", "Shot Noise", "Skewness", "Activity", "npix", "compact"]:
            label = QLabel(f"{name}: -")
            stats_layout.addWidget(label)
            self.stats_labels[name] = label

        layout.addWidget(stats_group)

        # actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self.save_btn = QPushButton("Save iscell.npy")
        self.save_btn.clicked.connect(self._on_save_clicked)
        actions_layout.addWidget(self.save_btn)

        self.export_btn = QPushButton("Export Training Data")
        self.export_btn.clicked.connect(self._on_export_clicked)
        actions_layout.addWidget(self.export_btn)

        layout.addWidget(actions_group)

        layout.addStretch()

    def _connect_model(self):
        """Connect to model signals."""
        self.model.roi_selected.connect(self._on_model_roi_selected)
        self.model.iscell_changed.connect(self._on_model_iscell_changed)
        self.model.iscell_batch_changed.connect(self._on_model_batch_changed)
        self.model.data_loaded.connect(self._on_model_data_loaded)

    @pyqtSlot()
    def _on_model_data_loaded(self):
        """Handle data loaded."""
        self._update_filter_ranges()
        self._update_roi_slider_range()
        self._update_info()
        self._update_stats()

    @pyqtSlot(int)
    def _on_model_roi_selected(self, roi_idx):
        """Handle ROI selection from model."""
        self._updating_from_model = True
        self._update_roi_slider_for_idx(roi_idx)
        self._update_stats()
        self._updating_from_model = False

    @pyqtSlot(int, bool)
    def _on_model_iscell_changed(self, roi_idx, is_cell):
        """Handle single cell change."""
        self._update_info()
        self._update_roi_slider_range()

    @pyqtSlot()
    def _on_model_batch_changed(self):
        """Handle batch iscell change."""
        self._update_info()
        self._update_roi_slider_range()

    def _update_filter_ranges(self):
        """Update filter slider ranges from data."""
        snr = self.model.snr
        if snr is not None:
            self._snr_min = float(np.min(snr))
            self._snr_max = float(np.max(snr))
            self._filter_snr_min = self._snr_min

        shot_noise = self.model.shot_noise
        if shot_noise is not None:
            self._shot_noise_min = float(np.min(shot_noise))
            self._shot_noise_max = float(np.max(shot_noise))
            self._filter_shot_noise_max = self._shot_noise_max

        skew = self.model.skewness
        if skew is not None:
            self._skew_min = float(np.min(skew))
            self._skew_max = float(np.max(skew))
            self._filter_skew_min = self._skew_min

        activity = self.model.activity
        if activity is not None:
            self._activity_min = float(np.min(activity))
            self._activity_max = float(np.max(activity))
            self._filter_activity_min = self._activity_min

        # update slider positions
        self._updating_from_model = True
        self.snr_slider.setValue(0)
        self.shot_noise_slider.setValue(1000)
        self.skew_slider.setValue(0)
        self.activity_slider.setValue(0)
        self._update_slider_labels()
        self._updating_from_model = False

    def _update_slider_labels(self):
        """Update slider value labels."""
        self.snr_label.setText(f"{self._filter_snr_min:.2f}")
        self.shot_noise_label.setText(f"{self._filter_shot_noise_max:.2f}")
        self.skew_label.setText(f"{self._filter_skew_min:.2f}")
        self.activity_label.setText(f"{self._filter_activity_min * 100:.1f}%")

    def _update_roi_slider_range(self):
        """Update ROI slider range based on visible ROIs."""
        if self.show_cells_only.isChecked():
            n_visible = len(self.model.cell_indices)
        else:
            n_visible = self.model.n_rois

        self.roi_slider.setMaximum(max(0, n_visible - 1))

    def _update_roi_slider_for_idx(self, roi_idx):
        """Update ROI slider to show given ROI index."""
        if self.show_cells_only.isChecked():
            indices = self.model.cell_indices
            if roi_idx in indices:
                pos = np.where(indices == roi_idx)[0][0]
                self.roi_slider.setValue(pos)
        else:
            self.roi_slider.setValue(roi_idx)

        self.roi_label.setText(str(roi_idx))

    def _update_info(self):
        """Update info labels."""
        path = self.model.plane_dir
        if path:
            self.path_label.setText(f"Path: {path.name}")
        else:
            self.path_label.setText("Path: -")

        n_rois = self.model.n_rois
        n_cells = self.model.n_cells
        self.counts_label.setText(f"ROIs: {n_rois} | Cells: {n_cells}")

    def _update_stats(self):
        """Update current ROI stats."""
        roi_idx = self.model.selected_roi

        if self.model.snr is not None and roi_idx < len(self.model.snr):
            self.stats_labels["SNR"].setText(f"SNR: {self.model.snr[roi_idx]:.2f}")
        else:
            self.stats_labels["SNR"].setText("SNR: -")

        if self.model.shot_noise is not None and roi_idx < len(self.model.shot_noise):
            self.stats_labels["Shot Noise"].setText(f"Shot Noise: {self.model.shot_noise[roi_idx]:.4f}")
        else:
            self.stats_labels["Shot Noise"].setText("Shot Noise: -")

        if self.model.skewness is not None and roi_idx < len(self.model.skewness):
            self.stats_labels["Skewness"].setText(f"Skewness: {self.model.skewness[roi_idx]:.2f}")
        else:
            self.stats_labels["Skewness"].setText("Skewness: -")

        if self.model.activity is not None and roi_idx < len(self.model.activity):
            self.stats_labels["Activity"].setText(f"Activity: {self.model.activity[roi_idx] * 100:.1f}%")
        else:
            self.stats_labels["Activity"].setText("Activity: -")

        if self.model.stat is not None and roi_idx < len(self.model.stat):
            s = self.model.stat[roi_idx]
            self.stats_labels["npix"].setText(f"npix: {s.get('npix', '-')}")
            self.stats_labels["compact"].setText(f"compact: {s.get('compact', '-'):.3f}" if s.get('compact') else "compact: -")
        else:
            self.stats_labels["npix"].setText("npix: -")
            self.stats_labels["compact"].setText("compact: -")

    def _on_roi_slider_changed(self, value):
        """Handle ROI slider change."""
        if self._updating_from_model:
            return

        if self.show_cells_only.isChecked():
            indices = self.model.cell_indices
            if value < len(indices):
                roi_idx = indices[value]
                self.model.selected_roi = roi_idx
                self.roi_label.setText(str(roi_idx))
        else:
            self.model.selected_roi = value
            self.roi_label.setText(str(value))

    def _on_snr_changed(self, value):
        """Handle SNR slider change."""
        if self._updating_from_model:
            return

        # map 0-1000 to snr range
        t = value / 1000.0
        self._filter_snr_min = self._snr_min + t * (self._snr_max - self._snr_min)
        self.snr_label.setText(f"{self._filter_snr_min:.2f}")
        self._apply_filters()

    def _on_shot_noise_changed(self, value):
        """Handle shot noise slider change."""
        if self._updating_from_model:
            return

        t = value / 1000.0
        self._filter_shot_noise_max = self._shot_noise_min + t * (self._shot_noise_max - self._shot_noise_min)
        self.shot_noise_label.setText(f"{self._filter_shot_noise_max:.2f}")
        self._apply_filters()

    def _on_skew_changed(self, value):
        """Handle skewness slider change."""
        if self._updating_from_model:
            return

        t = value / 1000.0
        self._filter_skew_min = self._skew_min + t * (self._skew_max - self._skew_min)
        self.skew_label.setText(f"{self._filter_skew_min:.2f}")
        self._apply_filters()

    def _on_activity_changed(self, value):
        """Handle activity slider change."""
        if self._updating_from_model:
            return

        t = value / 1000.0
        self._filter_activity_min = self._activity_min + t * (self._activity_max - self._activity_min)
        self.activity_label.setText(f"{self._filter_activity_min * 100:.1f}%")
        self._apply_filters()

    def _apply_filters(self):
        """Apply current filter thresholds."""
        self.model.apply_filter(
            snr_min=self._filter_snr_min,
            shot_noise_max=self._filter_shot_noise_max,
            skew_min=self._filter_skew_min,
            activity_min=self._filter_activity_min
        )

    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.model.reset_filters()
        self._update_filter_ranges()

    def _on_save_clicked(self):
        """Handle save button click."""
        if self.model.save_iscell():
            QMessageBox.information(self, "Saved", "iscell.npy saved successfully.")
        else:
            QMessageBox.warning(self, "Error", "Failed to save iscell.npy")

    def _on_export_clicked(self):
        """Handle export training data."""
        if self.model.stat is None or self.model.iscell is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return

        save_path = self.model.save_path
        if save_path is None:
            return

        # create training dir
        training_dir = save_path / "classifier_training"
        training_dir.mkdir(exist_ok=True)

        # extract features
        import time
        keys = ["npix_norm", "compact", "skew"]
        n_rois = len(self.model.stat)

        stats = np.zeros((n_rois, len(keys)), dtype=np.float32)
        for i, s in enumerate(self.model.stat):
            for j, k in enumerate(keys):
                stats[i, j] = s.get(k, 0.0)

        # get iscell
        if self.model.iscell.ndim == 2:
            iscell_labels = self.model.iscell[:, 0].astype(np.float32)
        else:
            iscell_labels = self.model.iscell.astype(np.float32)

        # save
        training_data = {
            "stats": stats,
            "iscell": iscell_labels,
            "keys": keys,
            "source_path": str(save_path),
        }

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = training_dir / f"training_{timestamp}.npy"
        np.save(export_path, training_data)

        QMessageBox.information(self, "Exported", f"Training data exported to:\n{export_path}")


class IntegratedClassifierWindow(QMainWindow):
    """Main window integrating Suite2p viewer with diagnostics panel.

    Provides bidirectional synchronization between mask visualization
    and filter-based classification.

    Parameters
    ----------
    plane_dir : str or Path, optional
        Path to suite2p plane directory to load.
    """

    def __init__(self, plane_dir=None):
        super().__init__()
        self.setWindowTitle("ROI Classifier")
        self.resize(1600, 900)

        # shared model
        self.model = SharedDataModel()

        self._setup_ui()
        self._setup_menu()

        if plane_dir:
            self.load_data(plane_dir)

    def _setup_ui(self):
        """Create the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # left: diagnostics panel (in scroll area)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(400)

        self.diagnostics = DiagnosticsPanel(self.model)
        scroll.setWidget(self.diagnostics)
        splitter.addWidget(scroll)

        # right: suite2p embedded
        self.suite2p = Suite2pEmbedded(self.model)
        splitter.addWidget(self.suite2p)

        # set initial sizes (25% diagnostics, 75% suite2p)
        splitter.setSizes([400, 1200])

    def _setup_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # file menu
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._on_load)
        file_menu.addAction(load_action)

        save_action = QAction("Save iscell", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # view menu
        view_menu = menubar.addMenu("View")

        reset_zoom = QAction("Reset Zoom", self)
        reset_zoom.setShortcut("Escape")
        reset_zoom.triggered.connect(self._on_reset_zoom)
        view_menu.addAction(reset_zoom)

    def load_data(self, plane_dir):
        """Load Suite2p results.

        Parameters
        ----------
        plane_dir : str or Path
            Path to plane directory.
        """
        plane_dir = Path(plane_dir)
        print(f"[IntegratedClassifierWindow] load_data: {plane_dir}")

        try:
            print("[IntegratedClassifierWindow] calling model.load_data...")
            self.model.load_data(plane_dir)
            print("[IntegratedClassifierWindow] model.load_data returned")
            self.setWindowTitle(f"ROI Classifier - {plane_dir.name}")
            self.statusBar().showMessage(
                f"Loaded {self.model.n_rois} ROIs, {self.model.n_cells} cells", 5000
            )
        except Exception as e:
            print(f"[IntegratedClassifierWindow] error: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Error loading: {e}", 5000)

    def _on_load(self):
        """Handle File > Load."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select plane directory"
        )
        if folder:
            self.load_data(folder)

    def _on_save(self):
        """Handle File > Save."""
        if self.model.save_iscell():
            self.statusBar().showMessage("Saved iscell.npy", 3000)
        else:
            self.statusBar().showMessage("Failed to save", 3000)

    def _on_reset_zoom(self):
        """Reset zoom on both panels."""
        ops = self.model.ops or {}
        Lx = ops.get("Lx", 512)
        Ly = ops.get("Ly", 512)
        self.suite2p.p1.setRange(xRange=[0, Lx], yRange=[0, Ly], padding=0)


def launch_classifier(plane_dir=None):
    """Launch the integrated classifier application.

    Parameters
    ----------
    plane_dir : str or Path, optional
        Path to plane directory to load.

    Returns
    -------
    IntegratedClassifierWindow
        The main window instance.
    """
    from PyQt6.QtWidgets import QApplication
    import sys

    # create application if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = IntegratedClassifierWindow(plane_dir)
    window.show()

    return window


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    plane_dir = sys.argv[1] if len(sys.argv) > 1 else None
    window = IntegratedClassifierWindow(plane_dir)
    window.show()

    sys.exit(app.exec())
