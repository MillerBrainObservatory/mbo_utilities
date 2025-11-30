"""
ROI Diagnostics Widget for Suite2p Results.

Provides comprehensive ROI visualization and diagnostics:
- Full FOV and zoomed ROI views
- Trace plotting with F, Fneu, dF/F
- Quality metrics scatter plots (SNR, Compactness, Skewness, Activity, Size)
- ROI navigation controls
"""

import numpy as np
from pathlib import Path
from imgui_bundle import imgui, implot, portable_file_dialogs as pfd

from mbo_utilities.graphics._availability import HAS_SUITE2P


class DiagnosticsWidget:
    """ROI diagnostics viewer for suite2p results."""

    def __init__(self):
        # Data
        self.stat = None
        self.iscell = None
        self.F = None
        self.Fneu = None
        self.ops = None
        self.loaded_path = None

        # ROI selection
        self.selected_roi = 0
        self.show_all_rois = True
        self.show_only_cells = True

        # Computed metrics
        self._snr = None
        self._dff = None
        self._compactness = None
        self._skewness = None
        self._activity = None
        self._shot_noise = None

        # View settings
        self.zoom_padding = 20
        self.trace_window = 500  # frames to show

    def load_results(self, plane_dir: Path):
        """Load suite2p results from a plane directory."""
        if not HAS_SUITE2P:
            raise ImportError("suite2p or lbm_suite2p_python not available")

        from lbm_suite2p_python import load_planar_results, load_ops

        plane_dir = Path(plane_dir)
        results = load_planar_results(plane_dir)

        self.stat = results.get("stat")
        self.iscell = results.get("iscell")
        self.F = results.get("F")
        self.Fneu = results.get("Fneu")
        self.ops = load_ops(plane_dir)
        self.loaded_path = plane_dir

        # Compute metrics
        self._compute_metrics()

        # Reset selection
        self.selected_roi = 0

    def _compute_metrics(self):
        """Compute quality metrics for all ROIs."""
        if self.F is None or self.stat is None:
            return

        n_rois = len(self.F)

        # dF/F
        if self.Fneu is not None:
            F_corrected = self.F - 0.7 * self.Fneu
        else:
            F_corrected = self.F

        baseline = np.percentile(F_corrected, 10, axis=1, keepdims=True)
        baseline = np.maximum(baseline, 1.0)  # Avoid division by zero
        self._dff = (F_corrected - baseline) / baseline

        # SNR
        noise = np.std(self._dff, axis=1)
        signal = np.max(self._dff, axis=1) - np.min(self._dff, axis=1)
        self._snr = np.where(noise > 0, signal / noise, 0)

        # Compactness, Skewness, Activity from stat
        self._compactness = np.zeros(n_rois)
        self._skewness = np.zeros(n_rois)
        self._activity = np.zeros(n_rois)
        self._shot_noise = np.zeros(n_rois)

        for i, s in enumerate(self.stat):
            self._compactness[i] = s.get("compact", 0)
            self._skewness[i] = s.get("skew", 0)
            self._activity[i] = np.sum(self._dff[i] > 0.5) / len(self._dff[i])
            # Shot noise approximation
            self._shot_noise[i] = s.get("std", np.std(self.F[i])) if self.F is not None else 0

    @property
    def n_rois(self):
        """Number of ROIs."""
        return len(self.stat) if self.stat is not None else 0

    @property
    def cell_indices(self):
        """Indices of ROIs classified as cells."""
        if self.iscell is None:
            return np.arange(self.n_rois)
        return np.where(self.iscell[:, 0] > 0.5)[0]

    @property
    def visible_indices(self):
        """Indices of currently visible ROIs."""
        if self.show_only_cells:
            return self.cell_indices
        return np.arange(self.n_rois)

    def draw(self):
        """Draw the diagnostics widget."""
        if self.stat is None:
            self._draw_load_ui()
            return

        # Layout: left panel (controls + FOV), right panel (plots)
        avail = imgui.get_content_region_avail()
        left_width = min(400, avail.x * 0.4)

        # Left panel
        with imgui.begin_child("DiagLeft", imgui.ImVec2(left_width, 0), imgui.ChildFlags_.border):
            self._draw_controls()
            imgui.separator()
            self._draw_fov()

        imgui.same_line()

        # Right panel
        with imgui.begin_child("DiagRight", imgui.ImVec2(0, 0), imgui.ChildFlags_.border):
            self._draw_trace_plot()
            imgui.separator()
            self._draw_scatter_plots()

    def _draw_load_ui(self):
        """Draw UI for loading results."""
        imgui.text("No suite2p results loaded")
        imgui.spacing()

        if imgui.button("Load Results Folder"):
            result = pfd.select_folder("Select plane folder with suite2p results", str(Path.home()))
            if result and result.result():
                try:
                    self.load_results(Path(result.result()))
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), f"Error: {e}")

    def _draw_controls(self):
        """Draw ROI navigation controls."""
        imgui.text(f"Loaded: {self.loaded_path.name if self.loaded_path else 'None'}")
        imgui.text(f"Total ROIs: {self.n_rois}, Cells: {len(self.cell_indices)}")
        imgui.spacing()

        # ROI selection
        visible = self.visible_indices
        if len(visible) > 0:
            # Map selected_roi to visible index
            if self.selected_roi >= len(visible):
                self.selected_roi = 0

            imgui.set_next_item_width(150)
            changed, new_val = imgui.slider_int(
                "ROI", self.selected_roi, 0, len(visible) - 1
            )
            if changed:
                self.selected_roi = new_val

            imgui.same_line()
            if imgui.button("<"):
                self.selected_roi = max(0, self.selected_roi - 1)
            imgui.same_line()
            if imgui.button(">"):
                self.selected_roi = min(len(visible) - 1, self.selected_roi + 1)

        imgui.spacing()
        _, self.show_only_cells = imgui.checkbox("Show only cells", self.show_only_cells)
        _, self.show_all_rois = imgui.checkbox("Show all ROIs in FOV", self.show_all_rois)

    def _draw_fov(self):
        """Draw full FOV and zoomed ROI views."""
        if self.ops is None:
            return

        visible = self.visible_indices
        if len(visible) == 0:
            imgui.text("No ROIs to display")
            return

        roi_idx = visible[self.selected_roi]
        s = self.stat[roi_idx]

        # Get ROI position
        ypix = s.get("ypix", [])
        xpix = s.get("xpix", [])

        if len(ypix) == 0 or len(xpix) == 0:
            imgui.text("ROI has no pixels")
            return

        # Display metrics for selected ROI
        imgui.text(f"ROI {roi_idx}")
        if self.iscell is not None:
            prob = self.iscell[roi_idx, 0] if self.iscell.ndim > 1 else self.iscell[roi_idx]
            imgui.same_line()
            color = imgui.ImVec4(0.2, 1.0, 0.2, 1.0) if prob > 0.5 else imgui.ImVec4(1.0, 0.5, 0.2, 1.0)
            imgui.text_colored(color, f"(p={prob:.2f})")

        imgui.text(f"Size: {len(ypix)} px")
        if self._snr is not None:
            imgui.text(f"SNR: {self._snr[roi_idx]:.2f}")
        if self._compactness is not None:
            imgui.text(f"Compactness: {self._compactness[roi_idx]:.2f}")

        # Draw zoomed view placeholder
        imgui.spacing()
        imgui.text("Zoomed ROI View")
        cy, cx = np.mean(ypix), np.mean(xpix)
        imgui.text(f"Center: ({cx:.0f}, {cy:.0f})")

    def _draw_trace_plot(self):
        """Draw fluorescence trace plot."""
        visible = self.visible_indices
        if len(visible) == 0 or self.F is None:
            return

        roi_idx = visible[self.selected_roi]

        imgui.text(f"Trace - ROI {roi_idx}")

        # Get trace data
        F_trace = self.F[roi_idx].astype(np.float64)
        n_frames = len(F_trace)
        xs = np.arange(n_frames, dtype=np.float64)

        plot_height = 200
        if implot.begin_plot(f"##Trace{roi_idx}", imgui.ImVec2(-1, plot_height)):
            implot.setup_axes("Frame", "Fluorescence")

            # F trace
            implot.plot_line("F", xs, F_trace)

            # Fneu trace
            if self.Fneu is not None:
                Fneu_trace = self.Fneu[roi_idx].astype(np.float64)
                implot.plot_line("Fneu", xs, Fneu_trace)

            implot.end_plot()

        # dF/F trace
        if self._dff is not None:
            if implot.begin_plot(f"##dFF{roi_idx}", imgui.ImVec2(-1, plot_height)):
                implot.setup_axes("Frame", "dF/F")
                dff_trace = self._dff[roi_idx].astype(np.float64)
                implot.plot_line("dF/F", xs, dff_trace)
                implot.end_plot()

    def _draw_scatter_plots(self):
        """Draw quality metrics scatter plots."""
        if self._snr is None or self.n_rois == 0:
            return

        visible = self.visible_indices
        if len(visible) == 0:
            return

        imgui.text("Quality Metrics (visible ROIs)")
        imgui.spacing()

        # Get visible metrics
        snr = self._snr[visible].astype(np.float64)
        compact = self._compactness[visible].astype(np.float64)
        skew = self._skewness[visible].astype(np.float64)
        activity = self._activity[visible].astype(np.float64)

        plot_size = imgui.ImVec2(200, 150)

        # SNR vs Compactness
        if implot.begin_plot("SNR vs Compact", plot_size):
            implot.setup_axes("SNR", "Compactness")
            implot.plot_scatter("##sc1", snr, compact)
            # Highlight selected
            sel_snr = np.array([self._snr[visible[self.selected_roi]]], dtype=np.float64)
            sel_comp = np.array([self._compactness[visible[self.selected_roi]]], dtype=np.float64)
            implot.set_next_marker_style(implot.Marker_.circle, 8)
            implot.plot_scatter("Selected", sel_snr, sel_comp)
            implot.end_plot()

        imgui.same_line()

        # SNR vs Skewness
        if implot.begin_plot("SNR vs Skew", plot_size):
            implot.setup_axes("SNR", "Skewness")
            implot.plot_scatter("##sc2", snr, skew)
            sel_snr = np.array([self._snr[visible[self.selected_roi]]], dtype=np.float64)
            sel_skew = np.array([self._skewness[visible[self.selected_roi]]], dtype=np.float64)
            implot.set_next_marker_style(implot.Marker_.circle, 8)
            implot.plot_scatter("Selected", sel_snr, sel_skew)
            implot.end_plot()

        # SNR vs Activity
        if implot.begin_plot("SNR vs Activity", plot_size):
            implot.setup_axes("SNR", "Activity")
            implot.plot_scatter("##sc3", snr, activity)
            sel_snr = np.array([self._snr[visible[self.selected_roi]]], dtype=np.float64)
            sel_act = np.array([self._activity[visible[self.selected_roi]]], dtype=np.float64)
            implot.set_next_marker_style(implot.Marker_.circle, 8)
            implot.plot_scatter("Selected", sel_snr, sel_act)
            implot.end_plot()
