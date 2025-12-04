"""
ROI Diagnostics Widget for Suite2p Results.

Provides comprehensive ROI visualization and diagnostics:
- Full FOV and zoomed ROI views
- Trace plotting with F, Fneu, dF/F
- Quality metrics scatter plots (SNR, Skewness, Activity, Shot Noise)
- ROI navigation controls
"""

import numpy as np
from pathlib import Path
from imgui_bundle import imgui, implot, portable_file_dialogs as pfd

from mbo_utilities.graphics._availability import HAS_SUITE2P
from mbo_utilities.preferences import get_last_dir, set_last_dir


class DiagnosticsWidget:
    """ROI diagnostics viewer for suite2p results."""

    def __init__(self):
        # Data
        self.stat = None
        self.iscell = None
        self.F = None
        self.Fneu = None
        self.spks = None  # Spike data if available
        self.ops = None
        self.loaded_path = None

        # ROI selection
        self.selected_roi = 0
        self.show_all_rois = True
        self.show_only_cells = True

        # Computed metrics
        self._snr = None
        self._dff = None
        self._skewness = None
        self._activity = None
        self._shot_noise = None
        self._spike_rate = None  # Spikes per second
        self._mean_spike_amp = None  # Mean spike amplitude

        # View settings
        self.zoom_padding = 20
        self.trace_window = 500  # frames to show

        # Stats popup state
        self._show_stats_popup = False
        self._stats_popup_open = False

        # Adjustable plot heights (fractions of available space)
        self._trace_height_frac = 0.40  # traces section (40% default)
        self._scatter_height_frac = 0.60  # scatter plots (60% default)
        self._min_plot_height = 80  # minimum height for any plot section

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

        # Load spike data if available
        spks_path = plane_dir / "spks.npy"
        if spks_path.exists():
            self.spks = np.load(spks_path)
        else:
            self.spks = None

        # load_ops expects the path to ops.npy file, not the directory
        ops_path = plane_dir / "ops.npy"
        if ops_path.exists():
            self.ops = load_ops(ops_path)
        else:
            self.ops = None
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

        # Skewness, Activity from stat
        self._skewness = np.zeros(n_rois)
        self._activity = np.zeros(n_rois)
        self._shot_noise = np.zeros(n_rois)

        for i, s in enumerate(self.stat):
            self._skewness[i] = s.get("skew", 0)
            self._activity[i] = np.sum(self._dff[i] > 0.5) / len(self._dff[i])
            # Shot noise approximation
            self._shot_noise[i] = s.get("std", np.std(self.F[i])) if self.F is not None else 0

        # Spike metrics if spks.npy available
        self._spike_rate = np.zeros(n_rois)
        self._mean_spike_amp = np.zeros(n_rois)

        if self.spks is not None:
            # Get frame rate from ops if available
            fs = 30.0  # Default frame rate
            if self.ops is not None:
                fs = self.ops.get("fs", 30.0)

            n_frames = self.spks.shape[1]
            duration_sec = n_frames / fs

            for i in range(n_rois):
                spk_trace = self.spks[i]
                # Count events where spike amplitude > threshold
                threshold = np.std(spk_trace) * 2
                spike_events = spk_trace > threshold
                n_spikes = np.sum(spike_events)
                self._spike_rate[i] = n_spikes / duration_sec if duration_sec > 0 else 0

                # Mean amplitude of spikes
                if n_spikes > 0:
                    self._mean_spike_amp[i] = np.mean(spk_trace[spike_events])
                else:
                    self._mean_spike_amp[i] = 0

    @property
    def n_rois(self):
        """Number of ROIs."""
        return len(self.stat) if self.stat is not None else 0

    @property
    def cell_indices(self):
        """Indices of ROIs classified as cells."""
        if self.iscell is None:
            return np.arange(self.n_rois)
        if self.iscell.ndim == 2:
            return np.where(self.iscell[:, 0] > 0.5)[0]
        return np.where(self.iscell[:] > 0.5)[0]

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
        left_width = min(300, avail.x * 0.3)

        # Left panel - compact controls and FOV
        if imgui.begin_child("DiagLeft", imgui.ImVec2(left_width, 0), imgui.ChildFlags_.borders):
            self._draw_controls()
            imgui.separator()
            self._draw_fov()
        imgui.end_child()

        imgui.same_line()

        # Right panel - traces and scatter plots with adjustable heights
        if imgui.begin_child("DiagRight", imgui.ImVec2(0, 0), imgui.ChildFlags_.borders):
            self._draw_right_panel()
        imgui.end_child()

        # Draw stats popup if open
        self._draw_stats_popup()

    def _draw_load_ui(self):
        """Draw UI for loading results."""
        if imgui.button("Load trace quality statistics"):
            default_dir = str(get_last_dir("suite2p_diagnostics") or Path.home())
            result = pfd.select_folder("Select plane folder with suite2p results", default_dir)
            if result and result.result():
                try:
                    set_last_dir("suite2p_diagnostics", result.result())
                    self.load_results(Path(result.result()))
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), f"Error: {e}")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Load suite2p results (F.npy, stat.npy, iscell.npy) to view\nROI traces, dF/F, SNR, compactness, and other quality metrics.")

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

        # ROI header with cell probability
        imgui.text(f"ROI {roi_idx}")
        if self.iscell is not None:
            prob = self.iscell[roi_idx, 0] if self.iscell.ndim > 1 else self.iscell[roi_idx]
            imgui.same_line()
            color = imgui.ImVec4(0.2, 1.0, 0.2, 1.0) if prob > 0.5 else imgui.ImVec4(1.0, 0.5, 0.2, 1.0)
            imgui.text_colored(color, f"(p={prob:.2f})")

        # Compact key metrics on one line
        imgui.text(f"px:{len(ypix)}")
        if self._snr is not None:
            imgui.same_line()
            imgui.text(f"SNR:{self._snr[roi_idx]:.1f}")
        if self._shot_noise is not None:
            imgui.same_line()
            imgui.text(f"SN:{self._shot_noise[roi_idx]:.0f}")

        # Button to open full stats popup
        if imgui.button("View All Stats..."):
            self._show_stats_popup = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Open detailed statistics popup for this ROI")

        # Draw zoomed view placeholder
        imgui.spacing()
        cy, cx = np.mean(ypix), np.mean(xpix)
        imgui.text_disabled(f"Center: ({cx:.0f}, {cy:.0f})")

    def _draw_right_panel(self):
        """Draw right panel with adjustable trace and scatter plot heights."""
        avail = imgui.get_content_region_avail()
        total_height = avail.y

        # Calculate heights based on fractions
        splitter_height = 8  # height of draggable splitter
        usable_height = total_height - splitter_height

        trace_height = max(self._min_plot_height, usable_height * self._trace_height_frac)
        scatter_height = max(self._min_plot_height, usable_height * self._scatter_height_frac)

        # Draw traces section
        if imgui.begin_child("TracesSection", imgui.ImVec2(-1, trace_height), imgui.ChildFlags_.none):
            self._draw_trace_plot()
        imgui.end_child()

        # Draw draggable splitter
        self._draw_splitter(splitter_height, usable_height)

        # Draw scatter plots section
        if imgui.begin_child("ScatterSection", imgui.ImVec2(-1, 0), imgui.ChildFlags_.none):
            self._draw_scatter_plots()
        imgui.end_child()

    def _draw_splitter(self, height: float, total_height: float):
        """Draw a horizontal draggable splitter between trace and scatter plots."""
        # Splitter bar
        cursor_pos = imgui.get_cursor_screen_pos()
        avail_width = imgui.get_content_region_avail().x

        # Draw splitter visual
        draw_list = imgui.get_window_draw_list()
        splitter_color = imgui.get_color_u32(imgui.Col_.separator)
        splitter_color_hovered = imgui.get_color_u32(imgui.Col_.separator_hovered)
        splitter_color_active = imgui.get_color_u32(imgui.Col_.separator_active)

        # Create invisible button for interaction
        imgui.invisible_button("##splitter", imgui.ImVec2(avail_width, height))
        is_hovered = imgui.is_item_hovered()
        is_active = imgui.is_item_active()

        # Set cursor for resize
        if is_hovered or is_active:
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ns)

        # Draw splitter bar with grip lines
        color = splitter_color_active if is_active else (splitter_color_hovered if is_hovered else splitter_color)
        draw_list.add_rect_filled(
            imgui.ImVec2(cursor_pos.x, cursor_pos.y),
            imgui.ImVec2(cursor_pos.x + avail_width, cursor_pos.y + height),
            color
        )

        # Draw grip lines in center
        center_y = cursor_pos.y + height / 2
        line_color = imgui.get_color_u32(imgui.Col_.text) if (is_hovered or is_active) else imgui.get_color_u32(imgui.Col_.text_disabled)
        for offset in [-8, 0, 8]:
            cx = cursor_pos.x + avail_width / 2 + offset
            draw_list.add_line(
                imgui.ImVec2(cx - 10, center_y),
                imgui.ImVec2(cx + 10, center_y),
                line_color, 1.0
            )

        # Handle dragging
        if is_active:
            delta = imgui.get_io().mouse_delta.y
            if delta != 0:
                # Calculate new fraction
                new_trace_frac = self._trace_height_frac + (delta / total_height)
                # Clamp to reasonable bounds (20% to 80%)
                new_trace_frac = max(0.15, min(0.85, new_trace_frac))
                self._trace_height_frac = new_trace_frac
                self._scatter_height_frac = 1.0 - new_trace_frac

    def _draw_trace_plot(self):
        """Draw fluorescence trace plot with inline metrics."""
        visible = self.visible_indices
        if len(visible) == 0 or self.F is None:
            return

        roi_idx = visible[self.selected_roi]

        # Header line with key metrics
        imgui.text(f"Trace - ROI {roi_idx}")
        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()
        if self._snr is not None:
            imgui.text(f"SNR: {self._snr[roi_idx]:.2f}")
            imgui.same_line()
        if self._shot_noise is not None:
            imgui.text(f"Shot Noise: {self._shot_noise[roi_idx]:.1f}")
            imgui.same_line()
        if self._activity is not None:
            imgui.text(f"Activity: {self._activity[roi_idx]:.1%}")
        # Add spike rate if available
        if self._spike_rate is not None and self.spks is not None:
            imgui.same_line()
            imgui.text(f"| Spk/s: {self._spike_rate[roi_idx]:.2f}")

        # Get trace data
        F_trace = self.F[roi_idx].astype(np.float64)
        n_frames = len(F_trace)
        xs = np.arange(n_frames, dtype=np.float64)

        # Calculate available height - use all space in this child window
        avail = imgui.get_content_region_avail()
        # 2 plots (F/Fneu and dF/F), possibly 3 if spks available
        n_trace_plots = 3 if self.spks is not None else 2
        header_height = 25  # space for header text
        plot_height = max(60, (avail.y - header_height) / n_trace_plots)

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

        # Spike trace if available
        if self.spks is not None:
            if implot.begin_plot(f"##Spks{roi_idx}", imgui.ImVec2(-1, plot_height)):
                implot.setup_axes("Frame", "Spikes")
                spk_trace = self.spks[roi_idx].astype(np.float64)
                implot.plot_line("Spks", xs[:len(spk_trace)], spk_trace)
                implot.end_plot()

    def _find_nearest_point(self, mouse_x: float, mouse_y: float,
                             x_data: np.ndarray, y_data: np.ndarray,
                             x_range: float, y_range: float) -> int | None:
        """Find the nearest data point to mouse position.

        Parameters
        ----------
        mouse_x, mouse_y : float
            Mouse position in plot coordinates
        x_data, y_data : np.ndarray
            Data arrays
        x_range, y_range : float
            Current axis ranges for normalization

        Returns
        -------
        int | None
            Index of nearest point, or None if no point is close enough
        """
        if len(x_data) == 0 or x_range == 0 or y_range == 0:
            return None

        # Normalize distances by axis ranges for fair comparison
        dx = (x_data - mouse_x) / x_range
        dy = (y_data - mouse_y) / y_range
        distances = np.sqrt(dx**2 + dy**2)

        min_idx = np.argmin(distances)
        # Threshold: within 5% of the plot diagonal
        if distances[min_idx] < 0.05:
            return int(min_idx)
        return None

    def _draw_scatter_plots(self):
        """Draw quality metrics scatter plots with click-to-select."""
        if self._snr is None or self.n_rois == 0:
            return

        visible = self.visible_indices
        if len(visible) == 0:
            return

        imgui.text("Quality Metrics (click to select)")
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Scatter plots showing relationships between ROI quality metrics.\n"
                "Click on a point to select that ROI.\n"
                "Hover over points to see values.\n"
                "Scroll to zoom, drag to pan."
            )

        # Get visible metrics
        snr = self._snr[visible].astype(np.float64)
        skew = self._skewness[visible].astype(np.float64)
        activity = self._activity[visible].astype(np.float64)
        shot_noise = self._shot_noise[visible].astype(np.float64)

        # Calculate available height for scatter plots - use all space in this child window
        avail = imgui.get_content_region_avail()
        header_height = 30  # space for header text
        # 3 plots need to fit in available height
        plot_height = max(80, (avail.y - header_height) / 3)
        plot_size = imgui.ImVec2(-1, plot_height)

        # Get cell classification for coloring
        if self.iscell is not None:
            if self.iscell.ndim == 2:
                is_cell = self.iscell[visible, 0] > 0.5
            else:
                is_cell = self.iscell[visible] > 0.5
        else:
            is_cell = np.ones(len(visible), dtype=bool)

        # Split data into cells and non-cells for different colors
        cell_mask = is_cell
        noncell_mask = ~is_cell

        def draw_metric_plot(title: str, tooltip: str, plot_id: str,
                             x_data: np.ndarray, y_data: np.ndarray,
                             x_label: str, y_label: str,
                             sel_x: float, sel_y: float):
            """Draw scatter plot with click selection and hover tooltips."""
            if implot.begin_plot(plot_id, plot_size):
                implot.setup_axes(x_label, y_label)

                # Plot non-cells in gray (smaller markers)
                if np.any(noncell_mask):
                    implot.set_next_marker_style(implot.Marker_.circle, 4)
                    implot.push_style_color(implot.Col_.marker_fill, imgui.ImVec4(0.5, 0.5, 0.5, 0.6))
                    implot.push_style_color(implot.Col_.marker_outline, imgui.ImVec4(0.4, 0.4, 0.4, 0.8))
                    implot.plot_scatter("Non-cells", x_data[noncell_mask], y_data[noncell_mask])
                    implot.pop_style_color(2)

                # Plot cells in blue (normal markers)
                if np.any(cell_mask):
                    implot.set_next_marker_style(implot.Marker_.circle, 5)
                    implot.push_style_color(implot.Col_.marker_fill, imgui.ImVec4(0.2, 0.6, 1.0, 0.7))
                    implot.push_style_color(implot.Col_.marker_outline, imgui.ImVec4(0.1, 0.4, 0.8, 0.9))
                    implot.plot_scatter("Cells", x_data[cell_mask], y_data[cell_mask])
                    implot.pop_style_color(2)

                # Plot selected point (large red marker)
                sel_x_arr = np.array([sel_x], dtype=np.float64)
                sel_y_arr = np.array([sel_y], dtype=np.float64)
                implot.set_next_marker_style(implot.Marker_.diamond, 12)
                implot.push_style_color(implot.Col_.marker_fill, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
                implot.push_style_color(implot.Col_.marker_outline, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
                implot.plot_scatter("Selected", sel_x_arr, sel_y_arr)
                implot.pop_style_color(2)

                # Handle click to select
                if implot.is_plot_hovered():
                    mouse_pos = implot.get_plot_mouse_pos()

                    # Get axis limits for distance normalization
                    x_limits = implot.get_plot_limits().x
                    y_limits = implot.get_plot_limits().y
                    x_range = x_limits.max - x_limits.min
                    y_range = y_limits.max - y_limits.min

                    # Find nearest point for hover tooltip
                    nearest_idx = self._find_nearest_point(
                        mouse_pos.x, mouse_pos.y, x_data, y_data, x_range, y_range
                    )

                    if nearest_idx is not None:
                        # Show hover tooltip
                        roi_global_idx = visible[nearest_idx]
                        prob = 0.0
                        if self.iscell is not None:
                            prob = self.iscell[roi_global_idx, 0] if self.iscell.ndim > 1 else self.iscell[roi_global_idx]
                        imgui.begin_tooltip()
                        imgui.text(f"ROI {roi_global_idx}")
                        imgui.text(f"{x_label}: {x_data[nearest_idx]:.2f}")
                        imgui.text(f"{y_label}: {y_data[nearest_idx]:.2f}")
                        imgui.text(f"Cell prob: {prob:.2f}")
                        imgui.end_tooltip()

                        # Click to select
                        if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                            self.selected_roi = nearest_idx

                implot.end_plot()

        roi_idx = visible[self.selected_roi]

        # SNR vs Skewness
        draw_metric_plot(
            "SNR vs Skewness",
            "Signal-to-Noise Ratio vs Skewness.\n"
            "Skewness: asymmetry of fluorescence distribution.\n"
            "Positive = calcium transients (expected for neurons).",
            "##snr_skew", snr, skew, "SNR", "Skewness",
            self._snr[roi_idx], self._skewness[roi_idx]
        )

        # SNR vs Activity
        draw_metric_plot(
            "SNR vs Activity",
            "Signal-to-Noise Ratio vs Activity.\n"
            "Activity: fraction of frames with dF/F > 0.5.\n"
            "Higher = more frequent calcium events.",
            "##snr_activity", snr, activity, "SNR", "Activity",
            self._snr[roi_idx], self._activity[roi_idx]
        )

        # SNR vs Shot Noise
        draw_metric_plot(
            "SNR vs Shot Noise",
            "Signal-to-Noise Ratio vs Shot Noise.\n"
            "Shot Noise: std of raw fluorescence (from stat.npy).\n"
            "Related to photon counting statistics.",
            "##snr_shotnoise", snr, shot_noise, "SNR", "Shot Noise",
            self._snr[roi_idx], self._shot_noise[roi_idx]
        )

    def _draw_stats_popup(self):
        """Draw comprehensive ROI statistics popup."""
        if self._show_stats_popup:
            self._stats_popup_open = True
            imgui.open_popup("ROI Statistics")
            self._show_stats_popup = False

        # Set popup size
        imgui.set_next_window_size(imgui.ImVec2(450, 500), imgui.Cond_.first_use_ever)

        opened, visible = imgui.begin_popup_modal(
            "ROI Statistics",
            p_open=True if self._stats_popup_open else None,
            flags=imgui.WindowFlags_.no_saved_settings
        )

        if opened:
            if not visible:
                self._stats_popup_open = False
                imgui.close_current_popup()
            else:
                visible_indices = self.visible_indices
                if len(visible_indices) == 0:
                    imgui.text("No ROIs selected")
                else:
                    roi_idx = visible_indices[self.selected_roi]
                    s = self.stat[roi_idx]

                    # Header
                    imgui.text(f"ROI {roi_idx} - Comprehensive Statistics")
                    imgui.separator()

                    # Use columns for organized display
                    if imgui.begin_child("StatsScroll", imgui.ImVec2(0, -35), imgui.ChildFlags_.borders):
                        # Classification
                        if imgui.collapsing_header("Classification", imgui.TreeNodeFlags_.default_open):
                            if self.iscell is not None:
                                prob = self.iscell[roi_idx, 0] if self.iscell.ndim > 1 else self.iscell[roi_idx]
                                is_cell = prob > 0.5
                                color = imgui.ImVec4(0.2, 1.0, 0.2, 1.0) if is_cell else imgui.ImVec4(1.0, 0.5, 0.2, 1.0)
                                imgui.text("Cell Probability:")
                                imgui.same_line()
                                imgui.text_colored(color, f"{prob:.4f}")
                                imgui.text(f"Classification: {'Cell' if is_cell else 'Not Cell'}")

                        # Morphology
                        if imgui.collapsing_header("Morphology", imgui.TreeNodeFlags_.default_open):
                            ypix = s.get("ypix", [])
                            xpix = s.get("xpix", [])
                            imgui.text(f"Pixel Count: {len(ypix)}")
                            if len(ypix) > 0:
                                cy, cx = np.mean(ypix), np.mean(xpix)
                                imgui.text(f"Centroid: ({cx:.1f}, {cy:.1f})")
                            imgui.text(f"Radius: {s.get('radius', 0):.2f}")
                            imgui.text(f"Aspect Ratio: {s.get('aspect_ratio', 0):.2f}")
                            imgui.text(f"Solidity: {s.get('solidity', 0):.3f}")
                            # Footprint extent
                            if len(ypix) > 0 and len(xpix) > 0:
                                imgui.text(f"Footprint: [{min(xpix)}-{max(xpix)}, {min(ypix)}-{max(ypix)}]")

                        # Signal Quality
                        if imgui.collapsing_header("Signal Quality", imgui.TreeNodeFlags_.default_open):
                            if self._snr is not None:
                                imgui.text(f"SNR: {self._snr[roi_idx]:.4f}")
                            if self._shot_noise is not None:
                                imgui.text(f"Shot Noise (std): {self._shot_noise[roi_idx]:.2f}")
                            if self._skewness is not None:
                                imgui.text(f"Skewness: {self._skewness[roi_idx]:.4f}")
                            if self._activity is not None:
                                imgui.text(f"Activity (dF/F>0.5): {self._activity[roi_idx]:.2%}")
                            # Raw trace statistics
                            if self.F is not None:
                                f_trace = self.F[roi_idx]
                                imgui.text(f"F Mean: {np.mean(f_trace):.2f}")
                                imgui.text(f"F Std: {np.std(f_trace):.2f}")
                                imgui.text(f"F Min/Max: {np.min(f_trace):.0f} / {np.max(f_trace):.0f}")
                            # dF/F statistics
                            if self._dff is not None:
                                dff_trace = self._dff[roi_idx]
                                imgui.text(f"dF/F Mean: {np.mean(dff_trace):.4f}")
                                imgui.text(f"dF/F Std: {np.std(dff_trace):.4f}")
                                imgui.text(f"dF/F Min/Max: {np.min(dff_trace):.3f} / {np.max(dff_trace):.3f}")

                        # Spike Characteristics (if spks.npy available)
                        if self.spks is not None:
                            if imgui.collapsing_header("Spiking Activity", imgui.TreeNodeFlags_.default_open):
                                spk_trace = self.spks[roi_idx]
                                imgui.text(f"Spike Rate: {self._spike_rate[roi_idx]:.3f} Hz")
                                imgui.text(f"Mean Spike Amplitude: {self._mean_spike_amp[roi_idx]:.2f}")
                                imgui.text(f"Spike Trace Mean: {np.mean(spk_trace):.4f}")
                                imgui.text(f"Spike Trace Max: {np.max(spk_trace):.4f}")
                                # Count threshold crossings
                                threshold = np.std(spk_trace) * 2
                                n_events = np.sum(spk_trace > threshold)
                                imgui.text(f"Events (>2 std): {n_events}")
                                # Frame rate info
                                if self.ops is not None:
                                    fs = self.ops.get("fs", 30.0)
                                    imgui.text(f"Frame Rate: {fs:.2f} Hz")

                        # Raw stat.npy values
                        if imgui.collapsing_header("Raw stat.npy Values"):
                            # Show all keys from stat dict
                            for key, value in sorted(s.items()):
                                if key in ("ypix", "xpix", "lam"):  # Skip large arrays
                                    imgui.text_disabled(f"{key}: [{len(value)} values]")
                                elif isinstance(value, (int, float, np.integer, np.floating)):
                                    imgui.text(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
                                elif isinstance(value, np.ndarray) and value.size <= 10:
                                    imgui.text(f"{key}: {value}")
                                else:
                                    imgui.text_disabled(f"{key}: [array shape {getattr(value, 'shape', 'N/A')}]")

                    imgui.end_child()

                # Close button
                imgui.spacing()
                if imgui.button("Close", imgui.ImVec2(100, 0)):
                    self._stats_popup_open = False
                    imgui.close_current_popup()

            imgui.end_popup()
