"""
ROI Diagnostics Widget for Suite2p Results.

Provides ROI filtering and visualization:
- dF/F trace with adjustable baseline (median or percentile)
- Metric histograms with threshold sliders for filtering
- Updates iscell based on filter thresholds
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
        self.iscell_original = None  # Keep original for reset
        self.F = None
        self.Fneu = None
        self.ops = None
        self.loaded_path = None

        # ROI selection
        self.selected_roi = 0
        self.show_only_cells = True

        # dF/F baseline settings
        self._baseline_method = 0  # 0 = percentile, 1 = median
        self._baseline_percentile = 10.0
        self._neuropil_coeff = 0.7

        # Computed metrics
        self._snr = None
        self._dff = None
        self._skewness = None
        self._activity = None
        self._shot_noise = None

        # Filter thresholds (min values)
        self._snr_min = 0.0
        self._snr_max = 100.0
        self._shot_noise_min = 0.0
        self._shot_noise_max = 1000.0
        self._skewness_min = -10.0
        self._skewness_max = 10.0
        self._activity_min = 0.0
        self._activity_max = 1.0

        # Current filter values
        self._filter_snr_min = 0.0
        self._filter_shot_noise_max = 1000.0
        self._filter_skewness_min = 0.0
        self._filter_activity_min = 0.0

        # Layout
        self._trace_height_frac = 0.30
        self._hist_height_frac = 0.70
        self._min_plot_height = 60

        # Stats popup
        self._show_stats_popup = False
        self._stats_popup_open = False

    def load_results(self, plane_dir: Path):
        """Load suite2p results from a plane directory."""
        if not HAS_SUITE2P:
            raise ImportError("suite2p or lbm_suite2p_python not available")

        from lbm_suite2p_python import load_planar_results, load_ops

        plane_dir = Path(plane_dir)
        results = load_planar_results(plane_dir)

        self.stat = results.get("stat")
        self.iscell = results.get("iscell")
        if self.iscell is not None:
            self.iscell_original = self.iscell.copy()
        self.F = results.get("F")
        self.Fneu = results.get("Fneu")

        ops_path = plane_dir / "ops.npy"
        if ops_path.exists():
            self.ops = load_ops(ops_path)
        else:
            self.ops = None
        self.loaded_path = plane_dir

        self._compute_metrics()
        self._update_filter_ranges()
        self.selected_roi = 0

    def _compute_metrics(self):
        """Compute quality metrics for all ROIs."""
        if self.F is None or self.stat is None:
            return

        self._recompute_dff()

        n_rois = len(self.F)

        # SNR from dF/F
        noise = np.std(self._dff, axis=1)
        signal = np.max(self._dff, axis=1) - np.min(self._dff, axis=1)
        self._snr = np.where(noise > 0, signal / noise, 0)

        # Skewness, Activity, Shot Noise from stat
        self._skewness = np.zeros(n_rois)
        self._activity = np.zeros(n_rois)
        self._shot_noise = np.zeros(n_rois)

        for i, s in enumerate(self.stat):
            self._skewness[i] = s.get("skew", 0)
            self._activity[i] = np.sum(self._dff[i] > 0.5) / len(self._dff[i])
            self._shot_noise[i] = s.get("std", np.std(self.F[i])) if self.F is not None else 0

    def _recompute_dff(self):
        """Recompute dF/F with current baseline settings."""
        if self.F is None:
            return

        if self.Fneu is not None:
            F_corrected = self.F - self._neuropil_coeff * self.Fneu
        else:
            F_corrected = self.F

        if self._baseline_method == 0:  # percentile
            baseline = np.percentile(F_corrected, self._baseline_percentile, axis=1, keepdims=True)
        else:  # median
            baseline = np.median(F_corrected, axis=1, keepdims=True)

        baseline = np.maximum(baseline, 1.0)
        self._dff = (F_corrected - baseline) / baseline

    def _update_filter_ranges(self):
        """Update filter slider ranges based on data."""
        if self._snr is not None:
            self._snr_min = float(np.min(self._snr))
            self._snr_max = float(np.max(self._snr))
            self._filter_snr_min = self._snr_min
        if self._shot_noise is not None:
            self._shot_noise_min = float(np.min(self._shot_noise))
            self._shot_noise_max = float(np.max(self._shot_noise))
            self._filter_shot_noise_max = self._shot_noise_max
        if self._skewness is not None:
            self._skewness_min = float(np.min(self._skewness))
            self._skewness_max = float(np.max(self._skewness))
            self._filter_skewness_min = self._skewness_min
        if self._activity is not None:
            self._activity_min = float(np.min(self._activity))
            self._activity_max = float(np.max(self._activity))
            self._filter_activity_min = self._activity_min

    def _apply_filters(self):
        """Apply current filter thresholds to iscell."""
        if self.iscell is None or self.iscell_original is None:
            return

        n_rois = len(self.stat)
        for i in range(n_rois):
            passes = True
            if self._snr is not None and self._snr[i] < self._filter_snr_min:
                passes = False
            if self._shot_noise is not None and self._shot_noise[i] > self._filter_shot_noise_max:
                passes = False
            if self._skewness is not None and self._skewness[i] < self._filter_skewness_min:
                passes = False
            if self._activity is not None and self._activity[i] < self._filter_activity_min:
                passes = False

            # Only modify if originally classified as cell
            orig_prob = self.iscell_original[i, 0] if self.iscell_original.ndim > 1 else self.iscell_original[i]
            if orig_prob > 0.5:
                if self.iscell.ndim > 1:
                    self.iscell[i, 0] = orig_prob if passes else 0.0
                else:
                    self.iscell[i] = orig_prob if passes else 0.0

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

        avail = imgui.get_content_region_avail()
        left_width = min(280, avail.x * 0.28)

        # Left panel - controls
        if imgui.begin_child("DiagLeft", imgui.ImVec2(left_width, 0), imgui.ChildFlags_.borders):
            self._draw_controls()
            imgui.separator()
            self._draw_baseline_settings()
        imgui.end_child()

        imgui.same_line()

        # Right panel - trace and histograms
        if imgui.begin_child("DiagRight", imgui.ImVec2(0, 0), imgui.ChildFlags_.borders):
            self._draw_right_panel()
        imgui.end_child()

        self._draw_stats_popup()

    def _draw_load_ui(self):
        """Draw UI for loading results."""
        if imgui.button("Load suite2p results"):
            default_dir = str(get_last_dir("suite2p_diagnostics") or Path.home())
            result = pfd.select_folder("Select plane folder", default_dir)
            if result and result.result():
                try:
                    set_last_dir("suite2p_diagnostics", result.result())
                    self.load_results(Path(result.result()))
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), f"Error: {e}")

    def _draw_controls(self):
        """Draw ROI navigation controls."""
        imgui.text(f"Path: {self.loaded_path.name if self.loaded_path else 'None'}")

        n_cells = len(self.cell_indices)
        n_orig = len(np.where(self.iscell_original[:, 0] > 0.5)[0]) if self.iscell_original is not None and self.iscell_original.ndim > 1 else n_cells
        imgui.text(f"ROIs: {self.n_rois} | Cells: {n_cells}/{n_orig}")

        imgui.spacing()

        visible = self.visible_indices
        if len(visible) > 0:
            if self.selected_roi >= len(visible):
                self.selected_roi = 0

            imgui.set_next_item_width(120)
            changed, new_val = imgui.slider_int("ROI", self.selected_roi, 0, len(visible) - 1)
            if changed:
                self.selected_roi = new_val

            imgui.same_line()
            if imgui.button("<"):
                self.selected_roi = max(0, self.selected_roi - 1)
            imgui.same_line()
            if imgui.button(">"):
                self.selected_roi = min(len(visible) - 1, self.selected_roi + 1)

            # Show current ROI info
            roi_idx = visible[self.selected_roi]
            imgui.text(f"ROI {roi_idx}")
            if self.iscell is not None:
                prob = self.iscell[roi_idx, 0] if self.iscell.ndim > 1 else self.iscell[roi_idx]
                color = imgui.ImVec4(0.2, 1.0, 0.2, 1.0) if prob > 0.5 else imgui.ImVec4(1.0, 0.5, 0.2, 1.0)
                imgui.same_line()
                imgui.text_colored(color, f"p={prob:.2f}")

        imgui.spacing()
        _, self.show_only_cells = imgui.checkbox("Show only cells", self.show_only_cells)

        if imgui.button("View Stats..."):
            self._show_stats_popup = True

        imgui.same_line()
        if imgui.button("Reset Filters"):
            self._update_filter_ranges()
            if self.iscell_original is not None:
                self.iscell = self.iscell_original.copy()

        imgui.same_line()
        if imgui.button("Save iscell"):
            self._save_iscell()

    def _draw_baseline_settings(self):
        """Draw dF/F baseline configuration."""
        imgui.text("dF/F Baseline")

        imgui.set_next_item_width(100)
        changed_method, self._baseline_method = imgui.combo(
            "Method", self._baseline_method, ["Percentile", "Median"]
        )

        if self._baseline_method == 0:
            imgui.set_next_item_width(100)
            changed_pct, self._baseline_percentile = imgui.slider_float(
                "Percentile", self._baseline_percentile, 1.0, 50.0, "%.0f%%"
            )
        else:
            changed_pct = False

        imgui.set_next_item_width(100)
        changed_coeff, self._neuropil_coeff = imgui.slider_float(
            "Fneu coeff", self._neuropil_coeff, 0.0, 1.0, "%.2f"
        )

        if changed_method or changed_pct or changed_coeff:
            self._recompute_dff()
            # Recompute SNR and activity since they depend on dF/F
            if self._dff is not None:
                noise = np.std(self._dff, axis=1)
                signal = np.max(self._dff, axis=1) - np.min(self._dff, axis=1)
                self._snr = np.where(noise > 0, signal / noise, 0)
                for i in range(len(self.stat)):
                    self._activity[i] = np.sum(self._dff[i] > 0.5) / len(self._dff[i])
                self._update_filter_ranges()

    def _draw_right_panel(self):
        """Draw right panel with trace and histograms."""
        avail = imgui.get_content_region_avail()
        trace_height = max(self._min_plot_height, avail.y * self._trace_height_frac)

        # dF/F trace section
        if imgui.begin_child("TraceSection", imgui.ImVec2(-1, trace_height), imgui.ChildFlags_.none):
            self._draw_dff_trace()
        imgui.end_child()

        imgui.separator()

        # Histograms section
        if imgui.begin_child("HistSection", imgui.ImVec2(-1, 0), imgui.ChildFlags_.none):
            self._draw_filter_histograms()
        imgui.end_child()

    def _draw_dff_trace(self):
        """Draw dF/F trace for selected ROI."""
        visible = self.visible_indices
        if len(visible) == 0 or self._dff is None:
            imgui.text("No data")
            return

        roi_idx = visible[self.selected_roi]

        # Header with metrics
        imgui.text(f"dF/F - ROI {roi_idx}")
        imgui.same_line()
        if self._snr is not None:
            imgui.text(f"| SNR:{self._snr[roi_idx]:.1f}")
            imgui.same_line()
        if self._skewness is not None:
            imgui.text(f"Skew:{self._skewness[roi_idx]:.2f}")
            imgui.same_line()
        if self._activity is not None:
            imgui.text(f"Act:{self._activity[roi_idx]:.1%}")

        dff_trace = self._dff[roi_idx].astype(np.float64)
        n_frames = len(dff_trace)
        xs = np.arange(n_frames, dtype=np.float64)

        avail = imgui.get_content_region_avail()
        plot_height = max(60, avail.y - 5)

        if implot.begin_plot("##dFF", imgui.ImVec2(-1, plot_height)):
            implot.setup_axes("Frame", "dF/F")
            implot.plot_line("dF/F", xs, dff_trace)
            implot.end_plot()

    def _draw_filter_histograms(self):
        """Draw histograms with filter sliders."""
        if self._snr is None:
            return

        imgui.text("Filter Thresholds (adjust to modify cell classification)")

        avail = imgui.get_content_region_avail()
        hist_height = max(60, (avail.y - 40) / 4)

        filters_changed = False

        # SNR histogram with min threshold
        filters_changed |= self._draw_histogram_with_slider(
            "SNR", self._snr, hist_height,
            "_filter_snr_min", self._snr_min, self._snr_max,
            is_min_filter=True
        )

        # Shot Noise histogram with max threshold
        filters_changed |= self._draw_histogram_with_slider(
            "Shot Noise", self._shot_noise, hist_height,
            "_filter_shot_noise_max", self._shot_noise_min, self._shot_noise_max,
            is_min_filter=False
        )

        # Skewness histogram with min threshold
        filters_changed |= self._draw_histogram_with_slider(
            "Skewness", self._skewness, hist_height,
            "_filter_skewness_min", self._skewness_min, self._skewness_max,
            is_min_filter=True
        )

        # Activity histogram with min threshold
        filters_changed |= self._draw_histogram_with_slider(
            "Activity %", self._activity * 100, hist_height,
            "_filter_activity_min", self._activity_min * 100, self._activity_max * 100,
            is_min_filter=True, scale=100
        )

        if filters_changed:
            self._apply_filters()

    def _draw_histogram_with_slider(self, label: str, data: np.ndarray, height: float,
                                     filter_attr: str, data_min: float, data_max: float,
                                     is_min_filter: bool, scale: float = 1.0) -> bool:
        """Draw a histogram with an integrated threshold slider.

        Returns True if filter value changed.
        """
        if data is None or len(data) == 0:
            return False

        # Get current filter value
        current_val = getattr(self, filter_attr)
        if scale != 1.0:
            current_val = current_val * scale

        # Compute histogram
        n_bins = 50
        hist_data = data * scale if scale != 1.0 else data
        hist_min = data_min * scale if scale != 1.0 else data_min
        hist_max = data_max * scale if scale != 1.0 else data_max

        if hist_max <= hist_min:
            hist_max = hist_min + 1

        counts, edges = np.histogram(hist_data, bins=n_bins, range=(hist_min, hist_max))
        bin_centers = (edges[:-1] + edges[1:]) / 2

        # Draw plot with histogram and threshold line
        if implot.begin_plot(f"##{label}", imgui.ImVec2(-1, height), implot.Flags_.no_legend):
            implot.setup_axes(label, "Count", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)

            # Draw histogram bars
            bar_width = (hist_max - hist_min) / n_bins * 0.9
            implot.plot_bars(label, bin_centers.astype(np.float64), counts.astype(np.float64), bar_width)

            # Draw threshold line
            threshold_x = np.array([current_val, current_val], dtype=np.float64)
            threshold_y = np.array([0, np.max(counts) * 1.1], dtype=np.float64)
            implot.push_style_color(implot.Col_.line, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
            implot.plot_line("threshold", threshold_x, threshold_y)
            implot.pop_style_color()

            # Draw shaded region for excluded values
            if is_min_filter:
                # Shade left of threshold (excluded)
                shade_x = np.array([hist_min, current_val, current_val, hist_min], dtype=np.float64)
            else:
                # Shade right of threshold (excluded)
                shade_x = np.array([current_val, hist_max, hist_max, current_val], dtype=np.float64)
            shade_y = np.array([0, 0, np.max(counts) * 1.1, np.max(counts) * 1.1], dtype=np.float64)
            implot.push_style_color(implot.Col_.fill, imgui.ImVec4(1.0, 0.3, 0.3, 0.2))
            implot.plot_shaded("excluded", shade_x[:2], shade_y[:2], shade_y[2:4])
            implot.pop_style_color()

            implot.end_plot()

        # Slider for threshold
        imgui.set_next_item_width(-1)
        slider_label = f"Min##{label}" if is_min_filter else f"Max##{label}"
        changed, new_val = imgui.slider_float(slider_label, current_val, hist_min, hist_max, "%.2f")

        if changed:
            if scale != 1.0:
                new_val = new_val / scale
            setattr(self, filter_attr, new_val)
            return True

        return False

    def _save_iscell(self):
        """Save modified iscell.npy to disk."""
        if self.iscell is None or self.loaded_path is None:
            return

        iscell_path = self.loaded_path / "iscell.npy"
        try:
            np.save(iscell_path, self.iscell)
            print(f"Saved iscell to {iscell_path}")
        except Exception as e:
            print(f"Error saving iscell: {e}")

    def _draw_stats_popup(self):
        """Draw ROI statistics popup."""
        if self._show_stats_popup:
            self._stats_popup_open = True
            imgui.open_popup("ROI Statistics")
            self._show_stats_popup = False

        imgui.set_next_window_size(imgui.ImVec2(400, 450), imgui.Cond_.first_use_ever)

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

                    imgui.text(f"ROI {roi_idx}")
                    imgui.separator()

                    if imgui.begin_child("StatsScroll", imgui.ImVec2(0, -35), imgui.ChildFlags_.borders):
                        # Classification
                        if imgui.collapsing_header("Classification", imgui.TreeNodeFlags_.default_open):
                            if self.iscell is not None:
                                prob = self.iscell[roi_idx, 0] if self.iscell.ndim > 1 else self.iscell[roi_idx]
                                orig_prob = self.iscell_original[roi_idx, 0] if self.iscell_original.ndim > 1 else self.iscell_original[roi_idx]
                                color = imgui.ImVec4(0.2, 1.0, 0.2, 1.0) if prob > 0.5 else imgui.ImVec4(1.0, 0.5, 0.2, 1.0)
                                imgui.text("Current probability:")
                                imgui.same_line()
                                imgui.text_colored(color, f"{prob:.4f}")
                                imgui.text(f"Original probability: {orig_prob:.4f}")

                        # Morphology
                        if imgui.collapsing_header("Morphology", imgui.TreeNodeFlags_.default_open):
                            ypix = s.get("ypix", [])
                            xpix = s.get("xpix", [])
                            imgui.text(f"Pixels: {len(ypix)}")
                            if len(ypix) > 0:
                                imgui.text(f"Center: ({np.mean(xpix):.1f}, {np.mean(ypix):.1f})")
                            imgui.text(f"Radius: {s.get('radius', 0):.2f}")
                            imgui.text(f"Aspect ratio: {s.get('aspect_ratio', 0):.2f}")

                        # Signal metrics
                        if imgui.collapsing_header("Signal Metrics", imgui.TreeNodeFlags_.default_open):
                            if self._snr is not None:
                                imgui.text(f"SNR: {self._snr[roi_idx]:.4f}")
                            if self._shot_noise is not None:
                                imgui.text(f"Shot noise: {self._shot_noise[roi_idx]:.2f}")
                            if self._skewness is not None:
                                imgui.text(f"Skewness: {self._skewness[roi_idx]:.4f}")
                            if self._activity is not None:
                                imgui.text(f"Activity: {self._activity[roi_idx]:.2%}")
                            if self._dff is not None:
                                dff = self._dff[roi_idx]
                                imgui.text(f"dF/F range: [{np.min(dff):.3f}, {np.max(dff):.3f}]")
                                imgui.text(f"dF/F std: {np.std(dff):.4f}")

                        # Raw stat values
                        if imgui.collapsing_header("stat.npy values"):
                            for key, value in sorted(s.items()):
                                if key in ("ypix", "xpix", "lam"):
                                    imgui.text_disabled(f"{key}: [{len(value)} values]")
                                elif isinstance(value, (int, float, np.integer, np.floating)):
                                    if isinstance(value, float):
                                        imgui.text(f"{key}: {value:.4f}")
                                    else:
                                        imgui.text(f"{key}: {value}")
                                elif isinstance(value, np.ndarray) and value.size <= 10:
                                    imgui.text(f"{key}: {value}")
                                else:
                                    imgui.text_disabled(f"{key}: [shape {getattr(value, 'shape', 'N/A')}]")

                    imgui.end_child()

                imgui.spacing()
                if imgui.button("Close", imgui.ImVec2(100, 0)):
                    self._stats_popup_open = False
                    imgui.close_current_popup()

            imgui.end_popup()
