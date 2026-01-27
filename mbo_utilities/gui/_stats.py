"""
Z-stats computation and display for time series data.

This module contains the z-stats computation logic and
the ImPlot-based visualization for signal quality analysis.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from imgui_bundle import imgui, implot

from mbo_utilities.gui._imgui_helpers import set_tooltip, style_seaborn_dark
from mbo_utilities.gui.widgets.progress_bar import reset_progress_state
from mbo_utilities.reader import imread


def _has_time_dim(arr: Any) -> bool:
    """Check if array has a time dimension."""
    dims = getattr(arr, "dims", None)
    if dims is not None:
        return "t" in dims
    # fallback: assume time dim exists if ndim >= 3 and not a views array
    return arr.ndim >= 3


def _get_zslice(arr: Any, t_slice: slice, z: int) -> np.ndarray:
    """Get a z-slice from an array, handling different dimension layouts.

    Uses dims property to correctly index arrays with non-standard layouts
    like IsoviewArray and ClusterPTArray which have a views/camera dimension.

    Returns a stack of frames (T, Y, X) for computing temporal statistics,
    or a single frame (Y, X) expanded to (1, Y, X) for single-timepoint data.

    Supports:
    - 3D: (T, Y, X) -> arr[t_slice]
    - 4D standard: (T, Z, Y, X) -> arr[t_slice, z]
    - 4D single-timepoint views: (Z, Views, Y, X) -> arr[z, 0] expanded to (1, Y, X)
    - 5D with views: (T, Z, Views, Y, X) -> arr[t_slice, z, 0]
    """
    dims = getattr(arr, "dims", None)

    # use dims property to determine correct indexing
    if dims is not None:
        # check if this is a views-type array (isoview, clusterpt)
        if "cm" in dims:
            if "t" in dims:
                # multi-timepoint: (T, Z, Views, Y, X)
                return arr[t_slice, z, 0]
            else:
                # single-timepoint: (Z, Views, Y, X) -> (Y, X)
                # expand to (1, Y, X) for consistent stats computation
                return arr[z, 0][np.newaxis, ...]
        elif "z" in dims:
            # has z but no views
            if "t" in dims:
                return arr[t_slice, z]
            else:
                # single z-slice, expand to (1, Y, X)
                return arr[z][np.newaxis, ...]
        else:
            # no z dimension, just time
            return arr[t_slice]

    # fallback for arrays without dims property
    if arr.ndim == 3:
        return arr[t_slice]
    elif arr.ndim == 4:
        return arr[t_slice, z]
    elif arr.ndim == 5:
        return arr[t_slice, z, 0]
    else:
        return arr[t_slice]


def compute_zstats_single_array(parent: Any, idx: int, arr: Any):
    """Compute z-stats for a single array."""
    # Check for pre-computed stats in zarr metadata (instant loading)
    # supports both 'stats' (new) and 'zstats' (legacy) properties
    pre_stats = None
    if hasattr(arr, "stats") and arr.stats is not None:
        pre_stats = arr.stats
    elif hasattr(arr, "zstats") and arr.zstats is not None:
        pre_stats = arr.zstats
    if pre_stats is not None:
        stats = pre_stats
        parent._zstats[idx - 1] = stats
        # Still need to compute mean images for visualization
        means = []
        tiff_lock = threading.Lock()
        # determine actual z-planes for this array
        z_range = [0] if arr.ndim == 3 else list(range(parent.nz))
        n_z_planes = len(z_range)
        for i, z in enumerate(z_range):
            with tiff_lock:
                stack = _get_zslice(arr, slice(None, None, 10), z)
                mean_img = np.mean(stack, axis=0)
                means.append(mean_img)
                parent._zstats_progress[idx - 1] = (i + 1) / n_z_planes
                parent._zstats_current_z[idx - 1] = z
        means_stack = np.stack(means)
        parent._zstats_means[idx - 1] = means_stack
        parent._zstats_mean_scalar[idx - 1] = means_stack.mean(axis=(1, 2))
        parent._zstats_done[idx - 1] = True
        parent._zstats_running[idx - 1] = False
        parent.logger.info(f"Loaded pre-computed z-stats from zarr metadata for array {idx}")
        return

    stats, means = {"mean": [], "std": [], "snr": []}, []
    tiff_lock = threading.Lock()

    # determine actual z-planes for this array
    z_range = [0] if arr.ndim == 3 else list(range(parent.nz))
    n_z_planes = len(z_range)

    for i, z in enumerate(z_range):
        with tiff_lock:
            stack = _get_zslice(arr, slice(None, None, 10), z).astype(np.float32)

            mean_img = np.mean(stack, axis=0)
            std_img = np.std(stack, axis=0)
            snr_img = np.divide(mean_img, std_img + 1e-5, where=(std_img > 1e-5))

            stats["mean"].append(float(np.mean(mean_img)))
            stats["std"].append(float(np.mean(std_img)))
            stats["snr"].append(float(np.mean(snr_img)))

            means.append(mean_img)
            parent._zstats_progress[idx - 1] = (i + 1) / n_z_planes
            parent._zstats_current_z[idx - 1] = z

    parent._zstats[idx - 1] = stats
    means_stack = np.stack(means)
    parent._zstats_means[idx - 1] = means_stack
    parent._zstats_mean_scalar[idx - 1] = means_stack.mean(axis=(1, 2))
    parent._zstats_done[idx - 1] = True
    parent._zstats_running[idx - 1] = False

    # Save stats to array metadata for persistence (zarr files)
    # prefer 'stats' property but fall back to 'zstats' for backwards compat
    if hasattr(arr, "stats"):
        try:
            arr.stats = stats
            parent.logger.info(f"Saved stats to array {idx} metadata")
        except Exception as e:
            parent.logger.debug(f"Could not save stats to array metadata: {e}")
    elif hasattr(arr, "zstats"):
        try:
            arr.zstats = stats
            parent.logger.info(f"Saved z-stats to array {idx} metadata")
        except Exception as e:
            parent.logger.debug(f"Could not save z-stats to array metadata: {e}")


def compute_zstats(parent: Any):
    """Compute z-stats for all graphics/arrays."""
    if not parent.image_widget or not parent.image_widget.data:
        return

    # Compute z-stats for each graphic (array)
    for idx, arr in enumerate(parent.image_widget.data, start=1):
        threading.Thread(
            target=compute_zstats_single_array,
            args=(parent, idx, arr),
            daemon=True,
        ).start()


def refresh_zstats(parent: Any):
    """
    Reset and recompute z-stats for all arrays.

    This is useful after loading new data or when z-stats need to be
    recalculated (e.g., after changing the number of z-planes).
    """
    if not parent.image_widget:
        return

    # Use num_graphics which matches len(iw.graphics)
    n = parent.num_graphics

    # Reset z-stats state
    parent._zstats = [{"mean": [], "std": [], "snr": []} for _ in range(n)]
    parent._zstats_means = [None] * n
    parent._zstats_mean_scalar = [0.0] * n
    parent._zstats_done = [False] * n
    parent._zstats_running = [False] * n
    parent._zstats_progress = [0.0] * n
    parent._zstats_current_z = [0] * n

    # Reset progress state for each graphic to allow new progress display
    for i in range(n):
        reset_progress_state(f"zstats_{i}")

    # Update nz based on dims property if available, else fallback to shape
    arr = parent.image_widget.data[0] if parent.image_widget.data else None
    dims = getattr(arr, "dims", None) if arr is not None else None
    if dims is not None and "z" in dims:
        z_idx = dims.index("z")
        parent.nz = parent.shape[z_idx]
    elif len(parent.shape) >= 4:
        parent.nz = parent.shape[1]
    elif len(parent.shape) == 3:
        parent.nz = 1
    else:
        parent.nz = 1

    parent.logger.info(f"Refreshing z-stats for {n} arrays, nz={parent.nz}")

    # Mark all as running before starting
    for i in range(n):
        parent._zstats_running[i] = True

    # Recompute z-stats
    compute_zstats(parent)


def draw_stats_section(parent: Any):
    """Draw the z-stats visualization section."""
    if not any(parent._zstats_done):
        return

    stats_list = parent._zstats
    is_single_zplane = parent.nz == 1  # Single bar for 1 plane
    is_dual_zplane = parent.nz == 2    # Grouped bars for 2 planes

    # Different title for single vs multi z-plane
    if is_single_zplane or is_dual_zplane:
        imgui.text_colored(
            imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Signal Quality Summary"
        )
    else:
        imgui.text_colored(
            imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Z-Plane Summary Stats"
        )

    # ROI selector
    array_labels = [
        f"graphic {i + 1}"
        for i in range(len(stats_list))
        if stats_list[i] and "mean" in stats_list[i]
    ]
    # Only show "Combined" if there are multiple arrays
    if len(array_labels) > 1:
        array_labels.append("Combined")

    # Ensure selected array is within bounds
    if parent._selected_array >= len(array_labels):
        parent._selected_array = 0

    avail = imgui.get_content_region_avail().x
    xpos = 0

    for i, label in enumerate(array_labels):
        if imgui.radio_button(label, parent._selected_array == i):
            parent._selected_array = i
        button_width = (
            imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 4
        )
        xpos += button_width + imgui.get_style().item_spacing.x

        if xpos >= avail:
            xpos = button_width
            imgui.new_line()
        else:
            imgui.same_line()

    imgui.separator()

    # Check if "Combined" view is selected (only valid if there are multiple arrays)
    has_combined = len(array_labels) > 1 and array_labels[-1] == "Combined"
    is_combined = has_combined and parent._selected_array == len(array_labels) - 1

    _draw_array_stats(parent, stats_list, is_single_zplane, is_dual_zplane, is_combined)


def _draw_array_stats(
    parent, stats_list, is_single_zplane, is_dual_zplane, is_combined
):
    """Draw stats for selected array or combined view."""
    # Get stats values based on combined or single array mode
    if is_combined:
        imgui.text("Stats for Combined graphics")
        mean_vals = np.mean(
            [np.array(s["mean"]) for s in stats_list if s and "mean" in s], axis=0
        )
        if len(mean_vals) == 0:
            return
        std_vals = np.mean(
            [np.array(s["std"]) for s in stats_list if s and "std" in s], axis=0
        )
        snr_vals = np.mean(
            [np.array(s["snr"]) for s in stats_list if s and "snr" in s], axis=0
        )
        array_idx = None
    else:
        array_idx = parent._selected_array
        stats = stats_list[array_idx]
        if not stats or "mean" not in stats:
            return
        imgui.text(f"Stats for graphic {array_idx + 1}")
        mean_vals = np.array(stats["mean"])
        std_vals = np.array(stats["std"])
        snr_vals = np.array(stats["snr"])
        n = min(len(mean_vals), len(std_vals), len(snr_vals))
        mean_vals, std_vals, snr_vals = mean_vals[:n], std_vals[:n], snr_vals[:n]

    # Convert to contiguous arrays for ImPlot
    z_vals = np.ascontiguousarray(np.arange(1, len(mean_vals) + 1, dtype=np.float64))
    mean_vals = np.ascontiguousarray(mean_vals, dtype=np.float64)
    std_vals = np.ascontiguousarray(std_vals, dtype=np.float64)

    # Draw table and chart based on z-plane count
    if is_single_zplane or is_dual_zplane:
        _draw_simple_stats_table(mean_vals, std_vals, snr_vals, is_dual_zplane, array_idx)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        if is_combined:
            _draw_signal_comparison_chart(parent, mean_vals, is_dual_zplane)
        else:
            _draw_signal_metrics_chart(mean_vals, std_vals, snr_vals, is_dual_zplane, array_idx)
    else:
        # Multi-z-plane: show table and line plot
        _draw_zplane_stats_table(z_vals, mean_vals, std_vals, snr_vals, array_idx)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        if is_combined:
            _draw_combined_zplane_plot(parent, z_vals, stats_list)
        else:
            _draw_zplane_signal_plot(z_vals, mean_vals, std_vals, array_idx)


def _draw_simple_stats_table(mean_vals, std_vals, snr_vals, is_dual_zplane, array_idx=None):
    """Draw simplified stats table for single/dual z-plane."""
    n_cols = 4 if is_dual_zplane else 3
    table_id = f"stats{array_idx}" if array_idx is not None else "Stats (averaged over graphics)"

    if imgui.begin_table(
        table_id,
        n_cols,
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
    ):
        if is_dual_zplane:
            for col in ["Metric", "Z1", "Z2", "Unit"]:
                imgui.table_setup_column(col, imgui.TableColumnFlags_.width_stretch)
        else:
            for col in ["Metric", "Value", "Unit"]:
                imgui.table_setup_column(col, imgui.TableColumnFlags_.width_stretch)
        imgui.table_headers_row()

        if is_dual_zplane:
            metrics = [
                ("Mean Fluorescence", mean_vals[0], mean_vals[1], "a.u."),
                ("Std. Deviation", std_vals[0], std_vals[1], "a.u."),
                ("Signal-to-Noise", snr_vals[0], snr_vals[1], "ratio"),
            ]
            for metric_name, val1, val2, unit in metrics:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(metric_name)
                imgui.table_next_column()
                imgui.text(f"{val1:.2f}")
                imgui.table_next_column()
                imgui.text(f"{val2:.2f}")
                imgui.table_next_column()
                imgui.text(unit)
        else:
            metrics = [
                ("Mean Fluorescence", mean_vals[0], "a.u."),
                ("Std. Deviation", std_vals[0], "a.u."),
                ("Signal-to-Noise", snr_vals[0], "ratio"),
            ]
            for metric_name, value, unit in metrics:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(metric_name)
                imgui.table_next_column()
                imgui.text(f"{value:.2f}")
                imgui.table_next_column()
                imgui.text(unit)
        imgui.end_table()


def _draw_zplane_stats_table(z_vals, mean_vals, std_vals, snr_vals, array_idx=None):
    """Draw z-plane stats table for multi-z data."""
    table_id = f"zstats{array_idx}" if array_idx is not None else "Stats, averaged over graphics"

    if imgui.begin_table(
        table_id,
        4,
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
    ):
        for col in ["Z", "Mean", "Std", "SNR"]:
            imgui.table_setup_column(col, imgui.TableColumnFlags_.width_stretch)
        imgui.table_headers_row()
        for i in range(len(z_vals)):
            imgui.table_next_row()
            for val in (z_vals[i], mean_vals[i], std_vals[i], snr_vals[i]):
                imgui.table_next_column()
                imgui.text(f"{val:.2f}")
        imgui.end_table()


def _draw_signal_comparison_chart(parent, mean_vals, is_dual_zplane):
    """Draw signal comparison bar chart."""
    imgui.text("Signal Quality Comparison")
    set_tooltip(
        "Comparison of mean fluorescence across all graphics"
        + (" and z-planes" if is_dual_zplane else ""),
        True,
    )

    plot_width = imgui.get_content_region_avail().x

    if is_dual_zplane:
        # Grouped bar chart for 2 z-planes
        graphic_means_z1 = [
            np.asarray(parent._zstats[r]["mean"][0], float)
            for r in range(parent.num_graphics)
            if parent._zstats[r] and "mean" in parent._zstats[r] and len(parent._zstats[r]["mean"]) >= 1
        ]
        graphic_means_z2 = [
            np.asarray(parent._zstats[r]["mean"][1], float)
            for r in range(parent.num_graphics)
            if parent._zstats[r] and "mean" in parent._zstats[r] and len(parent._zstats[r]["mean"]) >= 2
        ]

        if graphic_means_z1 and graphic_means_z2 and implot.begin_plot(
            "Signal Comparison", imgui.ImVec2(plot_width, 350)
        ):
            try:
                style_seaborn_dark()
                implot.setup_axes(
                    "Graphic",
                    "Mean Fluorescence (a.u.)",
                    implot.AxisFlags_.none.value,
                    implot.AxisFlags_.auto_fit.value,
                )

                n_graphics = len(graphic_means_z1)
                bar_width = 0.35
                x_pos = np.arange(n_graphics, dtype=np.float64)

                labels = [f"{i + 1}" for i in range(n_graphics)]
                implot.setup_axis_limits(
                    implot.ImAxis_.x1.value, -0.5, n_graphics - 0.5
                )
                implot.setup_axis_ticks(
                    implot.ImAxis_.x1.value, x_pos.tolist(), labels, False
                )

                # Z-plane 1 bars (offset left)
                x_z1 = x_pos - bar_width / 2
                heights_z1 = np.array(graphic_means_z1, dtype=np.float64)
                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                )
                implot.plot_bars("Z-Plane 1", x_z1, heights_z1, bar_width)
                implot.pop_style_color()
                implot.pop_style_var()

                # Z-plane 2 bars (offset right)
                x_z2 = x_pos + bar_width / 2
                heights_z2 = np.array(graphic_means_z2, dtype=np.float64)
                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.9, 0.4, 0.2, 0.8)
                )
                implot.plot_bars("Z-Plane 2", x_z2, heights_z2, bar_width)
                implot.pop_style_color()
                implot.pop_style_var()

            finally:
                implot.end_plot()
    else:
        # Single z-plane: simple bar chart
        graphic_means = [
            np.asarray(parent._zstats[r]["mean"][0], float)
            for r in range(parent.num_graphics)
            if parent._zstats[r] and "mean" in parent._zstats[r]
        ]

        if graphic_means and implot.begin_plot(
            "Signal Comparison", imgui.ImVec2(plot_width, 350)
        ):
            try:
                style_seaborn_dark()
                implot.setup_axes(
                    "Graphic",
                    "Mean Fluorescence (a.u.)",
                    implot.AxisFlags_.none.value,
                    implot.AxisFlags_.auto_fit.value,
                )

                x_pos = np.arange(len(graphic_means), dtype=np.float64)
                heights = np.array(graphic_means, dtype=np.float64)

                labels = [f"{i + 1}" for i in range(len(graphic_means))]
                implot.setup_axis_limits(
                    implot.ImAxis_.x1.value, -0.5, len(graphic_means) - 0.5
                )
                implot.setup_axis_ticks(
                    implot.ImAxis_.x1.value, x_pos.tolist(), labels, False
                )

                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                )
                implot.plot_bars(
                    "Graphic Signal",
                    x_pos,
                    heights,
                    0.6,
                )
                implot.pop_style_color()
                implot.pop_style_var()

                # Add mean line
                mean_line = np.full_like(heights, mean_vals[0])
                implot.push_style_var(implot.StyleVar_.line_weight.value, 2)
                implot.push_style_color(
                    implot.Col_.line.value, (1.0, 0.4, 0.2, 0.8)
                )
                implot.plot_line("Average", x_pos, mean_line)
                implot.pop_style_color()
                implot.pop_style_var()
            finally:
                implot.end_plot()


def _draw_signal_metrics_chart(mean_vals, std_vals, snr_vals, is_dual_zplane, array_idx):
    """Draw signal metrics bar chart for single array."""
    style_seaborn_dark()
    imgui.text("Signal Quality Metrics")
    set_tooltip(
        "Bar chart showing mean fluorescence, standard deviation, and SNR"
        + (" for each z-plane" if is_dual_zplane else ""),
        True,
    )

    plot_width = imgui.get_content_region_avail().x
    if implot.begin_plot(
        f"Signal Metrics {array_idx}", imgui.ImVec2(plot_width, 350)
    ):
        try:
            implot.setup_axes(
                "Metric",
                "Value",
                implot.AxisFlags_.none.value,
                implot.AxisFlags_.auto_fit.value,
            )

            x_pos = np.array([0.0, 1.0, 2.0], dtype=np.float64)
            implot.setup_axis_limits(implot.ImAxis_.x1.value, -0.5, 2.5)
            implot.setup_axis_ticks(
                implot.ImAxis_.x1.value, x_pos.tolist(), ["Mean", "Std Dev", "SNR"], False
            )

            if is_dual_zplane:
                # Grouped bars for Z1 and Z2
                bar_width = 0.35
                x_z1 = x_pos - bar_width / 2
                x_z2 = x_pos + bar_width / 2

                heights_z1 = np.array([mean_vals[0], std_vals[0], snr_vals[0]], dtype=np.float64)
                heights_z2 = np.array([mean_vals[1], std_vals[1], snr_vals[1]], dtype=np.float64)

                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                )
                implot.plot_bars("Z-Plane 1", x_z1, heights_z1, bar_width)
                implot.pop_style_color()
                implot.pop_style_var()

                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.9, 0.4, 0.2, 0.8)
                )
                implot.plot_bars("Z-Plane 2", x_z2, heights_z2, bar_width)
                implot.pop_style_color()
                implot.pop_style_var()
            else:
                # Single bars for single z-plane
                heights = np.array([mean_vals[0], std_vals[0], snr_vals[0]], dtype=np.float64)

                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                )
                implot.plot_bars("Signal Metrics", x_pos, heights, 0.6)
                implot.pop_style_color()
                implot.pop_style_var()
        finally:
            implot.end_plot()


def _draw_combined_zplane_plot(parent, z_vals, stats_list):
    """Draw combined z-plane signal plot."""
    imgui.text("Z-plane Signal: Combined")
    set_tooltip(
        "Gray = per-ROI z-profiles (mean over frames)."
        " Blue shade = across-ROI mean ± std; blue line = mean."
        " Hover gray lines for values.",
        True,
    )

    # build per-graphic series
    graphic_series = [
        np.asarray(parent._zstats[r]["mean"], float)
        for r in range(parent.num_graphics)
    ]

    L = min(len(s) for s in graphic_series)
    z = np.asarray(z_vals[:L], float)
    graphic_series = [s[:L] for s in graphic_series]
    stack = np.vstack(graphic_series)
    mean_vals = stack.mean(axis=0)
    std_vals = stack.std(axis=0)
    lower = mean_vals - std_vals
    upper = mean_vals + std_vals

    # Use available width to prevent cutoff
    plot_width = imgui.get_content_region_avail().x
    if implot.begin_plot(
        "Z-Plane Plot (Combined)", imgui.ImVec2(plot_width, 300)
    ):
        try:
            style_seaborn_dark()
            implot.setup_axes(
                "Z-Plane",
                "Mean Fluorescence",
                implot.AxisFlags_.none.value,
                implot.AxisFlags_.auto_fit.value,
            )

            implot.setup_axis_limits(
                implot.ImAxis_.x1.value, float(z[0]), float(z[-1])
            )
            implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")

            for i, ys in enumerate(graphic_series):
                label = f"ROI {i + 1}##roi{i}"
                implot.push_style_var(implot.StyleVar_.line_weight.value, 1)
                implot.push_style_color(
                    implot.Col_.line.value, (0.6, 0.6, 0.6, 0.35)
                )
                implot.plot_line(label, z, ys)
                implot.pop_style_color()
                implot.pop_style_var()

            implot.push_style_color(
                implot.Col_.fill.value, (0.2, 0.4, 0.8, 0.25)
            )
            implot.plot_shaded("Mean ± Std##band", z, lower, upper)
            implot.pop_style_color()

            implot.push_style_var(implot.StyleVar_.line_weight.value, 2)
            implot.plot_line("Mean##line", z, mean_vals)
            implot.pop_style_var()
        finally:
            implot.end_plot()


def _draw_zplane_signal_plot(z_vals, mean_vals, std_vals, array_idx):
    """Draw z-plane signal plot with error bars."""
    style_seaborn_dark()
    imgui.text("Z-plane Signal: Mean ± Std")
    plot_width = imgui.get_content_region_avail().x
    if implot.begin_plot(
        f"Z-Plane Signal {array_idx}", imgui.ImVec2(plot_width, 300)
    ):
        try:
            implot.setup_axes(
                "Z-Plane",
                "Mean Fluorescence",
                implot.AxisFlags_.auto_fit.value,
                implot.AxisFlags_.auto_fit.value,
            )
            implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")
            implot.plot_error_bars(
                f"Mean ± Std {array_idx}", z_vals, mean_vals, std_vals
            )
            implot.plot_line(f"Mean {array_idx}", z_vals, mean_vals)
        finally:
            implot.end_plot()
