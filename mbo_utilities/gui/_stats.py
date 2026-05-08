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


# accent color for the currently-active z-plane in tables and plots.
# matches the idle "Run" button green so the highlight is consistent with
# the rest of the GUI's run-state palette.
_ACTIVE_Z_COLOR = (0.13, 0.55, 0.13, 1.00)


def _active_z(parent: Any) -> int | None:
    """Return the 1-based z-index currently displayed by the image widget,
    or None when the dataset is single-plane / has no z slider.

    Z lives at indices[1] for the canonical TZYX 4D layout (T is at index 0).
    """
    iw = getattr(parent, "image_widget", None)
    if iw is None or getattr(iw, "n_sliders", 0) < 2:
        return None
    try:
        return int(iw.indices[1]) + 1
    except (IndexError, TypeError, ValueError):
        return None


def _has_time_dim(arr: Any) -> bool:
    """Check if array has a time dimension."""
    dims = getattr(arr, "dims", None)
    if dims is not None:
        return "t" in dims
    # fallback: assume time dim exists if ndim >= 3 and not a views array
    return arr.ndim >= 3


def _load_subsampled(arr: Any, subsample: int = 10) -> np.ndarray:
    """Load a stats buffer canonicalized to (T_sub, Z, Y, X).

    Uses arr.dims (if present) to identify the T, C, Z, Y, X axes and
    builds an index that subsamples T, pins C=0, and keeps Z/Y/X. Then
    inserts size-1 axes for any missing T or Z so the downstream stats
    loop can always index `[:, z, :, :]` regardless of the input rank.

    Falls back to positional TCZYX assumptions when dims is unavailable.

    Handles every shape the readers produce:
    - (Y, X)             → (1, 1, Y, X)
    - (T, Y, X)          → (T_sub, 1, Y, X)
    - (Z, Y, X)          → (1, Z, Y, X)           ← was crashing
    - (T, Z, Y, X)       → (T_sub, Z, Y, X)
    - (C, Z, Y, X)       → (1, Z, Y, X)           ← squeezed pollen etc.
    - (T, C, Z, Y, X)    → (T_sub, Z, Y, X)
    """
    cache_key = f"_stats_cache_{subsample}"
    if hasattr(arr, cache_key):
        return getattr(arr, cache_key)

    dims_lower = tuple(d.lower() for d in (getattr(arr, "dims", None) or ()))
    shape = arr.shape

    if dims_lower and len(dims_lower) == len(shape):
        # dim-aware: build index per-axis from labels
        idx = []
        for d in dims_lower:
            if d == "t":
                idx.append(slice(None, None, subsample))
            elif d == "c":
                idx.append(0)
            else:
                idx.append(slice(None))
        data = np.asarray(arr[tuple(idx)])
        has_t = "t" in dims_lower
        has_z = "z" in dims_lower
    else:
        # fallback: positional TCZYX assumption by rank
        ndim = len(shape)
        if ndim == 5:
            data = np.asarray(arr[::subsample, 0, :, :, :])
            has_t, has_z = True, True
        elif ndim == 4:
            data = np.asarray(arr[::subsample, :, :, :])
            has_t, has_z = True, True
        elif ndim == 3:
            data = np.asarray(arr[::subsample])
            has_t, has_z = True, False
        else:  # 2D or weird
            data = np.asarray(arr)
            has_t, has_z = False, False

    # canonicalize to (T_sub, Z, Y, X) — remaining dims (post C-pin) are
    # always a subset of (T, Z, Y, X) in that order, so insertions are
    # positional: T at axis 0, Z at axis 1.
    if not has_t:
        data = data[np.newaxis, ...]
    if not has_z:
        data = data[:, np.newaxis, ...]

    try:
        setattr(arr, cache_key, data)
    except (AttributeError, TypeError):
        pass
    return data


def _get_slice_range(parent: Any, arr: Any) -> tuple[list[int], str]:
    """Determine the range of slices to compute stats over.

    Returns (slice_indices, slice_label) where slice_label is 'z' or 'c'.
    For 3D data (TYX), returns ([0], 'z').
    For 4D data with z-planes, iterates over z.
    For 4D data with channels (no z), iterates over channels.
    For piezo arrays, uses num_slices to iterate z-slices within volumes.
    """
    # use parent widget's detected nz/nc which handles both 4D and 5D arrays
    nz = getattr(parent, "nz", None)
    nc = getattr(parent, "nc", None)

    # piezo arrays: enable averaging to get per-z-slice stats
    if hasattr(arr, "num_slices") and hasattr(arr, "average_frames"):
        if not arr.average_frames and arr.can_average:
            arr.average_frames = True
        if nz is not None:
            return list(range(nz)), "z"

    # fallback: detect from shape5d if available, else from parent
    if nz is None:
        nz = arr.nz if hasattr(arr, "nz") else 1
    if nc is None:
        nc = arr.nc if hasattr(arr, "nc") else 1

    if nz > 1:
        return list(range(nz)), "z"
    elif nc > 1:
        return list(range(nc)), "c"

    return [0], "z"


def _read_plane_subsampled(arr: Any, z: int, max_samples: int = 200) -> np.ndarray:
    """Read a single z-plane subsampled in T, capped at `max_samples` frames.

    This is the per-plane equivalent of `_load_subsampled` and is used by
    the stats compute loop so progress reports as each plane finishes
    rather than blocking on one giant bulk read of the whole stack. The
    per-plane budget keeps the network read bounded regardless of how
    many timepoints the recording has — without it, an N-thousand-frame
    LBM dataset on a slow share would pull tens of GB over the wire and
    leave the progress bar pinned at 0% for the duration.

    Returns a (T_sub, Y, X) stack of float32 ready for stats math.
    """
    dims_lower = tuple(d.lower() for d in (getattr(arr, "dims", None) or ()))
    shape = arr.shape
    has_dims = bool(dims_lower) and len(dims_lower) == len(shape)

    # find T axis size to compute the dynamic stride
    if has_dims and "t" in dims_lower:
        t_axis = dims_lower.index("t")
        n_t = int(shape[t_axis])
    elif has_dims:
        n_t = 1
    else:
        # positional fallback: assume axis 0 is T for ndim >= 3
        n_t = int(shape[0]) if len(shape) >= 3 else 1

    # ceiling division so max_samples is a true upper bound. floor div had
    # an off-by-one: for T in [max_samples+1, 2*max_samples-1], stride
    # collapsed to 1 and we read every frame — a 399-frame array pulled
    # 399 samples while a 4000-frame array pulled 200, so smaller data
    # could be slower than larger. ceiling keeps the sample count
    # monotone and bounded at max_samples for every T > 0.
    stride = max(1, (n_t + max(1, max_samples) - 1) // max(1, max_samples))

    if has_dims:
        idx = []
        for d in dims_lower:
            if d == "t":
                idx.append(slice(None, None, stride))
            elif d == "c":
                idx.append(0)
            elif d == "z":
                idx.append(int(z))
            else:
                idx.append(slice(None))
        data = np.asarray(arr[tuple(idx)])
    else:
        # positional 5D TCZYX or 4D TZYX assumption
        ndim = len(shape)
        if ndim == 5:
            data = np.asarray(arr[::stride, 0, int(z), :, :])
        elif ndim == 4:
            data = np.asarray(arr[::stride, int(z), :, :])
        elif ndim == 3:
            data = np.asarray(arr[::stride])
        else:
            data = np.asarray(arr)

    # we may end up with (T, Y, X) or (Y, X) depending on how many indexed
    # dims got squeezed. coerce to 3D so the caller can always axis-0 reduce.
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    return data.astype(np.float32, copy=False)


def compute_zstats_single_array(parent: Any, idx: int, arr: Any):
    """Compute slice-stats (z-plane or channel) for a single array."""
    # phase correction is irrelevant for mean/std/SNR and expensive per
    # chunk under the shared _tiff_lock — turn it off for all reads here.
    arr.fix_phase = False
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
        # Still need to compute mean images for visualization. read per
        # plane so progress reports as each one finishes — same per-plane
        # capped read the compute path uses below.
        means = []
        slice_range, slice_label = _get_slice_range(parent, arr)
        n_slices = len(slice_range)
        # nudge to a non-zero value so the progress bar shows "started"
        # before the first plane finishes (the first read can take a
        # while on a slow share even with the per-plane cap).
        parent._zstats_progress[idx - 1] = 0.01
        for i, s in enumerate(slice_range):
            stack = _read_plane_subsampled(arr, z=s, max_samples=200)
            mean_img = np.mean(stack, axis=0)
            means.append(mean_img)
            parent._zstats_progress[idx - 1] = (i + 1) / n_slices
            parent._zstats_current_z[idx - 1] = s
        means_stack = np.stack(means)
        parent._zstats_means[idx - 1] = means_stack
        parent._zstats_mean_scalar[idx - 1] = means_stack.mean(axis=(1, 2))
        parent._zstats_done[idx - 1] = True
        parent._zstats_running[idx - 1] = False
        parent.logger.info(f"Loaded pre-computed {slice_label}-stats from zarr metadata for array {idx}")
        return

    stats, means = {"mean": [], "std": [], "snr": []}, []

    # determine slice range (z-planes or channels)
    slice_range, slice_label = _get_slice_range(parent, arr)
    n_slices = len(slice_range)
    stats["slice_label"] = slice_label

    # nudge progress so the bar shows we've started before the first
    # plane finishes — on a network share the first read can take tens
    # of seconds and the user otherwise sees a stuck 0%.
    parent._zstats_progress[idx - 1] = 0.01

    # read each z-plane independently with a capped sample count. this
    # used to be a single bulk `_load_subsampled` call that pulled every
    # 10th frame across ALL planes at once — fine for a few-thousand
    # frame LBM file, catastrophic for a 388-file network dataset where
    # the bulk read is tens of GB. per-plane reads keep total bytes
    # bounded (max_samples * n_planes * Y * X) and let the progress
    # callback tick once per plane.
    for i, s in enumerate(slice_range):
        stack = _read_plane_subsampled(arr, z=s, max_samples=200)

        mean_img = np.mean(stack, axis=0)
        std_img = np.std(stack, axis=0)

        p80 = np.percentile(mean_img, 80)
        p50 = np.percentile(mean_img, 50)
        fg = mean_img >= p80
        bg = mean_img <= p50
        fg_mean = float(mean_img[fg].mean()) if fg.any() else 0.0
        bg_mean = float(mean_img[bg].mean()) if bg.any() else 0.0
        bg_std = float(mean_img[bg].std()) if bg.any() else 1.0
        snr_val = (fg_mean - bg_mean) / bg_std if bg_std > 0 else 0.0

        stats["mean"].append(float(np.mean(mean_img)))
        stats["std"].append(float(np.mean(std_img)))
        stats["snr"].append(snr_val)

        means.append(mean_img)
        parent._zstats_progress[idx - 1] = (i + 1) / n_slices
        parent._zstats_current_z[idx - 1] = s

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

    # Update nz from the array's dims tuple. Dims labels are not normalized:
    # some readers emit lowercase ("t", "z", "c"), others uppercase
    # ("T", "C", "Z", "Y", "X" — IsoViewCorrectedArray, ScanImageArray, etc.)
    # The case-sensitive `"z" in dims` test used to fail on uppercase
    # readers, so we'd fall through to `shape[1]` — which on a 5D TCZYX
    # array is the channel axis, not Z. Result: nz silently became nc, the
    # zstats slider ran 0..nc-1 and surfaced "z" stats for what were really
    # the first nc planes. Lowercase both sides before the lookup.
    arr = parent.image_widget.data[0] if parent.image_widget.data else None
    dims = getattr(arr, "dims", None) if arr is not None else None
    dims_lower = tuple(d.lower() for d in dims) if dims else None
    if dims_lower is not None and "z" in dims_lower:
        z_idx = dims_lower.index("z")
        parent.nz = parent.shape[z_idx]
    elif arr is not None and hasattr(arr, "nz"):
        # arrays with shape5d carry a canonical Z size even when dims
        # don't expose a "z" label — prefer that over a positional guess.
        parent.nz = int(arr.nz)
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

    # only draw the selector when there are multiple graphics to choose between
    if len(array_labels) > 1:
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
        # Multi-z-plane: show table and line plot. The currently displayed
        # z-plane (1-based) drives both the table-row highlight and the
        # in-plot accent line / bracketed tick label / inlay annotation.
        active_z = _active_z(parent)
        _draw_zplane_stats_table(
            z_vals, mean_vals, std_vals, snr_vals, array_idx, active_z=active_z
        )
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        if is_combined:
            _draw_combined_zplane_plot(
                parent, z_vals, stats_list, active_z=active_z
            )
        else:
            _draw_zplane_signal_plot(
                z_vals, mean_vals, std_vals, array_idx,
                active_z=active_z, parent=parent,
            )


SNR_TOOLTIP = "SNR = (mean_foreground - mean_background) / std_background, foreground = top 20% brightest pixels, background = bottom 50%"


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
                ("Signal-to-Noise (?)", snr_vals[0], snr_vals[1], "ratio"),
            ]
            for metric_name, val1, val2, unit in metrics:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(metric_name)
                if "(?)" in metric_name and imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text(SNR_TOOLTIP)
                    imgui.end_tooltip()
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
                ("Signal-to-Noise (?)", snr_vals[0], "ratio"),
            ]
            for metric_name, value, unit in metrics:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(metric_name)
                if "(?)" in metric_name and imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text(SNR_TOOLTIP)
                    imgui.end_tooltip()
                imgui.table_next_column()
                imgui.text(f"{value:.2f}")
                imgui.table_next_column()
                imgui.text(unit)
        imgui.end_table()


def _draw_zplane_stats_table(
    z_vals, mean_vals, std_vals, snr_vals, array_idx=None, *, active_z=None
):
    """Draw z-plane stats table for multi-z data.

    When `active_z` matches a row's z value, that row is tinted with the
    accent color and its cells are rendered in the accent for emphasis,
    plus a small marker glyph in the Z column so the active plane reads
    even on a quick glance.
    """
    table_id = f"zstats{array_idx}" if array_idx is not None else "Stats, averaged over graphics"

    if imgui.begin_table(
        table_id,
        4,
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
    ):
        for col in ["Z", "Mean", "Std", "SNR (?)"]:
            imgui.table_setup_column(col, imgui.TableColumnFlags_.width_stretch)
        imgui.table_headers_row()
        # tooltip for SNR header - hover anywhere on header row
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text(SNR_TOOLTIP)
            imgui.end_tooltip()

        # green tint for the active row — green is darker than white so
        # bump alpha to ~25% to keep the highlight readable.
        row_bg = imgui.color_convert_float4_to_u32(
            imgui.ImVec4(_ACTIVE_Z_COLOR[0], _ACTIVE_Z_COLOR[1],
                         _ACTIVE_Z_COLOR[2], 0.25)
        )
        # text in the active row stays pure white. Pushing the green
        # accent here (which is what we did initially) makes the digits
        # disappear into the green tint — white-on-green-tint is the
        # readable combination.
        active_text = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)

        for i in range(len(z_vals)):
            imgui.table_next_row()
            is_active = active_z is not None and int(z_vals[i]) == active_z
            if is_active:
                imgui.table_set_bg_color(
                    imgui.TableBgTarget_.row_bg0, row_bg
                )
                imgui.push_style_color(imgui.Col_.text, active_text)
            # Z renders as a plain integer (planes are 1, 2, 3 — never 1.50);
            # numeric stats keep two-decimal precision.
            imgui.table_next_column()
            imgui.text(f"{int(z_vals[i])}")
            for val in (mean_vals[i], std_vals[i], snr_vals[i]):
                imgui.table_next_column()
                imgui.text(f"{val:.2f}")
            if is_active:
                imgui.pop_style_color()
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


def _draw_combined_zplane_plot(parent, z_vals, stats_list, *, active_z=None):
    """Draw combined z-plane signal plot.

    When `active_z` is provided, an orange vertical line marks that plane
    on the x-axis, the corresponding tick label is rendered in brackets
    ("[N]") so it stands out visually, and an inlay annotation labels the
    active plane.
    """
    imgui.text("Z-plane Signal: Combined")
    set_tooltip(
        "Gray = per-ROI z-profiles (mean over frames)."
        " Blue shade = across-ROI mean ± std; blue line = mean."
        " Orange line = currently displayed plane."
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

    plot_width = imgui.get_content_region_avail().x
    # bold the plot's axis tick labels (and any other in-plot text) when a
    # bold font was loaded — implot doesn't support per-tick font styling,
    # so this is the cleanest way to render the bracketed active label
    # `[N]` in bold along with the rest of the axis.
    bold = getattr(parent, "_bold_font", None)
    pushed_bold = False
    if bold is not None:
        imgui.push_font(bold, bold.legacy_size)
        pushed_bold = True
    if implot.begin_plot(
        "Z-Plane Plot (Combined)",
        imgui.ImVec2(plot_width, 350),
        implot.Flags_.no_legend.value,
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
            # custom ticks so the active plane's label can be bracketed.
            # only swap to custom labels when there are <= 32 z-planes;
            # past that, integer auto-ticks read better than a forced label
            # at every z.
            if len(z) <= 32:
                tick_vals = z.tolist()
                tick_labels = [
                    (f"[{int(v)}]" if active_z is not None and int(v) == active_z
                     else f"{int(v)}")
                    for v in z
                ]
                implot.setup_axis_ticks(
                    implot.ImAxis_.x1.value, tick_vals, tick_labels, False
                )
            else:
                implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")

            # per-ROI traces — slightly more saturated than before so they
            # don't disappear into the dark background.
            for i, ys in enumerate(graphic_series):
                label = f"ROI {i + 1}##roi{i}"
                implot.push_style_var(implot.StyleVar_.line_weight.value, 1)
                implot.push_style_color(
                    implot.Col_.line.value, (0.65, 0.70, 0.78, 0.55)
                )
                implot.plot_line(label, z, ys)
                implot.pop_style_color()
                implot.pop_style_var()

            # shaded mean±std band
            implot.push_style_color(
                implot.Col_.fill.value, (0.30, 0.55, 0.95, 0.28)
            )
            implot.plot_shaded("Mean ± Std##band", z, lower, upper)
            implot.pop_style_color()

            # mean line — heavier and brighter, with circle markers when the
            # plane count is small enough that markers don't clutter.
            implot.push_style_var(implot.StyleVar_.line_weight.value, 2.5)
            implot.push_style_color(
                implot.Col_.line.value, (0.40, 0.75, 1.00, 1.00)
            )
            if len(z) <= 24:
                implot.set_next_marker_style(
                    implot.Marker_.circle.value, 4,
                    imgui.ImVec4(0.40, 0.75, 1.00, 1.00), 1.5,
                    imgui.ImVec4(0.13, 0.15, 0.18, 1.00),
                )
            implot.plot_line("Mean##line", z, mean_vals)
            implot.pop_style_color()
            implot.pop_style_var()

            # accent line + annotation for the active z-plane
            if active_z is not None and z[0] <= active_z <= z[-1]:
                implot.push_style_var(implot.StyleVar_.line_weight.value, 2.0)
                implot.push_style_color(
                    implot.Col_.line.value, _ACTIVE_Z_COLOR
                )
                implot.plot_inf_lines(
                    "Active plane##active_z",
                    np.array([float(active_z)], dtype=np.float64),
                )
                implot.pop_style_color()
                implot.pop_style_var()
                # find y for annotation — clamp to series so the label
                # sits on the curve rather than floating in the void.
                _idx = int(min(max(active_z - 1, 0), len(mean_vals) - 1))
                implot.annotation(
                    float(active_z), float(mean_vals[_idx]),
                    imgui.ImVec4(*_ACTIVE_Z_COLOR),
                    imgui.ImVec2(0, -18), True, f"Z = {active_z}",
                )
        finally:
            implot.end_plot()
    if pushed_bold:
        imgui.pop_font()


def _draw_zplane_signal_plot(
    z_vals, mean_vals, std_vals, array_idx, *, active_z=None, parent=None
):
    """Draw z-plane signal plot with error bars.

    Same active-plane treatment as the combined plot: green vertical
    accent line, bracketed tick label (when <= 32 planes), and an inlay
    annotation tagging "Z = N". When `parent` is provided and a bold
    font is loaded, axis tick labels render in bold so the bracketed
    active label `[N]` reads with extra visual weight.
    """
    style_seaborn_dark()
    imgui.text("Z-plane Signal: Mean ± Std")
    plot_width = imgui.get_content_region_avail().x
    bold = getattr(parent, "_bold_font", None) if parent is not None else None
    pushed_bold = False
    if bold is not None:
        imgui.push_font(bold, bold.legacy_size)
        pushed_bold = True
    if implot.begin_plot(
        f"Z-Plane Signal {array_idx}",
        imgui.ImVec2(plot_width, 350),
        implot.Flags_.no_legend.value,
    ):
        try:
            implot.setup_axes(
                "Z-Plane",
                "Mean Fluorescence",
                implot.AxisFlags_.auto_fit.value,
                implot.AxisFlags_.auto_fit.value,
            )

            z = np.asarray(z_vals, float)
            if len(z) <= 32:
                tick_vals = z.tolist()
                tick_labels = [
                    (f"[{int(v)}]" if active_z is not None and int(v) == active_z
                     else f"{int(v)}")
                    for v in z
                ]
                implot.setup_axis_ticks(
                    implot.ImAxis_.x1.value, tick_vals, tick_labels, False
                )
            else:
                implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")

            implot.plot_error_bars(
                f"Mean ± Std {array_idx}", z_vals, mean_vals, std_vals
            )

            # mean line — match the combined plot palette (sky-blue, heavier)
            # so both plots feel like the same family.
            implot.push_style_var(implot.StyleVar_.line_weight.value, 2.5)
            implot.push_style_color(
                implot.Col_.line.value, (0.40, 0.75, 1.00, 1.00)
            )
            if len(z) <= 24:
                implot.set_next_marker_style(
                    implot.Marker_.circle.value, 4,
                    imgui.ImVec4(0.40, 0.75, 1.00, 1.00), 1.5,
                    imgui.ImVec4(0.13, 0.15, 0.18, 1.00),
                )
            implot.plot_line(f"Mean {array_idx}", z_vals, mean_vals)
            implot.pop_style_color()
            implot.pop_style_var()

            # accent line + annotation for the active z-plane
            if (
                active_z is not None
                and len(z) > 0
                and float(z[0]) <= active_z <= float(z[-1])
            ):
                implot.push_style_var(implot.StyleVar_.line_weight.value, 2.0)
                implot.push_style_color(
                    implot.Col_.line.value, _ACTIVE_Z_COLOR
                )
                implot.plot_inf_lines(
                    f"Active plane##active_z_{array_idx}",
                    np.array([float(active_z)], dtype=np.float64),
                )
                implot.pop_style_color()
                implot.pop_style_var()
                _idx = int(min(max(active_z - 1, 0), len(mean_vals) - 1))
                implot.annotation(
                    float(active_z), float(mean_vals[_idx]),
                    imgui.ImVec4(*_ACTIVE_Z_COLOR),
                    imgui.ImVec2(0, -18), True, f"Z = {active_z}",
                )
        finally:
            implot.end_plot()
    if pushed_bold:
        imgui.pop_font()
