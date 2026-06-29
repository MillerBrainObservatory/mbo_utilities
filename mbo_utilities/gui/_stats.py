"""
Z-stats computation and display for time series data.

This module contains the z-stats computation logic and
the ImPlot-based visualization for signal quality analysis.
"""

from __future__ import annotations

import threading
import time
from itertools import product
from typing import Any

import numpy as np
from imgui_bundle import imgui, implot

from mbo_utilities.arrays.features import canonical_axis, find_slider_name
from mbo_utilities.arrays.features._summary_stats import (
    STATS_SUMMARY_VERSION,
    SummaryStatsSpec,
    build_summary_stats_spec,
    stats_signature,
)
from mbo_utilities.gui._imgui_helpers import set_tooltip, style_seaborn_dark
from mbo_utilities.gui.widgets.progress_bar import reset_progress_state
from mbo_utilities.reader import imread


# accent color for the currently-active z-plane in tables and plots.
# matches the idle "Run" button green so the highlight is consistent with
# the rest of the GUI's run-state palette.
_ACTIVE_Z_COLOR = (0.13, 0.55, 0.13, 1.00)


def _slider_labels(parent: Any) -> dict[str, str]:
    """Live slider labels keyed by canonical axis (for spec display names)."""
    iw = getattr(parent, "image_widget", None)
    names = tuple(getattr(iw, "_slider_dim_names", None) or ()) if iw else ()
    out: dict[str, str] = {}
    for canon in ("Z", "T", "C"):
        nm = find_slider_name(names, canon)
        if nm:
            out[canon] = nm
    return out


def _spec_for(parent: Any, arr: Any) -> SummaryStatsSpec:
    """Resolve the array's `SummaryStatsSpec` for the rendered (squeezed) view.

    The array owns the classification (`summary_stats_dim_role` /
    `summary_stats_metrics`); the GUI supplies the dims/shape it actually
    indexes, the series preference, and the live slider labels. Falls back to
    the default builder if the array predates the hooks.
    """
    dims = tuple(getattr(arr, "dims", None) or ())
    shape = tuple(arr.shape)
    if not dims or len(dims) != len(shape):
        from mbo_utilities.arrays.features._dim_labels import DEFAULT_DIMS
        dims = DEFAULT_DIMS.get(len(shape), tuple("TCZYX"[: len(shape)]))

    # piezo arrays: enable averaging so per-z-slice stats are meaningful
    if hasattr(arr, "num_slices") and hasattr(arr, "average_frames"):
        if not arr.average_frames and getattr(arr, "can_average", False):
            arr.average_frames = True

    pref = str(getattr(parent, "_stats_axis_pref", "z")).lower()
    labels = _slider_labels(parent)
    spec_fn = getattr(arr, "summary_stats_spec", None)
    if callable(spec_fn):
        return spec_fn(dims=dims, shape=shape, series_pref=pref, labels=labels)
    return build_summary_stats_spec(dims, shape, series_pref=pref, labels=labels)


def _read_stat_point(
    arr: Any, spec: SummaryStatsSpec, combo: tuple, s: int
) -> np.ndarray:
    """Read + reduce one (group-combo, series-index) point.

    Indexes the array per ``spec.dims``: the series axis at ``s``, each group
    axis at its ``combo`` value, and any reduce axis strided to at most
    ``budget.avg_max_samples`` frames (mean-reduced by the caller). Y/X are
    spatially binned by ``spec.spatial_bin`` so stored stats / mean-images stay
    small. Returns an ``(N, Yb, Xb)`` float32 stack (``N`` collapses every
    reduce axis; ``N == 1`` when there is none).
    """
    series_axis = spec.series.axis if spec.series else -1
    group_by_axis = {g.axis: int(combo[i]) for i, g in enumerate(spec.groups)}
    reduce_axes = {r.axis for r in spec.reduce}
    avg_max = spec.budget.avg_max_samples

    idx: list = []
    for ax, d in enumerate(spec.dims):
        canon = canonical_axis(d) or str(d).upper()
        if canon in ("Y", "X"):
            idx.append(slice(None))
        elif ax == series_axis:
            idx.append(int(s))
        elif ax in group_by_axis:
            idx.append(group_by_axis[ax])
        elif ax in reduce_axes:
            n = int(spec.shape[ax])
            stride = max(1, (n + avg_max - 1) // avg_max)
            idx.append(slice(None, None, stride))
        else:
            idx.append(0)

    data = np.asarray(arr[tuple(idx)])
    if spec.spatial_bin > 1 and data.ndim >= 2:
        data = data[..., :: spec.spatial_bin, :: spec.spatial_bin]

    # series + group axes are int-indexed (squeezed away); what remains is
    # (reduce axes..., Yb, Xb). Collapse to 3D so the caller can axis-0 reduce.
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim > 3:
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
    return data.astype(np.float32, copy=False)


def compute_zstats_single_array(parent: Any, idx: int, arr: Any):
    """Compute summary stats for one array per its `SummaryStatsSpec`.

    The array owns the layout (`arr.summary_stats_spec`): the *series* axis
    runs along the table rows / plot x-axis; the *group* axes (every other
    scrollable dim — tiles, cameras, views, channels) are each sampled and
    stored separately so the display can follow the sliders; *reduce* axes are
    collapsed per point. Each (group-combo, series-index) point is read
    subsampled + spatially binned to bound memory, then each metric in
    ``spec.metrics`` is applied. Storage per array: ``parent._zstats[idx-1]`` /
    ``_zstats_means[idx-1]`` / ``_zstats_mean_scalar[idx-1]`` are dicts keyed
    by the group-combo tuple (``()`` when there is no group), and
    ``_zstats_spec[idx-1]`` holds the spec used to map sliders back to a combo.
    """
    # phase correction is irrelevant for the metrics and expensive per chunk
    # under the shared _tiff_lock — turn it off for all reads here.
    arr.fix_phase = False
    t_total_start = time.perf_counter()

    spec = _spec_for(parent, arr)
    if not hasattr(parent, "_zstats_spec"):
        parent._zstats_spec = [None] * max(
            idx, int(getattr(parent, "num_graphics", idx))
        )
    parent._zstats_spec[idx - 1] = spec
    parent.logger.debug(f"[zstats] start array={idx} {spec.describe()}")

    parent._zstats[idx - 1] = {}
    parent._zstats_means[idx - 1] = {}
    parent._zstats_mean_scalar[idx - 1] = {}

    series_indices = spec.series.indices if spec.series else [0]
    n_slices = len(series_indices)
    metrics = spec.metrics
    has_reduce = spec.has_reduce
    series_key = spec.series.name.lower() if spec.series else "z"

    # record the 1-based series numbers actually sampled so the table/plot
    # axes stay truthful when the series axis is subsampled.
    if not hasattr(parent, "_zstats_z_indices"):
        parent._zstats_z_indices = [None] * max(
            idx, int(getattr(parent, "num_graphics", idx))
        )
    parent._zstats_z_indices[idx - 1] = [int(s) + 1 for s in series_indices]

    combos = (
        list(product(*[g.indices for g in spec.groups]))
        if spec.groups
        else [()]
    )
    total_steps = max(1, len(combos) * n_slices)
    parent._zstats_progress[idx - 1] = 0.01

    total_read_ms = 0.0
    total_compute_ms = 0.0
    total_bytes = 0
    prev_end = time.perf_counter()
    step = 0

    for combo in combos:
        stats: dict = {m.key: [] for m in metrics}
        stats["slice_label"] = series_key
        means: list[np.ndarray] = []
        for s in series_indices:
            t_iter_start = time.perf_counter()
            stack = _read_stat_point(arr, spec, combo, s)
            t_after_read = time.perf_counter()

            mean_img = np.mean(stack, axis=0)
            for m in metrics:
                stats[m.key].append(m.reducer(stack, mean_img, has_reduce))
            means.append(mean_img)

            step += 1
            parent._zstats_progress[idx - 1] = step / total_steps
            parent._zstats_current_z[idx - 1] = s

            t_after_compute = time.perf_counter()
            total_read_ms += (t_after_read - t_iter_start) * 1000.0
            total_compute_ms += (t_after_compute - t_after_read) * 1000.0
            total_bytes += int(stack.nbytes)
            prev_end = t_after_compute

        parent._zstats[idx - 1][combo] = stats
        means_stack = np.stack(means)
        parent._zstats_means[idx - 1][combo] = means_stack
        parent._zstats_mean_scalar[idx - 1][combo] = means_stack.mean(axis=(1, 2))

    parent._zstats_done[idx - 1] = True
    parent._zstats_running[idx - 1] = False
    parent.logger.debug(
        f"[zstats] done array={idx} "
        f"total={(time.perf_counter() - t_total_start) * 1000:.0f}ms "
        f"reads={total_read_ms:.0f}ms compute={total_compute_ms:.0f}ms "
        f"bytes={total_bytes / 1e6:.1f}MB combos={len(combos)} "
        f"slices={n_slices}"
    )

    # Persist the full per-combo stats (+ mean stack) to the backing zarr
    # store so the next open hydrates instead of recomputing.
    _persist_stats(parent, idx, arr, spec)


def _base_array(arr: Any) -> Any:
    """Unwrap GUI view wrappers (Squeezed/Axial/Phasecorr) to the disk array.

    These wrappers hold the source on ``_base`` (SqueezedView) or ``_source``
    (AxialShiftView / PhaseCorrectedView) and do not forward underscore
    attributes, so reach the innermost array to call its persistence hooks.
    """
    cur = arr
    for _ in range(8):
        nxt = getattr(cur, "_base", None) or getattr(cur, "_source", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def _persist_stats(parent: Any, idx: int, arr: Any, spec: SummaryStatsSpec) -> None:
    """Write the array's computed stats (+ binned mean stack) to its store."""
    base = _base_array(arr)
    save = getattr(base, "save_summary_stats", None)
    if not callable(save):
        return
    zstats = parent._zstats[idx - 1]
    if not zstats:
        return
    combos = list(zstats.keys())
    payload = {
        "version": STATS_SUMMARY_VERSION,
        **stats_signature(
            spec.dims, spec.shape,
            spec.series.name if spec.series else None,
            [m.key for m in spec.metrics],
        ),
        "series_pref": str(getattr(parent, "_stats_axis_pref", "z")).lower(),
        "series_indices": [int(s) for s in (parent._zstats_z_indices[idx - 1] or [])],
        "spatial_bin": int(spec.spatial_bin),
        "combos": [[int(v) for v in c] for c in combos],
        "stats": [zstats[c] for c in combos],
    }
    means = None
    means_map = parent._zstats_means[idx - 1]
    try:
        stacks = [means_map.get(c) for c in combos]
        if stacks and all(s is not None for s in stacks):
            means = np.stack([np.asarray(s, dtype=np.float32) for s in stacks])
    except Exception:
        means = None
    try:
        if save(payload, means):
            parent.logger.debug(f"[zstats] persisted array={idx} ({len(combos)} combos)")
    except Exception as e:
        parent.logger.debug(f"[zstats] persist array={idx} failed: {e}")


def _hydrate_one(parent: Any, idx: int, arr: Any) -> bool:
    """Load cached stats for one array and populate parent state. Returns True
    when the cache matched the current dims/shape/series and was applied."""
    base = _base_array(arr)
    loader = getattr(base, "load_summary_stats", None)
    if not callable(loader):
        return False
    loaded = loader()
    if not loaded:
        return False
    payload, means = loaded

    spec = _spec_for(parent, arr)
    sig = stats_signature(
        spec.dims, spec.shape,
        spec.series.name if spec.series else None,
        [m.key for m in spec.metrics],
    )
    if any(payload.get(k) != sig[k] for k in ("dims", "shape", "series", "metrics")):
        return False

    combos = [tuple(int(v) for v in c) for c in payload.get("combos", [])]
    stats_list = payload.get("stats", [])
    if len(combos) != len(stats_list):
        return False

    parent._zstats_spec[idx - 1] = spec
    parent._zstats[idx - 1] = {c: stats_list[k] for k, c in enumerate(combos)}
    parent._zstats_z_indices[idx - 1] = [int(s) for s in payload.get("series_indices", [])]

    mean_map: dict = {}
    scalar_map: dict = {}
    if means is not None and len(means) == len(combos):
        for k, c in enumerate(combos):
            mimg = np.asarray(means[k], dtype=np.float32)
            mean_map[c] = mimg
            scalar_map[c] = mimg.mean(axis=(1, 2))
    parent._zstats_means[idx - 1] = mean_map
    parent._zstats_mean_scalar[idx - 1] = scalar_map

    parent._zstats_done[idx - 1] = True
    parent._zstats_running[idx - 1] = False
    parent._zstats_progress[idx - 1] = 1.0
    parent.logger.debug(f"[zstats] hydrated array={idx} from store · {spec.describe()}")
    return True


def hydrate_zstats(parent: Any) -> list[bool]:
    """Populate stats from each array's cached store. Returns a per-array
    ``hydrated`` flag list; arrays that returned False must be computed."""
    n = parent.num_graphics
    out = [False] * n
    if not parent.image_widget or not parent.image_widget.data:
        return out
    for i, arr in enumerate(parent.image_widget.data):
        if i >= n:
            break
        try:
            out[i] = _hydrate_one(parent, i + 1, arr)
        except Exception as e:
            parent.logger.debug(f"[zstats] hydrate array={i + 1} failed: {e}")
    return out


def compute_zstats(parent: Any, only: list[int] | None = None):
    """Compute z-stats for all graphics/arrays (or only the given 0-based indices)."""
    if not parent.image_widget or not parent.image_widget.data:
        return

    # Compute z-stats for each graphic (array)
    for idx, arr in enumerate(parent.image_widget.data, start=1):
        if only is not None and (idx - 1) not in only:
            continue
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

    # Reset z-stats state. Per-array slots are dicts keyed by the group-combo
    # tuple (``()`` when there is no group), populated by
    # `compute_zstats_single_array`; ``_zstats_spec[i]`` holds that array's
    # `SummaryStatsSpec` so the display can map sliders back to a combo.
    parent._zstats = [{} for _ in range(n)]
    parent._zstats_means = [{} for _ in range(n)]
    parent._zstats_mean_scalar = [{} for _ in range(n)]
    parent._zstats_spec = [None] * n
    parent._zstats_done = [False] * n
    parent._zstats_running = [False] * n
    parent._zstats_progress = [0.0] * n
    parent._zstats_current_z = [0] * n
    parent._zstats_z_indices = [None] * n

    # Reset progress state for each graphic to allow new progress display
    for i in range(n):
        reset_progress_state(f"zstats_{i}")

    # Update nz from the array's dims tuple. Dims labels are not normalized:
    # some readers emit lowercase ("t", "z", "c"), others uppercase
    # ("T", "C", "Z", "Y", "X" — IsoviewArray, ScanImageArray, etc.)
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
        # lazy arrays carry a canonical Z size (.nz) even when dims
        # don't expose a "z" label — prefer that over a positional guess.
        parent.nz = int(arr.nz)
    elif len(parent.shape) >= 4:
        parent.nz = parent.shape[1]
    elif len(parent.shape) == 3:
        parent.nz = 1
    else:
        parent.nz = 1

    parent.logger.debug(f"Refreshing z-stats for {n} arrays, nz={parent.nz}")

    # Mark all as running before starting
    for i in range(n):
        parent._zstats_running[i] = True

    # Recompute z-stats
    compute_zstats(parent)


def _ref_spec(parent: Any) -> SummaryStatsSpec | None:
    """First populated `SummaryStatsSpec` across graphics (labels / radios)."""
    specs = getattr(parent, "_zstats_spec", None)
    if not specs:
        return None
    for s in specs:
        if s is not None:
            return s
    return None


def current_breakout_key(parent: Any, idx: int) -> tuple:
    """Nearest sampled group combo for graphic ``idx`` from the sliders.

    Reads each group axis's current slider position and snaps it to the
    nearest sampled index, so the table/plot follow the scrollable dims even
    though only a strided subset of tiles/cameras was computed. Returns
    ``()`` when the array has no group axes.
    """
    specs = getattr(parent, "_zstats_spec", None)
    spec = specs[idx] if specs and idx < len(specs) else None
    if spec is None or not spec.groups:
        return ()
    iw = getattr(parent, "image_widget", None)
    names = tuple(getattr(iw, "_slider_dim_names", None) or ()) if iw else ()
    key: list[int] = []
    for g in spec.groups:
        v = 0
        name = find_slider_name(names, g.name)
        if iw is not None and name:
            try:
                v = int(iw.indices[name])
            except (KeyError, IndexError, TypeError, ValueError):
                v = 0
        key.append(min(g.indices, key=lambda k: abs(k - v)))
    return tuple(key)


def _series_for(parent: Any, i: int) -> tuple[dict | None, tuple | None]:
    """(stats_dict, combo_key) for graphic ``i`` at the current sliders.

    Falls back to the first computed combo if the slider-derived combo is not
    populated yet. Returns ``(None, None)`` when nothing is available.
    """
    slot = parent._zstats[i] if i < len(parent._zstats) else None
    if not isinstance(slot, dict) or not slot:
        return None, None
    key = current_breakout_key(parent, i)
    stats = slot.get(key)
    if stats is None:
        key = next(iter(slot))
        stats = slot.get(key)
    if not stats or "mean" not in stats:
        return None, None
    return stats, key


def _combo_caption(spec: SummaryStatsSpec | None, key: tuple) -> str:
    """Compact "Tile 3, Cam 1" label for the displayed group combo."""
    if not spec or not spec.groups or not key:
        return ""
    return ", ".join(
        f"{g.label} {int(v) + 1}" for g, v in zip(spec.groups, key)
    )


def _draw_axis_pick(parent: Any, spec: SummaryStatsSpec) -> None:
    """Radio to pick which scrollable dim drives the series (e.g. Zplane vs Timepoint)."""
    pref = str(getattr(parent, "_stats_axis_pref", "z")).lower()
    cur = spec.series.name.lower() if spec.series else pref
    imgui.text("Series axis:")
    changed = False
    for cand in spec.series_candidates:
        imgui.same_line()
        sel = cand.name.lower() == cur
        if imgui.radio_button(cand.label, sel) and not sel:
            parent._stats_axis_pref = cand.name.lower()
            changed = True
    if changed:
        parent.refresh_zstats()
    imgui.separator()


def draw_stats_section(parent: Any):
    """Draw the summary-stats visualization section."""
    if not any(parent._zstats_done):
        return

    stats_list = parent._zstats
    spec = _ref_spec(parent)
    n_stat = (
        len(spec.series.indices) if spec is not None and spec.series
        else int(getattr(parent, "nz", 1))
    )
    stat_label = spec.series.label if spec is not None and spec.series else "Z-Plane"
    is_single_zplane = n_stat == 1  # Single bar for 1 series point
    is_dual_zplane = n_stat == 2    # Grouped bars for 2 series points

    if is_single_zplane or is_dual_zplane:
        imgui.text_colored(
            imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Signal Quality Summary"
        )
    else:
        imgui.text_colored(
            imgui.ImVec4(0.8, 1.0, 0.2, 1.0), f"{stat_label} Summary Stats"
        )

    # more than one series candidate -> let the user pick which drives the x-axis
    if spec is not None and spec.both_series:
        _draw_axis_pick(parent, spec)

    # ROI selector — show a graphic only when it has a populated series.
    array_labels = [
        f"graphic {i + 1}"
        for i in range(len(stats_list))
        if _series_for(parent, i)[0] is not None
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

    # Caption the displayed group combo (it follows the sliders, snapped to
    # the nearest sampled tile/camera/etc.).
    if spec is not None and spec.groups:
        sel = 0 if parent._selected_array >= len(stats_list) else parent._selected_array
        _, key = _series_for(parent, sel)
        caption = _combo_caption(spec, key or ())
        if caption:
            imgui.text_disabled(caption)

    # Check if "Combined" view is selected (only valid if there are multiple arrays)
    has_combined = len(array_labels) > 1 and array_labels[-1] == "Combined"
    is_combined = has_combined and parent._selected_array == len(array_labels) - 1

    _draw_array_stats(
        parent, stats_list, spec, is_single_zplane, is_dual_zplane, is_combined
    )


def _z_axis_values(parent: Any, array_idx: int | None, n: int) -> np.ndarray:
    """1-based plane numbers for the n stats points, honoring z-subsampling.

    Uses the sampled plane numbers recorded by ``compute_zstats_single_array``
    so a subsampled z axis shows the real (evenly-strided) plane numbers.
    Falls back to a contiguous 1..n when no record matches (channel-stats
    mode, or a length mismatch).
    """
    idxs = getattr(parent, "_zstats_z_indices", None)
    picked = None
    if idxs:
        if array_idx is not None and 0 <= array_idx < len(idxs) and idxs[array_idx]:
            picked = idxs[array_idx]
        else:
            picked = next((v for v in idxs if v), None)
    if picked is not None and len(picked) >= n:
        return np.ascontiguousarray(np.asarray(picked[:n], dtype=np.float64))
    return np.ascontiguousarray(np.arange(1, n + 1, dtype=np.float64))


def _active_stat(parent: Any, spec: SummaryStatsSpec | None) -> int | None:
    """1-based current position along the series axis (from its slider).

    For a zplane series it follows the Z slider, for a timepoint series the
    T slider, etc. Returns None when the series axis has no slider (single
    point) or the lookup fails.
    """
    if spec is None or spec.series is None:
        return None
    iw = getattr(parent, "image_widget", None)
    if iw is None or getattr(iw, "n_sliders", 0) < 1:
        return None
    names = tuple(getattr(iw, "_slider_dim_names", None) or ())
    name = find_slider_name(names, spec.series.name)
    if name is None:
        return None
    try:
        return int(iw.indices[name]) + 1
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _combined_stats(parent, metrics) -> dict | None:
    """Average each metric series across graphics at their current combos."""
    per = [
        s for i in range(len(parent._zstats))
        for s in (_series_for(parent, i)[0],) if s is not None
    ]
    if not per:
        return None
    out: dict = {}
    for m in metrics:
        arrs = [np.asarray(s[m.key], float) for s in per if m.key in s]
        if not arrs:
            continue
        L = min(len(a) for a in arrs)
        out[m.key] = np.mean([a[:L] for a in arrs], axis=0)
    return out or None


def _draw_array_stats(
    parent, stats_list, spec, is_single_zplane, is_dual_zplane, is_combined
):
    """Draw stats for the selected array (or combined) at the current combo."""
    from mbo_utilities.arrays.features import DEFAULT_METRICS
    metrics = spec.metrics if spec is not None else DEFAULT_METRICS
    stat_label = spec.series.label if spec is not None and spec.series else "Z-Plane"

    if is_combined:
        imgui.text("Stats for Combined graphics")
        stats = _combined_stats(parent, metrics)
        array_idx = None
    else:
        array_idx = parent._selected_array
        stats, _ = _series_for(parent, array_idx)
    if not stats or "mean" not in stats:
        return

    mean_vals = np.asarray(stats.get("mean", []), dtype=np.float64)
    n = len(mean_vals)
    if n == 0:
        return
    std_vals = np.ascontiguousarray(
        np.asarray(stats.get("std", np.zeros(n)), dtype=np.float64)[:n]
    )
    # series x positions: real (possibly subsampled) 1-based numbers.
    z_vals = _z_axis_values(parent, array_idx, n)
    mean_vals = np.ascontiguousarray(mean_vals[:n])

    if is_single_zplane or is_dual_zplane:
        _draw_simple_stats_table(stats, metrics, is_dual_zplane, array_idx, stat_label)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        if is_combined:
            _draw_signal_comparison_chart(parent, mean_vals, is_dual_zplane, stat_label)
        else:
            snr_vals = np.asarray(stats.get("snr", np.zeros(n)), dtype=np.float64)[:n]
            _draw_signal_metrics_chart(
                mean_vals, std_vals, snr_vals, is_dual_zplane, array_idx, stat_label
            )
    else:
        # Multi-point series: the current series position (1-based) drives the
        # table-row highlight and the in-plot accent line.
        active_z = _active_stat(parent, spec)
        _draw_zplane_stats_table(
            z_vals, stats, metrics, array_idx, active_z=active_z, stat_label=stat_label
        )
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        if is_combined:
            _draw_combined_zplane_plot(
                parent, z_vals, stats_list, active_z=active_z, stat_label=stat_label
            )
        else:
            _draw_zplane_signal_plot(
                z_vals, mean_vals, std_vals, array_idx,
                active_z=active_z, parent=parent, stat_label=stat_label,
            )


def _draw_simple_stats_table(stats, metrics, is_dual_zplane, array_idx=None, stat_label="Z-Plane"):
    """Draw the metric table for a single/dual-point series (one row per metric)."""
    n_cols = 4 if is_dual_zplane else 3
    table_id = f"stats{array_idx}" if array_idx is not None else "Stats (averaged over graphics)"
    short = stat_label[:1].upper() or "Z"

    if imgui.begin_table(
        table_id,
        n_cols,
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
    ):
        cols = (
            ["Metric", f"{short}1", f"{short}2", "Unit"]
            if is_dual_zplane else ["Metric", "Value", "Unit"]
        )
        for col in cols:
            imgui.table_setup_column(col, imgui.TableColumnFlags_.width_stretch)
        imgui.table_headers_row()

        for m in metrics:
            vals = np.asarray(stats.get(m.key, []), dtype=np.float64)
            if vals.size == 0:
                continue
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(m.label + (" (?)" if m.tooltip else ""))
            if m.tooltip and imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.text(m.tooltip)
                imgui.end_tooltip()
            imgui.table_next_column()
            imgui.text(f"{vals[0]:.2f}")
            if is_dual_zplane:
                imgui.table_next_column()
                imgui.text(f"{vals[1]:.2f}" if vals.size > 1 else "—")
            imgui.table_next_column()
            imgui.text(m.unit)
        imgui.end_table()


def _draw_zplane_stats_table(
    z_vals, stats, metrics, array_idx=None, *, active_z=None, stat_label="Z-Plane"
):
    """Draw the per-series-point metric table (one column per metric).

    When `active_z` matches a row's series value, that row is tinted with the
    accent color so the active series point reads at a glance.
    """
    table_id = f"zstats{array_idx}" if array_idx is not None else "Stats, averaged over graphics"
    short = stat_label[:1].upper() or "Z"
    cols = [m for m in metrics if np.asarray(stats.get(m.key, []), float).size]
    tips = "  ".join(m.tooltip for m in cols if m.tooltip)

    if imgui.begin_table(
        table_id,
        1 + len(cols),
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
    ):
        imgui.table_setup_column(short, imgui.TableColumnFlags_.width_stretch)
        for m in cols:
            header = m.header + (" (?)" if m.tooltip else "")
            imgui.table_setup_column(header, imgui.TableColumnFlags_.width_stretch)
        imgui.table_headers_row()
        if tips and imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text(tips)
            imgui.end_tooltip()

        # green tint for the active row, white text on the tint stays readable.
        row_bg = imgui.color_convert_float4_to_u32(
            imgui.ImVec4(_ACTIVE_Z_COLOR[0], _ACTIVE_Z_COLOR[1],
                         _ACTIVE_Z_COLOR[2], 0.25)
        )
        active_text = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
        series = [np.asarray(stats[m.key], dtype=np.float64) for m in cols]

        for i in range(len(z_vals)):
            imgui.table_next_row()
            is_active = active_z is not None and int(z_vals[i]) == active_z
            if is_active:
                imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, row_bg)
                imgui.push_style_color(imgui.Col_.text, active_text)
            imgui.table_next_column()
            imgui.text(f"{int(z_vals[i])}")
            for arr in series:
                imgui.table_next_column()
                imgui.text(f"{arr[i]:.2f}" if i < len(arr) else "—")
            if is_active:
                imgui.pop_style_color()
        imgui.end_table()


def _draw_signal_comparison_chart(parent, mean_vals, is_dual_zplane, stat_label="Z-Plane"):
    """Draw signal comparison bar chart across graphics at the current combo."""
    short = stat_label[:1].upper() or "Z"
    imgui.text("Signal Quality Comparison")
    set_tooltip(
        "Comparison of mean fluorescence across all graphics"
        + (f" and {stat_label.lower()}s" if is_dual_zplane else ""),
        True,
    )

    plot_width = imgui.get_content_region_avail().x

    if is_dual_zplane:
        # Grouped bar chart for 2 series points
        per_r = [_series_for(parent, r)[0] for r in range(parent.num_graphics)]
        graphic_means_z1 = [
            np.asarray(s["mean"][0], float)
            for s in per_r if s and "mean" in s and len(s["mean"]) >= 1
        ]
        graphic_means_z2 = [
            np.asarray(s["mean"][1], float)
            for s in per_r if s and "mean" in s and len(s["mean"]) >= 2
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

                # series-point 1 bars (offset left)
                x_z1 = x_pos - bar_width / 2
                heights_z1 = np.array(graphic_means_z1, dtype=np.float64)
                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                )
                implot.plot_bars(f"{short}1", x_z1, heights_z1, bar_width)
                implot.pop_style_color()
                implot.pop_style_var()

                # series-point 2 bars (offset right)
                x_z2 = x_pos + bar_width / 2
                heights_z2 = np.array(graphic_means_z2, dtype=np.float64)
                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.9, 0.4, 0.2, 0.8)
                )
                implot.plot_bars(f"{short}2", x_z2, heights_z2, bar_width)
                implot.pop_style_color()
                implot.pop_style_var()

            finally:
                implot.end_plot()
    else:
        # Single series point: simple bar chart
        per_r = [_series_for(parent, r)[0] for r in range(parent.num_graphics)]
        graphic_means = [
            np.asarray(s["mean"][0], float)
            for s in per_r if s and "mean" in s and len(s["mean"]) >= 1
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


def _draw_signal_metrics_chart(
    mean_vals, std_vals, snr_vals, is_dual_zplane, array_idx, stat_label="Z-Plane"
):
    """Draw signal metrics bar chart for single array."""
    short = stat_label[:1].upper() or "Z"
    style_seaborn_dark()
    imgui.text("Signal Quality Metrics")
    set_tooltip(
        "Bar chart showing mean fluorescence, standard deviation, and SNR"
        + (f" for each {stat_label.lower()}" if is_dual_zplane else ""),
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
                implot.plot_bars(f"{short}1", x_z1, heights_z1, bar_width)
                implot.pop_style_color()
                implot.pop_style_var()

                implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                implot.push_style_color(
                    implot.Col_.fill.value, (0.9, 0.4, 0.2, 0.8)
                )
                implot.plot_bars(f"{short}2", x_z2, heights_z2, bar_width)
                implot.pop_style_color()
                implot.pop_style_var()
            else:
                # Single bars for single series point
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


def _draw_combined_zplane_plot(
    parent, z_vals, stats_list, *, active_z=None, stat_label="Z-Plane"
):
    """Draw combined series signal plot across graphics at the current combo.

    When `active_z` is provided, an accent vertical line marks that series
    point on the x-axis, the corresponding tick label is rendered in brackets
    ("[N]") so it stands out, and an inlay annotation labels it.
    """
    imgui.text(f"{stat_label} Signal: Combined")
    set_tooltip(
        f"Gray = per-ROI {stat_label.lower()} profiles."
        " Blue shade = across-ROI mean ± std; blue line = mean."
        " Accent line = current series point."
        " Hover gray lines for values.",
        True,
    )

    # build per-graphic series at the current breakout combo
    graphic_series = [
        np.asarray(s["mean"], float)
        for s in (_series_for(parent, r)[0] for r in range(parent.num_graphics))
        if s and "mean" in s and len(s["mean"]) > 0
    ]
    if not graphic_series:
        return

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
                stat_label,
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
                short = stat_label[:1].upper() or "Z"
                implot.annotation(
                    float(active_z), float(mean_vals[_idx]),
                    imgui.ImVec4(*_ACTIVE_Z_COLOR),
                    imgui.ImVec2(0, -18), True, f"{short} = {active_z}",
                )
        finally:
            implot.end_plot()
    if pushed_bold:
        imgui.pop_font()


def _draw_zplane_signal_plot(
    z_vals, mean_vals, std_vals, array_idx, *,
    active_z=None, parent=None, stat_label="Z-Plane"
):
    """Draw the series signal plot with error bars.

    Same active-point treatment as the combined plot: accent vertical
    line, bracketed tick label (when <= 32 points), and an inlay
    annotation tagging the current series point. When `parent` is provided
    and a bold font is loaded, axis tick labels render in bold so the
    bracketed active label `[N]` reads with extra visual weight.
    """
    short = stat_label[:1].upper() or "Z"
    style_seaborn_dark()
    imgui.text(f"{stat_label} Signal: Mean ± Std")
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
                stat_label,
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
                    imgui.ImVec2(0, -18), True, f"{short} = {active_z}",
                )
        finally:
            implot.end_plot()
    if pushed_bold:
        imgui.pop_font()
