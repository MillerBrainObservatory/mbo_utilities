"""
composite poster figures: 2x2 image square (left) + traces (right).

versions:
  - max-proj, correlation, mean-enhanced backgrounds (masks colored per-plane)
  - highlight: all ROIs in one muted color, traced cells in plane color

each version at 50 and 100 cells per plane.

usage:
    python poster_figures.py
    python poster_figures.py --planes 2 5 8 12 --window 20
"""
import sys
sys.path.insert(0, r"C:\Users\RBO\repos\LBM-Suite2p-Python")

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from scipy.ndimage import percentile_filter, uniform_filter1d

from scipy.ndimage import distance_transform_edt
from lbm_suite2p_python.zplane import get_background_image, normalize99, stat_to_mask

RESULTS = Path(r"D:\demo\results")
SAVE = RESULTS / "poster" / "composite"
DPI = 300

# -- dark theme (default) --
DARK = dict(
    bg="#080808",
    plane_colors=["#5ef0ff", "#ff5a9e", "#c0ff4a", "#ffbe2e"],
    muted_mask="#2a2a3a",
    all_mask_color="#3ddc84",
    multicell_trace_colors=["#5ef0ff", "#ff5a9e", "#c9a0ff", "#ffbe2e"],
    cmap_corr=LinearSegmentedColormap.from_list("_corr_d", ["#080808", "#1a1a1a", "#3a3a3a", "#d0d0d0"]),
    bg_dim=0.5,
    bg_floor=0.0,
    trace_lw=0.6,
    invert=False,
    heatmap_cmap="inferno",
    style="dark_background",
)

# -- light theme --
LIGHT = dict(
    bg="#f8f8f8",
    plane_colors=["#0033aa", "#a00040", "#0a5500", "#b06800"],
    muted_mask="#b0b0b0",
    all_mask_color="#0a5500",
    multicell_trace_colors=["#0033aa", "#a00040", "#8b00c0", "#b06800"],
    cmap_corr=LinearSegmentedColormap.from_list("_corr_l", ["#f8f8f8", "#909090", "#404040", "#0a0a0a"]),
    bg_dim=1.0,
    bg_floor=0.15,
    trace_lw=1.0,
    invert=True,
    heatmap_cmap="YlOrRd",
    style="default",
)

# active theme — set in main()
T = DARK


def load_plane(idx):
    d = RESULTS / f"plane{idx:02d}_stitched"
    ops = np.load(d / "ops.npy", allow_pickle=True).item()
    stat = np.load(d / "stat.npy", allow_pickle=True)
    iscell = np.load(d / "iscell.npy")
    return dict(
        ops=ops, stat=stat, iscell=iscell,
        F=np.load(d / "F.npy"),
        Fneu=np.load(d / "Fneu.npy"),
    )


def compute_dff(F_raw, Fneu, fs):
    F_corr = F_raw - 0.7 * Fneu
    bwin = int(60 * fs) | 1
    out = np.zeros_like(F_corr)
    for i in range(F_corr.shape[0]):
        bl = percentile_filter(F_corr[i], percentile=8, size=bwin)
        bl = np.maximum(bl, 1.0)
        raw = (F_corr[i] - bl) / bl
        mu, sd = raw.mean(), raw.std()
        z = np.clip((raw - mu) / (sd + 1e-6), 0, None)
        out[i] = uniform_filter1d(z, size=max(int(0.3 * fs), 1))
    return out


def _get_bg_and_coords(ops, img_key):
    """get background image, normalized, and coordinate offsets."""
    if img_key in ("max_proj", "Vcorr"):
        img, yoff, xoff = get_background_image(ops, img_key)
    else:
        img = ops.get(img_key, np.zeros((ops["Ly"], ops["Lx"])))
        yoff, xoff = 0, 0
    return img, yoff, xoff


def _build_canvas(img):
    """normalize image and build RGB canvas blended with theme bg."""
    norm = normalize99(img)
    floor = T.get("bg_floor", 0.0)
    norm = np.clip((norm - floor) / (1.0 - floor), 0, 1)

    bg_rgb = np.array(to_rgb(T["bg"]))
    strength = T["bg_dim"]
    Ly, Lx = img.shape[:2]

    if T.get("invert", False):
        # light mode: high signal = dark, low signal = bg (white)
        target = 1.0 - norm
        target = np.stack([target] * 3, axis=-1)
    else:
        # dark mode: high signal = bright, low signal = bg (black)
        target = np.stack([norm] * 3, axis=-1)

    canvas = np.ones((Ly, Lx, 3)) * bg_rgb
    alpha = norm[..., None] * strength
    canvas = canvas * (1 - alpha) + target * alpha
    return np.clip(canvas, 0, 1).copy()


def _color_mask_overlay(canvas, mask, color_map, max_alpha=0.9, edge_width=3):
    """blend colored masks onto canvas with feathered edges.

    color_map: dict mapping cell label (int) -> RGB tuple.
    max_alpha: opacity at mask center.
    edge_width: pixels over which mask fades to transparent at borders.
    """
    # build feathered alpha for entire mask: distance from edge
    binary = mask > 0
    if not np.any(binary):
        return canvas
    dist = distance_transform_edt(binary)
    feather = np.clip(dist / edge_width, 0, 1) * max_alpha

    for label, col in color_map.items():
        px = mask == label
        if not np.any(px):
            continue
        col_rgb = np.array(to_rgb(col))
        a = feather[px]
        for k in range(3):
            canvas[px, k] = (1 - a) * canvas[px, k] + a * col_rgb[k]
    return canvas


def _make_labeled_mask(stat, indices, Ly, Lx, yoff, xoff):
    """build a labeled mask from a subset of stat entries.
    returns mask where pixel value = position in indices + 1."""
    subset = [stat[i] for i in indices]
    return stat_to_mask(subset, Ly, Lx, yoff, xoff)


def build_panel_highlight(ops, stat, all_accepted, highlight_indices, color_rgb,
                          img_key="max_proj", cmap=None):
    """all ROIs in muted color, highlighted subset in bright color."""
    img, yoff, xoff = _get_bg_and_coords(ops, img_key)
    Ly, Lx = img.shape[:2]
    canvas = _build_canvas(img)

    # background ROIs
    bg_set = set(highlight_indices)
    bg_rois = [i for i in all_accepted if i not in bg_set]
    bg_mask = _make_labeled_mask(stat, bg_rois, Ly, Lx, yoff, xoff)
    bg_colors = {i + 1: T["muted_mask"] for i in range(len(bg_rois))}
    _color_mask_overlay(canvas, bg_mask, bg_colors, max_alpha=0.4, edge_width=2)

    # highlighted ROIs
    hl_mask = _make_labeled_mask(stat, highlight_indices, Ly, Lx, yoff, xoff)
    hl_colors = {i + 1: color_rgb for i in range(len(highlight_indices))}
    _color_mask_overlay(canvas, hl_mask, hl_colors, max_alpha=0.95, edge_width=3)

    return np.clip(canvas, 0, 1)


def build_panel_multicell(ops, stat, all_accepted, highlight_indices, trace_color,
                          img_key="max_proj", cmap=None):
    """all ROIs in uniform green, highlighted subset in one bright trace color."""
    img, yoff, xoff = _get_bg_and_coords(ops, img_key)
    Ly, Lx = img.shape[:2]
    canvas = _build_canvas(img)

    bg_set = set(highlight_indices)
    bg_rois = [i for i in all_accepted if i not in bg_set]
    bg_mask = _make_labeled_mask(stat, bg_rois, Ly, Lx, yoff, xoff)
    bg_colors = {i + 1: T["all_mask_color"] for i in range(len(bg_rois))}
    _color_mask_overlay(canvas, bg_mask, bg_colors, max_alpha=0.45, edge_width=2)

    hl_mask = _make_labeled_mask(stat, highlight_indices, Ly, Lx, yoff, xoff)
    hl_colors = {i + 1: trace_color for i in range(len(highlight_indices))}
    _color_mask_overlay(canvas, hl_mask, hl_colors, max_alpha=0.95, edge_width=3)

    return np.clip(canvas, 0, 1)


def prepare_plane(pidx, n_cells):
    p = load_plane(pidx)
    ops, stat, iscell = p["ops"], p["stat"], p["iscell"]
    fs = ops.get("fs", 30.0)
    accepted = np.where(iscell[:, 0] > 0.5)[0]
    n_show = min(n_cells, len(accepted))
    dff = compute_dff(p["F"], p["Fneu"], fs)
    peaks = np.max(dff[accepted], axis=1)
    top = accepted[np.argsort(peaks)[::-1]][:n_show]
    return dict(ops=ops, stat=stat, dff=dff, top=top, fs=fs,
                n_show=n_show, pidx=pidx, accepted=accepted)


def compute_trace_layout(plane_data, wf, f0=0):
    # reading order: top-left at top, bottom-right at bottom
    all_shifted, all_colors = [], []
    global_offset = 0.0
    for d, color in zip(plane_data, T["plane_colors"]):
        traces = d["dff"][d["top"], f0:f0 + wf]
        n = d["n_show"]
        col_rgb = to_rgb(color)
        p10 = np.percentile(traces, 10, axis=1)
        p90 = np.percentile(traces, 90, axis=1)
        step = max(np.median(p90 - p10) * 1.0, 0.5)
        for i in range(n):
            bl = np.percentile(traces[i], 8)
            all_shifted.append((traces[i] - bl) + global_offset + i * step)
            all_colors.append(col_rgb)
        global_offset += n * step + step * 1.5
    return all_shifted, all_colors


def compute_plane_traces(d, wf, color, f0=0, smooth=0):
    """compute trace data for a single plane group.
    smooth: kernel size in frames for temporal smoothing (0 = off).
    returns (shifted, colors, raw, step) where each is a list over cells."""
    traces = d["dff"][d["top"], f0:f0 + wf]
    if smooth > 1:
        traces = uniform_filter1d(traces, size=smooth, axis=1)
    n = d["n_show"]
    col_rgb = to_rgb(color)
    p10 = np.percentile(traces, 10, axis=1)
    p90 = np.percentile(traces, 90, axis=1)
    step = max(np.median(p90 - p10) * 1.0, 0.5)
    shifted, colors, raw = [], [], []
    for i in range(n):
        bl = np.percentile(traces[i], 8)
        shifted.append((traces[i] - bl) + i * step)
        colors.append(col_rgb)
        raw.append(traces[i] - bl)
    return shifted, colors, raw, step


def draw_plane_traces(ax, time, shifted, colors):
    """draw traces for one plane group into an axis."""
    n = len(shifted)
    for i in range(n - 1, -1, -1):
        z = n - i
        ax.fill_between(time, shifted[i], y2=shifted[i].min() - 1.0,
                         color=T["bg"], zorder=z - 0.5)
        ax.plot(time, shifted[i], color=colors[i], lw=T.get("trace_lw", 0.6), zorder=z, alpha=1.0)
    ax.set_xlim(0, time[-1])
    y_min = min(s.min() for s in shifted)
    y_max = max(s.max() for s in shifted)
    yr = y_max - y_min
    ax.set_ylim(y_min - yr * 0.005, y_max + yr * 0.02)
    ax.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # bounds: bottom of first trace to percentile of top trace (avoids spike overshoot)
    top_p25 = np.percentile(shifted[-1], 25)
    ax._trace_ybounds = (y_min, top_p25)


def draw_plane_heatmap(ax, time, raw, step, vmin, vmax, color=None):
    """draw heatmap for one plane group. colormap goes bg -> color."""
    import matplotlib as mpl
    bg_rgb = to_rgb(T["bg"])
    if color is not None:
        cmap_obj = LinearSegmentedColormap.from_list("_h", [bg_rgb, to_rgb(color)])
    else:
        cmap = T.get("heatmap_cmap", "inferno")
        cmap_obj = mpl.colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
    norm_fn = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    mat = np.array(raw)
    colored = cmap_obj(norm_fn(mat))[:, :, :3]
    ax.imshow(colored, aspect="auto", interpolation="none", rasterized=True)
    ax.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_traces(ax, time, all_shifted, all_colors):
    """draw all traces into a single axis (used by make_figure)."""
    n = len(all_shifted)
    for i in range(n - 1, -1, -1):
        z = n - i
        ax.fill_between(time, all_shifted[i], y2=all_shifted[i].min() - 1.0,
                         color=T["bg"], zorder=z - 0.5)
        ax.plot(time, all_shifted[i], color=all_colors[i], lw=T.get("trace_lw", 0.6), zorder=z, alpha=1.0)
    ax.set_xlim(0, time[-1])
    y_min = min(s.min() for s in all_shifted)
    y_max = max(s.max() for s in all_shifted)
    yr = y_max - y_min
    ax.set_ylim(y_min - yr * 0.005, y_max + yr * 0.005)
    ax.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_figure(plane_data, window_s, img_key, cmap, highlight=True):
    """2x2 square + traces. one function for all versions."""
    fs = plane_data[0]["fs"]
    wf = min(int(window_s * fs), plane_data[0]["dff"].shape[1])
    time = np.arange(wf) / fs
    all_shifted, all_colors = compute_trace_layout(plane_data, wf)

    fig = plt.figure(figsize=(22, 14), facecolor=T["bg"])
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.005,
                              left=0.01, right=0.99, top=0.99, bottom=0.01)
    mask_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], wspace=0.0, hspace=0.04)

    for i, (d, color) in enumerate(zip(plane_data, T["plane_colors"])):
        ax = fig.add_subplot(mask_gs[i // 2, i % 2], facecolor=T["bg"])
        panel = build_panel_highlight(d["ops"], d["stat"], d["accepted"], d["top"],
                                      color, img_key=img_key, cmap=cmap)
        ax.imshow(panel, interpolation="nearest", aspect="auto")
        ax.axis("off")

    ax_t = fig.add_subplot(outer[1], facecolor=T["bg"])
    draw_traces(ax_t, time, all_shifted, all_colors)
    return fig


def _style_left_spine(ax, color, bounded=False, lw=3.5):
    """draw a thick left bar as a manual line to avoid spine rendering quirks."""
    from matplotlib.lines import Line2D
    ax.spines["left"].set_visible(False)
    if bounded and hasattr(ax, "_trace_ybounds"):
        ylo, yhi = ax._trace_ybounds
        # data coords -> axes coords
        ylim = ax.get_ylim()
        yr = ylim[1] - ylim[0]
        a0 = (ylo - ylim[0]) / yr
        a1 = (yhi - ylim[0]) / yr
    else:
        a0, a1 = 0.0, 1.0
    line = Line2D([0, 0], [a0, a1], transform=ax.transAxes,
                  color=to_rgb(color), linewidth=lw, solid_capstyle="butt",
                  clip_on=False)
    ax.add_line(line)


def make_figure_multicell(plane_data, window_s, img_key, cmap, offset_s=0.0, smooth_s=0.0):
    """2x2 square with heatmap insets + traces column."""
    fs = plane_data[0]["fs"]
    n_total = plane_data[0]["dff"].shape[1]
    f0 = min(int(offset_s * fs), n_total - 1)
    wf = min(int(window_s * fs), n_total - f0)
    smooth_frames = max(int(smooth_s * fs), 0)
    time = np.arange(wf) / fs
    colors = T["multicell_trace_colors"]
    n_planes = len(plane_data)

    # per-plane trace data in reading order
    groups = []
    group_colors = list(colors)
    for d, col in zip(plane_data, group_colors):
        groups.append(compute_plane_traces(d, wf, col, f0=f0, smooth=smooth_frames))

    # global heatmap color scale
    all_raw_flat = [r for _, _, raw, _ in groups for r in raw]
    raw_all = np.concatenate(all_raw_flat)
    vmin = np.percentile(raw_all, 1)
    vmax = np.percentile(raw_all, 99)

    h_ratios = [len(g[0]) for g in groups]

    fig = plt.figure(figsize=(22, 14), facecolor=T["bg"])
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.01,
                              left=0.01, right=0.99, top=0.99, bottom=0.01)

    # 2x2 mask images with heatmap insets
    mask_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], wspace=0.0, hspace=0.04)
    for i, (d, color) in enumerate(zip(plane_data, colors)):
        ax = fig.add_subplot(mask_gs[i // 2, i % 2], facecolor=T["bg"])
        panel = build_panel_multicell(d["ops"], d["stat"], d["accepted"], d["top"],
                                      color, img_key=img_key, cmap=cmap)
        ax.imshow(panel, interpolation="nearest", aspect="auto")
        ax.axis("off")

        # heatmap inset in bottom-right corner
        _, _, raw, step = groups[i]
        inset = ax.inset_axes([0.65, 0.03, 0.33, 0.28])
        draw_plane_heatmap(inset, time, raw, step, vmin, vmax, color=group_colors[i])
        for sp in inset.spines.values():
            sp.set_visible(False)
        _style_left_spine(inset, color, lw=2.5)

    # traces column: one row per plane group
    trace_gs = gridspec.GridSpecFromSubplotSpec(
        n_planes, 1, subplot_spec=outer[1], hspace=0.08, height_ratios=h_ratios)
    for i, (shifted, grp_colors, _, _) in enumerate(groups):
        ax = fig.add_subplot(trace_gs[i], facecolor=T["bg"])
        draw_plane_traces(ax, time, shifted, grp_colors)
        _style_left_spine(ax, group_colors[i], bounded=True)

    return fig


def save_panel(img_array, path, pad=0.05, spine_color=None):
    """save a single image array as its own figure with equal padding."""
    h, w = img_array.shape[:2]
    aspect = w / h
    fig_h = 6
    fig_w = fig_h * aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=T["bg"])
    ax.imshow(img_array, interpolation="nearest")
    ax.axis("off")
    if spine_color is not None:
        _style_left_spine(ax, spine_color)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=DPI, facecolor=T["bg"], bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(f"  saved: {path}")



def main():
    parser = argparse.ArgumentParser(description="composite poster figures")
    parser.add_argument("--planes", nargs=4, type=int, default=[3, 4, 5, 6])
    parser.add_argument("--ncells", type=int, default=100)
    parser.add_argument("--window", type=float, default=25.0, help="duration in seconds")
    parser.add_argument("--offset", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--smooth", type=float, default=0.0, help="temporal smoothing in seconds")
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--pad", type=float, default=0.0)
    parser.add_argument("--light", action="store_true", help="white background version")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    global T
    T = LIGHT if args.light else DARK

    plt.style.use(T["style"])
    bg = T["bg"]
    plt.rcParams.update({"figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg})

    save_dir = Path(args.save_dir) if args.save_dir else SAVE
    save_dir.mkdir(parents=True, exist_ok=True)
    ptag = "_".join(str(p) for p in args.planes)
    suffix = "_light" if args.light else ""
    nc = args.ncells

    print(f"loading {nc} cells per plane...")
    plane_data = []
    for pidx in args.planes:
        print(f"  plane {pidx}...", flush=True)
        plane_data.append(prepare_plane(pidx, nc))

    fs = plane_data[0]["fs"]
    n_total = plane_data[0]["dff"].shape[1]
    max_s = n_total / fs
    f0 = int(args.offset * fs)
    if f0 >= n_total:
        print(f"  WARNING: offset {args.offset:.1f}s exceeds recording length ({max_s:.1f}s), clamping")
        f0 = max(n_total - int(args.window * fs), 0)
    wf = min(int(args.window * fs), n_total - f0)
    print(f"  showing frames {f0}-{f0+wf} ({f0/fs:.1f}-{(f0+wf)/fs:.1f}s of {max_s:.1f}s)")
    time = np.arange(wf) / fs

    # build and save each mask panel individually
    for i, (d, color) in enumerate(zip(plane_data, T["multicell_trace_colors"])):
        panel = build_panel_multicell(d["ops"], d["stat"], d["accepted"], d["top"],
                                      color, img_key="Vcorr", cmap=T["cmap_corr"])
        pos = ["topleft", "topright", "bottomleft", "bottomright"][i]
        save_panel(panel, save_dir / f"mask_{pos}_plane{d['pidx']}_{ptag}_{nc}cells{suffix}.png",
                   pad=args.pad, spine_color=color)

    # per-plane trace groups in reading order
    colors = T["multicell_trace_colors"]
    group_colors = list(colors)
    groups = []
    smooth_frames = max(int(args.smooth * fs), 0)
    for d, col in zip(plane_data, group_colors):
        groups.append(compute_plane_traces(d, wf, col, f0=f0, smooth=smooth_frames))

    # global heatmap scale
    all_raw_flat = [r for _, _, raw, _ in groups for r in raw]
    raw_all = np.concatenate(all_raw_flat)
    vmin = np.percentile(raw_all, 1)
    vmax = np.percentile(raw_all, 99)

    n_planes = len(groups)
    h_ratios = [len(g[0]) for g in groups]

    # save traces as stacked column
    fig, axes = plt.subplots(n_planes, 1, figsize=(8, 14), facecolor=T["bg"],
                              gridspec_kw={"hspace": 0.08, "height_ratios": h_ratios})
    for ax, (shifted, grp_colors, _, _), col in zip(axes, groups, group_colors):
        ax.set_facecolor(T["bg"])
        draw_plane_traces(ax, time, shifted, grp_colors)
        _style_left_spine(ax, col, bounded=True)
    pad_t = max(args.pad, 0.05)
    fig.savefig(save_dir / f"traces_{ptag}_{nc}cells{suffix}.png",
                dpi=DPI, facecolor=T["bg"], bbox_inches="tight", pad_inches=pad_t)
    plt.close(fig)
    print(f"  saved: traces_{ptag}_{nc}cells{suffix}.png")

    # save heatmap as stacked column
    fig, axes = plt.subplots(n_planes, 1, figsize=(4, 14), facecolor=T["bg"],
                              gridspec_kw={"hspace": 0.08, "height_ratios": h_ratios})
    for ax, (_, _, raw, step), col in zip(axes, groups, group_colors):
        ax.set_facecolor(T["bg"])
        draw_plane_heatmap(ax, time, raw, step, vmin, vmax, color=col)
        _style_left_spine(ax, col)
    fig.savefig(save_dir / f"heatmap_{ptag}_{nc}cells{suffix}.png",
                dpi=DPI, facecolor=T["bg"], bbox_inches="tight", pad_inches=args.pad)
    plt.close(fig)
    print(f"  saved: heatmap_{ptag}_{nc}cells{suffix}.png")

    # also save the full composite
    print("  composite...", flush=True)
    fig = make_figure_multicell(plane_data, args.window, "Vcorr", T["cmap_corr"],
                                offset_s=args.offset, smooth_s=args.smooth)
    fig.savefig(save_dir / f"corr_multicell_{ptag}_{nc}cells{suffix}.png",
                dpi=args.dpi, facecolor=T["bg"], bbox_inches="tight", pad_inches=args.pad)
    plt.close(fig)
    print(f"  saved: corr_multicell_{ptag}_{nc}cells{suffix}.png")

    print("\ndone")


if __name__ == "__main__":
    main()
