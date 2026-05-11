# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///
"""Compare two suite2p pipelines on the same dataset.

For each plane, renders a 4x2 figure with rows = summary images
(``meanImg``, ``meanImgE``, ``max_proj``, ``Vcorr``) and columns =
pipeline. Accepted-cell ROIs are overlaid using LBM-Suite2p-Python's
lam-weighted feathered blending (see ``zplane.plot_masks`` at
``LBM-Suite2p-Python/lbm_suite2p_python/zplane.py:844``), so colours
and feathering match the existing ``*_segmentation.png`` figures.

The two pipelines have different folder layouts:
  - LBM-Suite2p-Python: ``<root>/zplane{NN}_tp00001-64033/``
  - Reference / kbarber:  ``<root>/plane_{N}/``

Usage:
    uv run scripts/compare_pipelines.py \\
        --lbm   D:/kbarber/2025-03-13_mk301/suite2p \\
        --other D:/kbarber/2025-03-13_mk301/suite2p_full_output \\
        --out   D:/kbarber/2025-03-13_mk301/pipeline_compare
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

PROJECTIONS = ("meanImg", "meanImgE", "max_proj", "Vcorr")
PROJ_TITLES = {
    "meanImg": "Mean",
    "meanImgE": "Enhanced mean",
    "max_proj": "Max projection",
    "Vcorr": "Correlation map",
}
CROPPED_PROJS = {"max_proj", "Vcorr"}

LBM_DIR_RE = re.compile(r"^zplane(\d+)_tp", re.IGNORECASE)
OTHER_DIR_RE = re.compile(r"^plane[_ ]?(\d+)$", re.IGNORECASE)

# skew of the dF/F trace is suite2p's built-in activity-sparseness proxy;
# its classifier uses skew >= 1.0 by default, so we re-use that as the
# "active cell" threshold here.
ACTIVE_SKEW = 1.0
LBM_LABEL = "LBM-Suite2p-Python"
OTHER_LABEL = "Reference pipeline"
# Wong colour-blind-safe palette
LBM_COLOR = "#0173B2"
OTHER_COLOR = "#DE8F05"


def discover_planes(root: Path, regex: re.Pattern) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = regex.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def load_plane(plane_dir: Path) -> tuple[dict, np.ndarray, np.ndarray] | None:
    ops_path = plane_dir / "ops.npy"
    stat_path = plane_dir / "stat.npy"
    iscell_path = plane_dir / "iscell.npy"
    if not (ops_path.exists() and stat_path.exists() and iscell_path.exists()):
        return None
    ops = np.load(ops_path, allow_pickle=True).item()
    stat = np.load(stat_path, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)
    accepted = iscell[:, 0].astype(bool) if iscell.ndim == 2 else iscell.astype(bool)
    return ops, stat, accepted


def _stat_offsets(ops: dict, img_shape: tuple[int, int], proj_key: str) -> tuple[int, int]:
    """Translate full-frame stat coords into the projection's coord space.

    Suite2p stores ``max_proj`` and ``Vcorr`` already cropped to
    ``yrange`` / ``xrange``; ``meanImg`` / ``meanImgE`` are full frame.
    """
    if proj_key not in CROPPED_PROJS:
        return 0, 0
    full_Ly, full_Lx = img_shape
    yoff = int(ops.get("yrange", [0, full_Ly])[0])
    xoff = int(ops.get("xrange", [0, full_Lx])[0])
    return yoff, xoff


def build_overlay_canvas(
    img: np.ndarray,
    stat: np.ndarray,
    accepted: np.ndarray,
    ops: dict,
    proj_key: str,
    colors: np.ndarray,
) -> np.ndarray:
    """Port of LBM ``plot_masks`` canvas building.

    Percentile-stretched grayscale tiled to RGB, then for each accepted
    ROI blend ``0.5 * canvas + 0.5 * color * (lam / lam.max())``. The
    per-pixel ``lam`` weight is what gives the soft feathered edges that
    match LBM's ``*_segmentation.png`` output.
    """
    img = np.asarray(img, dtype=np.float32)
    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99)
    norm = np.clip((img - vmin) / (vmax - vmin + 1e-6), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    canvas = np.tile(norm, (3, 1, 1)).transpose(1, 2, 0).astype(np.float32)

    Ly, Lx = img.shape[:2]
    yoff, xoff = _stat_offsets(ops, (Ly, Lx), proj_key)

    c = 0
    for n, s in enumerate(stat):
        if not accepted[n]:
            continue
        ypix = np.asarray(s["ypix"]) - yoff
        xpix = np.asarray(s["xpix"]) - xoff
        lam = np.asarray(s["lam"], dtype=np.float32)

        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        if not valid.any():
            c += 1
            continue
        ypix = ypix[valid]
        xpix = xpix[valid]
        lam = lam[valid]
        lam = lam / (lam.max() + 1e-10)
        col = colors[c % len(colors)]
        c += 1
        for k in range(3):
            canvas[ypix, xpix, k] = 0.5 * canvas[ypix, xpix, k] + 0.5 * col[k] * lam
    return canvas


def available_projections(*ops_dicts: dict) -> tuple[str, ...]:
    """Projections in canonical order that exist (non-None) in every ops dict."""
    return tuple(
        k for k in PROJECTIONS
        if all((k in o) and (o[k] is not None) for o in ops_dicts)
    )


def render_plane(
    plane_idx: int,
    lbm: tuple[dict, np.ndarray, np.ndarray],
    other: tuple[dict, np.ndarray, np.ndarray],
    out_path: Path,
    dpi: int = 200,
    projections: tuple[str, ...] | None = None,
) -> None:
    pipelines = [
        (LBM_LABEL, lbm),
        (OTHER_LABEL, other),
    ]
    if projections is None:
        projections = available_projections(lbm[0], other[0])
    if not projections:
        return

    n_rows = len(projections)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(10, 4.5 * n_rows),
        facecolor="black",
        squeeze=False,
    )

    for col, (label, (ops, stat, accepted)) in enumerate(pipelines):
        n_acc = int(accepted.sum())
        n_tot = int(accepted.size)
        colors = plt.cm.hsv(np.linspace(0, 1, max(n_acc, 1) + 1))[:, :3]

        for row, key in enumerate(projections):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            img = ops[key]
            canvas = build_overlay_canvas(img, stat, accepted, ops, key, colors)
            ax.imshow(canvas, interpolation="nearest")

            if row == 0:
                ax.set_title(
                    f"{label}\nplane {plane_idx} — {n_acc}/{n_tot} cells",
                    fontsize=10, color="white", fontweight="bold",
                )
            if col == 0:
                ax.set_ylabel(
                    PROJ_TITLES[key],
                    color="white", fontsize=11, fontweight="bold",
                )

    fig.suptitle(
        f"Plane {plane_idx}: LBM-Suite2p-Python vs reference pipeline",
        color="white", fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="black")
    plt.close(fig)


def _stat_field(stat: np.ndarray, key: str) -> np.ndarray:
    vals = np.full(len(stat), np.nan, dtype=np.float64)
    for i, s in enumerate(stat):
        v = s.get(key) if isinstance(s, dict) else None
        if v is None:
            continue
        try:
            vals[i] = float(v)
        except (TypeError, ValueError):
            continue
    return vals


def compute_plane_stats(plane: int, label: str, stat: np.ndarray, accepted: np.ndarray) -> dict:
    skew = _stat_field(stat, "skew")
    npix = _stat_field(stat, "npix")
    acc_skew = skew[accepted]
    acc_npix = npix[accepted]
    n_active = int(np.nansum(acc_skew >= ACTIVE_SKEW))
    n_total = int(accepted.size)
    n_accepted = int(accepted.sum())
    n_rejected = n_total - n_accepted
    return {
        "plane": plane,
        "pipeline": label,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_total": n_total,
        "accept_rate": n_accepted / n_total if n_total else float("nan"),
        "median_skew": float(np.nanmedian(acc_skew)) if acc_skew.size else float("nan"),
        "mean_skew": float(np.nanmean(acc_skew)) if acc_skew.size else float("nan"),
        "median_npix": float(np.nanmedian(acc_npix)) if acc_npix.size else float("nan"),
        "n_active": n_active,
    }


def write_stats_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _by_pipeline(rows: list[dict], planes: list[int], label: str, key: str) -> np.ndarray:
    by_plane = {r["plane"]: r[key] for r in rows if r["pipeline"] == label}
    return np.array([by_plane.get(p, np.nan) for p in planes], dtype=np.float64)


_PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 11,
}


def render_stats_summary(rows: list[dict], path: Path, dpi: int = 300) -> None:
    if not rows:
        return
    planes = sorted({r["plane"] for r in rows})
    x = np.arange(len(planes))
    w = 0.4
    bar_kw = {"alpha": 0.9, "edgecolor": "white", "linewidth": 0.5}

    n_lbm = _by_pipeline(rows, planes, LBM_LABEL, "n_accepted")
    n_oth = _by_pipeline(rows, planes, OTHER_LABEL, "n_accepted")
    rej_lbm = _by_pipeline(rows, planes, LBM_LABEL, "n_rejected")
    rej_oth = _by_pipeline(rows, planes, OTHER_LABEL, "n_rejected")
    skew_lbm = _by_pipeline(rows, planes, LBM_LABEL, "median_skew")
    skew_oth = _by_pipeline(rows, planes, OTHER_LABEL, "median_skew")
    act_lbm = _by_pipeline(rows, planes, LBM_LABEL, "n_active")
    act_oth = _by_pipeline(rows, planes, OTHER_LABEL, "n_active")

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor="white")

        ax = axes[0, 0]
        ax.bar(x - w / 2, n_lbm, w, color=LBM_COLOR, label=LBM_LABEL, **bar_kw)
        ax.bar(x + w / 2, n_oth, w, color=OTHER_COLOR, label=OTHER_LABEL, **bar_kw)
        ax.set_title("Accepted cells")
        ax.set_ylabel("number of cells")

        ax = axes[0, 1]
        ax.bar(x - w / 2, rej_lbm, w, color=LBM_COLOR, **bar_kw)
        ax.bar(x + w / 2, rej_oth, w, color=OTHER_COLOR, **bar_kw)
        ax.set_title("Rejected cells")
        ax.set_ylabel("number of cells")

        ax = axes[1, 0]
        ax.plot(planes, skew_lbm, "o-", color=LBM_COLOR, linewidth=1.8, markersize=6)
        ax.plot(planes, skew_oth, "s-", color=OTHER_COLOR, linewidth=1.8, markersize=6)
        ax.set_title("Median skew")
        ax.set_ylabel("skew")
        ax.set_xlabel("plane")

        ax = axes[1, 1]
        ax.bar(x - w / 2, act_lbm, w, color=LBM_COLOR, **bar_kw)
        ax.bar(x + w / 2, act_oth, w, color=OTHER_COLOR, **bar_kw)
        ax.set_title(f"Cells with skew ≥ {ACTIVE_SKEW:g}")
        ax.set_ylabel("number of cells")
        ax.set_xlabel("plane")

        for ax_row, ax in zip([0, 0, 1, 1], axes.ravel()):
            pass
        for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
            ax.set_xticks(x)
            ax.set_xticklabels(planes)
        axes[1, 0].set_xticks(planes)
        for ax in axes.ravel():
            ax.xaxis.grid(False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def compute_registration_stats(ops: dict) -> dict:
    """Per-plane registration quality summary from suite2p ops."""
    xoff = np.asarray(ops.get("xoff", []), dtype=np.float64)
    yoff = np.asarray(ops.get("yoff", []), dtype=np.float64)
    if xoff.size and yoff.size:
        shift = np.sqrt(xoff ** 2 + yoff ** 2)
        mean_shift = float(np.mean(shift))
    else:
        mean_shift = float("nan")
    corrXY = np.asarray(ops.get("corrXY", []), dtype=np.float64)
    mean_corrXY = float(np.mean(corrXY)) if corrXY.size else float("nan")
    Vcorr = ops.get("Vcorr")
    mean_Vcorr = float(np.nanmean(Vcorr)) if Vcorr is not None else float("nan")
    return {
        "mean_shift": mean_shift,
        "mean_corrXY": mean_corrXY,
        "mean_Vcorr": mean_Vcorr,
    }


def render_registration_quality(plane_records: list[dict], path: Path, dpi: int = 300) -> None:
    if not plane_records:
        return
    planes = [r["plane"] for r in plane_records]
    x = np.arange(len(planes))
    w = 0.4
    bar_kw = {"alpha": 0.9, "edgecolor": "white", "linewidth": 0.5}

    def col(pipe, key):
        return np.array([r[pipe]["reg"][key] for r in plane_records], dtype=np.float64)

    panels = [
        ("mean_shift", "Mean frame shift", "pixels"),
        ("mean_corrXY", "Mean frame–template correlation", "correlation"),
        ("mean_Vcorr", "Mean local correlation (Vcorr)", "correlation"),
    ]

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white")
        for ax, (key, title, ylabel) in zip(axes, panels):
            lbm = col("lbm", key)
            ref = col("ref", key)
            ax.bar(x - w / 2, lbm, w, color=LBM_COLOR, label=LBM_LABEL, **bar_kw)
            ax.bar(x + w / 2, ref, w, color=OTHER_COLOR, label=OTHER_LABEL, **bar_kw)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("plane")
            ax.set_xticks(x)
            ax.set_xticklabels(planes)
            ax.xaxis.grid(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def _violin_pair(ax, values_a: np.ndarray, values_b: np.ndarray, ylabel: str, title: str) -> None:
    data = []
    positions = []
    colors = []
    for i, vals in enumerate((values_a, values_b)):
        v = vals[~np.isnan(vals)] if vals.size else vals
        if v.size:
            data.append(v)
            positions.append(i)
            colors.append(LBM_COLOR if i == 0 else OTHER_COLOR)
    if not data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        parts = ax.violinplot(data, positions=positions, widths=0.7, showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.8)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.6)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LBM_LABEL, OTHER_LABEL], rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.xaxis.grid(False)


def render_quality_distributions(plane_records: list[dict], path: Path, dpi: int = 300) -> None:
    if not plane_records:
        return
    metrics = [("skew", "Skew"), ("compact", "Compactness"), ("npix", "Footprint size (pixels)")]

    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        "lbm": {k: [] for k, _ in metrics},
        "ref": {k: [] for k, _ in metrics},
    }
    for r in plane_records:
        for pipe in ("lbm", "ref"):
            stat = r[pipe]["stat"]
            accepted = r[pipe]["accepted"]
            for key, _ in metrics:
                vals = _stat_field(stat, key)[accepted]
                pooled[pipe][key].append(vals)

    flat = {pipe: {k: (np.concatenate(v) if v else np.array([])) for k, v in d.items()} for pipe, d in pooled.items()}

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), facecolor="white")
        for ax, (key, label) in zip(axes, metrics):
            _violin_pair(ax, flat["lbm"][key], flat["ref"][key], label, label)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def mutual_match(meds_a: np.ndarray, meds_b: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbour ROI matching within ``threshold`` pixels.

    Returns (matched_a_idx, matched_b_idx) so that for each k,
    ``meds_a[matched_a_idx[k]]`` is the nearest in A to ``meds_b[matched_b_idx[k]]``,
    and vice-versa, and their distance is <= threshold.
    """
    if meds_a.size == 0 or meds_b.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    tree_b = cKDTree(meds_b)
    tree_a = cKDTree(meds_a)
    dist_ab, idx_ab = tree_b.query(meds_a, k=1)
    _, idx_ba = tree_a.query(meds_b, k=1)
    a_idx, b_idx = [], []
    for i, j in enumerate(idx_ab):
        if dist_ab[i] <= threshold and idx_ba[j] == i:
            a_idx.append(i)
            b_idx.append(j)
    return np.array(a_idx, dtype=int), np.array(b_idx, dtype=int)


def compute_per_plane_unique_masks(
    lbm_stat: np.ndarray,
    lbm_accepted: np.ndarray,
    ref_stat: np.ndarray,
    ref_accepted: np.ndarray,
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the "LBM only" and "Reference only" buckets.

    A cell is in "LBM only" iff LBM accepted it AND there is no mutually
    matched reference cell that also accepted it. Symmetric for
    "Reference only". Detection-only differences and matched-pair
    classification disagreements both fall into the bucket.
    """
    if len(lbm_stat) == 0 or len(ref_stat) == 0:
        return lbm_accepted.copy(), ref_accepted.copy()

    lbm_med = np.array([s.get("med", (np.nan, np.nan)) for s in lbm_stat], dtype=np.float64)
    ref_med = np.array([s.get("med", (np.nan, np.nan)) for s in ref_stat], dtype=np.float64)

    a_idx, b_idx = mutual_match(lbm_med, ref_med, threshold=threshold)

    matched_lbm = np.zeros(len(lbm_stat), dtype=bool)
    matched_ref = np.zeros(len(ref_stat), dtype=bool)
    matched_lbm[a_idx] = True
    matched_ref[b_idx] = True

    lbm_pair_ref_accepts = np.zeros(len(lbm_stat), dtype=bool)
    ref_pair_lbm_accepts = np.zeros(len(ref_stat), dtype=bool)
    if a_idx.size:
        lbm_pair_ref_accepts[a_idx] = ref_accepted[b_idx]
        ref_pair_lbm_accepts[b_idx] = lbm_accepted[a_idx]

    lbm_only = lbm_accepted & ~(matched_lbm & lbm_pair_ref_accepts)
    ref_only = ref_accepted & ~(matched_ref & ref_pair_lbm_accepts)
    return lbm_only, ref_only


def render_plane_unique(
    plane_idx: int,
    lbm: tuple[dict, np.ndarray, np.ndarray],
    other: tuple[dict, np.ndarray, np.ndarray],
    lbm_only_mask: np.ndarray,
    ref_only_mask: np.ndarray,
    out_path: Path,
    dpi: int = 300,
    projections: tuple[str, ...] | None = None,
) -> None:
    pipelines = [
        (f"{LBM_LABEL} only", lbm, lbm_only_mask),
        (f"{OTHER_LABEL} only", other, ref_only_mask),
    ]
    if projections is None:
        projections = available_projections(lbm[0], other[0])
    if not projections:
        return

    n_rows = len(projections)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4.5 * n_rows), facecolor="black", squeeze=False)

    for col, (label, (ops, stat, _accepted), only_mask) in enumerate(pipelines):
        n_only = int(only_mask.sum())
        colors = plt.cm.hsv(np.linspace(0, 1, max(n_only, 1) + 1))[:, :3]

        for row, key in enumerate(projections):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            img = ops[key]
            canvas = build_overlay_canvas(img, stat, only_mask, ops, key, colors)
            ax.imshow(canvas, interpolation="nearest")

            if row == 0:
                ax.set_title(
                    f"{label}\nplane {plane_idx} — {n_only} cells",
                    fontsize=10, color="white", fontweight="bold",
                )
            if col == 0:
                ax.set_ylabel(
                    PROJ_TITLES[key],
                    color="white", fontsize=11, fontweight="bold",
                )

    fig.suptitle(
        f"Plane {plane_idx}: cells accepted by only one pipeline",
        color="white", fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="black")
    plt.close(fig)


def collect_disagreement_buckets(plane_records: list[dict], threshold: float = 5.0) -> dict[str, dict[str, np.ndarray]]:
    """Cross-pipeline classification buckets with their per-cell quality metrics.

    Buckets: 'both_accept', 'lbm_only', 'ref_only', 'both_reject'.
    "lbm_only" includes both matched-pair disagreements (LBM accept, ref reject)
    and LBM-unique detections that LBM accepted; symmetric for "ref_only".
    """
    keys = ("skew", "npix")
    buckets: dict[str, dict[str, list[np.ndarray]]] = {
        b: {k: [] for k in keys} for b in ("both_accept", "lbm_only", "ref_only", "both_reject")
    }

    def get_metric(stat, key):
        return _stat_field(stat, key)

    def push(bucket_name, stat, key, idx):
        if idx.size == 0:
            return
        vals = get_metric(stat, key)[idx]
        buckets[bucket_name][key].append(vals)

    for r in plane_records:
        lbm_stat = r["lbm"]["stat"]
        ref_stat = r["ref"]["stat"]
        lbm_acc = r["lbm"]["accepted"]
        ref_acc = r["ref"]["accepted"]
        lbm_med = np.array([s.get("med", (np.nan, np.nan)) for s in lbm_stat], dtype=np.float64)
        ref_med = np.array([s.get("med", (np.nan, np.nan)) for s in ref_stat], dtype=np.float64)
        if lbm_med.size == 0 or ref_med.size == 0:
            continue

        a_idx, b_idx = mutual_match(lbm_med, ref_med, threshold=threshold)
        matched_lbm = np.zeros(len(lbm_stat), dtype=bool)
        matched_ref = np.zeros(len(ref_stat), dtype=bool)
        matched_lbm[a_idx] = True
        matched_ref[b_idx] = True

        for k in keys:
            la = lbm_acc[a_idx]
            ra = ref_acc[b_idx]
            push("both_accept", lbm_stat, k, a_idx[la & ra])
            push("both_reject", lbm_stat, k, a_idx[(~la) & (~ra)])
            # matched but only one accepts
            push("lbm_only", lbm_stat, k, a_idx[la & (~ra)])
            push("ref_only", ref_stat, k, b_idx[ra & (~la)])
            # unique detections (no match in the other pipeline)
            unique_lbm = np.where(~matched_lbm & lbm_acc)[0]
            unique_ref = np.where(~matched_ref & ref_acc)[0]
            push("lbm_only", lbm_stat, k, unique_lbm)
            push("ref_only", ref_stat, k, unique_ref)

    return {b: {k: (np.concatenate(v) if v else np.array([])) for k, v in d.items()} for b, d in buckets.items()}


def render_classification_disagreement(plane_records: list[dict], path: Path, dpi: int = 300, threshold: float = 5.0) -> None:
    if not plane_records:
        return
    buckets = collect_disagreement_buckets(plane_records, threshold=threshold)
    bucket_order = ("both_accept", "lbm_only", "ref_only", "both_reject")
    bucket_labels = {
        "both_accept": "both accepted",
        "lbm_only": "LBM only",
        "ref_only": "Reference only",
        "both_reject": "both rejected",
    }
    bucket_colors = {
        "both_accept": "#4C9F70",
        "lbm_only": LBM_COLOR,
        "ref_only": OTHER_COLOR,
        "both_reject": "#9E9E9E",
    }

    metrics = [("skew", "Skew"), ("npix", "Footprint size (pixels)")]

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), facecolor="white")
        for ax, (key, label) in zip(axes, metrics):
            data = []
            positions = []
            colors = []
            counts = []
            for i, b in enumerate(bucket_order):
                v = buckets[b][key]
                v = v[~np.isnan(v)] if v.size else v
                counts.append(v.size)
                if v.size:
                    data.append(v)
                    positions.append(i)
                    colors.append(bucket_colors[b])
            if data:
                parts = ax.violinplot(data, positions=positions, widths=0.75, showmedians=True, showextrema=False)
                for pc, c in zip(parts["bodies"], colors):
                    pc.set_facecolor(c)
                    pc.set_alpha(0.8)
                    pc.set_edgecolor("black")
                    pc.set_linewidth(0.6)
                if "cmedians" in parts:
                    parts["cmedians"].set_color("black")
                    parts["cmedians"].set_linewidth(1.2)
            ax.set_xticks(range(len(bucket_order)))
            ax.set_xticklabels(
                [f"{bucket_labels[b]}\nn={counts[i]:,}" for i, b in enumerate(bucket_order)],
                fontsize=9,
            )
            ax.set_title(label)
            ax.set_ylabel(label)
            ax.xaxis.grid(False)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def print_stats_table(rows: list[dict]) -> None:
    if not rows:
        return
    planes = sorted({r["plane"] for r in rows})
    by = {(r["plane"], r["pipeline"]): r for r in rows}

    def fmt(v, spec):
        return "    -" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)

    print()
    print(f"  {'plane':>5} | {'pipeline':<22} | {'accepted':>8} {'rejected':>8} {'accept%':>7} {'med_skew':>8} {'active':>6}")
    print("  " + "-" * 80)
    tot = {LBM_LABEL: [0, 0, 0], OTHER_LABEL: [0, 0, 0]}
    for p in planes:
        for label in (LBM_LABEL, OTHER_LABEL):
            r = by.get((p, label))
            if r is None:
                continue
            print(
                f"  {p:>5} | {label:<22} | "
                f"{r['n_accepted']:>8} {r['n_rejected']:>8} "
                f"{fmt(100 * r['accept_rate'], '6.1f'):>7} "
                f"{fmt(r['median_skew'], '8.3f'):>8} {r['n_active']:>6}"
            )
            tot[label][0] += r["n_accepted"]
            tot[label][1] += r["n_rejected"]
            tot[label][2] += r["n_active"]
    print("  " + "-" * 80)
    for label in (LBM_LABEL, OTHER_LABEL):
        a, rej, act = tot[label]
        total = a + rej
        rate = 100 * a / total if total else float("nan")
        print(f"  {'TOTAL':>5} | {label:<22} | {a:>8} {rej:>8} {rate:>7.1f} {'':>8} {act:>6}")
    print()


def parse_planes(spec: str | None, available: list[int]) -> list[int]:
    if spec is None or spec.lower() == "all":
        return available
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return [p for p in out if p in available]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lbm", required=True, type=Path, help="LBM-Suite2p-Python output root (contains zplaneNN_tpXXXXX-XXXXX/).")
    ap.add_argument("--other", required=True, type=Path, help="Reference pipeline root (contains plane_N/).")
    ap.add_argument("--out", required=True, type=Path, help="Directory to write comparison PNGs into.")
    ap.add_argument("--planes", default="all", help='Comma-separated plane indices, e.g. "1,3-5", or "all" (default).')
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    lbm_planes = discover_planes(args.lbm, LBM_DIR_RE)
    other_planes = discover_planes(args.other, OTHER_DIR_RE)
    common = sorted(set(lbm_planes) & set(other_planes))
    if not common:
        raise SystemExit(
            f"No matching planes found.\n  LBM:   {sorted(lbm_planes)}\n  other: {sorted(other_planes)}"
        )

    only_lbm = sorted(set(lbm_planes) - set(other_planes))
    only_other = sorted(set(other_planes) - set(lbm_planes))
    if only_lbm:
        print(f"[skip] in LBM only:   {only_lbm}")
    if only_other:
        print(f"[skip] in other only: {only_other}")

    targets = parse_planes(args.planes, common)
    if not targets:
        raise SystemExit(f"No planes selected. Available: {common}")

    print(f"Rendering {len(targets)} plane(s) -> {args.out}")
    stats_rows: list[dict] = []
    plane_records: list[dict] = []
    for p in targets:
        lbm_data = load_plane(lbm_planes[p])
        other_data = load_plane(other_planes[p])
        if lbm_data is None or other_data is None:
            missing = "LBM" if lbm_data is None else "other"
            print(f"  plane {p}: skipped ({missing} missing ops/stat/iscell)")
            continue
        out_path = args.out / "compare" / f"plane_{p:02d}.png"
        render_plane(p, lbm_data, other_data, out_path, dpi=args.dpi)
        lbm_ops, lbm_stat, lbm_acc = lbm_data
        ref_ops, ref_stat, ref_acc = other_data
        lbm_only, ref_only = compute_per_plane_unique_masks(lbm_stat, lbm_acc, ref_stat, ref_acc)
        unique_path = args.out / "unique" / f"plane_{p:02d}.png"
        render_plane_unique(p, lbm_data, other_data, lbm_only, ref_only, unique_path, dpi=args.dpi)
        stats_rows.append(compute_plane_stats(p, LBM_LABEL, lbm_stat, lbm_acc))
        stats_rows.append(compute_plane_stats(p, OTHER_LABEL, ref_stat, ref_acc))
        plane_records.append({
            "plane": p,
            "lbm": {"stat": lbm_stat, "accepted": lbm_acc, "reg": compute_registration_stats(lbm_ops)},
            "ref": {"stat": ref_stat, "accepted": ref_acc, "reg": compute_registration_stats(ref_ops)},
        })
        n_lbm = int(lbm_acc.sum())
        n_other = int(ref_acc.sum())
        n_lo = int(lbm_only.sum())
        n_ro = int(ref_only.sum())
        print(
            f"  plane {p:>2}: LBM={n_lbm:>4} cells, other={n_other:>4} cells, "
            f"unique LBM={n_lo:>4} / ref={n_ro:>4}"
        )

    if stats_rows:
        csv_path = args.out / "stats.csv"
        summary_path = args.out / "stats_summary.png"
        reg_path = args.out / "registration_quality.png"
        qual_path = args.out / "quality_distributions.png"
        agree_path = args.out / "classification_disagreement.png"
        write_stats_csv(stats_rows, csv_path)
        render_stats_summary(stats_rows, summary_path, dpi=args.dpi)
        render_registration_quality(plane_records, reg_path, dpi=args.dpi)
        render_quality_distributions(plane_records, qual_path, dpi=args.dpi)
        render_classification_disagreement(plane_records, agree_path, dpi=args.dpi)
        print_stats_table(stats_rows)
        print(f"  stats CSV          -> {csv_path}")
        print(f"  stats summary      -> {summary_path}")
        print(f"  registration       -> {reg_path}")
        print(f"  quality dists      -> {qual_path}")
        print(f"  classification     -> {agree_path}")


if __name__ == "__main__":
    main()
