"""View a suite2p volume (max projection + ROI masks) in napari.

Reads each plane's ops['max_proj'] and stat.npy directly; no consolidation step.
A slider in the napari dock controls the iscell probability threshold.

With ``--merge`` ROIs from adjacent z-planes that overlap (IoU >= ``--iou``)
are linked into connected components, so one physical cell that suite2p
detected on multiple planes shows up as a single label spanning those planes.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import napari
import numpy as np
from magicgui.widgets import CheckBox, Container, FloatSlider, PushButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _plane_dirs(suite2p_path: Path) -> list[Path]:
    dirs = sorted(suite2p_path.glob("zplane*"))
    if not dirs:
        dirs = sorted(suite2p_path.glob("plane*"))
    return [d for d in dirs if (d / "ops.npy").exists()]


def _plane_index(name: str) -> int:
    m = re.search(r"z?plane(\d+)", name)
    if m is None:
        raise ValueError(f"cannot parse plane index from {name!r}")
    return int(m.group(1))


def load_movie_4d(suite2p_path: Path):
    """Lazy (T, Z, Y, X) dask array stacked from each plane's ``data.bin``.

    Returns None if any plane has no data.bin or dask isn't available.
    """
    try:
        import dask.array as da
    except ImportError:
        print("dask not installed; --movie disabled (uv pip install dask)")
        return None

    plane_dirs = _plane_dirs(suite2p_path)
    arrays = []
    for pdir in plane_dirs:
        ops = np.load(pdir / "ops.npy", allow_pickle=True).item()
        bin_path = pdir / "data.bin"
        if not bin_path.exists():
            print(f"  {pdir.name}: no data.bin, can't build 4D movie")
            return None
        Ly_p, Lx_p, T_p = ops["Ly"], ops["Lx"], ops["nframes"]
        mm = np.memmap(bin_path, dtype=np.int16, mode="r",
                       shape=(T_p, Ly_p, Lx_p))
        arrays.append(da.from_array(mm, chunks=(1, Ly_p, Lx_p)))

    T = min(a.shape[0] for a in arrays)
    Ly = max(a.shape[1] for a in arrays)
    Lx = max(a.shape[2] for a in arrays)
    trimmed = []
    for a in arrays:
        a = a[:T]
        if a.shape[1] != Ly or a.shape[2] != Lx:
            a = da.pad(a, ((0, 0), (0, Ly - a.shape[1]), (0, Lx - a.shape[2])))
        trimmed.append(a)
    return da.stack(trimmed, axis=1)  # (T, Z, Y, X)


def _load_dff(pdir: Path) -> np.ndarray | None:
    if (pdir / "dff.npy").exists():
        return np.load(pdir / "dff.npy")
    F_path = pdir / "F.npy"
    Fneu_path = pdir / "Fneu.npy"
    if F_path.exists() and Fneu_path.exists():
        F = np.load(F_path)
        Fneu = np.load(Fneu_path)
        corr = F - 0.7 * Fneu
        base = np.percentile(corr, 20, axis=1, keepdims=True)
        return (corr - base) / (np.abs(base) + 1e-6)
    return None


@dataclass
class Roi:
    z: int
    ypix: np.ndarray
    xpix: np.ndarray
    prob: float
    med: tuple[float, float]
    skew: float = 0.0
    trace: np.ndarray | None = None
    plane_name: str = ""


@dataclass
class MergedRoi:
    members: list[Roi]
    prob: float = 0.0  # max prob across members
    z_planes: list[int] = field(default_factory=list)

    @property
    def trace(self) -> np.ndarray | None:
        traces = [m.trace for m in self.members if m.trace is not None]
        if not traces:
            return None
        T = min(t.shape[0] for t in traces)
        return np.mean(np.stack([t[:T] for t in traces], axis=0), axis=0)

    @property
    def best_member(self) -> Roi:
        return max(self.members, key=lambda m: m.prob)


def load_volume(suite2p_path: Path):
    """Return (proj_vol, rois, plane_labels).

    rois is a list of Roi — one per ROI across all planes.
    """
    plane_dirs = _plane_dirs(suite2p_path)
    if not plane_dirs:
        raise SystemExit(f"no plane folders in {suite2p_path}")

    ops0 = np.load(plane_dirs[0] / "ops.npy", allow_pickle=True).item()
    Ly, Lx = ops0["Ly"], ops0["Lx"]
    nz = len(plane_dirs)
    proj_vol = np.zeros((nz, Ly, Lx), dtype=np.float32)

    rois: list[Roi] = []
    plane_labels: list[int] = []

    for z, pdir in enumerate(plane_dirs):
        ops = np.load(pdir / "ops.npy", allow_pickle=True).item()
        img = ops.get("max_proj")
        if img is None:
            print(f"  plane {pdir.name}: no max_proj, falling back to meanImg")
            img = ops.get("meanImgE", ops.get("meanImg"))
        if img is not None:
            img = img.astype(np.float32, copy=False)
            if img.shape == (Ly, Lx):
                proj_vol[z] = img
            else:
                yr = ops.get("yrange", (0, img.shape[0]))
                xr = ops.get("xrange", (0, img.shape[1]))
                proj_vol[z, yr[0]:yr[0] + img.shape[0], xr[0]:xr[0] + img.shape[1]] = img

        plane_labels.append(_plane_index(pdir.name))

        stat_f = pdir / "stat.npy"
        iscell_f = pdir / "iscell.npy"
        if not stat_f.exists() or not iscell_f.exists():
            continue
        stat = np.load(stat_f, allow_pickle=True)
        iscell = np.load(iscell_f)
        dff = _load_dff(pdir)

        for i, s in enumerate(stat):
            ypix, xpix = s["ypix"], s["xpix"]
            valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
            if not valid.any():
                continue
            med = s.get("med", (float(ypix.mean()), float(xpix.mean())))
            rois.append(Roi(
                z=z, ypix=ypix[valid], xpix=xpix[valid],
                prob=float(iscell[i, 1]),
                med=(float(med[0]), float(med[1])),
                skew=float(s.get("skew", 0.0)),
                trace=dff[i] if dff is not None else None,
                plane_name=pdir.name,
            ))
        print(f"  plane {pdir.name}: {len(stat)} ROIs"
              f"{' (no trace data)' if dff is None else ''}")

    return proj_vol, rois, plane_labels


def merge_adjacent_planes(
    rois: list[Roi], iou_thresh: float = 0.3, max_dz: int = 1,
    centroid_radius: float = 20.0,
) -> list[MergedRoi]:
    """Union-find merge of ROIs whose IoU >= iou_thresh on planes within max_dz.

    centroid_radius prunes the O(n^2) candidate set by xy-distance.
    """
    n = len(rois)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_plane: dict[int, list[int]] = {}
    for i, r in enumerate(rois):
        by_plane.setdefault(r.z, []).append(i)

    pixsets = [set(zip(r.ypix.tolist(), r.xpix.tolist())) for r in rois]
    meds = np.array([r.med for r in rois], dtype=np.float32)

    planes = sorted(by_plane)
    for zi in planes:
        for zj in planes:
            if zj <= zi or zj - zi > max_dz:
                continue
            for i in by_plane[zi]:
                for j in by_plane[zj]:
                    if np.linalg.norm(meds[i] - meds[j]) > centroid_radius:
                        continue
                    inter = len(pixsets[i] & pixsets[j])
                    if inter == 0:
                        continue
                    iou = inter / len(pixsets[i] | pixsets[j])
                    if iou >= iou_thresh:
                        union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged: list[MergedRoi] = []
    for members_idx in clusters.values():
        members = [rois[i] for i in members_idx]
        merged.append(MergedRoi(
            members=members,
            prob=max(r.prob for r in members),
            z_planes=sorted({r.z for r in members}),
        ))
    return merged


@dataclass
class Ellipsoid:
    label_id: int          # stable index + 1
    prob: float
    bbox: tuple[slice, slice, slice]
    inside: np.ndarray     # bool, shape matches bbox
    z_planes: list[int]    # voxel-z indices covered


def _fit_ellipsoid(
    members: list[Roi], shape, scale, sigma_thresh: float,
    z_min_spread_vox: float = 0.5, xy_min_spread_vox: float = 0.5,
) -> tuple[tuple[slice, slice, slice], np.ndarray] | None:
    """Fit a 3D Gaussian to a cell's pixel cloud (in physical units) and
    return ``(bbox_in_voxels, bool_mask_inside_bbox)`` for the level set at
    ``sigma_thresh`` Mahalanobis distance.
    """
    sz, sy, sx = scale
    zs, ys, xs = [], [], []
    for r in members:
        n = len(r.ypix)
        if n == 0:
            continue
        zs.append(np.full(n, r.z * sz, dtype=np.float64))
        ys.append(r.ypix.astype(np.float64) * sy)
        xs.append(r.xpix.astype(np.float64) * sx)
    if not zs:
        return None
    P = np.column_stack([np.concatenate(zs), np.concatenate(ys), np.concatenate(xs)])
    mu = P.mean(axis=0)
    Pc = P - mu
    cov = (Pc.T @ Pc) / max(len(P) - 1, 1)
    cov[0, 0] += (z_min_spread_vox * sz) ** 2
    cov[1, 1] += (xy_min_spread_vox * sy) ** 2
    cov[2, 2] += (xy_min_spread_vox * sx) ** 2
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None
    evals, _ = np.linalg.eigh(cov)
    max_radius_phys = float(np.sqrt(max(evals, default=0.0))) * sigma_thresh
    mu_vox = mu / np.array([sz, sy, sx])
    rad_vox = max_radius_phys / np.array([sz, sy, sx])

    z0 = max(0, int(np.floor(mu_vox[0] - rad_vox[0])))
    y0 = max(0, int(np.floor(mu_vox[1] - rad_vox[1])))
    x0 = max(0, int(np.floor(mu_vox[2] - rad_vox[2])))
    z1 = min(shape[0], int(np.ceil(mu_vox[0] + rad_vox[0])) + 1)
    y1 = min(shape[1], int(np.ceil(mu_vox[1] + rad_vox[1])) + 1)
    x1 = min(shape[2], int(np.ceil(mu_vox[2] + rad_vox[2])) + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return None

    zz = np.arange(z0, z1, dtype=np.float64) * sz - mu[0]
    yy = np.arange(y0, y1, dtype=np.float64) * sy - mu[1]
    xx = np.arange(x0, x1, dtype=np.float64) * sx - mu[2]
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
    pts = np.stack([Z, Y, X], axis=-1)
    d2 = np.einsum("...i,ij,...j->...", pts, inv_cov, pts)
    inside = d2 <= sigma_thresh ** 2
    if not inside.any():
        return None
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1)), inside


def fit_ellipsoids_flat(
    rois: list[Roi], shape, scale, sigma_thresh: float,
) -> list[Ellipsoid | None]:
    out: list[Ellipsoid | None] = []
    for idx, r in enumerate(rois):
        result = _fit_ellipsoid([r], shape, scale, sigma_thresh)
        if result is None:
            out.append(None)
            continue
        bbox, inside = result
        out.append(Ellipsoid(
            label_id=idx + 1, prob=r.prob, bbox=bbox, inside=inside,
            z_planes=[r.z],
        ))
    return out


def fit_ellipsoids_merged(
    merged: list[MergedRoi], shape, scale, sigma_thresh: float,
) -> list[Ellipsoid | None]:
    out: list[Ellipsoid | None] = []
    for idx, m in enumerate(merged):
        result = _fit_ellipsoid(m.members, shape, scale, sigma_thresh)
        if result is None:
            out.append(None)
            continue
        bbox, inside = result
        out.append(Ellipsoid(
            label_id=idx + 1, prob=m.prob, bbox=bbox, inside=inside,
            z_planes=list(m.z_planes),
        ))
    return out


def render_ellipsoids(
    ellipsoids: list[Ellipsoid | None], shape, threshold: float,
) -> np.ndarray:
    """Paint pre-fit ellipsoid masks; first-written-wins for overlapping voxels."""
    mask = np.zeros(shape, dtype=np.uint32)
    for e in ellipsoids:
        if e is None or e.prob <= threshold:
            continue
        sub = mask[e.bbox]
        write = e.inside & (sub == 0)
        sub[write] = e.label_id
    return mask


def render_masks_flat(rois: list[Roi], shape, threshold: float) -> np.ndarray:
    """Stable labels: ``mask[r.z, ypix, xpix] = roi_index + 1``."""
    mask = np.zeros(shape, dtype=np.uint32)
    for idx, r in enumerate(rois):
        if r.prob <= threshold:
            continue
        mask[r.z, r.ypix, r.xpix] = idx + 1
    return mask


def render_masks_merged(
    merged: list[MergedRoi], shape, threshold: float,
) -> np.ndarray:
    """Stable labels: ``mask[...] = cluster_index + 1`` (same value across planes)."""
    mask = np.zeros(shape, dtype=np.uint32)
    for idx, m in enumerate(merged):
        if m.prob <= threshold:
            continue
        for r in m.members:
            mask[r.z, r.ypix, r.xpix] = idx + 1
    return mask


def _format_info(
    args,
    proj_vol_shape,
    plane_labels: list[int],
    rois: list[Roi],
    merged: list[MergedRoi] | None,
    threshold: float,
    refined: bool = False,
) -> str:
    nz, ny, nx = proj_vol_shape
    n_total = len(rois)

    probs = np.array([r.prob for r in rois], dtype=np.float32) if rois else np.zeros(0)
    n_acc = int((probs > threshold).sum())
    n_rej = n_total - n_acc

    per_plane_acc = np.zeros(nz, dtype=int)
    per_plane_tot = np.zeros(nz, dtype=int)
    for r in rois:
        per_plane_tot[r.z] += 1
        if r.prob > threshold:
            per_plane_acc[r.z] += 1

    rows = [
        '<table style="border-collapse:collapse;">'
        '<tr><th style="text-align:left;padding-right:8px;">source</th>'
        f'<td style="word-break:break-all;">{args.suite2p_path}</td></tr>'
        f'<tr><th style="text-align:left;padding-right:8px;">volume</th>'
        f'<td>{nz} &times; {ny} &times; {nx}</td></tr>'
        f'<tr><th style="text-align:left;padding-right:8px;">scale (Z,Y,X)</th>'
        f'<td>{args.scale[0]:g}, {args.scale[1]:g}, {args.scale[2]:g}</td></tr>'
        '</table>'
    ]

    rows.append('<hr style="margin:4px 0;">')
    rows.append(
        '<b>threshold</b><br>'
        f'iscell[:,1] &gt; <b>{threshold:.2f}</b><br>'
        f'accepted: <b>{n_acc}</b>&nbsp;&nbsp;rejected: <b>{n_rej}</b><br>'
        f'({n_acc / max(n_total, 1):.1%} of {n_total})'
    )

    if merged is not None:
        multi = [m for m in merged if len(m.z_planes) > 1]
        merged_probs = np.array([m.prob for m in merged], dtype=np.float32)
        n_groups_visible = int((merged_probs > threshold).sum())
        multi_probs = np.array([m.prob for m in multi], dtype=np.float32)
        n_multi_visible = int((multi_probs > threshold).sum()) if multi else 0
        sizes = [len(m.z_planes) for m in merged]
        size_hist: dict[int, int] = {}
        for s in sizes:
            size_hist[s] = size_hist.get(s, 0) + 1
        rows.append('<hr style="margin:4px 0;">')
        rows.append(
            '<b>merge</b><br>'
            f'IoU &ge; <b>{args.iou:.2f}</b>, max &Delta;z = <b>{args.max_dz}</b><br>'
            f'{n_total} ROIs &rarr; <b>{len(merged)}</b> groups<br>'
            f'span &gt;1 plane: <b>{len(multi)}</b><br>'
            f'visible: <b>{n_groups_visible}</b> ({n_multi_visible} multi-plane)<br>'
            'span (planes &rarr; #groups):<br>'
            + ', '.join(f'{k}&rarr;{v}' for k, v in sorted(size_hist.items()))
        )

    if refined:
        rows.append('<hr style="margin:4px 0;">')
        rows.append(
            '<b>refine</b><br>'
            f'3D ellipsoid fit at <b>{args.sigma_fit:.2f}&sigma;</b><br>'
            'masks are smooth ovals fit to the pixel cloud<br>'
            '(anisotropy from --scale)'
        )

    rows.append('<hr style="margin:4px 0;">')
    per_plane_rows = ['<b>per plane</b><br>'
                      '<table style="border-collapse:collapse;font-family:monospace;">'
                      '<tr><th style="text-align:right;padding:0 6px;">plane</th>'
                      '<th style="text-align:right;padding:0 6px;">acc</th>'
                      '<th style="text-align:right;padding:0 6px;">total</th></tr>']
    for z in range(nz):
        per_plane_rows.append(
            f'<tr><td style="text-align:right;padding:0 6px;">{plane_labels[z]}</td>'
            f'<td style="text-align:right;padding:0 6px;">{per_plane_acc[z]}</td>'
            f'<td style="text-align:right;padding:0 6px;">{per_plane_tot[z]}</td></tr>'
        )
    per_plane_rows.append('</table>')
    rows.append(''.join(per_plane_rows))

    return ''.join(rows)


class TracePanel(QWidget):
    """Embedded matplotlib trace + status label."""

    def __init__(self) -> None:
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        self.status = QLabel(
            "click an ROI to view its trace<br>"
            "<span style='color:#888;'>tip: press <b>5</b> for napari's eyedropper "
            "(precise 3D picking)</span>"
        )
        self.status.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText)
        v.addWidget(self.status)

        self.fig = Figure(figsize=(3.2, 1.8), facecolor="#1e1e1e")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(140)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="#aaa", labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_color("#444")
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.18)
        v.addWidget(self.canvas)

        self.cursor = None
        self.trace_len = 0

    def show_trace(self, title: str, trace: np.ndarray | None) -> None:
        self.status.setText(title)
        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="#aaa", labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_color("#444")
        self.cursor = None
        self.trace_len = 0
        if trace is None:
            self.ax.text(0.5, 0.5, "no trace data", color="#888",
                         ha="center", va="center", transform=self.ax.transAxes)
        else:
            self.ax.plot(trace, color="#7ec8ff", linewidth=0.8)
            self.ax.set_xlim(0, len(trace) - 1)
            lo, hi = np.percentile(trace, [1, 99.5])
            margin = (hi - lo) * 0.1 + 1e-3
            self.ax.set_ylim(lo - margin, hi + margin)
            self.ax.set_xlabel("frame", color="#aaa", fontsize=7)
            self.ax.set_ylabel("Δf/f", color="#aaa", fontsize=7)
            self.cursor = self.ax.axvline(0, color="#ff5555", lw=0.8, alpha=0.85)
            self.trace_len = len(trace)
        self.canvas.draw_idle()

    def set_cursor(self, t: int) -> None:
        if self.cursor is None or self.trace_len == 0:
            return
        t = int(np.clip(t, 0, self.trace_len - 1))
        self.cursor.set_xdata([t, t])
        self.canvas.draw_idle()


def _build_dock(controls: Container, info_label: QLabel,
                trace_panel: TracePanel) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(6)
    layout.addWidget(controls.native)
    line1 = QFrame()
    line1.setFrameShape(QFrame.HLine)
    line1.setFrameShadow(QFrame.Sunken)
    layout.addWidget(line1)
    layout.addWidget(trace_panel)
    line2 = QFrame()
    line2.setFrameShape(QFrame.HLine)
    line2.setFrameShadow(QFrame.Sunken)
    layout.addWidget(line2)
    scroll = QScrollArea()
    scroll.setWidget(info_label)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    layout.addWidget(scroll, stretch=1)
    return container


def _record_orbit(viewer, out_path: Path, n_steps: int = 240) -> None:
    try:
        from napari_animation import Animation
    except ImportError:
        print("napari-animation not installed; install with: "
              "uv pip install napari-animation")
        return
    viewer.dims.ndisplay = 3
    anim = Animation(viewer)
    viewer.camera.angles = (0, 0, 90)
    anim.capture_keyframe(steps=1)
    viewer.camera.angles = (0, 180, 90)
    anim.capture_keyframe(steps=n_steps // 2)
    viewer.camera.angles = (0, 359.9, 90)
    anim.capture_keyframe(steps=n_steps // 2)
    print(f"rendering orbit to {out_path} ...")
    anim.animate(str(out_path), fps=30, quality=7, canvas_only=True)
    print(f"saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite2p_path", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="initial iscell probability threshold (default 0.5)")
    ap.add_argument("--scale", nargs=3, type=float, default=(1.0, 1.0, 1.0),
                    metavar=("Z", "Y", "X"))
    ap.add_argument("--ndisplay", type=int, default=3, choices=(2, 3))
    ap.add_argument("--merge", action="store_true",
                    help="merge ROIs across adjacent planes via IoU")
    ap.add_argument("--iou", type=float, default=0.3,
                    help="IoU threshold for merging (default 0.3)")
    ap.add_argument("--max-dz", type=int, default=1,
                    help="max plane distance for merging (default 1)")
    ap.add_argument("--no-movie", action="store_true",
                    help="skip loading the lazy 4D movie from data.bin")
    ap.add_argument("--refine", action="store_true",
                    help="fit 3D ellipsoids and render them as smooth surface "
                         "meshes (uses --scale for anisotropy)")
    ap.add_argument("--sigma-fit", type=float, default=1.5,
                    help="ellipsoid radius in std devs (default 1.5)")
    args = ap.parse_args()

    proj_vol, rois, plane_labels = load_volume(args.suite2p_path)
    print(f"volume: {proj_vol.shape}  planes: {plane_labels}  total ROIs: {len(rois)}")

    merged: list[MergedRoi] | None = None
    if args.merge:
        merged = merge_adjacent_planes(
            rois, iou_thresh=args.iou, max_dz=args.max_dz,
        )
        multi = [m for m in merged if len(m.z_planes) > 1]
        print(f"merged: {len(rois)} -> {len(merged)} groups "
              f"({len(multi)} span >1 plane)")

    ellipsoids: list[Ellipsoid | None] | None = None
    if args.refine:
        print(f"fitting ellipsoids (sigma_thresh={args.sigma_fit}) ...")
        if merged is not None:
            ellipsoids = fit_ellipsoids_merged(
                merged, proj_vol.shape, tuple(args.scale), args.sigma_fit,
            )
        else:
            ellipsoids = fit_ellipsoids_flat(
                rois, proj_vol.shape, tuple(args.scale), args.sigma_fit,
            )
        n_ok = sum(1 for e in ellipsoids if e is not None)
        print(f"  fitted {n_ok}/{len(ellipsoids)} ellipsoids")
        renderer = lambda thr: render_ellipsoids(ellipsoids, proj_vol.shape, thr)
        layer_name = "ROIs (ellipsoid)" if merged is None else "ROIs (merged, ellipsoid)"
    elif merged is not None:
        renderer = lambda thr: render_masks_merged(merged, proj_vol.shape, thr)
        layer_name = "ROIs (merged)"
    else:
        renderer = lambda thr: render_masks_flat(rois, proj_vol.shape, thr)
        layer_name = "ROIs"

    mask_vol = renderer(args.threshold)
    print(f"visible @ threshold={args.threshold}: {int(mask_vol.max())}")

    movie_4d = None if args.no_movie else load_movie_4d(args.suite2p_path)
    has_movie = movie_4d is not None
    if has_movie:
        print(f"movie: shape (T,Z,Y,X)={movie_4d.shape}, dtype={movie_4d.dtype}")

    viewer = napari.Viewer(ndisplay=args.ndisplay, title="suite2p volume")
    viewer.dims.axis_labels = ("T", "Z", "Y", "X") if has_movie else ("Z", "Y", "X")
    lo, hi = np.percentile(proj_vol[proj_vol > 0], [1, 99]) if proj_vol.any() else (0, 1)
    image_layer = viewer.add_image(
        proj_vol, name="max_proj", scale=tuple(args.scale),
        colormap="gray", contrast_limits=[float(lo), float(hi)],
        blending="translucent", opacity=0.8,
        visible=not has_movie,
    )
    movie_layer = None
    if has_movie:
        sample = np.asarray(movie_4d[0])
        m_lo, m_hi = np.percentile(sample, [1, 99.5])
        movie_layer = viewer.add_image(
            movie_4d, name="data.bin (T,Z,Y,X)",
            scale=(1.0,) + tuple(args.scale),
            colormap="gray", contrast_limits=[float(m_lo), float(m_hi)],
            blending="translucent", opacity=0.9,
        )
    mask_layer = viewer.add_labels(
        mask_vol, name=layer_name, scale=tuple(args.scale),
        opacity=0.7, blending="translucent",
    )

    info_label = QLabel()
    info_label.setTextFormat(Qt.RichText)
    info_label.setWordWrap(True)
    info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    info_label.setStyleSheet("QLabel { font-size: 11px; padding: 4px; }")

    def refresh_info(thr: float) -> None:
        info_label.setText(
            _format_info(args, proj_vol.shape, plane_labels, rois, merged, thr,
                         refined=ellipsoids is not None)
        )

    threshold_w = FloatSlider(
        name="threshold", min=0.0, max=1.0, step=0.01, value=args.threshold,
    )
    ndisplay_chk = CheckBox(name="3D view", value=(args.ndisplay == 3))
    isolate_chk = CheckBox(name="isolate selected ROI", value=False)
    record_btn = PushButton(name="record orbit (mp4)")

    def on_threshold(v: float) -> None:
        mask_layer.data = renderer(v)
        refresh_info(v)

    def on_record() -> None:
        out = args.suite2p_path / "volume_orbit.mp4"
        _record_orbit(viewer, out)

    threshold_w.changed.connect(on_threshold)
    ndisplay_chk.changed.connect(
        lambda v: setattr(viewer.dims, "ndisplay", 3 if v else 2)
    )
    isolate_chk.changed.connect(
        lambda v: setattr(mask_layer, "show_selected_label", bool(v))
    )
    record_btn.clicked.connect(on_record)

    # keep the dock checkbox in sync if user uses napari's built-in 2D/3D button
    def _sync_ndisplay(event=None):
        ndisplay_chk.value = viewer.dims.ndisplay == 3
    viewer.dims.events.ndisplay.connect(_sync_ndisplay)

    controls = Container(
        widgets=[threshold_w, ndisplay_chk, isolate_chk, record_btn],
        labels=True,
    )

    trace_panel = TracePanel()
    mask_layer.show_selected_label = False

    def lookup(label_id: int) -> tuple[str, np.ndarray | None]:
        if label_id <= 0:
            return "click an ROI to view its trace", None
        idx = label_id - 1
        if merged is not None:
            if idx >= len(merged):
                return "label out of range", None
            m = merged[idx]
            best = m.best_member
            title = (f"<b>cluster #{idx + 1}</b>  planes={m.z_planes}  "
                     f"members={len(m.members)}  "
                     f"prob={m.prob:.2f}  skew={best.skew:.2f}")
            return title, m.trace
        if idx >= len(rois):
            return "label out of range", None
        r = rois[idx]
        title = (f"<b>ROI #{idx + 1}</b>  {r.plane_name}  "
                 f"prob={r.prob:.2f}  skew={r.skew:.2f}")
        return title, r.trace

    def show_label(label_id: int) -> None:
        title, trace = lookup(label_id)
        trace_panel.status.setTextFormat(Qt.RichText)
        trace_panel.show_trace(title, trace)
        if has_movie and trace is not None:
            trace_panel.set_cursor(int(viewer.dims.current_step[0]))

    # napari fires this whenever the labels layer's selected_label changes —
    # via our manual click below, the built-in eyedropper (mode='pick',
    # shortcut '5'), or the spinbox in the layer-controls panel.
    def on_selected_label(event=None):
        show_label(int(mask_layer.selected_label))

    mask_layer.events.selected_label.connect(on_selected_label)

    @mask_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        if event.type != "mouse_press":
            return
        if "Shift" in event.modifiers or "Control" in event.modifiers:
            return
        try:
            value = layer.get_value(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            label_id = int(value or 0)
        except (TypeError, ValueError):
            label_id = 0
        if label_id == 0:
            trace_panel.show_trace(
                "no ROI under cursor — try napari's eyedropper "
                "(press <b>5</b>) for precise 3D picking", None,
            )
            return
        # setting selected_label triggers on_selected_label which updates the panel
        layer.selected_label = label_id

    if has_movie:
        def on_step(event=None):
            trace_panel.set_cursor(int(viewer.dims.current_step[0]))

        viewer.dims.events.current_step.connect(on_step)

    refresh_info(args.threshold)
    # focus the Z slider so up/down arrow keys + scroll wheel walk z-planes
    viewer.dims.last_used = 1 if has_movie else 0
    dock = _build_dock(controls, info_label, trace_panel)
    viewer.window.add_dock_widget(dock, area="right", name="ROIs")
    napari.run()


if __name__ == "__main__":
    main()
