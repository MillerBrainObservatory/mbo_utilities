"""Render an mp4 of the top-N suite2p traces beside the per-frame max projection.

For each plane in ``suite2p_path``:
  - reads ``stat.npy``, ``iscell.npy``, ``F.npy``, ``Fneu.npy`` (or ``dff.npy`` if present),
  - memory-maps ``data.bin`` for per-frame display.

Picks the top-N accepted cells across all planes by ``--metric``
(``skew``, ``prob``, ``snr``, or ``skew*prob`` — the default).

Layout:
  left  = Z max-projection of the registered movie at frame t
          with the top-N ROI footprints outlined in their trace colors
  right = stacked Δf/f traces, vertical cursor at frame t

Output: ``<suite2p_path>/traces_movie.mp4`` (override with ``--out``).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import to_rgba


def _plane_dirs(suite2p_path: Path) -> list[Path]:
    dirs = sorted(suite2p_path.glob("zplane*"))
    if not dirs:
        dirs = sorted(suite2p_path.glob("plane*"))
    return [d for d in dirs if (d / "ops.npy").exists()]


def _plane_index(name: str) -> int:
    m = re.search(r"z?plane(\d+)", name)
    return int(m.group(1)) if m else -1


def _compute_dff(F: np.ndarray, Fneu: np.ndarray, neucoeff: float = 0.7) -> np.ndarray:
    corr = F - neucoeff * Fneu
    base = np.percentile(corr, 20, axis=1, keepdims=True)
    return (corr - base) / (np.abs(base) + 1e-6)


def _score(cell: dict, metric: str) -> float:
    skew = abs(cell.get("skew", 0.0))
    prob = float(cell.get("prob", 0.0))
    snr = float(cell.get("snr", 0.0))
    if metric == "skew":
        return skew
    if metric == "prob":
        return prob
    if metric == "snr":
        return snr
    if metric == "skew*prob":
        return skew * prob
    raise ValueError(f"unknown metric: {metric}")


def load_planes(suite2p_path: Path) -> tuple[list[dict], int, int, int]:
    plane_dirs = _plane_dirs(suite2p_path)
    if not plane_dirs:
        raise SystemExit(f"no plane folders in {suite2p_path}")

    planes: list[dict] = []
    Ly = Lx = None
    T = None
    for pd in plane_dirs:
        ops = np.load(pd / "ops.npy", allow_pickle=True).item()
        if not (pd / "stat.npy").exists() or not (pd / "iscell.npy").exists():
            print(f"  skip {pd.name}: missing stat/iscell")
            continue
        stat = np.load(pd / "stat.npy", allow_pickle=True)
        iscell = np.load(pd / "iscell.npy")
        F = np.load(pd / "F.npy") if (pd / "F.npy").exists() else None
        Fneu = np.load(pd / "Fneu.npy") if (pd / "Fneu.npy").exists() else None
        if (pd / "dff.npy").exists():
            dff = np.load(pd / "dff.npy")
        elif F is not None and Fneu is not None:
            dff = _compute_dff(F, Fneu)
        else:
            print(f"  skip {pd.name}: no trace data")
            continue

        Ly_p, Lx_p = ops["Ly"], ops["Lx"]
        T_p = ops["nframes"]
        if Ly is None:
            Ly, Lx = Ly_p, Lx_p
        if T is None:
            T = T_p
        else:
            T = min(T, T_p)

        bin_path = pd / "data.bin"
        mm = None
        if bin_path.exists():
            mm = np.memmap(bin_path, dtype=np.int16, mode="r",
                           shape=(T_p, Ly_p, Lx_p))

        planes.append({
            "name": pd.name,
            "z": _plane_index(pd.name),
            "ops": ops,
            "stat": stat,
            "iscell": iscell,
            "dff": dff[:, :T_p],
            "mm": mm,
            "Ly": Ly_p, "Lx": Lx_p, "T": T_p,
        })
        print(f"  {pd.name}: {len(stat)} ROIs ({int((iscell[:,0]==1).sum())} accepted), T={T_p}")

    return planes, Ly, Lx, T


def pick_top_cells(planes: list[dict], n_top: int, metric: str,
                   min_prob: float = 0.5) -> list[dict]:
    cells: list[dict] = []
    for plane in planes:
        stat = plane["stat"]
        iscell = plane["iscell"]
        dff = plane["dff"]
        for i, s in enumerate(stat):
            if iscell[i, 0] == 0 or iscell[i, 1] < min_prob:
                continue
            trace = dff[i]
            # cheap MAD-based SNR
            noise = np.median(np.abs(np.diff(trace))) / 0.6745 + 1e-6
            snr = float(np.std(trace) / noise)
            cells.append({
                "plane_name": plane["name"],
                "z_label": plane["z"],
                "ypix": s["ypix"], "xpix": s["xpix"],
                "trace": trace,
                "prob": float(iscell[i, 1]),
                "skew": float(s.get("skew", 0.0)),
                "snr": snr,
            })
    if not cells:
        raise SystemExit("no accepted ROIs above min_prob")
    cells.sort(key=lambda c: -_score(c, metric))
    return cells[:n_top]


def build_animation(
    planes: list[dict], top: list[dict], Ly: int, Lx: int, T: int,
    fps: int, out_path: Path, dpi: int,
) -> None:
    n_top = len(top)
    fig = plt.figure(figsize=(12, 0.5 * n_top + 2), dpi=dpi, facecolor="black")
    gs = fig.add_gridspec(n_top, 2, width_ratios=[1.4, 1.0],
                          wspace=0.05, hspace=0.0, left=0.02, right=0.98,
                          top=0.97, bottom=0.05)

    ax_vol = fig.add_subplot(gs[:, 0])
    ax_vol.set_facecolor("black")
    ax_vol.set_xticks([]); ax_vol.set_yticks([])
    for sp in ax_vol.spines.values():
        sp.set_color("white")

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, n_top))

    def get_frame_max(t: int) -> np.ndarray:
        frames = []
        for p in planes:
            if p["mm"] is None:
                frames.append(np.zeros((Ly, Lx), dtype=np.int16))
            else:
                f = p["mm"][t]
                if f.shape != (Ly, Lx):
                    pad = np.zeros((Ly, Lx), dtype=f.dtype)
                    pad[:f.shape[0], :f.shape[1]] = f
                    f = pad
                frames.append(f)
        return np.max(np.stack(frames, axis=0), axis=0)

    img0 = get_frame_max(0)
    samples = [get_frame_max(t) for t in np.linspace(0, T - 1, 8, dtype=int)]
    vmin, vmax = np.percentile(np.stack(samples), [2, 99.5])
    im = ax_vol.imshow(img0, cmap="gray", vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="equal")
    ax_vol.set_title(f"max-Z projection (t=0/{T})", color="white", fontsize=10)

    for cell, color in zip(top, colors):
        ax_vol.scatter(cell["xpix"], cell["ypix"],
                       c=[color], s=2, alpha=0.6, linewidths=0)

    cursors = []
    for i, (cell, color) in enumerate(zip(top, colors)):
        ax = fig.add_subplot(gs[i, 1])
        ax.set_facecolor("black")
        trace = cell["trace"][:T]
        ax.plot(trace, color=color, lw=0.7)
        cur = ax.axvline(0, color="white", lw=0.7, alpha=0.8)
        cursors.append(cur)
        ax.set_xlim(0, T - 1)
        lo, hi = np.percentile(trace, [1, 99.5])
        margin = (hi - lo) * 0.1 + 1e-3
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.text(0.01, 0.5,
                f"z{cell['z_label']}  skew={cell['skew']:.1f}  p={cell['prob']:.2f}",
                transform=ax.transAxes, color=to_rgba(color, 0.95),
                fontsize=7, va="center", ha="left")

    title = ax_vol.title

    def update(t: int):
        im.set_data(get_frame_max(int(t)))
        title.set_text(f"max-Z projection (t={int(t)}/{T})")
        for cur in cursors:
            cur.set_xdata([t, t])
        return [im, title, *cursors]

    print(f"rendering {T} frames at {fps} fps -> {out_path}")
    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    ani.save(str(out_path), writer=writer, dpi=dpi,
             savefig_kwargs={"facecolor": "black"})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suite2p_path", type=Path)
    ap.add_argument("--n-top", type=int, default=12)
    ap.add_argument("--metric", default="skew*prob",
                    choices=("skew", "prob", "snr", "skew*prob"))
    ap.add_argument("--min-prob", type=float, default=0.5,
                    help="minimum iscell prob to be eligible (default 0.5)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames (debug; default: all)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"loading planes from {args.suite2p_path}")
    planes, Ly, Lx, T = load_planes(args.suite2p_path)
    if args.max_frames:
        T = min(T, args.max_frames)
    print(f"\n{len(planes)} planes, FOV={Ly}x{Lx}, T={T}")

    top = pick_top_cells(planes, args.n_top, args.metric, args.min_prob)
    print(f"\ntop {len(top)} cells by {args.metric}:")
    for c in top:
        print(f"  z{c['z_label']:>3}  skew={c['skew']:6.2f}  "
              f"p={c['prob']:.3f}  snr={c['snr']:.2f}")

    out_path = args.out or (args.suite2p_path / "traces_movie.mp4")
    build_animation(planes, top, Ly, Lx, T, args.fps, out_path, args.dpi)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
