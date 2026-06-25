"""
Scan-phase analysis for bidirectional scanning correction.

Measures the even/odd row offset to determine optimal correction parameters.
Two diagnostics, each computed with FFT (subpixel) and without FFT (integer):

- offset vs temporal averaging window length
- offset spatial distribution across the FOV
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm

from mbo_utilities import log
from mbo_utilities.analysis.phasecorr import _phase_corr_2d

logger = log.get("analysis.scanphase")


# MBO dark theme colors (consistent with benchmarks.py and docs/_static/custom.css)
MBO_DARK_THEME = {
    "background": "#121212",
    "surface": "#1e1e1e",
    "text": "#e0e0e0",
    "text_muted": "#9e9e9e",
    "border": "#333333",
    "primary": "#82aaff",  # blue
    "secondary": "#c792ea",  # purple
    "success": "#c3e88d",  # green
    "warning": "#ffcb6b",  # yellow
    "error": "#f07178",  # red
    "accent": "#89ddff",  # cyan
    "orange": "#f78c6c",
}


@dataclass
class ScanPhaseResults:
    """Results from scan-phase analysis."""

    # window length sweep (offset vs frames averaged)
    window_sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    window_offsets_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    window_stds_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    window_offsets_int: np.ndarray = field(default_factory=lambda: np.array([]))
    window_stds_int: np.ndarray = field(default_factory=lambda: np.array([]))

    # spatial grid offsets across the FOV
    patch_size: int = 0
    grid_offsets_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    grid_offsets_int: np.ndarray = field(default_factory=lambda: np.array([]))
    grid_valid: np.ndarray = field(default_factory=lambda: np.array([]))

    # flat per-patch offsets (for summary stats)
    offsets_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_int: np.ndarray = field(default_factory=lambda: np.array([]))

    # metadata
    num_timepoints: int = 0
    num_planes: int = 1
    num_rois: int = 1
    frame_shape: tuple = ()
    pixel_resolution_um: float = 0.0
    analysis_time: float = 0.0

    def compute_stats(self, arr):
        arr = np.asarray(arr, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
        }

    def get_summary(self):
        summary = {
            "metadata": {
                "num_timepoints": self.num_timepoints,
                "num_planes": self.num_planes,
                "num_rois": self.num_rois,
                "frame_shape": self.frame_shape,
                "analysis_time": self.analysis_time,
            }
        }
        if len(self.offsets_fft) > 0:
            summary["fft"] = self.compute_stats(self.offsets_fft)
        if len(self.offsets_int) > 0:
            summary["int"] = self.compute_stats(self.offsets_int)
        return summary


def _apply_mbo_style(ax, fig=None):
    """Apply MBO dark theme to matplotlib axes."""
    colors = MBO_DARK_THEME
    ax.set_facecolor(colors["surface"])
    if fig:
        fig.patch.set_facecolor(colors["background"])
    for spine in ax.spines.values():
        spine.set_color(colors["border"])
    ax.tick_params(colors=colors["text_muted"], which="both")
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])
    ax.grid(True, alpha=0.2, color=colors["text_muted"], linestyle="-", linewidth=0.5)


def _mbo_fig(*args, **kwargs):
    """Create figure with MBO dark theme."""
    import matplotlib.pyplot as plt
    colors = MBO_DARK_THEME
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(colors["background"])
    if hasattr(axes, "__iter__"):
        for ax in np.array(axes).flat:
            _apply_mbo_style(ax, fig)
    else:
        _apply_mbo_style(axes, fig)
    return fig, axes


def _mbo_colorbar(im, ax, label=None):
    """Create colorbar with MBO dark theme styling."""
    import matplotlib.pyplot as plt
    colors = MBO_DARK_THEME
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=colors["text_muted"])
    cbar.outline.set_edgecolor(colors["border"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=colors["text_muted"])
    if label:
        cbar.set_label(label, color=colors["text"])
    return cbar


class ScanPhaseAnalyzer:
    """
    Analyzer for bidirectional scan-phase offset.

    Accepts any mbo LazyArray (canonical 5D TCZYX) or a raw 2D-5D ndarray.
    Measures the offset as a function of temporal averaging window and of
    spatial position, each with and without subpixel FFT refinement.
    """

    def __init__(self, data, channel=0, plane=0):
        self.data = data
        self.channel = channel
        self.plane = plane
        self.shape = tuple(data.shape)
        self.ndim = len(self.shape)

        nt, nz, ny, nx = self._dims_from_shape(self.ndim, self.shape)
        self.nt = nt
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.num_planes = int(getattr(data, "num_planes", nz) or nz)

        self.num_rois = int(getattr(data, "num_rois", 1) or 1)

        dx = getattr(data, "dx", None)
        if dx is None:
            md = getattr(data, "metadata", None) or {}
            dx = md.get("dx", 0.0)
        self.pixel_resolution_um = float(dx or 0.0)

        self.results = ScanPhaseResults(
            num_timepoints=self.nt,
            num_planes=self.num_planes,
            num_rois=self.num_rois,
            frame_shape=(self.ny, self.nx),
            pixel_resolution_um=self.pixel_resolution_um,
        )
        logger.info(
            f"ScanPhaseAnalyzer: {self.nt} timepoints, {self.num_planes} planes, "
            f"shape={self.shape}"
        )

    @staticmethod
    def _dims_from_shape(ndim, shape):
        """Return (nt, nz, ny, nx) from a 2D-5D shape."""
        if ndim == 2:
            return 1, 1, shape[0], shape[1]
        if ndim == 3:
            return shape[0], 1, shape[1], shape[2]
        if ndim == 4:
            return shape[0], shape[1], shape[2], shape[3]
        # 5D canonical TCZYX (LazyArray or raw)
        return shape[0], shape[2], shape[3], shape[4]

    def _frame(self, t):
        """Return a single 2D frame for timepoint `t` at the active channel/plane."""
        d = self.data
        if self.ndim == 2:
            img = np.asarray(d)
        elif self.ndim == 3:
            img = np.asarray(d[t])
        elif self.ndim == 4:
            img = np.asarray(d[t, self.plane])
        else:
            img = np.asarray(d[t, self.channel, self.plane])
        while img.ndim > 2:
            img = img[0]
        return img

    def _mean_frame(self, indices):
        """Mean frame over `indices`, accumulated one frame at a time."""
        acc = None
        n = 0
        for i in indices:
            f = self._frame(i).astype(np.float64)
            acc = f if acc is None else acc + f
            n += 1
        if acc is None:
            return None
        return (acc / n).astype(np.float32)

    def _window_starts(self, ws, num_samples):
        """Evenly spaced start indices for non-overlapping windows of length `ws`."""
        n_possible = max(1, self.nt // ws)
        n = min(num_samples, n_possible)
        if n <= 1:
            return [0]
        return np.linspace(0, self.nt - ws, n, dtype=int).tolist()

    def analyze_windows(self, sizes=None, border=4, max_offset=10, num_samples=5):
        """Offset vs temporal averaging window, with and without FFT."""
        if sizes is None:
            sizes = [s for s in (1, 10, 50, 100, 200, 500) if s < self.nt]
        sizes = sorted(set(int(s) for s in sizes if s <= self.nt) | {self.nt})

        off_fft, std_fft, off_int, std_int = [], [], [], []
        for ws in tqdm(sizes, desc="window sizes", leave=False):
            s_fft, s_int = [], []
            for start in self._window_starts(ws, num_samples):
                mean_img = self._mean_frame(range(start, min(start + ws, self.nt)))
                if mean_img is None:
                    continue
                s_fft.append(_phase_corr_2d(mean_img, border, max_offset, use_fft=True))
                s_int.append(_phase_corr_2d(mean_img, border, max_offset, use_fft=False))
            off_fft.append(np.mean(s_fft) if s_fft else np.nan)
            std_fft.append(np.std(s_fft) if len(s_fft) > 1 else 0.0)
            off_int.append(np.mean(s_int) if s_int else np.nan)
            std_int.append(np.std(s_int) if len(s_int) > 1 else 0.0)

        self.results.window_sizes = np.array(sizes)
        self.results.window_offsets_fft = np.array(off_fft)
        self.results.window_stds_fft = np.array(std_fft)
        self.results.window_offsets_int = np.array(off_int)
        self.results.window_stds_int = np.array(std_int)
        logger.info(f"window sizes: {len(sizes)} tested")

    def analyze_spatial(self, patch_size=32, max_offset=10, num_frames=200):
        """Offset across a grid of FOV patches, with and without FFT."""
        idx = np.linspace(0, self.nt - 1, min(num_frames, self.nt), dtype=int)
        mean_img = self._mean_frame(idx)
        if mean_img is None:
            return

        h, w = mean_img.shape
        n_rows, n_cols = h // patch_size, w // patch_size
        if n_rows < 1 or n_cols < 1:
            logger.info("spatial: frame smaller than patch size, skipped")
            return

        thr = max(1.0, 0.05 * float(mean_img.mean()))
        grid_fft = np.full((n_rows, n_cols), np.nan)
        grid_int = np.full((n_rows, n_cols), np.nan)
        valid = np.zeros((n_rows, n_cols), dtype=bool)

        for row in tqdm(range(n_rows), desc="spatial grid", leave=False):
            for col in range(n_cols):
                y0, y1 = row * patch_size, (row + 1) * patch_size
                x0, x1 = col * patch_size, (col + 1) * patch_size
                patch = mean_img[y0:y1, x0:x1]
                if patch.mean() < thr:
                    continue
                try:
                    grid_fft[row, col] = _phase_corr_2d(patch, 0, max_offset, use_fft=True)
                    grid_int[row, col] = _phase_corr_2d(patch, 0, max_offset, use_fft=False)
                    valid[row, col] = True
                except Exception:
                    pass

        self.results.patch_size = patch_size
        self.results.grid_offsets_fft = grid_fft
        self.results.grid_offsets_int = grid_int
        self.results.grid_valid = valid
        self.results.offsets_fft = grid_fft[valid]
        self.results.offsets_int = grid_int[valid]

        n_valid = int(valid.sum())
        if n_valid:
            stats = self.results.compute_stats(grid_fft[valid])
            logger.info(f"spatial: {n_valid} patches, mean(fft)={stats['mean']:.3f}")

    def run(self, border=4, max_offset=10, patch_size=32):
        """Run both diagnostics."""
        start = time.perf_counter()
        steps = [
            ("window lengths", lambda: self.analyze_windows(
                border=border, max_offset=max_offset)),
            ("spatial distribution", lambda: self.analyze_spatial(
                patch_size=patch_size, max_offset=max_offset)),
        ]
        for _name, func in tqdm(steps, desc="scan-phase analysis"):
            func()
        self.results.analysis_time = time.perf_counter() - start
        logger.info(f"complete in {self.results.analysis_time:.1f}s")
        return self.results

    def generate_figures(self, output_dir=None, fmt="png", dpi=150, show=False):
        """Generate the two analysis figures."""
        import matplotlib.pyplot as plt
        colors = MBO_DARK_THEME

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        def _save_fig(fig, name):
            if output_dir:
                path = output_dir / f"{name}.{fmt}"
                fig.savefig(path, dpi=dpi, facecolor=colors["background"],
                            edgecolor="none", bbox_inches="tight")
                saved.append(path)
            if show:
                plt.show()
            plt.close(fig)

        if len(self.results.window_sizes) > 0:
            _save_fig(self._fig_windows(), "window_lengths")
        if self.results.grid_valid.any():
            _save_fig(self._fig_spatial(), "spatial_distribution")

        return saved

    def _fig_windows(self):
        """Offset and precision vs averaging window length, with/without FFT."""
        colors = MBO_DARK_THEME
        fig, axes = _mbo_fig(1, 2, figsize=(12, 5))

        ws = self.results.window_sizes
        off_fft = self.results.window_offsets_fft
        std_fft = self.results.window_stds_fft
        off_int = self.results.window_offsets_int
        std_int = self.results.window_stds_int

        ax = axes[0]
        ax.fill_between(ws, off_fft - std_fft, off_fft + std_fft, alpha=0.25, color=colors["primary"])
        ax.plot(ws, off_fft, "o-", color=colors["primary"], ms=6, lw=2, label="with FFT (subpixel)")
        ax.fill_between(ws, off_int - std_int, off_int + std_int, alpha=0.2, color=colors["orange"])
        ax.plot(ws, off_int, "s--", color=colors["orange"], ms=6, lw=2, label="without FFT (integer)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(lambda x, p: f"{int(x)}" if x >= 1 else "")
        ax.set_xlabel("Window Length (frames)")
        ax.set_ylabel("Offset (px)")
        ax.set_title("Offset vs Window Length", fontweight="bold")
        ax.legend(loc="best", facecolor=colors["surface"],
                  edgecolor=colors["border"], labelcolor=colors["text"])

        ax = axes[1]
        ax.plot(ws, std_fft, "o-", color=colors["primary"], ms=6, lw=2, label="with FFT")
        ax.plot(ws, std_int, "s--", color=colors["orange"], ms=6, lw=2, label="without FFT")
        ax.axhline(0.1, color=colors["warning"], ls="--", lw=1.5, alpha=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(lambda x, p: f"{int(x)}" if x >= 1 else "")
        ax.set_xlabel("Window Length (frames)")
        ax.set_ylabel("Std of Estimate (px)")
        ax.set_title("Estimation Precision", fontweight="bold")
        ax.legend(loc="best", facecolor=colors["surface"],
                  edgecolor=colors["border"], labelcolor=colors["text"])

        fig.tight_layout()
        return fig

    def _fig_spatial(self):
        """Spatial offset heatmaps across the FOV, with/without FFT."""
        from matplotlib.colors import TwoSlopeNorm
        colors = MBO_DARK_THEME

        valid = self.results.grid_valid
        ps = self.results.patch_size
        disp_fft = np.where(valid, self.results.grid_offsets_fft, np.nan)
        disp_int = np.where(valid, self.results.grid_offsets_int, np.nan)

        vmax = max(0.5, np.nanmax(np.abs(np.concatenate([disp_fft.ravel(), disp_int.ravel()]))))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, axes = _mbo_fig(1, 2, figsize=(12, 5.5))
        for ax, disp, label in (
            (axes[0], disp_fft, "with FFT (subpixel)"),
            (axes[1], disp_int, "without FFT (integer)"),
        ):
            ax.grid(False)
            im = ax.imshow(disp, norm=norm, aspect="equal", interpolation="nearest")
            _mbo_colorbar(im, ax, "Offset (px)")
            vals = disp[~np.isnan(disp)]
            if len(vals) > 0:
                ax.set_title(f"{label}\nmean={np.mean(vals):.3f}, std={np.std(vals):.3f} px",
                             fontweight="bold")
            else:
                ax.set_title(label, fontweight="bold")
            ax.set_xlabel(f"X ({ps}px patches)")
            ax.set_ylabel(f"Y ({ps}px patches)")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        return fig

    def save_results(self, path):
        """Save results to npz."""
        path = Path(path)
        r = self.results
        np.savez_compressed(
            path,
            window_sizes=r.window_sizes,
            window_offsets_fft=r.window_offsets_fft,
            window_stds_fft=r.window_stds_fft,
            window_offsets_int=r.window_offsets_int,
            window_stds_int=r.window_stds_int,
            patch_size=r.patch_size,
            grid_offsets_fft=r.grid_offsets_fft,
            grid_offsets_int=r.grid_offsets_int,
            grid_valid=r.grid_valid,
            offsets_fft=r.offsets_fft,
            offsets_int=r.offsets_int,
            num_timepoints=r.num_timepoints,
            num_planes=r.num_planes,
            num_rois=r.num_rois,
            frame_shape=r.frame_shape,
            pixel_resolution_um=r.pixel_resolution_um,
            analysis_time=r.analysis_time,
        )
        logger.info(f"saved to {path}")
        return path


def run_scanphase_analysis(
    data_path=None,
    output_dir=None,
    image_format="png",
    show_plots=False,
    patch_size=32,
):
    """Run scan-phase analysis."""
    from mbo_utilities import imread

    if data_path is None:
        from mbo_utilities.gui import select_files
        paths = select_files(title="Select data for scan-phase analysis")
        if not paths:
            return None
        data_path = paths[0] if len(paths) == 1 else paths

    if isinstance(data_path, (list, tuple)):
        if len(data_path) == 0:
            raise ValueError("empty list of paths")
        first_path = Path(data_path[0])
        if output_dir is None:
            output_dir = first_path.parent / f"{first_path.parent.name}_scanphase_analysis"
        logger.info(f"loading {len(data_path)} tiff files")
        arr = imread(data_path)
    else:
        data_path = Path(data_path)
        if output_dir is None:
            output_dir = data_path.parent / f"{data_path.stem}_scanphase_analysis"
        logger.info(f"loading {data_path}")
        arr = imread(data_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ScanPhaseAnalyzer(arr)
    results = analyzer.run(patch_size=patch_size)
    analyzer.generate_figures(output_dir=output_dir, fmt=image_format, show=show_plots)
    analyzer.save_results(output_dir / "scanphase_results.npz")

    return results
