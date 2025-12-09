"""
Scan-phase analysis for bidirectional scanning correction.

Analyzes phase offset characteristics to determine optimal correction parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import time

import numpy as np
from tqdm.auto import tqdm

from mbo_utilities import log
from mbo_utilities.phasecorr import _phase_corr_2d

logger = log.get("analysis.scanphase")

# smaller patches for finer spatial resolution
DEFAULT_PATCH_SIZES = [8, 16, 32, 64]


@dataclass
class ScanPhaseResults:
    """Results from scan-phase analysis."""

    # per-frame offsets
    offsets_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_int: np.ndarray = field(default_factory=lambda: np.array([]))

    # window size analysis
    window_sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    window_offsets_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    window_offsets_int: np.ndarray = field(default_factory=lambda: np.array([]))
    window_std_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    window_std_int: np.ndarray = field(default_factory=lambda: np.array([]))

    # spatial grid (per ROI if applicable)
    grid_offsets: dict = field(default_factory=dict)  # {patch_size: 2D array}
    grid_valid: dict = field(default_factory=dict)

    # per-roi offsets (if multiple ROIs)
    roi_offsets_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    roi_offsets_int: np.ndarray = field(default_factory=lambda: np.array([]))

    # metadata
    num_frames: int = 0
    num_rois: int = 1
    frame_shape: tuple = ()
    roi_yslices: list = field(default_factory=list)
    analysis_time: float = 0.0

    def compute_stats(self, arr):
        """Compute basic statistics for an array."""
        arr = np.asarray(arr)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        return {
            'mean': float(np.mean(valid)),
            'median': float(np.median(valid)),
            'std': float(np.std(valid)),
            'min': float(np.min(valid)),
            'max': float(np.max(valid)),
        }

    def get_summary(self):
        """Get summary dict for CLI output."""
        summary = {
            'metadata': {
                'num_frames': self.num_frames,
                'num_rois': self.num_rois,
                'frame_shape': self.frame_shape,
                'analysis_time': self.analysis_time,
            }
        }
        if len(self.offsets_fft) > 0:
            summary['fft'] = self.compute_stats(self.offsets_fft)
        if len(self.offsets_int) > 0:
            summary['int'] = self.compute_stats(self.offsets_int)
        return summary


class ScanPhaseAnalyzer:
    """
    Analyzer for scan-phase offset characteristics.

    Handles both single-ROI and multi-ROI (vertically tiled) data.
    """

    def __init__(self, data, roi_yslices=None):
        """
        Initialize analyzer.

        Parameters
        ----------
        data : array-like
            Input data, shape (T, Y, X) or (T, Z, Y, X).
            For multi-ROI data, Y dimension contains vertically stacked ROIs.
        roi_yslices : list of slice, optional
            Y slices for each ROI if data contains multiple vertically stacked ROIs.
            If None, treats entire Y dimension as single ROI.
        """
        self.data = data
        self.shape = data.shape
        self.ndim = len(self.shape)

        # determine frame count
        if hasattr(data, 'num_frames'):
            self.num_frames = data.num_frames
        else:
            self.num_frames = self.shape[0]

        # handle ROI structure
        if roi_yslices is not None:
            self.roi_yslices = roi_yslices
            self.num_rois = len(roi_yslices)
        else:
            # single ROI = full frame
            self.roi_yslices = [slice(None)]
            self.num_rois = 1

        # frame dimensions (full tiled frame)
        self.frame_height = self.shape[-2]
        self.frame_width = self.shape[-1]

        self.results = ScanPhaseResults(
            num_frames=self.num_frames,
            num_rois=self.num_rois,
            frame_shape=(self.frame_height, self.frame_width),
            roi_yslices=self.roi_yslices,
        )

        logger.info(f"ScanPhaseAnalyzer: {self.num_frames} frames, {self.num_rois} ROIs, shape={self.shape}")

    def _get_frame(self, idx):
        """Get a single 2D frame."""
        if self.ndim == 2:
            return np.asarray(self.data)
        elif self.ndim == 3:
            return np.asarray(self.data[idx])
        elif self.ndim == 4:
            # assume (T, Z, Y, X), take first z-plane
            return np.asarray(self.data[idx, 0])
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

    def _get_roi_frame(self, frame, roi_idx):
        """Extract a single ROI from a frame using y-slice."""
        yslice = self.roi_yslices[roi_idx]
        return frame[yslice, :]

    def analyze_per_frame(self, use_fft=True, fft_method="1d", upsample=10, border=4, max_offset=10):
        """
        Analyze offset for each frame, averaging across ROIs if multiple.

        Returns array of offsets, one per frame.
        """
        offsets = []
        desc = "frames (FFT)" if use_fft else "frames (int)"

        for i in tqdm(range(self.num_frames), desc=desc, leave=False):
            frame = self._get_frame(i)

            # analyze each ROI separately and average
            roi_offsets = []
            for roi_idx in range(self.num_rois):
                roi_frame = self._get_roi_frame(frame, roi_idx)
                try:
                    offset = _phase_corr_2d(
                        roi_frame,
                        upsample=upsample,
                        border=border,
                        max_offset=max_offset,
                        use_fft=use_fft,
                        fft_method=fft_method,
                    )
                    roi_offsets.append(offset)
                except Exception:
                    pass

            if roi_offsets:
                offsets.append(np.mean(roi_offsets))
            else:
                offsets.append(np.nan)

        offsets = np.array(offsets)

        if use_fft:
            self.results.offsets_fft = offsets
        else:
            self.results.offsets_int = offsets

        stats = self.results.compute_stats(offsets)
        method = "FFT" if use_fft else "int"
        logger.info(f"Per-frame ({method}): mean={stats['mean']:.4f}, std={stats['std']:.4f}")

        return offsets

    def analyze_window_sizes(self, fft_method="1d", upsample=10, border=4, max_offset=10, num_samples=10):
        """
        Analyze how offset estimate varies with temporal window size.

        Tests window sizes from 1 to num_frames to determine minimum
        window needed for stable offset estimation.
        """
        # generate window sizes: 1, 2, 5, 10, 20, 50, 100, ... up to num_frames
        sizes = []
        for base in [1, 2, 5]:
            for mult in [1, 10, 100, 1000, 10000]:
                val = base * mult
                if val <= self.num_frames:
                    sizes.append(val)
        sizes = sorted(set(sizes))

        # ensure num_frames is included
        if self.num_frames not in sizes:
            sizes.append(self.num_frames)
        sizes = sorted(sizes)

        self.results.window_sizes = np.array(sizes)
        offsets_fft = []
        offsets_int = []
        std_fft = []
        std_int = []

        for ws in tqdm(sizes, desc="window sizes", leave=False):
            # sample multiple non-overlapping windows
            n_possible = self.num_frames // ws
            n_samples = min(num_samples, n_possible)

            if n_samples == n_possible:
                starts = [i * ws for i in range(n_samples)]
            else:
                starts = np.linspace(0, self.num_frames - ws, n_samples, dtype=int).tolist()

            fft_vals = []
            int_vals = []

            for start in starts:
                # average frames in window
                indices = range(start, min(start + ws, self.num_frames))
                frames = [self._get_frame(i) for i in indices]
                mean_frame = np.mean(frames, axis=0)

                # analyze each ROI
                roi_fft = []
                roi_int = []
                for roi_idx in range(self.num_rois):
                    roi_frame = self._get_roi_frame(mean_frame, roi_idx)
                    try:
                        off_fft = _phase_corr_2d(
                            roi_frame, upsample=upsample, border=border,
                            max_offset=max_offset, use_fft=True, fft_method=fft_method
                        )
                        roi_fft.append(off_fft)
                    except Exception:
                        pass
                    try:
                        off_int = _phase_corr_2d(
                            roi_frame, upsample=upsample, border=border,
                            max_offset=max_offset, use_fft=False, fft_method=fft_method
                        )
                        roi_int.append(off_int)
                    except Exception:
                        pass

                if roi_fft:
                    fft_vals.append(np.mean(roi_fft))
                if roi_int:
                    int_vals.append(np.mean(roi_int))

            offsets_fft.append(np.mean(fft_vals) if fft_vals else np.nan)
            offsets_int.append(np.mean(int_vals) if int_vals else np.nan)
            std_fft.append(np.std(fft_vals) if len(fft_vals) > 1 else 0)
            std_int.append(np.std(int_vals) if len(int_vals) > 1 else 0)

        self.results.window_offsets_fft = np.array(offsets_fft)
        self.results.window_offsets_int = np.array(offsets_int)
        self.results.window_std_fft = np.array(std_fft)
        self.results.window_std_int = np.array(std_int)

        logger.info(f"Window analysis: {len(sizes)} sizes from 1 to {sizes[-1]}")
        return sizes, offsets_fft, std_fft

    def analyze_spatial_grid(self, patch_sizes=None, fft_method="1d", upsample=10, max_offset=10, num_frames=100):
        """
        Analyze spatial distribution of offsets using small patches.

        Creates a grid of patches and computes offset for each to identify
        spatial variation across the FOV.
        """
        if patch_sizes is None:
            patch_sizes = [ps for ps in DEFAULT_PATCH_SIZES if ps <= min(self.frame_height, self.frame_width) // 2]

        if not patch_sizes:
            logger.warning("No valid patch sizes for this frame size")
            return {}

        # average some frames for robust spatial analysis
        sample_indices = np.linspace(0, self.num_frames - 1, min(num_frames, self.num_frames), dtype=int)
        frames = [self._get_frame(i) for i in sample_indices]
        mean_frame = np.mean(frames, axis=0)

        # analyze first ROI for spatial grid
        roi_frame = self._get_roi_frame(mean_frame, 0)
        even_rows = roi_frame[::2]
        odd_rows = roi_frame[1::2]
        m = min(even_rows.shape[0], odd_rows.shape[0])
        even_rows = even_rows[:m]
        odd_rows = odd_rows[:m]
        h, w = even_rows.shape

        for patch_size in tqdm(patch_sizes, desc="spatial grid", leave=False):
            n_rows = h // patch_size
            n_cols = w // patch_size

            if n_rows < 1 or n_cols < 1:
                continue

            offsets = np.full((n_rows, n_cols), np.nan)
            valid = np.zeros((n_rows, n_cols), dtype=bool)

            for row in range(n_rows):
                for col in range(n_cols):
                    y0, y1 = row * patch_size, (row + 1) * patch_size
                    x0, x1 = col * patch_size, (col + 1) * patch_size

                    patch_even = even_rows[y0:y1, x0:x1]
                    patch_odd = odd_rows[y0:y1, x0:x1]

                    # skip low-intensity patches
                    if patch_even.mean() < 10 or patch_odd.mean() < 10:
                        continue

                    # reconstruct 2D frame for phase_corr
                    combined = np.zeros((patch_size * 2, patch_size))
                    combined[::2] = patch_even
                    combined[1::2] = patch_odd

                    try:
                        offset = _phase_corr_2d(
                            combined, upsample=upsample, border=0,
                            max_offset=max_offset, use_fft=True, fft_method=fft_method
                        )
                        offsets[row, col] = offset
                        valid[row, col] = True
                    except Exception:
                        pass

            self.results.grid_offsets[patch_size] = offsets
            self.results.grid_valid[patch_size] = valid

            n_valid = valid.sum()
            if n_valid > 0:
                stats = self.results.compute_stats(offsets[valid])
                logger.info(f"Grid {patch_size}px: {n_valid}/{valid.size} patches, mean={stats['mean']:.4f}")

        return self.results.grid_offsets

    def analyze_per_roi(self, fft_method="1d", upsample=10, border=4, max_offset=10, num_frames=100):
        """Analyze offset for each ROI separately."""
        if self.num_rois <= 1:
            return

        sample_indices = np.linspace(0, self.num_frames - 1, min(num_frames, self.num_frames), dtype=int)
        frames = [self._get_frame(i) for i in sample_indices]
        mean_frame = np.mean(frames, axis=0)

        fft_offsets = []
        int_offsets = []

        for roi_idx in range(self.num_rois):
            roi_frame = self._get_roi_frame(mean_frame, roi_idx)
            try:
                off_fft = _phase_corr_2d(
                    roi_frame, upsample=upsample, border=border,
                    max_offset=max_offset, use_fft=True, fft_method=fft_method
                )
                fft_offsets.append(off_fft)
            except Exception:
                fft_offsets.append(np.nan)
            try:
                off_int = _phase_corr_2d(
                    roi_frame, upsample=upsample, border=border,
                    max_offset=max_offset, use_fft=False, fft_method=fft_method
                )
                int_offsets.append(off_int)
            except Exception:
                int_offsets.append(np.nan)

        self.results.roi_offsets_fft = np.array(fft_offsets)
        self.results.roi_offsets_int = np.array(int_offsets)

        logger.info(f"Per-ROI: {self.num_rois} ROIs analyzed")

    def run(self, fft_method="1d", upsample=10, border=4, max_offset=10):
        """Run full analysis."""
        start = time.time()

        steps = [
            ("per-frame (FFT)", lambda: self.analyze_per_frame(
                use_fft=True, fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
            ("per-frame (int)", lambda: self.analyze_per_frame(
                use_fft=False, fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
            ("window sizes", lambda: self.analyze_window_sizes(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
            ("spatial grid", lambda: self.analyze_spatial_grid(
                fft_method=fft_method, upsample=upsample, max_offset=max_offset)),
        ]

        if self.num_rois > 1:
            steps.append(("per-ROI", lambda: self.analyze_per_roi(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)))

        for name, func in tqdm(steps, desc="scan-phase analysis"):
            func()

        self.results.analysis_time = time.time() - start
        logger.info(f"Analysis complete in {self.results.analysis_time:.1f}s")

        return self.results

    def generate_figures(self, output_dir=None, fmt="png", dpi=150, show=False):
        """Generate analysis figures."""
        import matplotlib.pyplot as plt

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        # figure 1: temporal analysis
        fig = self._fig_temporal()
        if output_dir:
            path = output_dir / f"temporal.{fmt}"
            fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
            saved.append(path)
        if show:
            plt.show()
        plt.close(fig)

        # figure 2: window size convergence
        fig = self._fig_window_convergence()
        if output_dir:
            path = output_dir / f"window_convergence.{fmt}"
            fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
            saved.append(path)
        if show:
            plt.show()
        plt.close(fig)

        # figure 3: spatial grid (if data available)
        if self.results.grid_offsets:
            fig = self._fig_spatial()
            if output_dir:
                path = output_dir / f"spatial.{fmt}"
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                saved.append(path)
            if show:
                plt.show()
            plt.close(fig)

        return saved

    def _fig_temporal(self):
        """Create temporal analysis figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # time series
        ax = axes[0]
        if len(self.results.offsets_fft) > 0:
            ax.plot(self.results.offsets_fft, 'b-', lw=0.5, alpha=0.7, label='FFT')
            mean_fft = np.nanmean(self.results.offsets_fft)
            ax.axhline(mean_fft, color='blue', ls='--', lw=1.5)
        if len(self.results.offsets_int) > 0:
            ax.plot(self.results.offsets_int, 'g-', lw=0.5, alpha=0.5, label='int')
        ax.set_xlabel('frame')
        ax.set_ylabel('offset (px)')
        ax.set_title('per-frame offset')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # histogram
        ax = axes[1]
        if len(self.results.offsets_fft) > 0:
            valid = self.results.offsets_fft[~np.isnan(self.results.offsets_fft)]
            ax.hist(valid, bins=50, alpha=0.7, label='FFT', color='blue', density=True)
            stats = self.results.compute_stats(valid)
            txt = f"mean: {stats['mean']:.3f}\nstd: {stats['std']:.3f}"
            ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=9,
                    va='top', ha='right', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if len(self.results.offsets_int) > 0:
            valid = self.results.offsets_int[~np.isnan(self.results.offsets_int)]
            ax.hist(valid, bins=50, alpha=0.5, label='int', color='green', density=True)
        ax.set_xlabel('offset (px)')
        ax.set_ylabel('density')
        ax.set_title('distribution')
        ax.legend()

        plt.tight_layout()
        return fig

    def _fig_window_convergence(self):
        """Create window size convergence figure - key diagnostic."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ws = self.results.window_sizes
        fft_off = self.results.window_offsets_fft
        fft_std = self.results.window_std_fft
        int_off = self.results.window_offsets_int

        # offset vs window size
        ax = axes[0]
        ax.errorbar(ws, fft_off, yerr=fft_std, fmt='b-o', capsize=3, ms=4, label='FFT')
        ax.plot(ws, int_off, 'g-s', ms=4, label='int')
        ax.set_xscale('log')
        ax.set_xlabel('window size (frames)')
        ax.set_ylabel('offset (px)')
        ax.set_title('offset vs window size')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # format x-axis without scientific notation
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

        # estimation variance vs window size
        ax = axes[1]
        ax.plot(ws, fft_std, 'b-o', ms=4, label='FFT std')
        ax.set_xscale('log')
        ax.set_xlabel('window size (frames)')
        ax.set_ylabel('std of estimate (px)')
        ax.set_title('estimation variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

        # add annotation for convergence
        if len(fft_std) > 0:
            # find window size where std drops below threshold
            threshold = 0.1
            converged_idx = np.where(fft_std < threshold)[0]
            if len(converged_idx) > 0:
                conv_ws = ws[converged_idx[0]]
                ax.axvline(conv_ws, color='red', ls='--', alpha=0.7)
                ax.text(conv_ws, ax.get_ylim()[1] * 0.9, f'{conv_ws}', color='red', ha='center')

        plt.tight_layout()
        return fig

    def _fig_spatial(self):
        """Create spatial grid figure."""
        import matplotlib.pyplot as plt

        # use smallest available patch size for finest resolution
        patch_sizes = sorted(self.results.grid_offsets.keys())
        if not patch_sizes:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'no spatial data', ha='center', va='center', transform=ax.transAxes)
            return fig

        n_sizes = len(patch_sizes)
        fig, axes = plt.subplots(1, n_sizes, figsize=(4 * n_sizes, 4))
        if n_sizes == 1:
            axes = [axes]

        for ax, ps in zip(axes, patch_sizes):
            offsets = self.results.grid_offsets[ps]
            valid = self.results.grid_valid[ps]
            masked = np.where(valid, offsets, np.nan)

            vmax = max(0.5, np.nanmax(np.abs(masked)))
            im = ax.imshow(masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            plt.colorbar(im, ax=ax, label='offset (px)')
            ax.set_xlabel('x patch')
            ax.set_ylabel('y patch')
            ax.set_title(f'{ps}px patches')

            # stats annotation
            valid_vals = offsets[valid]
            if len(valid_vals) > 0:
                stats = self.results.compute_stats(valid_vals)
                ax.set_xlabel(f'x patch  [mean={stats["mean"]:.3f}]')

        plt.tight_layout()
        return fig

    def save_results(self, path):
        """Save results to npz file."""
        path = Path(path)
        data = {
            'offsets_fft': self.results.offsets_fft,
            'offsets_int': self.results.offsets_int,
            'window_sizes': self.results.window_sizes,
            'window_offsets_fft': self.results.window_offsets_fft,
            'window_offsets_int': self.results.window_offsets_int,
            'window_std_fft': self.results.window_std_fft,
            'window_std_int': self.results.window_std_int,
            'num_frames': self.results.num_frames,
            'num_rois': self.results.num_rois,
            'frame_shape': self.results.frame_shape,
            'analysis_time': self.results.analysis_time,
        }
        for ps, grid in self.results.grid_offsets.items():
            data[f'grid_{ps}'] = grid
            data[f'grid_{ps}_valid'] = self.results.grid_valid[ps]

        np.savez_compressed(path, **data)
        logger.info(f"Results saved to {path}")
        return path


def run_scanphase_analysis(
    data_path=None,
    output_dir=None,
    fft_method="1d",
    image_format="png",
    show_plots=False,
):
    """
    Run scan-phase analysis from CLI or programmatically.

    Parameters
    ----------
    data_path : str, Path, or list of Path, optional
        Path to data file, folder, or list of tiff files. If None, opens file dialog.
    output_dir : str or Path, optional
        Output directory. If None, creates <input>_scanphase_analysis/
    fft_method : str
        FFT method: '1d' (fast) or '2d'.
    image_format : str
        Output image format.
    show_plots : bool
        Show plots interactively.

    Returns
    -------
    ScanPhaseResults or None
        Analysis results, or None if user cancelled.
    """
    from pathlib import Path
    from mbo_utilities import open as mbo_open

    # handle file selection
    if data_path is None:
        from mbo_utilities.graphics import select_file
        data_path = select_file(title="Select data for scan-phase analysis")
        if data_path is None:
            return None

    # handle list of paths (from --num-tifs)
    if isinstance(data_path, (list, tuple)):
        # list of tiff files - pass directly to mbo_open
        if len(data_path) == 0:
            raise ValueError("Empty list of paths")
        first_path = Path(data_path[0])
        if output_dir is None:
            output_dir = first_path.parent / f"{first_path.parent.name}_scanphase_analysis"
        logger.info(f"Loading {len(data_path)} tiff files")
        arr = mbo_open(data_path)
    else:
        data_path = Path(data_path)
        if output_dir is None:
            output_dir = data_path.parent / f"{data_path.stem}_scanphase_analysis"
        logger.info(f"Loading {data_path}")
        arr = mbo_open(data_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # check for ROI structure (MboRawArray)
    roi_yslices = None
    if hasattr(arr, 'yslices') and hasattr(arr, 'num_rois'):
        if arr.num_rois > 1:
            roi_yslices = arr.yslices
            logger.info(f"Detected {arr.num_rois} ROIs with y-slices: {roi_yslices}")

    # create analyzer
    analyzer = ScanPhaseAnalyzer(arr, roi_yslices=roi_yslices)

    # run analysis
    results = analyzer.run(fft_method=fft_method)

    # generate figures
    analyzer.generate_figures(output_dir=output_dir, fmt=image_format, show=show_plots)

    # save numerical results
    analyzer.save_results(output_dir / "scanphase_results.npz")

    return results


# convenience function
def analyze_scanphase(data, output_dir=None, **kwargs):
    """Run scan-phase analysis on array data."""
    analyzer = ScanPhaseAnalyzer(data)
    results = analyzer.run(**kwargs)
    if output_dir:
        analyzer.generate_figures(output_dir=output_dir)
        analyzer.save_results(Path(output_dir) / "scanphase_results.npz")
    return results
