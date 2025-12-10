"""
Scan-phase analysis for bidirectional scanning correction.

Measures phase offset to determine optimal correction parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm

from mbo_utilities import log
from mbo_utilities.phasecorr import _phase_corr_2d
from mbo_utilities.metadata import get_param

logger = log.get("analysis.scanphase")


@dataclass
class ScanPhaseResults:
    """results from scan-phase analysis"""

    # per-frame offsets
    offsets_fft: np.ndarray = field(default_factory=lambda: np.array([]))

    # window size analysis
    window_sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    window_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    window_stds: np.ndarray = field(default_factory=lambda: np.array([]))

    # spatial grid offsets {patch_size: 2D array}
    grid_offsets: dict = field(default_factory=dict)
    grid_valid: dict = field(default_factory=dict)

    # z-plane offsets
    plane_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    plane_depths_um: np.ndarray = field(default_factory=lambda: np.array([]))

    # parameter sweep (offset vs signal intensity)
    intensity_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    offset_by_intensity: np.ndarray = field(default_factory=lambda: np.array([]))
    offset_std_by_intensity: np.ndarray = field(default_factory=lambda: np.array([]))

    # metadata
    num_frames: int = 0
    num_planes: int = 1
    frame_shape: tuple = ()
    pixel_resolution_um: float = 0.0
    analysis_time: float = 0.0

    def compute_stats(self, arr):
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
        summary = {
            'metadata': {
                'num_frames': self.num_frames,
                'num_planes': self.num_planes,
                'frame_shape': self.frame_shape,
                'analysis_time': self.analysis_time,
            }
        }
        if len(self.offsets_fft) > 0:
            summary['fft'] = self.compute_stats(self.offsets_fft)
        return summary


def _setup_dark_style():
    """setup dark theme for matplotlib figures"""
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')


def _dark_fig(*args, **kwargs):
    """create figure with dark background"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor('#1a1a2e')
    if hasattr(axes, '__iter__'):
        for ax in np.array(axes).flat:
            ax.set_facecolor('#1a1a2e')
    else:
        axes.set_facecolor('#1a1a2e')
    return fig, axes


class ScanPhaseAnalyzer:
    """
    Analyzer for scan-phase offset.

    Measures per-frame offset, window size effects, spatial variation, and z-plane dependence.
    """

    def __init__(self, data, roi_yslices=None):
        """
        Parameters
        ----------
        data : array-like
            input data, shape (T, Y, X) or (T, Z, Y, X)
        roi_yslices : list of slice, optional
            y slices for vertically stacked ROIs
        """
        self.data = data
        self.shape = data.shape
        self.ndim = len(self.shape)

        # frame count
        if hasattr(data, 'num_frames'):
            self.num_frames = data.num_frames
        else:
            self.num_frames = self.shape[0]

        # z-planes
        if hasattr(data, 'num_planes'):
            self.num_planes = data.num_planes
        elif self.ndim == 4:
            self.num_planes = self.shape[1]
        else:
            self.num_planes = 1

        # roi structure
        if roi_yslices is not None:
            self.roi_yslices = roi_yslices
            self.num_rois = len(roi_yslices)
        else:
            self.roi_yslices = [slice(None)]
            self.num_rois = 1

        # frame dimensions
        self.frame_height = self.shape[-2]
        self.frame_width = self.shape[-1]

        # pixel resolution if available
        md = getattr(data, 'metadata', None)
        self.pixel_resolution_um = get_param(md, "dx", default=0.0)

        self.results = ScanPhaseResults(
            num_frames=self.num_frames,
            num_planes=self.num_planes,
            frame_shape=(self.frame_height, self.frame_width),
            pixel_resolution_um=self.pixel_resolution_um,
        )

        logger.info(f"ScanPhaseAnalyzer: {self.num_frames} frames, {self.num_planes} planes, shape={self.shape}")

    def _get_frame(self, idx, plane=0):
        """get a single 2D frame"""
        if self.ndim == 2:
            frame = np.asarray(self.data)
        elif self.ndim == 3:
            frame = np.asarray(self.data[idx])
        elif self.ndim == 4:
            frame = np.asarray(self.data[idx, plane])
        else:
            raise ValueError(f"unsupported ndim: {self.ndim}")

        while frame.ndim > 2:
            frame = frame[0]
        return frame

    def _get_roi_frame(self, frame, roi_idx):
        """extract single ROI from frame"""
        while frame.ndim > 2:
            frame = frame[0]
        yslice = self.roi_yslices[roi_idx]
        return frame[yslice, :]

    def _compute_offset(self, frame, fft_method="1d", upsample=10, border=4, max_offset=10):
        """compute offset for a 2D frame, averaging across rois"""
        roi_offsets = []
        for roi_idx in range(self.num_rois):
            roi_frame = self._get_roi_frame(frame, roi_idx)
            try:
                offset = _phase_corr_2d(
                    roi_frame, upsample=upsample, border=border,
                    max_offset=max_offset, use_fft=True, fft_method=fft_method
                )
                roi_offsets.append(offset)
            except Exception:
                pass
        return np.mean(roi_offsets) if roi_offsets else np.nan

    def analyze_per_frame(self, fft_method="1d", upsample=10, border=4, max_offset=10):
        """
        compute offset for each frame.

        primary measurement - shows temporal stability of the offset.
        """
        offsets = []
        for i in tqdm(range(self.num_frames), desc="per-frame", leave=False):
            frame = self._get_frame(i)
            offsets.append(self._compute_offset(frame, fft_method, upsample, border, max_offset))

        self.results.offsets_fft = np.array(offsets)
        stats = self.results.compute_stats(self.results.offsets_fft)
        logger.info(f"per-frame: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        return self.results.offsets_fft

    def analyze_window_sizes(self, fft_method="1d", upsample=10, border=4, max_offset=10, num_samples=5):
        """
        analyze how offset estimate varies with temporal window size.

        key diagnostic - shows how many frames are needed for stable estimation.
        small windows = noisy estimates, large windows = converged estimate.
        """
        # window sizes from 1 to num_frames
        sizes = []
        for base in [1, 2, 5]:
            for mult in [1, 10, 100, 1000, 10000]:
                val = base * mult
                if val <= self.num_frames:
                    sizes.append(val)
        sizes = sorted(set(sizes))
        if self.num_frames not in sizes:
            sizes.append(self.num_frames)
        sizes = sorted(sizes)

        self.results.window_sizes = np.array(sizes)
        window_offsets = []
        window_stds = []

        for ws in tqdm(sizes, desc="window sizes", leave=False):
            # sample multiple windows and measure variance
            n_possible = self.num_frames // ws
            n_samp = min(num_samples, n_possible)

            if n_samp == n_possible:
                starts = [i * ws for i in range(n_samp)]
            else:
                starts = np.linspace(0, self.num_frames - ws, n_samp, dtype=int).tolist()

            sample_offsets = []
            for start in starts:
                # average frames in window
                indices = range(start, min(start + ws, self.num_frames))
                frames = [self._get_frame(i) for i in indices]
                mean_frame = np.mean(frames, axis=0)
                offset = self._compute_offset(mean_frame, fft_method, upsample, border, max_offset)
                if not np.isnan(offset):
                    sample_offsets.append(offset)

            if sample_offsets:
                window_offsets.append(np.mean(sample_offsets))
                window_stds.append(np.std(sample_offsets) if len(sample_offsets) > 1 else 0)
            else:
                window_offsets.append(np.nan)
                window_stds.append(np.nan)

        self.results.window_offsets = np.array(window_offsets)
        self.results.window_stds = np.array(window_stds)
        logger.info(f"window sizes: {len(sizes)} sizes tested")

    def analyze_spatial_grid(self, patch_sizes=(32, 64), fft_method="1d", upsample=10, max_offset=10, num_frames=100):
        """
        compute offset in a grid of patches across the fov.

        shows spatial variation - edges often differ from center.
        """
        sample_indices = np.linspace(0, self.num_frames - 1, min(num_frames, self.num_frames), dtype=int)
        frames = [self._get_frame(i) for i in sample_indices]
        mean_frame = np.mean(frames, axis=0)

        roi_frame = self._get_roi_frame(mean_frame, 0)
        while roi_frame.ndim > 2:
            roi_frame = roi_frame[0]

        even_rows = roi_frame[::2]
        odd_rows = roi_frame[1::2]
        m = min(even_rows.shape[0], odd_rows.shape[0])
        even_rows = even_rows[:m]
        odd_rows = odd_rows[:m]
        h, w = even_rows.shape[-2], even_rows.shape[-1]

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

                    if patch_even.mean() < 10 or patch_odd.mean() < 10:
                        continue

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
                logger.info(f"grid {patch_size}px: {n_valid} patches, mean={stats['mean']:.3f}")

    def analyze_z_planes(self, fft_method="1d", upsample=10, border=4, max_offset=10, num_frames=100):
        """
        compute offset for each z-plane.

        different depths may have different offsets.
        """
        if self.num_planes <= 1:
            return

        sample_indices = np.linspace(0, self.num_frames - 1, min(num_frames, self.num_frames), dtype=int)
        plane_offsets = []

        for plane in tqdm(range(self.num_planes), desc="z-planes", leave=False):
            frames = [self._get_frame(i, plane=plane) for i in sample_indices]
            mean_frame = np.mean(frames, axis=0)
            offset = self._compute_offset(mean_frame, fft_method, upsample, border, max_offset)
            plane_offsets.append(offset)

        self.results.plane_offsets = np.array(plane_offsets)

        if self.pixel_resolution_um > 0:
            self.results.plane_depths_um = np.arange(self.num_planes) * self.pixel_resolution_um
        else:
            self.results.plane_depths_um = np.arange(self.num_planes)

        logger.info(f"z-planes: {self.num_planes} planes")

    def analyze_parameters(self, fft_method="1d", upsample=10, border=4, max_offset=10, num_frames=50):
        """
        analyze offset reliability vs signal intensity.

        low signal regions produce unreliable offsets - helps set max_offset.
        """
        sample_indices = np.linspace(0, self.num_frames - 1, min(num_frames, self.num_frames), dtype=int)

        intensities = []
        offsets = []

        for idx in sample_indices:
            frame = self._get_frame(idx)
            roi_frame = self._get_roi_frame(frame, 0)
            while roi_frame.ndim > 2:
                roi_frame = roi_frame[0]

            even_rows = roi_frame[::2]
            odd_rows = roi_frame[1::2]
            m = min(even_rows.shape[0], odd_rows.shape[0])
            even_rows = even_rows[:m]
            odd_rows = odd_rows[:m]

            patch_size = 32
            h, w = even_rows.shape[-2], even_rows.shape[-1]

            for row in range(h // patch_size):
                for col in range(w // patch_size):
                    y0, y1 = row * patch_size, (row + 1) * patch_size
                    x0, x1 = col * patch_size, (col + 1) * patch_size

                    patch_even = even_rows[y0:y1, x0:x1]
                    patch_odd = odd_rows[y0:y1, x0:x1]

                    intensity = (patch_even.mean() + patch_odd.mean()) / 2
                    intensities.append(intensity)

                    combined = np.zeros((patch_size * 2, patch_size))
                    combined[::2] = patch_even
                    combined[1::2] = patch_odd

                    try:
                        offset = _phase_corr_2d(
                            combined, upsample=upsample, border=0,
                            max_offset=max_offset, use_fft=True, fft_method=fft_method
                        )
                        offsets.append(abs(offset))
                    except Exception:
                        offsets.append(np.nan)

        intensities = np.array(intensities)
        offsets = np.array(offsets)

        valid = ~np.isnan(offsets) & (intensities > 0)
        if valid.sum() < 10:
            return

        percentiles = np.percentile(intensities[valid], np.linspace(0, 100, 11))
        bins = np.unique(percentiles)

        bin_centers = []
        bin_means = []
        bin_stds = []

        for i in range(len(bins) - 1):
            mask = valid & (intensities >= bins[i]) & (intensities < bins[i + 1])
            if mask.sum() > 5:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(np.mean(offsets[mask]))
                bin_stds.append(np.std(offsets[mask]))

        self.results.intensity_bins = np.array(bin_centers)
        self.results.offset_by_intensity = np.array(bin_means)
        self.results.offset_std_by_intensity = np.array(bin_stds)
        logger.info(f"parameters: {len(offsets)} patches")

    def run(self, fft_method="1d", upsample=10, border=4, max_offset=10):
        """run full analysis"""
        start = time.time()

        steps = [
            ("per-frame", lambda: self.analyze_per_frame(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
            ("window sizes", lambda: self.analyze_window_sizes(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
            ("spatial grid", lambda: self.analyze_spatial_grid(
                patch_sizes=(32, 64), fft_method=fft_method,
                upsample=upsample, max_offset=max_offset)),
            ("parameters", lambda: self.analyze_parameters(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
        ]

        if self.num_planes > 1:
            steps.append(("z-planes", lambda: self.analyze_z_planes(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)))

        for name, func in tqdm(steps, desc="scan-phase analysis"):
            func()

        self.results.analysis_time = time.time() - start
        logger.info(f"complete in {self.results.analysis_time:.1f}s")
        return self.results

    def generate_figures(self, output_dir=None, fmt="png", dpi=150, show=False):
        """generate analysis figures with dark theme"""
        import matplotlib.pyplot as plt
        _setup_dark_style()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        # temporal: per-frame offset over movie
        fig = self._fig_temporal()
        if output_dir:
            path = output_dir / f"temporal.{fmt}"
            fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
            saved.append(path)
        if show:
            plt.show()
        plt.close(fig)

        # window sizes: convergence analysis
        if len(self.results.window_sizes) > 0:
            fig = self._fig_windows()
            if output_dir:
                path = output_dir / f"windows.{fmt}"
                fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
                saved.append(path)
            if show:
                plt.show()
            plt.close(fig)

        # spatial heatmaps
        if self.results.grid_offsets:
            fig = self._fig_spatial()
            if output_dir:
                path = output_dir / f"spatial.{fmt}"
                fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
                saved.append(path)
            if show:
                plt.show()
            plt.close(fig)

        # z-planes
        if len(self.results.plane_offsets) > 1:
            fig = self._fig_zplanes()
            if output_dir:
                path = output_dir / f"zplanes.{fmt}"
                fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
                saved.append(path)
            if show:
                plt.show()
            plt.close(fig)

        # parameters
        if len(self.results.intensity_bins) > 0:
            fig = self._fig_parameters()
            if output_dir:
                path = output_dir / f"parameters.{fmt}"
                fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
                saved.append(path)
            if show:
                plt.show()
            plt.close(fig)

        return saved

    def _fig_temporal(self):
        """per-frame offset over movie"""
        fig, axes = _dark_fig(1, 2, figsize=(10, 4))

        offsets = self.results.offsets_fft
        valid = offsets[~np.isnan(offsets)]

        # time series
        ax = axes[0]
        ax.plot(offsets, color='#4da6ff', lw=0.5, alpha=0.8)
        if len(valid) > 0:
            mean_val = np.mean(valid)
            ax.axhline(mean_val, color='#ff6b6b', ls='--', lw=1.5, label=f'mean={mean_val:.2f} px')
            ax.legend(facecolor='#1a1a2e', edgecolor='white')
        ax.set_xlabel('frame', color='white')
        ax.set_ylabel('offset (px)', color='white')
        ax.set_title('offset over time', color='white')
        ax.grid(True, alpha=0.2, color='white')

        # histogram
        ax = axes[1]
        if len(valid) > 0:
            ax.hist(valid, bins=50, alpha=0.8, color='#4da6ff', edgecolor='#1a1a2e')
            stats = self.results.compute_stats(valid)
            txt = f"mean: {stats['mean']:.3f} px\nstd: {stats['std']:.3f} px"
            ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=9,
                    va='top', ha='right', fontfamily='monospace', color='white',
                    bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='white', alpha=0.8))
        ax.set_xlabel('offset (px)', color='white')
        ax.set_ylabel('count', color='white')
        ax.set_title('distribution', color='white')

        fig.tight_layout()
        return fig

    def _fig_windows(self):
        """window size convergence"""
        fig, axes = _dark_fig(1, 2, figsize=(10, 4))

        ws = self.results.window_sizes
        offs = self.results.window_offsets
        stds = self.results.window_stds

        # offset vs window size
        ax = axes[0]
        ax.errorbar(ws, offs, yerr=stds, fmt='o-', color='#4da6ff', capsize=3, ms=5, ecolor='#ff6b6b')
        ax.set_xscale('log')
        ax.set_xlabel('window size (frames)', color='white')
        ax.set_ylabel('offset (px)', color='white')
        ax.set_title('offset vs window size', color='white')
        ax.grid(True, alpha=0.2, color='white')
        # format x ticks without scientific notation
        ax.xaxis.set_major_formatter(lambda x, p: f'{int(x)}' if x >= 1 else f'{x:.1f}')

        # std vs window size
        ax = axes[1]
        ax.plot(ws, stds, 'o-', color='#50fa7b', ms=5)
        ax.set_xscale('log')
        ax.set_xlabel('window size (frames)', color='white')
        ax.set_ylabel('std of estimate (px)', color='white')
        ax.set_title('estimation variance', color='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.xaxis.set_major_formatter(lambda x, p: f'{int(x)}' if x >= 1 else f'{x:.1f}')

        # mark convergence threshold
        if len(stds) > 2:
            threshold = 0.1
            below = np.where(np.array(stds) < threshold)[0]
            if len(below) > 0:
                conv_ws = ws[below[0]]
                ax.axvline(conv_ws, color='#ff6b6b', ls='--', alpha=0.7)
                ax.text(conv_ws * 1.2, ax.get_ylim()[1] * 0.8, f'{conv_ws} frames',
                        color='#ff6b6b', fontsize=9)

        fig.tight_layout()
        return fig

    def _fig_spatial(self):
        """spatial heatmaps with interpolation"""
        import matplotlib.pyplot as plt
        from scipy.ndimage import zoom

        patch_sizes = sorted(self.results.grid_offsets.keys())
        n = len(patch_sizes)

        fig, axes = _dark_fig(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, ps in zip(axes, patch_sizes):
            offsets = self.results.grid_offsets[ps]
            valid = self.results.grid_valid[ps]
            masked = np.where(valid, offsets, np.nan)

            # interpolate for smoother display
            if masked.shape[0] > 1 and masked.shape[1] > 1:
                # fill nans with nearest valid for interpolation
                from scipy.ndimage import generic_filter
                filled = masked.copy()
                nan_mask = np.isnan(filled)
                if nan_mask.any() and not nan_mask.all():
                    # simple nearest fill
                    from scipy.interpolate import griddata
                    y, x = np.mgrid[0:filled.shape[0], 0:filled.shape[1]]
                    valid_pts = ~nan_mask
                    if valid_pts.sum() > 3:
                        filled = griddata(
                            (y[valid_pts], x[valid_pts]), filled[valid_pts],
                            (y, x), method='nearest'
                        )
                # zoom for smoother display
                zoom_factor = max(1, 100 // max(filled.shape))
                if zoom_factor > 1:
                    filled = zoom(filled, zoom_factor, order=1)
                display = filled
            else:
                display = masked

            vmax = max(0.5, np.nanmax(np.abs(display)))
            im = ax.imshow(display, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          aspect='auto', interpolation='bilinear')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('offset (px)', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            ax.set_xlabel('x', color='white')
            ax.set_ylabel('y', color='white')

            valid_vals = offsets[valid]
            if len(valid_vals) > 0:
                mean_val = np.mean(valid_vals)
                ax.set_title(f'{ps}x{ps} patches  (mean={mean_val:.2f} px)', color='white')
            else:
                ax.set_title(f'{ps}x{ps} patches', color='white')

            # remove tick labels (they're misleading after zoom)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        return fig

    def _fig_zplanes(self):
        """offset vs z-plane depth"""
        fig, ax = _dark_fig(figsize=(6, 4))

        offsets = self.results.plane_offsets
        depths = self.results.plane_depths_um

        if self.pixel_resolution_um > 0:
            ax.plot(depths, offsets, 'o-', color='#4da6ff', ms=6)
            ax.set_xlabel('depth (µm)', color='white')
        else:
            ax.plot(np.arange(len(offsets)), offsets, 'o-', color='#4da6ff', ms=6)
            ax.set_xlabel('z-plane', color='white')

        ax.set_ylabel('offset (px)', color='white')
        ax.set_title('offset by depth', color='white')
        ax.grid(True, alpha=0.2, color='white')

        valid = offsets[~np.isnan(offsets)]
        if len(valid) > 0:
            ax.axhline(np.mean(valid), color='#ff6b6b', ls='--', alpha=0.7,
                      label=f'mean={np.mean(valid):.2f} px')
            ax.legend(facecolor='#1a1a2e', edgecolor='white')

        fig.tight_layout()
        return fig

    def _fig_parameters(self):
        """offset reliability vs signal intensity"""
        fig, axes = _dark_fig(1, 2, figsize=(10, 4))

        bins = self.results.intensity_bins
        means = self.results.offset_by_intensity
        stds = self.results.offset_std_by_intensity

        # offset vs intensity
        ax = axes[0]
        ax.errorbar(bins, means, yerr=stds, fmt='o-', color='#4da6ff', capsize=3, ms=5, ecolor='#ff6b6b')
        ax.set_xlabel('signal intensity (a.u.)', color='white')
        ax.set_ylabel('|offset| (px)', color='white')
        ax.set_title('offset vs signal', color='white')
        ax.grid(True, alpha=0.2, color='white')

        # mark threshold
        if len(means) > 2:
            threshold_idx = np.argmax(means < means[-1] * 1.5)
            if threshold_idx > 0:
                threshold_int = bins[threshold_idx]
                ax.axvline(threshold_int, color='#ff6b6b', ls='--', alpha=0.7)
                ax.text(threshold_int * 1.1, ax.get_ylim()[1] * 0.85,
                        f'threshold\n~{threshold_int:.0f}', fontsize=8, color='#ff6b6b')

        # std vs intensity
        ax = axes[1]
        ax.plot(bins, stds, 'o-', color='#50fa7b', ms=5)
        ax.set_xlabel('signal intensity (a.u.)', color='white')
        ax.set_ylabel('std of offset (px)', color='white')
        ax.set_title('variability vs signal', color='white')
        ax.grid(True, alpha=0.2, color='white')

        fig.tight_layout()
        return fig

    def save_results(self, path):
        """save results to npz"""
        path = Path(path)
        data = {
            'offsets_fft': self.results.offsets_fft,
            'window_sizes': self.results.window_sizes,
            'window_offsets': self.results.window_offsets,
            'window_stds': self.results.window_stds,
            'plane_offsets': self.results.plane_offsets,
            'plane_depths_um': self.results.plane_depths_um,
            'intensity_bins': self.results.intensity_bins,
            'offset_by_intensity': self.results.offset_by_intensity,
            'offset_std_by_intensity': self.results.offset_std_by_intensity,
            'num_frames': self.results.num_frames,
            'num_planes': self.results.num_planes,
            'frame_shape': self.results.frame_shape,
            'pixel_resolution_um': self.results.pixel_resolution_um,
            'analysis_time': self.results.analysis_time,
        }
        for ps, grid in self.results.grid_offsets.items():
            data[f'grid_{ps}'] = grid
            data[f'grid_{ps}_valid'] = self.results.grid_valid[ps]

        np.savez_compressed(path, **data)
        logger.info(f"saved to {path}")
        return path


def run_scanphase_analysis(
    data_path=None,
    output_dir=None,
    fft_method="1d",
    image_format="png",
    show_plots=False,
):
    """run scan-phase analysis"""
    from pathlib import Path
    from mbo_utilities import imread

    if data_path is None:
        from mbo_utilities.graphics import select_file
        data_path = select_file(title="Select data for scan-phase analysis")
        if data_path is None:
            return None

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

    roi_yslices = None
    if hasattr(arr, 'yslices') and hasattr(arr, 'num_rois'):
        if arr.num_rois > 1:
            roi_yslices = arr.yslices
            logger.info(f"detected {arr.num_rois} ROIs")

    analyzer = ScanPhaseAnalyzer(arr, roi_yslices=roi_yslices)
    results = analyzer.run(fft_method=fft_method)
    analyzer.generate_figures(output_dir=output_dir, fmt=image_format, show=show_plots)
    analyzer.save_results(output_dir / "scanphase_results.npz")

    return results


def analyze_scanphase(data, output_dir=None, **kwargs):
    """run scan-phase analysis on array data"""
    analyzer = ScanPhaseAnalyzer(data)
    results = analyzer.run(**kwargs)
    if output_dir:
        analyzer.generate_figures(output_dir=output_dir)
        analyzer.save_results(Path(output_dir) / "scanphase_results.npz")
    return results
