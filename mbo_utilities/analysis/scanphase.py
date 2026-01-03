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
from mbo_utilities.analysis.phasecorr import _phase_corr_2d
from mbo_utilities.metadata import get_param

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
    """results from scan-phase analysis."""

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

    # horizontal tile analysis (offset vs x-position)
    horizontal_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    horizontal_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    horizontal_stds: np.ndarray = field(default_factory=lambda: np.array([]))

    # vertical tile analysis (offset vs y-position)
    vertical_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    vertical_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    vertical_stds: np.ndarray = field(default_factory=lambda: np.array([]))

    # temporal-spatial heatmap (time x spatial bins)
    temporal_spatial_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    temporal_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    spatial_bins: np.ndarray = field(default_factory=lambda: np.array([]))

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
                "num_frames": self.num_frames,
                "num_planes": self.num_planes,
                "frame_shape": self.frame_shape,
                "analysis_time": self.analysis_time,
            }
        }
        if len(self.offsets_fft) > 0:
            summary["fft"] = self.compute_stats(self.offsets_fft)
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
        if hasattr(data, "num_frames"):
            self.num_frames = data.num_frames
        else:
            self.num_frames = self.shape[0]

        # z-planes
        if hasattr(data, "num_planes"):
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
        md = getattr(data, "metadata", None)
        self.pixel_resolution_um = get_param(md, "dx", default=0.0)

        self.results = ScanPhaseResults(
            num_frames=self.num_frames,
            num_planes=self.num_planes,
            frame_shape=(self.frame_height, self.frame_width),
            pixel_resolution_um=self.pixel_resolution_um,
        )

        logger.info(f"ScanPhaseAnalyzer: {self.num_frames} frames, {self.num_planes} planes, shape={self.shape}")

    def _get_frame(self, idx, plane=0):
        """Get a single 2D frame."""
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
        """Extract single ROI from frame."""
        while frame.ndim > 2:
            frame = frame[0]
        yslice = self.roi_yslices[roi_idx]
        return frame[yslice, :]

    def _compute_offset(self, frame, upsample=10, border=4, max_offset=10):
        """Compute offset for a 2D frame, averaging across rois."""
        roi_offsets = []
        for roi_idx in range(self.num_rois):
            roi_frame = self._get_roi_frame(frame, roi_idx)
            try:
                offset = _phase_corr_2d(
                    roi_frame, upsample=upsample, border=border,
                    max_offset=max_offset, use_fft=True
                )
                roi_offsets.append(offset)
            except Exception:
                pass
        return np.mean(roi_offsets) if roi_offsets else np.nan

    def analyze_per_frame(self, upsample=10, border=4, max_offset=10):
        """
        Compute offset for each frame.

        primary measurement - shows temporal stability of the offset.
        """
        offsets = []
        for i in tqdm(range(self.num_frames), desc="per-frame", leave=False):
            frame = self._get_frame(i)
            offsets.append(self._compute_offset(frame, upsample, border, max_offset))

        self.results.offsets_fft = np.array(offsets)
        stats = self.results.compute_stats(self.results.offsets_fft)
        logger.info(f"per-frame: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        return self.results.offsets_fft

    def analyze_window_sizes(self, upsample=10, border=4, max_offset=10, num_samples=5):
        """
        Analyze how offset estimate varies with temporal window size.

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
                offset = self._compute_offset(mean_frame, upsample, border, max_offset)
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

    def analyze_spatial_grid(self, patch_sizes=(32, 64), upsample=10, max_offset=10, num_frames=100):
        """
        Compute offset in a grid of patches across the fov.

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
                            max_offset=max_offset, use_fft=True
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

    def analyze_z_planes(self, upsample=10, border=4, max_offset=10, num_frames=100):
        """
        Compute offset for each z-plane.

        different depths may have different offsets.
        """
        if self.num_planes <= 1:
            return

        sample_indices = np.linspace(0, self.num_frames - 1, min(num_frames, self.num_frames), dtype=int)
        plane_offsets = []

        for plane in tqdm(range(self.num_planes), desc="z-planes", leave=False):
            frames = [self._get_frame(i, plane=plane) for i in sample_indices]
            mean_frame = np.mean(frames, axis=0)
            offset = self._compute_offset(mean_frame, upsample, border, max_offset)
            plane_offsets.append(offset)

        self.results.plane_offsets = np.array(plane_offsets)

        if self.pixel_resolution_um > 0:
            self.results.plane_depths_um = np.arange(self.num_planes) * self.pixel_resolution_um
        else:
            self.results.plane_depths_um = np.arange(self.num_planes)

        logger.info(f"z-planes: {self.num_planes} planes")

    def analyze_parameters(self, upsample=10, border=4, max_offset=10, num_frames=50):
        """
        Analyze offset reliability vs signal intensity.

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
                            max_offset=max_offset, use_fft=True
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

    def analyze_horizontal_tiles(self, upsample=10, max_offset=10, num_frames=100, n_bins=16):
        """
        Analyze offset variation across horizontal (x) position.

        Shows how offset varies from left to right edge of FOV - important for
        detecting scan mirror nonlinearities.
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

        # divide into vertical strips (horizontal position bins)
        strip_width = w // n_bins
        if strip_width < 16:
            n_bins = max(4, w // 16)
            strip_width = w // n_bins

        positions = []
        offsets = []
        stds = []

        for i in range(n_bins):
            x0 = i * strip_width
            x1 = (i + 1) * strip_width if i < n_bins - 1 else w

            strip_even = even_rows[:, x0:x1]
            strip_odd = odd_rows[:, x0:x1]

            # subdivide strip into patches for variance estimate
            patch_h = min(64, h)
            n_patches = h // patch_h
            patch_offsets = []

            for p in range(n_patches):
                y0, y1 = p * patch_h, (p + 1) * patch_h
                pe = strip_even[y0:y1]
                po = strip_odd[y0:y1]

                if pe.mean() < 10 or po.mean() < 10:
                    continue

                combined = np.zeros((patch_h * 2, x1 - x0))
                combined[::2] = pe
                combined[1::2] = po

                try:
                    offset = _phase_corr_2d(
                        combined, upsample=upsample, border=0,
                        max_offset=max_offset, use_fft=True
                    )
                    patch_offsets.append(offset)
                except Exception:
                    pass

            if patch_offsets:
                positions.append((x0 + x1) / 2)
                offsets.append(np.mean(patch_offsets))
                stds.append(np.std(patch_offsets) if len(patch_offsets) > 1 else 0)

        self.results.horizontal_positions = np.array(positions)
        self.results.horizontal_offsets = np.array(offsets)
        self.results.horizontal_stds = np.array(stds)
        logger.info(f"horizontal tiles: {len(positions)} bins")

    def analyze_vertical_tiles(self, upsample=10, max_offset=10, num_frames=100, n_bins=16):
        """
        Analyze offset variation across vertical (y) position.

        Shows how offset varies from top to bottom of FOV.
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

        # divide into horizontal strips (vertical position bins)
        strip_height = h // n_bins
        if strip_height < 16:
            n_bins = max(4, h // 16)
            strip_height = h // n_bins

        positions = []
        offsets = []
        stds = []

        for i in range(n_bins):
            y0 = i * strip_height
            y1 = (i + 1) * strip_height if i < n_bins - 1 else h

            strip_even = even_rows[y0:y1]
            strip_odd = odd_rows[y0:y1]

            # subdivide strip into patches for variance estimate
            patch_w = min(64, w)
            n_patches = w // patch_w
            patch_offsets = []

            for p in range(n_patches):
                x0, x1_patch = p * patch_w, (p + 1) * patch_w
                pe = strip_even[:, x0:x1_patch]
                po = strip_odd[:, x0:x1_patch]

                if pe.mean() < 10 or po.mean() < 10:
                    continue

                combined = np.zeros(((y1 - y0) * 2, patch_w))
                combined[::2] = pe
                combined[1::2] = po

                try:
                    offset = _phase_corr_2d(
                        combined, upsample=upsample, border=0,
                        max_offset=max_offset, use_fft=True
                    )
                    patch_offsets.append(offset)
                except Exception:
                    pass

            if patch_offsets:
                positions.append((y0 + y1) / 2 * 2)  # multiply by 2 for original coords
                offsets.append(np.mean(patch_offsets))
                stds.append(np.std(patch_offsets) if len(patch_offsets) > 1 else 0)

        self.results.vertical_positions = np.array(positions)
        self.results.vertical_offsets = np.array(offsets)
        self.results.vertical_stds = np.array(stds)
        logger.info(f"vertical tiles: {len(positions)} bins")

    def analyze_temporal_spatial(self, upsample=10, max_offset=10, n_time_bins=20, n_spatial_bins=8):
        """
        Analyze offset variation over both time and spatial position.

        Creates a 2D heatmap showing how offset changes across the movie
        and across the FOV simultaneously - reveals drift patterns.
        """
        time_bins = np.linspace(0, self.num_frames - 1, n_time_bins + 1, dtype=int)
        heatmap = np.full((n_time_bins, n_spatial_bins), np.nan)

        for t_idx in tqdm(range(n_time_bins), desc="temporal-spatial", leave=False):
            t0, t1 = time_bins[t_idx], time_bins[t_idx + 1]
            if t1 <= t0:
                t1 = t0 + 1

            # average frames in time bin
            frames = [self._get_frame(i) for i in range(t0, min(t1, self.num_frames))]
            if not frames:
                continue
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

            # divide into spatial bins (horizontal)
            strip_width = w // n_spatial_bins

            for s_idx in range(n_spatial_bins):
                x0 = s_idx * strip_width
                x1 = (s_idx + 1) * strip_width if s_idx < n_spatial_bins - 1 else w

                strip_even = even_rows[:, x0:x1]
                strip_odd = odd_rows[:, x0:x1]

                if strip_even.mean() < 10 or strip_odd.mean() < 10:
                    continue

                combined = np.zeros((h * 2, x1 - x0))
                combined[::2] = strip_even
                combined[1::2] = strip_odd

                try:
                    offset = _phase_corr_2d(
                        combined, upsample=upsample, border=0,
                        max_offset=max_offset, use_fft=True
                    )
                    heatmap[t_idx, s_idx] = offset
                except Exception:
                    pass

        self.results.temporal_spatial_offsets = heatmap
        self.results.temporal_bins = (time_bins[:-1] + time_bins[1:]) / 2
        self.results.spatial_bins = np.linspace(0, self.frame_width, n_spatial_bins + 1)[:-1] + self.frame_width / (2 * n_spatial_bins)
        logger.info(f"temporal-spatial: {n_time_bins}x{n_spatial_bins} grid")

    def run(self, upsample=10, border=4, max_offset=10):
        """Run full analysis."""
        start = time.perf_counter()

        steps = [
            ("per-frame", lambda: self.analyze_per_frame(
                upsample=upsample, border=border, max_offset=max_offset)),
            ("window sizes", lambda: self.analyze_window_sizes(
                upsample=upsample, border=border, max_offset=max_offset)),
            ("spatial grid", lambda: self.analyze_spatial_grid(
                patch_sizes=(32, 64), upsample=upsample, max_offset=max_offset)),
            ("horizontal tiles", lambda: self.analyze_horizontal_tiles(
                upsample=upsample, max_offset=max_offset)),
            ("vertical tiles", lambda: self.analyze_vertical_tiles(
                upsample=upsample, max_offset=max_offset)),
            ("temporal-spatial", lambda: self.analyze_temporal_spatial(
                upsample=upsample, max_offset=max_offset)),
            ("parameters", lambda: self.analyze_parameters(
                upsample=upsample, border=border, max_offset=max_offset)),
        ]

        if self.num_planes > 1:
            steps.append(("z-planes", lambda: self.analyze_z_planes(
                upsample=upsample, border=border, max_offset=max_offset)))

        for _name, func in tqdm(steps, desc="scan-phase analysis"):
            func()

        self.results.analysis_time = time.perf_counter() - start
        logger.info(f"complete in {self.results.analysis_time:.1f}s")
        return self.results

    def generate_figures(self, output_dir=None, fmt="png", dpi=150, show=False):
        """Generate analysis figures with MBO dark theme."""
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

        # 1. Summary dashboard - key metrics at a glance
        fig = self._fig_summary()
        _save_fig(fig, "summary")

        # 2. Temporal: per-frame offset over movie
        fig = self._fig_temporal()
        _save_fig(fig, "temporal")

        # 3. Window sizes: convergence analysis
        if len(self.results.window_sizes) > 0:
            fig = self._fig_windows()
            _save_fig(fig, "windows")

        # 4. Spatial heatmaps
        if self.results.grid_offsets:
            fig = self._fig_spatial()
            _save_fig(fig, "spatial")

        # 5. Horizontal tile analysis
        if len(self.results.horizontal_positions) > 0:
            fig = self._fig_horizontal()
            _save_fig(fig, "horizontal")

        # 6. Vertical tile analysis
        if len(self.results.vertical_positions) > 0:
            fig = self._fig_vertical()
            _save_fig(fig, "vertical")

        # 7. Temporal-spatial heatmap
        if self.results.temporal_spatial_offsets.size > 0:
            fig = self._fig_temporal_spatial()
            _save_fig(fig, "temporal_spatial")

        # 8. Z-planes
        if len(self.results.plane_offsets) > 1:
            fig = self._fig_zplanes()
            _save_fig(fig, "zplanes")

        # 9. Parameters (intensity analysis)
        if len(self.results.intensity_bins) > 0:
            fig = self._fig_parameters()
            _save_fig(fig, "parameters")

        return saved

    def _fig_summary(self):
        """Summary dashboard with key metrics at a glance."""
        import matplotlib.pyplot as plt
        colors = MBO_DARK_THEME

        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor(colors["background"])

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                             left=0.06, right=0.94, top=0.92, bottom=0.06)

        # Title
        fig.suptitle("Scan-Phase Analysis Summary", fontsize=16, fontweight="bold",
                    color=colors["text"], y=0.97)

        # 1. Offset time series (top left, spans 2 cols)
        ax1 = fig.add_subplot(gs[0, :2])
        _apply_mbo_style(ax1)
        offsets = self.results.offsets_fft
        valid = offsets[~np.isnan(offsets)]
        ax1.plot(offsets, color=colors["primary"], lw=0.8, alpha=0.9)
        if len(valid) > 0:
            mean_val = np.mean(valid)
            ax1.axhline(mean_val, color=colors["error"], ls="--", lw=2,
                       label=f"mean = {mean_val:.2f} px")
            ax1.fill_between(range(len(offsets)),
                           mean_val - np.std(valid), mean_val + np.std(valid),
                           alpha=0.2, color=colors["primary"])
            ax1.legend(loc="upper right", facecolor=colors["surface"],
                      edgecolor=colors["border"], labelcolor=colors["text"])
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Offset (px)")
        ax1.set_title("Temporal Stability", fontsize=11, fontweight="bold")

        # 2. Stats panel (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor(colors["surface"])
        ax2.axis("off")

        stats_text = []
        if len(valid) > 0:
            stats = self.results.compute_stats(valid)
            stats_text = [
                f"Frames: {self.results.num_frames:,}",
                f"Shape: {self.results.frame_shape[1]}x{self.results.frame_shape[0]}",
                "",
                "Offset Statistics",
                f"  Mean:   {stats['mean']:+.3f} px",
                f"  Median: {stats['median']:+.3f} px",
                f"  Std:    {stats['std']:.3f} px",
                f"  Range:  [{stats['min']:.2f}, {stats['max']:.2f}]",
                "",
                f"Analysis: {self.results.analysis_time:.1f}s",
            ]

        for i, line in enumerate(stats_text):
            weight = "bold" if "Statistics" in line or "Frames" in line else "normal"
            color = colors["accent"] if "Mean:" in line else colors["text"]
            ax2.text(0.1, 0.9 - i * 0.085, line, transform=ax2.transAxes,
                    fontsize=10, fontfamily="monospace", color=color,
                    fontweight=weight)

        # 3. Histogram (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        _apply_mbo_style(ax3)
        if len(valid) > 0:
            _n, bins_hist, patches = ax3.hist(valid, bins=40, alpha=0.85,
                                             color=colors["primary"],
                                             edgecolor=colors["surface"])
            # color bars by distance from mean
            for i, p in enumerate(patches):
                bin_center = (bins_hist[i] + bins_hist[i+1]) / 2
                dist = abs(bin_center - np.mean(valid)) / (np.std(valid) + 1e-6)
                if dist > 2:
                    p.set_facecolor(colors["error"])
                elif dist > 1:
                    p.set_facecolor(colors["warning"])
        ax3.set_xlabel("Offset (px)")
        ax3.set_ylabel("Count")
        ax3.set_title("Distribution", fontsize=11, fontweight="bold")

        # 4. Window convergence (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        _apply_mbo_style(ax4)
        if len(self.results.window_sizes) > 0:
            ws = self.results.window_sizes
            stds = self.results.window_stds
            ax4.fill_between(ws, 0, stds, alpha=0.3, color=colors["success"])
            ax4.plot(ws, stds, "o-", color=colors["success"], ms=5, lw=2)
            ax4.set_xscale("log")
            ax4.axhline(0.1, color=colors["warning"], ls="--", alpha=0.7, lw=1.5)
            ax4.set_xlabel("Window Size (frames)")
            ax4.set_ylabel("Std (px)")
            ax4.set_title("Convergence", fontsize=11, fontweight="bold")
            ax4.xaxis.set_major_formatter(lambda x, p: f"{int(x)}" if x >= 1 else "")

        # 5. Spatial heatmap preview (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        _apply_mbo_style(ax5)
        ax5.grid(False)
        if self.results.grid_offsets:
            ps = max(self.results.grid_offsets.keys())
            offsets_grid = self.results.grid_offsets[ps]
            valid_grid = self.results.grid_valid[ps]
            display = np.where(valid_grid, offsets_grid, np.nan)
            vmax = max(0.5, np.nanmax(np.abs(display)))
            im = ax5.imshow(display, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            _mbo_colorbar(im, ax5, "px")
            ax5.set_title("Spatial Variation", fontsize=11, fontweight="bold")
            ax5.set_xticks([])
            ax5.set_yticks([])

        # 6. Horizontal variation (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        _apply_mbo_style(ax6)
        if len(self.results.horizontal_positions) > 0:
            pos = self.results.horizontal_positions
            offs = self.results.horizontal_offsets
            stds = self.results.horizontal_stds
            ax6.fill_between(pos, offs - stds, offs + stds, alpha=0.3, color=colors["secondary"])
            ax6.plot(pos, offs, "o-", color=colors["secondary"], ms=4, lw=2)
            ax6.axhline(np.nanmean(offs), color=colors["error"], ls="--", alpha=0.7)
        ax6.set_xlabel("X Position (px)")
        ax6.set_ylabel("Offset (px)")
        ax6.set_title("Horizontal Profile", fontsize=11, fontweight="bold")

        # 7. Vertical variation (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        _apply_mbo_style(ax7)
        if len(self.results.vertical_positions) > 0:
            pos = self.results.vertical_positions
            offs = self.results.vertical_offsets
            stds = self.results.vertical_stds
            ax7.fill_between(pos, offs - stds, offs + stds, alpha=0.3, color=colors["accent"])
            ax7.plot(pos, offs, "o-", color=colors["accent"], ms=4, lw=2)
            ax7.axhline(np.nanmean(offs), color=colors["error"], ls="--", alpha=0.7)
        ax7.set_xlabel("Y Position (px)")
        ax7.set_ylabel("Offset (px)")
        ax7.set_title("Vertical Profile", fontsize=11, fontweight="bold")

        # 8. Temporal-spatial heatmap preview (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        _apply_mbo_style(ax8)
        ax8.grid(False)
        if self.results.temporal_spatial_offsets.size > 0:
            heatmap = self.results.temporal_spatial_offsets
            vmax = max(0.5, np.nanmax(np.abs(heatmap)))
            im = ax8.imshow(heatmap, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            _mbo_colorbar(im, ax8, "px")
            ax8.set_xlabel("X Position")
            ax8.set_ylabel("Time")
            ax8.set_title("Drift Pattern", fontsize=11, fontweight="bold")

        return fig

    def _fig_temporal(self):
        """Per-frame offset over movie with detailed view."""
        colors = MBO_DARK_THEME
        fig, axes = _mbo_fig(2, 2, figsize=(12, 8))

        offsets = self.results.offsets_fft
        valid = offsets[~np.isnan(offsets)]
        stats = self.results.compute_stats(valid) if len(valid) > 0 else {}

        # 1. Full time series (top left)
        ax = axes[0, 0]
        ax.plot(offsets, color=colors["primary"], lw=0.5, alpha=0.9)
        if len(valid) > 0:
            mean_val = np.mean(valid)
            std_val = np.std(valid)
            ax.axhline(mean_val, color=colors["error"], ls="--", lw=1.5)
            ax.fill_between(range(len(offsets)), mean_val - std_val, mean_val + std_val,
                           alpha=0.15, color=colors["primary"])
            ax.fill_between(range(len(offsets)), mean_val - 2*std_val, mean_val + 2*std_val,
                           alpha=0.08, color=colors["primary"])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Offset (px)")
        ax.set_title("Offset Time Series", fontweight="bold")

        # 2. Rolling mean (top right)
        ax = axes[0, 1]
        if len(valid) > 10:
            window = min(100, len(offsets) // 10)
            rolling = np.convolve(offsets, np.ones(window)/window, mode="valid")
            ax.plot(range(window//2, len(rolling) + window//2), rolling,
                   color=colors["success"], lw=2, label=f"{window}-frame avg")
            ax.axhline(np.mean(valid), color=colors["error"], ls="--", lw=1.5, label="Global mean")
            ax.legend(loc="upper right", facecolor=colors["surface"],
                     edgecolor=colors["border"], labelcolor=colors["text"], fontsize=9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Offset (px)")
        ax.set_title("Rolling Average (Drift Detection)", fontweight="bold")

        # 3. Histogram (bottom left)
        ax = axes[1, 0]
        if len(valid) > 0:
            _n, bins_h, _patches = ax.hist(valid, bins=50, alpha=0.85,
                                        color=colors["primary"],
                                        edgecolor=colors["surface"], lw=0.5)
            ax.axvline(stats["mean"], color=colors["error"], ls="-", lw=2, label="Mean")
            ax.axvline(stats["median"], color=colors["warning"], ls="--", lw=2, label="Median")

            # Add gaussian fit overlay
            from scipy.stats import norm
            x = np.linspace(valid.min(), valid.max(), 100)
            pdf = norm.pdf(x, stats["mean"], stats["std"]) * len(valid) * (bins_h[1] - bins_h[0])
            ax.plot(x, pdf, color=colors["accent"], lw=2, ls="-", alpha=0.8, label="Normal fit")

            ax.legend(loc="upper right", facecolor=colors["surface"],
                     edgecolor=colors["border"], labelcolor=colors["text"], fontsize=9)
        ax.set_xlabel("Offset (px)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution", fontweight="bold")

        # 4. Stats box (bottom right)
        ax = axes[1, 1]
        ax.axis("off")
        if len(valid) > 0:
            box_text = [
                ("Total Frames", f"{self.results.num_frames:,}"),
                ("Valid Measurements", f"{len(valid):,} ({100*len(valid)/len(offsets):.1f}%)"),
                ("", ""),
                ("Mean Offset", f'{stats["mean"]:+.4f} px'),
                ("Median Offset", f'{stats["median"]:+.4f} px'),
                ("Std Deviation", f'{stats["std"]:.4f} px'),
                ("Range", f'[{stats["min"]:.3f}, {stats["max"]:.3f}] px'),
                ("", ""),
                ("Recommended Correction", f'{round(stats["mean"]*2)/2:+.1f} px'),
            ]

            for i, (label, value) in enumerate(box_text):
                y = 0.9 - i * 0.095
                if label:
                    ax.text(0.05, y, label + ":", transform=ax.transAxes,
                           fontsize=11, color=colors["text_muted"], fontweight="bold")
                    color = colors["accent"] if "Recommended" in label else colors["text"]
                    ax.text(0.55, y, value, transform=ax.transAxes,
                           fontsize=11, color=color, fontfamily="monospace")

        fig.tight_layout()
        return fig

    def _fig_windows(self):
        """Window size convergence analysis."""
        colors = MBO_DARK_THEME
        fig, axes = _mbo_fig(1, 3, figsize=(14, 4))

        ws = self.results.window_sizes
        offs = self.results.window_offsets
        stds = self.results.window_stds

        # 1. Offset vs window size
        ax = axes[0]
        ax.errorbar(ws, offs, yerr=stds, fmt="o-", color=colors["primary"],
                   capsize=4, ms=6, lw=2, ecolor=colors["error"], capthick=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Window Size (frames)")
        ax.set_ylabel("Offset (px)")
        ax.set_title("Offset Estimate vs Averaging Window", fontweight="bold")
        ax.xaxis.set_major_formatter(lambda x, p: f"{int(x)}" if x >= 1 else "")

        # Add final value line
        if len(offs) > 0:
            final_val = offs[-1]
            ax.axhline(final_val, color=colors["success"], ls="--", lw=1.5, alpha=0.7)
            ax.text(ws[0], final_val, f" {final_val:.3f} px", va="bottom",
                   color=colors["success"], fontsize=9)

        # 2. Std vs window size (convergence)
        ax = axes[1]
        ax.fill_between(ws, 0, stds, alpha=0.3, color=colors["success"])
        ax.plot(ws, stds, "o-", color=colors["success"], ms=6, lw=2)
        ax.set_xscale("log")
        ax.set_xlabel("Window Size (frames)")
        ax.set_ylabel("Std of Estimate (px)")
        ax.set_title("Estimation Precision", fontweight="bold")
        ax.xaxis.set_major_formatter(lambda x, p: f"{int(x)}" if x >= 1 else "")

        # Mark convergence threshold
        threshold = 0.1
        ax.axhline(threshold, color=colors["warning"], ls="--", lw=1.5, alpha=0.8)
        ax.text(ws[-1], threshold * 1.1, f"{threshold} px threshold",
               ha="right", va="bottom", color=colors["warning"], fontsize=9)

        if len(stds) > 2:
            below = np.where(np.array(stds) < threshold)[0]
            if len(below) > 0:
                conv_ws = ws[below[0]]
                ax.axvline(conv_ws, color=colors["accent"], ls="-", lw=2, alpha=0.7)
                ax.text(conv_ws * 1.2, ax.get_ylim()[1] * 0.7,
                       f"Converges at\n{conv_ws} frames",
                       color=colors["accent"], fontsize=10, fontweight="bold")

        # 3. Relative error
        ax = axes[2]
        if len(offs) > 0 and offs[-1] != 0:
            rel_error = 100 * stds / (np.abs(offs[-1]) + 1e-6)
            ax.fill_between(ws, 0, rel_error, alpha=0.3, color=colors["secondary"])
            ax.plot(ws, rel_error, "o-", color=colors["secondary"], ms=6, lw=2)
            ax.axhline(10, color=colors["warning"], ls="--", lw=1.5, alpha=0.8)
            ax.text(ws[-1], 11, "10% error", ha="right", color=colors["warning"], fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Window Size (frames)")
        ax.set_ylabel("Relative Error (%)")
        ax.set_title("Measurement Uncertainty", fontweight="bold")
        ax.xaxis.set_major_formatter(lambda x, p: f"{int(x)}" if x >= 1 else "")

        fig.tight_layout()
        return fig

    def _fig_spatial(self):
        """Spatial heatmaps showing offset variation across FOV."""
        from matplotlib.colors import TwoSlopeNorm
        colors = MBO_DARK_THEME

        patch_sizes = sorted(self.results.grid_offsets.keys())
        n = len(patch_sizes)

        fig, axes = _mbo_fig(2, n, figsize=(5 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        for col, ps in enumerate(patch_sizes):
            offsets = self.results.grid_offsets[ps]
            valid = self.results.grid_valid[ps]
            display = np.where(valid, offsets, np.nan)

            vmax = max(0.5, np.nanmax(np.abs(display)))

            # Top row: heatmap
            ax = axes[0, col]
            ax.grid(False)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(display, cmap="coolwarm", norm=norm,
                          aspect="equal", interpolation="nearest")
            _mbo_colorbar(im, ax, "Offset (px)")

            valid_vals = offsets[valid]
            if len(valid_vals) > 0:
                mean_val = np.mean(valid_vals)
                std_val = np.std(valid_vals)
                ax.set_title(f"{ps}x{ps} Patches\nmean={mean_val:.3f}, std={std_val:.3f} px",
                           fontweight="bold")
            else:
                ax.set_title(f"{ps}x{ps} Patches", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            # Bottom row: row/column averages
            ax = axes[1, col]
            if valid.any():
                # Column means (x profile)
                col_means = np.nanmean(display, axis=0)
                col_stds = np.nanstd(display, axis=0)
                x_pos = np.arange(len(col_means))
                ax.fill_between(x_pos, col_means - col_stds, col_means + col_stds,
                               alpha=0.3, color=colors["primary"])
                ax.plot(x_pos, col_means, "o-", color=colors["primary"], ms=4, lw=2,
                       label="X profile")

                # Row means (y profile) - on secondary axis
                ax2 = ax.twinx()
                row_means = np.nanmean(display, axis=1)
                np.nanstd(display, axis=1)
                y_pos = np.arange(len(row_means))
                ax2.plot(y_pos, row_means, "s-", color=colors["secondary"], ms=4, lw=2,
                        label="Y profile")
                ax2.set_ylabel("Y Profile Offset (px)", color=colors["secondary"])
                ax2.tick_params(axis="y", colors=colors["secondary"])

                ax.axhline(0, color=colors["text_muted"], ls="--", alpha=0.5)
                ax.set_xlabel("Position (patches)")
                ax.set_ylabel("X Profile Offset (px)", color=colors["primary"])
                ax.tick_params(axis="y", colors=colors["primary"])
                ax.set_title("Spatial Profiles", fontweight="bold")

                # Combined legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                         facecolor=colors["surface"], edgecolor=colors["border"],
                         labelcolor=colors["text"], fontsize=8)

        fig.tight_layout()
        return fig

    def _fig_horizontal(self):
        """Detailed horizontal (x-position) tile analysis."""
        colors = MBO_DARK_THEME
        fig, axes = _mbo_fig(1, 2, figsize=(12, 5))

        pos = self.results.horizontal_positions
        offs = self.results.horizontal_offsets
        stds = self.results.horizontal_stds

        # 1. Profile with error bands
        ax = axes[0]
        ax.fill_between(pos, offs - stds, offs + stds, alpha=0.3, color=colors["secondary"])
        ax.fill_between(pos, offs - 2*stds, offs + 2*stds, alpha=0.15, color=colors["secondary"])
        ax.plot(pos, offs, "o-", color=colors["secondary"], ms=6, lw=2)

        # Add mean line and polynomial fit
        mean_off = np.nanmean(offs)
        ax.axhline(mean_off, color=colors["error"], ls="--", lw=2, label=f"Mean = {mean_off:.3f} px")

        if len(pos) > 3:
            # Fit polynomial to detect scan nonlinearity
            valid_mask = ~np.isnan(offs)
            if valid_mask.sum() > 3:
                coeffs = np.polyfit(pos[valid_mask], offs[valid_mask], 2)
                fit_x = np.linspace(pos.min(), pos.max(), 100)
                fit_y = np.polyval(coeffs, fit_x)
                ax.plot(fit_x, fit_y, "-", color=colors["warning"], lw=2, alpha=0.8,
                       label="Quadratic fit")

        ax.legend(loc="best", facecolor=colors["surface"],
                 edgecolor=colors["border"], labelcolor=colors["text"])
        ax.set_xlabel("X Position (px)")
        ax.set_ylabel("Offset (px)")
        ax.set_title("Horizontal Scan-Phase Profile", fontweight="bold")

        # 2. Deviation from mean
        ax = axes[1]
        deviation = offs - mean_off
        colors_bars = [colors["success"] if abs(d) < np.std(offs) else colors["warning"]
                      if abs(d) < 2*np.std(offs) else colors["error"] for d in deviation]
        ax.bar(pos, deviation, width=pos[1]-pos[0] if len(pos) > 1 else 50,
              color=colors_bars, alpha=0.8, edgecolor=colors["surface"])
        ax.axhline(0, color=colors["text_muted"], ls="-", lw=1)
        ax.axhline(np.std(offs), color=colors["warning"], ls="--", lw=1, alpha=0.7)
        ax.axhline(-np.std(offs), color=colors["warning"], ls="--", lw=1, alpha=0.7)
        ax.set_xlabel("X Position (px)")
        ax.set_ylabel("Deviation from Mean (px)")
        ax.set_title("Spatial Non-Uniformity", fontweight="bold")

        fig.tight_layout()
        return fig

    def _fig_vertical(self):
        """Detailed vertical (y-position) tile analysis."""
        colors = MBO_DARK_THEME
        fig, axes = _mbo_fig(1, 2, figsize=(12, 5))

        pos = self.results.vertical_positions
        offs = self.results.vertical_offsets
        stds = self.results.vertical_stds

        # 1. Profile with error bands
        ax = axes[0]
        ax.fill_between(pos, offs - stds, offs + stds, alpha=0.3, color=colors["accent"])
        ax.fill_between(pos, offs - 2*stds, offs + 2*stds, alpha=0.15, color=colors["accent"])
        ax.plot(pos, offs, "o-", color=colors["accent"], ms=6, lw=2)

        mean_off = np.nanmean(offs)
        ax.axhline(mean_off, color=colors["error"], ls="--", lw=2, label=f"Mean = {mean_off:.3f} px")

        if len(pos) > 3:
            valid_mask = ~np.isnan(offs)
            if valid_mask.sum() > 3:
                coeffs = np.polyfit(pos[valid_mask], offs[valid_mask], 1)
                fit_x = np.linspace(pos.min(), pos.max(), 100)
                fit_y = np.polyval(coeffs, fit_x)
                ax.plot(fit_x, fit_y, "-", color=colors["warning"], lw=2, alpha=0.8,
                       label=f"Linear fit (slope={coeffs[0]*1000:.2f} px/1000)")

        ax.legend(loc="best", facecolor=colors["surface"],
                 edgecolor=colors["border"], labelcolor=colors["text"])
        ax.set_xlabel("Y Position (px)")
        ax.set_ylabel("Offset (px)")
        ax.set_title("Vertical Scan-Phase Profile", fontweight="bold")

        # 2. Deviation from mean
        ax = axes[1]
        deviation = offs - mean_off
        colors_bars = [colors["success"] if abs(d) < np.std(offs) else colors["warning"]
                      if abs(d) < 2*np.std(offs) else colors["error"] for d in deviation]
        ax.barh(pos, deviation, height=pos[1]-pos[0] if len(pos) > 1 else 50,
               color=colors_bars, alpha=0.8, edgecolor=colors["surface"])
        ax.axvline(0, color=colors["text_muted"], ls="-", lw=1)
        ax.axvline(np.std(offs), color=colors["warning"], ls="--", lw=1, alpha=0.7)
        ax.axvline(-np.std(offs), color=colors["warning"], ls="--", lw=1, alpha=0.7)
        ax.set_ylabel("Y Position (px)")
        ax.set_xlabel("Deviation from Mean (px)")
        ax.set_title("Spatial Non-Uniformity", fontweight="bold")
        ax.invert_yaxis()

        fig.tight_layout()
        return fig

    def _fig_temporal_spatial(self):
        """Temporal-spatial heatmap showing drift patterns."""
        from matplotlib.colors import TwoSlopeNorm
        colors = MBO_DARK_THEME

        fig, axes = _mbo_fig(2, 2, figsize=(12, 9))

        heatmap = self.results.temporal_spatial_offsets

        vmax = max(0.5, np.nanmax(np.abs(heatmap)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        # 1. Main heatmap (top left)
        ax = axes[0, 0]
        ax.grid(False)
        im = ax.imshow(heatmap, cmap="coolwarm", norm=norm, aspect="auto",
                      interpolation="nearest", origin="upper")
        _mbo_colorbar(im, ax, "Offset (px)")
        ax.set_xlabel("Spatial Bin (X position)")
        ax.set_ylabel("Time Bin (frames)")
        ax.set_title("Offset Drift Over Time and Space", fontweight="bold")

        # 2. Temporal mean profile (top right)
        ax = axes[0, 1]
        time_means = np.nanmean(heatmap, axis=1)
        time_stds = np.nanstd(heatmap, axis=1)
        ax.fill_betweenx(range(len(time_means)), time_means - time_stds, time_means + time_stds,
                        alpha=0.3, color=colors["primary"])
        ax.plot(time_means, range(len(time_means)), "o-", color=colors["primary"], ms=4, lw=2)
        ax.axvline(np.nanmean(time_means), color=colors["error"], ls="--", lw=2)
        ax.set_xlabel("Mean Offset (px)")
        ax.set_ylabel("Time Bin")
        ax.set_title("Temporal Drift", fontweight="bold")
        ax.invert_yaxis()

        # 3. Spatial mean profile (bottom left)
        ax = axes[1, 0]
        spatial_means = np.nanmean(heatmap, axis=0)
        spatial_stds = np.nanstd(heatmap, axis=0)
        ax.fill_between(range(len(spatial_means)), spatial_means - spatial_stds,
                       spatial_means + spatial_stds, alpha=0.3, color=colors["secondary"])
        ax.plot(range(len(spatial_means)), spatial_means, "o-",
               color=colors["secondary"], ms=4, lw=2)
        ax.axhline(np.nanmean(spatial_means), color=colors["error"], ls="--", lw=2)
        ax.set_xlabel("Spatial Bin")
        ax.set_ylabel("Mean Offset (px)")
        ax.set_title("Spatial Variation", fontweight="bold")

        # 4. Stats and interpretation (bottom right)
        ax = axes[1, 1]
        ax.axis("off")

        # Compute drift statistics
        if heatmap.size > 0:
            temporal_range = np.nanmax(time_means) - np.nanmin(time_means)
            spatial_range = np.nanmax(spatial_means) - np.nanmin(spatial_means)
            total_std = np.nanstd(heatmap)

            stats_text = [
                ("Temporal Drift", f"{temporal_range:.3f} px"),
                ("Spatial Variation", f"{spatial_range:.3f} px"),
                ("Overall Std", f"{total_std:.3f} px"),
                ("", ""),
                ("Interpretation:", ""),
            ]

            if temporal_range > 0.2:
                stats_text.append(("", "Significant temporal drift detected"))
            else:
                stats_text.append(("", "Offset is temporally stable"))

            if spatial_range > 0.2:
                stats_text.append(("", "Spatial non-uniformity present"))
            else:
                stats_text.append(("", "Spatially uniform offset"))

            for i, (label, value) in enumerate(stats_text):
                y = 0.85 - i * 0.1
                if label:
                    ax.text(0.1, y, label, transform=ax.transAxes, fontsize=11,
                           color=colors["text"], fontweight="bold")
                    ax.text(0.6, y, value, transform=ax.transAxes, fontsize=11,
                           color=colors["accent"], fontfamily="monospace")
                else:
                    ax.text(0.15, y, value, transform=ax.transAxes, fontsize=10,
                           color=colors["text_muted"])

        fig.tight_layout()
        return fig

    def _fig_zplanes(self):
        """Offset vs z-plane depth."""
        colors = MBO_DARK_THEME
        fig, axes = _mbo_fig(1, 2, figsize=(12, 5))

        offsets = self.results.plane_offsets
        depths = self.results.plane_depths_um

        # 1. Offset vs depth
        ax = axes[0]
        if self.pixel_resolution_um > 0:
            ax.plot(depths, offsets, "o-", color=colors["primary"], ms=8, lw=2)
            ax.set_xlabel("Depth (um)")
        else:
            ax.plot(np.arange(len(offsets)), offsets, "o-", color=colors["primary"], ms=8, lw=2)
            ax.set_xlabel("Z-plane Index")

        valid = offsets[~np.isnan(offsets)]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            ax.axhline(mean_val, color=colors["error"], ls="--", lw=2,
                      label=f"Mean = {mean_val:.3f} px")
            ax.fill_between(ax.get_xlim(), mean_val - np.std(valid), mean_val + np.std(valid),
                           alpha=0.2, color=colors["primary"])
            ax.legend(loc="best", facecolor=colors["surface"],
                     edgecolor=colors["border"], labelcolor=colors["text"])

        ax.set_ylabel("Offset (px)")
        ax.set_title("Offset by Imaging Depth", fontweight="bold")

        # 2. Deviation from mean
        ax = axes[1]
        if len(valid) > 0:
            deviation = offsets - mean_val
            colors_pts = [colors["success"] if abs(d) < np.std(valid) else colors["warning"]
                         if abs(d) < 2*np.std(valid) else colors["error"] for d in deviation]

            if self.pixel_resolution_um > 0:
                for i, (d, dev, c) in enumerate(zip(depths, deviation, colors_pts, strict=False)):
                    ax.scatter([d], [dev], c=c, s=80, zorder=3)
                ax.plot(depths, deviation, "-", color=colors["text_muted"], lw=1, alpha=0.5)
                ax.set_xlabel("Depth (um)")
            else:
                for i, (dev, c) in enumerate(zip(deviation, colors_pts, strict=False)):
                    ax.scatter([i], [dev], c=c, s=80, zorder=3)
                ax.plot(range(len(deviation)), deviation, "-", color=colors["text_muted"], lw=1, alpha=0.5)
                ax.set_xlabel("Z-plane Index")

            ax.axhline(0, color=colors["text_muted"], ls="-", lw=1)
            ax.axhline(np.std(valid), color=colors["warning"], ls="--", lw=1, alpha=0.7)
            ax.axhline(-np.std(valid), color=colors["warning"], ls="--", lw=1, alpha=0.7)

        ax.set_ylabel("Deviation from Mean (px)")
        ax.set_title("Z-Plane Variation", fontweight="bold")

        fig.tight_layout()
        return fig

    def _fig_parameters(self):
        """Offset reliability vs signal intensity."""
        colors = MBO_DARK_THEME
        fig, axes = _mbo_fig(1, 3, figsize=(14, 4))

        bins = self.results.intensity_bins
        means = self.results.offset_by_intensity
        stds = self.results.offset_std_by_intensity

        # 1. Offset vs intensity
        ax = axes[0]
        ax.errorbar(bins, means, yerr=stds, fmt="o-", color=colors["primary"],
                   capsize=4, ms=6, lw=2, ecolor=colors["error"], capthick=1.5)
        ax.set_xlabel("Signal Intensity (a.u.)")
        ax.set_ylabel("|Offset| (px)")
        ax.set_title("Offset vs Signal Strength", fontweight="bold")

        # Add stable region indicator
        if len(means) > 2:
            stable_val = means[-1]
            ax.axhline(stable_val, color=colors["success"], ls="--", lw=1.5, alpha=0.7)
            ax.fill_between(ax.get_xlim(), stable_val * 0.9, stable_val * 1.1,
                           alpha=0.15, color=colors["success"])

        # 2. Std vs intensity (reliability)
        ax = axes[1]
        ax.fill_between(bins, 0, stds, alpha=0.3, color=colors["warning"])
        ax.plot(bins, stds, "o-", color=colors["warning"], ms=6, lw=2)
        ax.set_xlabel("Signal Intensity (a.u.)")
        ax.set_ylabel("Std of Offset (px)")
        ax.set_title("Measurement Reliability", fontweight="bold")

        # Mark good/bad regions
        if len(stds) > 2:
            good_threshold = 0.2
            ax.axhline(good_threshold, color=colors["success"], ls="--", lw=1.5, alpha=0.8)
            ax.text(bins[-1], good_threshold * 1.1, "Reliable", ha="right",
                   color=colors["success"], fontsize=9)

        # 3. SNR-like metric
        ax = axes[2]
        if len(means) > 0 and len(stds) > 0:
            snr = np.abs(means) / (stds + 1e-6)
            ax.fill_between(bins, 0, snr, alpha=0.3, color=colors["success"])
            ax.plot(bins, snr, "o-", color=colors["success"], ms=6, lw=2)
            ax.axhline(3, color=colors["error"], ls="--", lw=1.5, alpha=0.8)
            ax.text(bins[-1], 3.2, "SNR = 3", ha="right", color=colors["error"], fontsize=9)
        ax.set_xlabel("Signal Intensity (a.u.)")
        ax.set_ylabel("Offset / Std")
        ax.set_title("Detection Confidence", fontweight="bold")

        fig.tight_layout()
        return fig

    def save_results(self, path):
        """Save results to npz."""
        path = Path(path)
        data = {
            "offsets_fft": self.results.offsets_fft,
            "window_sizes": self.results.window_sizes,
            "window_offsets": self.results.window_offsets,
            "window_stds": self.results.window_stds,
            "plane_offsets": self.results.plane_offsets,
            "plane_depths_um": self.results.plane_depths_um,
            "intensity_bins": self.results.intensity_bins,
            "offset_by_intensity": self.results.offset_by_intensity,
            "offset_std_by_intensity": self.results.offset_std_by_intensity,
            "horizontal_positions": self.results.horizontal_positions,
            "horizontal_offsets": self.results.horizontal_offsets,
            "horizontal_stds": self.results.horizontal_stds,
            "vertical_positions": self.results.vertical_positions,
            "vertical_offsets": self.results.vertical_offsets,
            "vertical_stds": self.results.vertical_stds,
            "temporal_spatial_offsets": self.results.temporal_spatial_offsets,
            "temporal_bins": self.results.temporal_bins,
            "spatial_bins": self.results.spatial_bins,
            "num_frames": self.results.num_frames,
            "num_planes": self.results.num_planes,
            "frame_shape": self.results.frame_shape,
            "pixel_resolution_um": self.results.pixel_resolution_um,
            "analysis_time": self.results.analysis_time,
        }
        for ps, grid in self.results.grid_offsets.items():
            data[f"grid_{ps}"] = grid
            data[f"grid_{ps}_valid"] = self.results.grid_valid[ps]

        np.savez_compressed(path, **data)
        logger.info(f"saved to {path}")
        return path


def run_scanphase_analysis(
    data_path=None,
    output_dir=None,
    image_format="png",
    show_plots=False,
):
    """Run scan-phase analysis."""
    from pathlib import Path
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

    roi_yslices = None
    if hasattr(arr, "yslices") and hasattr(arr, "num_rois") and arr.num_rois > 1:
        roi_yslices = arr.yslices
        logger.info(f"detected {arr.num_rois} ROIs")

    analyzer = ScanPhaseAnalyzer(arr, roi_yslices=roi_yslices)
    results = analyzer.run()
    analyzer.generate_figures(output_dir=output_dir, fmt=image_format, show=show_plots)
    analyzer.save_results(output_dir / "scanphase_results.npz")

    return results


def analyze_scanphase(data, output_dir=None, **kwargs):
    """Run scan-phase analysis on array data."""
    analyzer = ScanPhaseAnalyzer(data)
    results = analyzer.run(**kwargs)
    if output_dir:
        analyzer.generate_figures(output_dir=output_dir)
        analyzer.save_results(Path(output_dir) / "scanphase_results.npz")
    return results
