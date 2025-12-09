"""
Comprehensive scan-phase analysis for bidirectional scanning correction.

This module provides exhaustive analysis of phase offset characteristics across:
- Temporal domain (every frame, rolling statistics)
- Spatial domain (gridded patches across FOV)
- Z-planes (per-plane analysis)
- Window sizes (extensive range of temporal averaging)
- Method comparison (FFT vs integer-only)

The analysis generates detailed statistics and comprehensive visualizations
to fully characterize the scan-phase behavior of imaging data.
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

# Extensive window sizes for thorough analysis
DEFAULT_WINDOW_SIZES = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]

# Patch sizes for spatial grid analysis
DEFAULT_PATCH_SIZES = [32, 64, 128, 256]

# Rolling window sizes
DEFAULT_ROLLING_WINDOW_SIZES = [3, 5, 7, 10, 15, 20, 30, 50, 100]


@dataclass
class ScanPhaseResults:
    """
    Comprehensive results from scan-phase analysis.

    Contains all computed offsets, statistics, and distributions.
    """

    # === Temporal Analysis ===
    # Per-frame offsets (FFT and integer methods)
    offsets_by_frame_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_by_frame_int: np.ndarray = field(default_factory=lambda: np.array([]))
    frame_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    # Temporal statistics
    temporal_stats_fft: dict = field(default_factory=dict)
    temporal_stats_int: dict = field(default_factory=dict)

    # === Window Size Analysis ===
    window_sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_by_window_fft: dict = field(default_factory=dict)  # {size: offset}
    offsets_by_window_int: dict = field(default_factory=dict)

    # Multiple samples per window size for variance estimation
    window_samples_fft: dict = field(default_factory=dict)  # {size: [offsets]}
    window_samples_int: dict = field(default_factory=dict)

    # === Spatial Grid Analysis ===
    # Grid offsets for different patch sizes
    grid_offsets: dict = field(default_factory=dict)  # {patch_size: 2D array}
    grid_x_centers: dict = field(default_factory=dict)
    grid_y_centers: dict = field(default_factory=dict)
    grid_intensities: dict = field(default_factory=dict)
    grid_valid_mask: dict = field(default_factory=dict)

    # Spatial statistics per patch size
    spatial_stats: dict = field(default_factory=dict)  # {patch_size: stats_dict}

    # === Z-Plane Analysis ===
    offsets_by_plane_fft: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_by_plane_int: np.ndarray = field(default_factory=lambda: np.array([]))
    plane_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    # === X/Y Distribution Analysis ===
    # Offset distribution along X (columns)
    x_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_vs_x: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_vs_x_std: np.ndarray = field(default_factory=lambda: np.array([]))

    # Offset distribution along Y (rows)
    y_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_vs_y: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_vs_y_std: np.ndarray = field(default_factory=lambda: np.array([]))

    # === Rolling Mean Analysis ===
    rolling_window_sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_offsets_fft: dict = field(default_factory=dict)  # {size: (mean, std, median)}
    rolling_offsets_int: dict = field(default_factory=dict)

    # === Metadata ===
    num_frames: int = 0
    num_planes: int = 1
    frame_shape: tuple = ()
    analysis_time_seconds: float = 0.0

    def compute_statistics(self, arr: np.ndarray) -> dict:
        """Compute comprehensive statistics for an array."""
        valid = arr[~np.isnan(arr)] if len(arr) > 0 else np.array([])
        if len(valid) == 0:
            return {}

        return {
            'mean': float(np.mean(valid)),
            'median': float(np.median(valid)),
            'std': float(np.std(valid)),
            'var': float(np.var(valid)),
            'min': float(np.min(valid)),
            'max': float(np.max(valid)),
            'range': float(np.ptp(valid)),
            'p5': float(np.percentile(valid, 5)),
            'p25': float(np.percentile(valid, 25)),
            'p75': float(np.percentile(valid, 75)),
            'p95': float(np.percentile(valid, 95)),
            'iqr': float(np.percentile(valid, 75) - np.percentile(valid, 25)),
            'mad': float(np.median(np.abs(valid - np.median(valid)))),  # median absolute deviation
            'n': len(valid),
        }

    def get_summary(self) -> dict:
        """Get comprehensive summary of all results."""
        summary = {
            'metadata': {
                'num_frames': self.num_frames,
                'num_planes': self.num_planes,
                'frame_shape': self.frame_shape,
                'analysis_time_seconds': self.analysis_time_seconds,
            }
        }

        if len(self.offsets_by_frame_fft) > 0:
            summary['temporal_fft'] = self.compute_statistics(self.offsets_by_frame_fft)
        if len(self.offsets_by_frame_int) > 0:
            summary['temporal_int'] = self.compute_statistics(self.offsets_by_frame_int)

        if len(self.offsets_by_plane_fft) > 0:
            summary['zplane_fft'] = self.compute_statistics(self.offsets_by_plane_fft)
        if len(self.offsets_by_plane_int) > 0:
            summary['zplane_int'] = self.compute_statistics(self.offsets_by_plane_int)

        # Spatial stats per patch size
        for patch_size, stats in self.spatial_stats.items():
            summary[f'spatial_{patch_size}px'] = stats

        return summary


class ScanPhaseAnalysis:
    """
    Comprehensive scan-phase analyzer for bidirectional scanning data.

    Performs exhaustive analysis across temporal, spatial, and z-plane dimensions
    using both FFT and integer-only methods.

    Parameters
    ----------
    data : array-like
        Input data array. Can be 2D (YX), 3D (TYX or ZYX), or 4D (TZYX).
    num_planes : int, optional
        Number of z-planes. If None, inferred from data.

    Examples
    --------
    >>> from mbo_utilities import imread
    >>> from mbo_utilities.analysis import ScanPhaseAnalysis
    >>> data = imread("path/to/data.tiff")
    >>> analyzer = ScanPhaseAnalysis(data)
    >>> results = analyzer.run_full_analysis()
    >>> analyzer.generate_plots("output_dir/")
    """

    def __init__(self, data, num_planes: Optional[int] = None):
        self.data = data
        self._num_planes = num_planes

        # Determine data properties
        if hasattr(data, 'shape'):
            self.shape = data.shape
            self.ndim = len(self.shape)
        else:
            self.shape = np.array(data).shape
            self.ndim = len(self.shape)

        # Infer number of planes
        if num_planes is not None:
            self.num_planes = num_planes
        elif hasattr(data, 'num_channels'):
            self.num_planes = data.num_channels
        elif hasattr(data, 'metadata') and 'num_planes' in data.metadata:
            self.num_planes = data.metadata['num_planes']
        elif self.ndim == 4:
            self.num_planes = self.shape[1]
        else:
            self.num_planes = 1

        # Determine number of frames
        if hasattr(data, 'num_frames'):
            self.num_frames = data.num_frames
        elif self.ndim >= 3:
            self.num_frames = self.shape[0]
        else:
            self.num_frames = 1

        # Frame dimensions
        self.frame_height = self.shape[-2]
        self.frame_width = self.shape[-1]

        # Results storage
        self.results = ScanPhaseResults(
            num_frames=self.num_frames,
            num_planes=self.num_planes,
            frame_shape=(self.frame_height, self.frame_width),
        )

        logger.info(
            f"ScanPhaseAnalysis initialized: shape={self.shape}, "
            f"frames={self.num_frames}, planes={self.num_planes}"
        )

    def _get_frame(self, frame_idx: int, plane_idx: int = 0) -> np.ndarray:
        """Get a single 2D frame from the data."""
        if self.ndim == 2:
            return np.asarray(self.data)
        elif self.ndim == 3:
            return np.asarray(self.data[frame_idx])
        elif self.ndim == 4:
            return np.asarray(self.data[frame_idx, plane_idx])
        else:
            raise ValueError(f"Unsupported data dimensionality: {self.ndim}")

    def _get_mean_frame(self, frame_indices: list, plane_idx: int = 0) -> np.ndarray:
        """Get mean of specified frames."""
        frames = [self._get_frame(idx, plane_idx) for idx in frame_indices]
        return np.mean(frames, axis=0)

    def analyze_temporal_all_frames(
        self,
        use_fft: bool = True,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 4,
        max_offset: int = 10,
        plane_idx: int = 0,
        progress_callback=None,
    ) -> np.ndarray:
        """
        Analyze phase offset for EVERY frame individually.

        Parameters
        ----------
        use_fft : bool
            Use FFT-based method.
        fft_method : str
            '1d' or '2d'.
        plane_idx : int
            Z-plane to analyze.
        progress_callback : callable
            Function(current, total, message) for progress.

        Returns
        -------
        np.ndarray
            Offset for each frame.
        """
        offsets = []
        method_name = "FFT" if use_fft else "Integer"

        for i in tqdm(range(self.num_frames), desc=f"frames ({method_name})", leave=False):
            frame = self._get_frame(i, plane_idx)
            offset = _phase_corr_2d(
                frame,
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=use_fft,
                fft_method=fft_method,
            )
            offsets.append(offset)

        offsets = np.array(offsets)

        if use_fft:
            self.results.offsets_by_frame_fft = offsets
        else:
            self.results.offsets_by_frame_int = offsets
        self.results.frame_indices = np.arange(self.num_frames)

        stats = self.results.compute_statistics(offsets)
        method_name = "FFT" if use_fft else "Integer"
        logger.info(
            f"Temporal analysis ({method_name}): {self.num_frames} frames, "
            f"mean={stats.get('mean', 0):.4f}, median={stats.get('median', 0):.4f}, "
            f"std={stats.get('std', 0):.4f}"
        )

        return offsets

    def analyze_window_sizes(
        self,
        window_sizes: Optional[list] = None,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 4,
        max_offset: int = 10,
        plane_idx: int = 0,
        num_samples_per_size: int = 10,
        progress_callback=None,
    ) -> dict:
        """
        Analyze offset across many temporal window sizes with multiple samples.

        For each window size, computes offsets from multiple non-overlapping
        segments of the recording to estimate variance.

        Parameters
        ----------
        window_sizes : list, optional
            Sizes to test. Default: extensive range from 1 to 5000.
        num_samples_per_size : int
            Number of independent samples per window size.

        Returns
        -------
        dict
            {size: {'fft': (mean, std), 'int': (mean, std)}}
        """
        if window_sizes is None:
            window_sizes = [ws for ws in DEFAULT_WINDOW_SIZES if ws <= self.num_frames]

        self.results.window_sizes = np.array(window_sizes)
        results = {}

        for ws in tqdm(window_sizes, desc="window sizes", leave=False):

            # Determine sample positions (non-overlapping if possible)
            max_samples = max(1, self.num_frames // ws)
            actual_samples = min(num_samples_per_size, max_samples)

            if actual_samples == max_samples:
                # Use all non-overlapping windows
                sample_starts = [i * ws for i in range(actual_samples)]
            else:
                # Evenly space samples
                sample_starts = np.linspace(0, self.num_frames - ws, actual_samples, dtype=int).tolist()

            fft_offsets = []
            int_offsets = []

            for start in sample_starts:
                # Get mean frame for this window
                indices = list(range(start, min(start + ws, self.num_frames)))
                mean_frame = self._get_mean_frame(indices, plane_idx)

                # FFT method
                offset_fft = _phase_corr_2d(
                    mean_frame,
                    upsample=upsample,
                    border=border,
                    max_offset=max_offset,
                    use_fft=True,
                    fft_method=fft_method,
                )
                fft_offsets.append(float(offset_fft))

                # Integer method
                offset_int = _phase_corr_2d(
                    mean_frame,
                    upsample=upsample,
                    border=border,
                    max_offset=max_offset,
                    use_fft=False,
                    fft_method=fft_method,
                )
                int_offsets.append(float(offset_int))

            # Store all samples
            self.results.window_samples_fft[ws] = fft_offsets
            self.results.window_samples_int[ws] = int_offsets

            # Store mean offset
            self.results.offsets_by_window_fft[ws] = float(np.mean(fft_offsets))
            self.results.offsets_by_window_int[ws] = float(np.mean(int_offsets))

            results[ws] = {
                'fft': (np.mean(fft_offsets), np.std(fft_offsets)),
                'int': (np.mean(int_offsets), np.std(int_offsets)),
            }

        logger.info(f"Window size analysis complete: {len(window_sizes)} sizes tested")
        return results

    def analyze_spatial_grid(
        self,
        patch_sizes: Optional[list] = None,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 0,
        max_offset: int = 10,
        min_intensity: float = 10.0,
        num_frames_to_average: int = 100,
        plane_idx: int = 0,
        progress_callback=None,
    ) -> dict:
        """
        Analyze spatial distribution of offsets using gridded patches.

        Creates a regular grid of patches across the FOV and computes
        the offset for each patch. Tests multiple patch sizes.

        Parameters
        ----------
        patch_sizes : list, optional
            Patch sizes to test. Default: [32, 64, 128, 256].
        num_frames_to_average : int
            Frames to average for robust spatial analysis.
        min_intensity : float
            Minimum patch intensity to include.

        Returns
        -------
        dict
            {patch_size: {'offsets': 2D array, 'stats': dict}}
        """
        if patch_sizes is None:
            # Only use patch sizes that fit in the frame
            patch_sizes = [ps for ps in DEFAULT_PATCH_SIZES
                          if ps <= min(self.frame_height // 2, self.frame_width)]

        # Get averaged frame
        sample_indices = np.linspace(
            0, self.num_frames - 1,
            min(num_frames_to_average, self.num_frames),
            dtype=int
        ).tolist()
        mean_frame = self._get_mean_frame(sample_indices, plane_idx)

        # Split into even/odd rows
        even_rows = mean_frame[::2]
        odd_rows = mean_frame[1::2]
        m = min(even_rows.shape[0], odd_rows.shape[0])
        even_rows = even_rows[:m]
        odd_rows = odd_rows[:m]

        h, w = even_rows.shape

        results = {}

        for patch_size in tqdm(patch_sizes, desc="spatial grid", leave=False):
            # Calculate grid dimensions
            n_rows = h // patch_size
            n_cols = w // patch_size

            if n_rows < 1 or n_cols < 1:
                logger.warning(f"Patch size {patch_size} too large for frame, skipping")
                continue

            # Initialize arrays
            offsets = np.full((n_rows, n_cols), np.nan)
            intensities = np.full((n_rows, n_cols), np.nan)
            valid_mask = np.zeros((n_rows, n_cols), dtype=bool)
            x_centers = np.zeros(n_cols)
            y_centers = np.zeros(n_rows)

            for row in range(n_rows):
                y_start = row * patch_size
                y_end = y_start + patch_size
                y_centers[row] = (y_start + y_end) / 2

                for col in range(n_cols):
                    x_start = col * patch_size
                    x_end = x_start + patch_size
                    x_centers[col] = (x_start + x_end) / 2

                    # Extract patches
                    patch_even = even_rows[y_start:y_end, x_start:x_end]
                    patch_odd = odd_rows[y_start:y_end, x_start:x_end]

                    # Check intensity
                    mean_int = (patch_even.mean() + patch_odd.mean()) / 2
                    intensities[row, col] = mean_int

                    if mean_int < min_intensity:
                        continue

                    try:
                        # Reconstruct combined frame for phase correlation
                        combined = np.zeros((patch_size * 2, patch_size))
                        combined[::2] = patch_even
                        combined[1::2] = patch_odd

                        offset = _phase_corr_2d(
                            combined,
                            upsample=upsample,
                            border=border,
                            max_offset=max_offset,
                            use_fft=True,
                            fft_method=fft_method,
                        )
                        offsets[row, col] = float(offset)
                        valid_mask[row, col] = True

                    except Exception as e:
                        logger.debug(f"Patch ({row}, {col}) failed: {e}")

            # Store results
            self.results.grid_offsets[patch_size] = offsets
            self.results.grid_x_centers[patch_size] = x_centers
            self.results.grid_y_centers[patch_size] = y_centers
            self.results.grid_intensities[patch_size] = intensities
            self.results.grid_valid_mask[patch_size] = valid_mask

            # Compute statistics for valid patches
            valid_offsets = offsets[valid_mask]
            stats = self.results.compute_statistics(valid_offsets)
            stats['n_patches'] = int(valid_mask.sum())
            stats['n_total'] = int(valid_mask.size)
            stats['coverage'] = float(valid_mask.sum() / valid_mask.size)
            self.results.spatial_stats[patch_size] = stats

            results[patch_size] = {'offsets': offsets, 'stats': stats}

            logger.info(
                f"Spatial grid {patch_size}px: {stats['n_patches']}/{stats['n_total']} patches, "
                f"mean={stats.get('mean', 0):.4f}, std={stats.get('std', 0):.4f}"
            )

        return results

    def analyze_x_distribution(
        self,
        n_strips: int = 20,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 0,
        max_offset: int = 10,
        num_frames_to_average: int = 100,
        plane_idx: int = 0,
        progress_callback=None,
    ) -> tuple:
        """
        Analyze offset distribution along X (horizontal) axis.

        Divides the image into vertical strips and computes offset for each.

        Returns
        -------
        tuple
            (x_positions, mean_offsets, std_offsets)
        """
        # Get averaged frame
        sample_indices = np.linspace(
            0, self.num_frames - 1,
            min(num_frames_to_average, self.num_frames),
            dtype=int
        ).tolist()
        mean_frame = self._get_mean_frame(sample_indices, plane_idx)

        even_rows = mean_frame[::2]
        odd_rows = mean_frame[1::2]
        m = min(even_rows.shape[0], odd_rows.shape[0])
        even_rows = even_rows[:m]
        odd_rows = odd_rows[:m]

        h, w = even_rows.shape
        strip_width = w // n_strips

        x_positions = []
        offsets_mean = []
        offsets_std = []

        for i in range(n_strips):
            x_start = i * strip_width
            x_end = x_start + strip_width if i < n_strips - 1 else w
            x_center = (x_start + x_end) / 2

            strip_even = even_rows[:, x_start:x_end]
            strip_odd = odd_rows[:, x_start:x_end]

            # Compute offset for strip
            combined = np.zeros((h * 2, x_end - x_start))
            combined[::2] = strip_even
            combined[1::2] = strip_odd

            try:
                offset = _phase_corr_2d(
                    combined,
                    upsample=upsample,
                    border=border,
                    max_offset=max_offset,
                    use_fft=True,
                    fft_method=fft_method,
                )
                x_positions.append(x_center)
                offsets_mean.append(float(offset))
                offsets_std.append(0.0)  # Single measurement
            except Exception:
                pass

        self.results.x_positions = np.array(x_positions)
        self.results.offsets_vs_x = np.array(offsets_mean)
        self.results.offsets_vs_x_std = np.array(offsets_std)

        return self.results.x_positions, self.results.offsets_vs_x, self.results.offsets_vs_x_std

    def analyze_y_distribution(
        self,
        n_strips: int = 10,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 0,
        max_offset: int = 10,
        num_frames_to_average: int = 100,
        plane_idx: int = 0,
        progress_callback=None,
    ) -> tuple:
        """
        Analyze offset distribution along Y (vertical) axis.

        Divides the image into horizontal strips and computes offset for each.

        Returns
        -------
        tuple
            (y_positions, mean_offsets, std_offsets)
        """
        # Get averaged frame
        sample_indices = np.linspace(
            0, self.num_frames - 1,
            min(num_frames_to_average, self.num_frames),
            dtype=int
        ).tolist()
        mean_frame = self._get_mean_frame(sample_indices, plane_idx)

        even_rows = mean_frame[::2]
        odd_rows = mean_frame[1::2]
        m = min(even_rows.shape[0], odd_rows.shape[0])
        even_rows = even_rows[:m]
        odd_rows = odd_rows[:m]

        h, w = even_rows.shape
        strip_height = h // n_strips

        y_positions = []
        offsets_mean = []
        offsets_std = []

        for i in range(n_strips):
            y_start = i * strip_height
            y_end = y_start + strip_height if i < n_strips - 1 else h
            y_center = (y_start + y_end) / 2

            strip_even = even_rows[y_start:y_end, :]
            strip_odd = odd_rows[y_start:y_end, :]

            # Compute offset for strip
            sh = y_end - y_start
            combined = np.zeros((sh * 2, w))
            combined[::2] = strip_even
            combined[1::2] = strip_odd

            try:
                offset = _phase_corr_2d(
                    combined,
                    upsample=upsample,
                    border=border,
                    max_offset=max_offset,
                    use_fft=True,
                    fft_method=fft_method,
                )
                y_positions.append(y_center)
                offsets_mean.append(float(offset))
                offsets_std.append(0.0)
            except Exception:
                pass

        self.results.y_positions = np.array(y_positions)
        self.results.offsets_vs_y = np.array(offsets_mean)
        self.results.offsets_vs_y_std = np.array(offsets_std)

        return self.results.y_positions, self.results.offsets_vs_y, self.results.offsets_vs_y_std

    def analyze_z_planes(
        self,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 4,
        max_offset: int = 10,
        frames_per_plane: int = 50,
        progress_callback=None,
    ) -> tuple:
        """
        Analyze offset for each z-plane.

        Returns
        -------
        tuple
            (plane_indices, offsets_fft, offsets_int)
        """
        if self.num_planes <= 1:
            return np.array([0]), np.array([0.0]), np.array([0.0])

        offsets_fft = []
        offsets_int = []

        for plane_idx in range(self.num_planes):
            if progress_callback:
                progress_callback(plane_idx, self.num_planes, f"Z-plane {plane_idx}")

            # Get mean frame for this plane
            sample_indices = np.linspace(
                0, self.num_frames - 1,
                min(frames_per_plane, self.num_frames),
                dtype=int
            ).tolist()
            mean_frame = self._get_mean_frame(sample_indices, plane_idx)

            # FFT method
            offset_fft = _phase_corr_2d(
                mean_frame,
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=True,
                fft_method=fft_method,
            )
            offsets_fft.append(float(offset_fft))

            # Integer method
            offset_int = _phase_corr_2d(
                mean_frame,
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=False,
                fft_method=fft_method,
            )
            offsets_int.append(float(offset_int))

        self.results.plane_indices = np.arange(self.num_planes)
        self.results.offsets_by_plane_fft = np.array(offsets_fft)
        self.results.offsets_by_plane_int = np.array(offsets_int)

        logger.info(
            f"Z-plane analysis: {self.num_planes} planes, "
            f"FFT range=[{min(offsets_fft):.3f}, {max(offsets_fft):.3f}], "
            f"INT range=[{min(offsets_int):.3f}, {max(offsets_int):.3f}]"
        )

        return self.results.plane_indices, self.results.offsets_by_plane_fft, self.results.offsets_by_plane_int

    def analyze_rolling_mean_subtraction(
        self,
        rolling_sizes: Optional[list] = None,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 4,
        max_offset: int = 10,
        plane_idx: int = 0,
        num_samples: int = 20,
        progress_callback=None,
    ) -> dict:
        """
        Analyze offset using rolling mean subtraction (high-pass filtering).

        For each rolling window size, subtracts a rolling mean from frames
        before computing phase correlation. This can help with data that
        has slow intensity drift.

        Returns
        -------
        dict
            {window_size: {'fft': (mean, std, median), 'int': (mean, std, median)}}
        """
        if rolling_sizes is None:
            rolling_sizes = [rs for rs in DEFAULT_ROLLING_WINDOW_SIZES if rs < self.num_frames // 2]

        if len(rolling_sizes) == 0:
            logger.warning("Not enough frames for rolling mean analysis")
            return {}

        self.results.rolling_window_sizes = np.array(rolling_sizes)
        results = {}

        for rs in rolling_sizes:
            if progress_callback:
                progress_callback(0, 1, f"Rolling window {rs}")

            # Sample positions in the recording
            valid_range = self.num_frames - rs
            if valid_range <= 0:
                continue

            sample_positions = np.linspace(rs // 2, self.num_frames - rs // 2 - 1, num_samples, dtype=int)

            fft_offsets = []
            int_offsets = []

            for pos in sample_positions:
                # Get frames for rolling mean
                start = max(0, pos - rs // 2)
                end = min(self.num_frames, pos + rs // 2)

                frames = [self._get_frame(i, plane_idx).astype(float) for i in range(start, end)]
                rolling_mean = np.mean(frames, axis=0)

                # Current frame minus rolling mean
                current = self._get_frame(pos, plane_idx).astype(float)
                subtracted = current - rolling_mean
                subtracted = subtracted - subtracted.min() + 1  # Shift to positive

                # FFT
                try:
                    offset_fft = _phase_corr_2d(
                        subtracted,
                        upsample=upsample,
                        border=border,
                        max_offset=max_offset,
                        use_fft=True,
                        fft_method=fft_method,
                    )
                    fft_offsets.append(float(offset_fft))
                except Exception:
                    pass

                # Integer
                try:
                    offset_int = _phase_corr_2d(
                        subtracted,
                        upsample=upsample,
                        border=border,
                        max_offset=max_offset,
                        use_fft=False,
                        fft_method=fft_method,
                    )
                    int_offsets.append(float(offset_int))
                except Exception:
                    pass

            if fft_offsets:
                self.results.rolling_offsets_fft[rs] = (
                    float(np.mean(fft_offsets)),
                    float(np.std(fft_offsets)),
                    float(np.median(fft_offsets))
                )
            if int_offsets:
                self.results.rolling_offsets_int[rs] = (
                    float(np.mean(int_offsets)),
                    float(np.std(int_offsets)),
                    float(np.median(int_offsets))
                )

            results[rs] = {
                'fft': self.results.rolling_offsets_fft.get(rs, (0, 0, 0)),
                'int': self.results.rolling_offsets_int.get(rs, (0, 0, 0)),
            }

        logger.info(f"Rolling mean analysis: {len(rolling_sizes)} window sizes")
        return results

    def run_full_analysis(
        self,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 4,
        max_offset: int = 10,
        progress_callback=None,
    ) -> ScanPhaseResults:
        """
        Run comprehensive scan-phase analysis.

        Performs all available analyses:
        1. Per-frame temporal analysis (FFT and integer)
        2. Window size effects (extensive range)
        3. Spatial grid analysis (multiple patch sizes)
        4. X and Y distribution analysis
        5. Z-plane analysis (if multi-plane)
        6. Rolling mean subtraction analysis

        Parameters
        ----------
        fft_method : str
            '1d' (faster) or '2d' (more accurate).
        upsample : int
            Subpixel precision factor.
        border : int
            Border pixels to exclude.
        max_offset : int
            Maximum offset to search.
        progress_callback : callable
            Function(current, total, message) for progress.

        Returns
        -------
        ScanPhaseResults
            Comprehensive analysis results.
        """
        start_time = time.time()

        steps = [
            ("temporal (FFT)", lambda: self.analyze_temporal_all_frames(
                use_fft=True, fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
            ("temporal (int)", lambda: self.analyze_temporal_all_frames(
                use_fft=False, fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
            ("window sizes", lambda: self.analyze_window_sizes(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)),
            ("spatial grid", lambda: self.analyze_spatial_grid(
                fft_method=fft_method, upsample=upsample,
                border=0, max_offset=max_offset)),
            ("x distribution", lambda: self.analyze_x_distribution(
                fft_method=fft_method, upsample=upsample, max_offset=max_offset)),
            ("y distribution", lambda: self.analyze_y_distribution(
                fft_method=fft_method, upsample=upsample, max_offset=max_offset)),
        ]

        # conditional steps
        if self.num_planes > 1:
            steps.append(("z-planes", lambda: self.analyze_z_planes(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)))
        if self.num_frames > 10:
            steps.append(("rolling mean", lambda: self.analyze_rolling_mean_subtraction(
                fft_method=fft_method, upsample=upsample,
                border=border, max_offset=max_offset)))

        for name, func in tqdm(steps, desc="scan-phase analysis"):
            func()

        self.results.analysis_time_seconds = time.time() - start_time
        logger.info(f"Full analysis complete in {self.results.analysis_time_seconds:.1f}s")

        return self.results

    def generate_plots(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        format: str = "png",
        dpi: int = 150,
        show: bool = False,
    ) -> list:
        """
        Generate visualization figures.

        Creates 2-3 focused figures:
        1. Temporal analysis (time series, histogram, window sizes)
        2. Spatial analysis (grid heatmap, X/Y profiles)
        3. Z-plane analysis (only if multi-plane data)

        Returns
        -------
        list
            Paths to saved figures.
        """
        import matplotlib.pyplot as plt

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # 1. temporal analysis figure
        fig = self._create_temporal_figure()
        if output_dir:
            path = output_dir / f"temporal.{format}"
            fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
            saved_files.append(path)
        if show:
            plt.show()
        plt.close(fig)

        # 2. spatial analysis figure
        fig = self._create_spatial_figure()
        if output_dir:
            path = output_dir / f"spatial.{format}"
            fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
            saved_files.append(path)
        if show:
            plt.show()
        plt.close(fig)

        # 3. z-plane figure (only if multi-plane)
        if self.num_planes > 1:
            fig = self._create_zplane_figure()
            if output_dir:
                path = output_dir / f"zplanes.{format}"
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                saved_files.append(path)
            if show:
                plt.show()
            plt.close(fig)

        logger.info(f"Generated {len(saved_files)} figures")
        return saved_files

    def _create_summary_figure(self):
        """Create comprehensive summary overview."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

        fig.suptitle(
            f"Scan-Phase Analysis Summary\n"
            f"Frames: {self.num_frames} | Planes: {self.num_planes} | "
            f"Size: {self.frame_width}×{self.frame_height} | "
            f"Analysis time: {self.results.analysis_time_seconds:.1f}s",
            fontsize=14, fontweight='bold'
        )

        # Row 1: Temporal overview
        # 1.1 FFT time series (small)
        ax1 = fig.add_subplot(gs[0, 0:2])
        if len(self.results.offsets_by_frame_fft) > 0:
            ax1.plot(self.results.frame_indices, self.results.offsets_by_frame_fft,
                     'b-', linewidth=0.3, alpha=0.7)
            mean_val = np.mean(self.results.offsets_by_frame_fft)
            ax1.axhline(mean_val, color='r', linestyle='--', linewidth=1.5)
            ax1.set_ylabel('Offset (px)')
            ax1.set_title(f'Temporal (FFT): mean={mean_val:.4f}')
        ax1.set_xlabel('Frame')
        ax1.grid(True, alpha=0.3)

        # 1.2 Histogram comparison
        ax2 = fig.add_subplot(gs[0, 2])
        if len(self.results.offsets_by_frame_fft) > 0:
            ax2.hist(self.results.offsets_by_frame_fft, bins=50, alpha=0.7,
                     label='FFT', color='blue', density=True)
        if len(self.results.offsets_by_frame_int) > 0:
            ax2.hist(self.results.offsets_by_frame_int, bins=50, alpha=0.7,
                     label='Integer', color='green', density=True)
        ax2.set_xlabel('Offset (px)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution Comparison')
        ax2.legend()

        # 1.3 Statistics table
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.axis('off')
        stats_text = self._format_stats_table()
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax3.set_title('Summary Statistics')

        # Row 2: Window size and spatial
        # 2.1 Window size
        ax4 = fig.add_subplot(gs[1, 0:2])
        if self.results.offsets_by_window_fft:
            ws = sorted(self.results.offsets_by_window_fft.keys())
            offs_fft = [self.results.offsets_by_window_fft[w] for w in ws]
            offs_int = [self.results.offsets_by_window_int.get(w, np.nan) for w in ws]
            ax4.semilogx(ws, offs_fft, 'b-o', label='FFT', markersize=4)
            ax4.semilogx(ws, offs_int, 'g-s', label='Integer', markersize=4)
            ax4.set_xlabel('Window Size (frames)')
            ax4.set_ylabel('Offset (px)')
            ax4.set_title('Window Size Effect')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 2.2 Largest spatial grid
        ax5 = fig.add_subplot(gs[1, 2:4])
        if self.results.grid_offsets:
            largest_patch = max(self.results.grid_offsets.keys())
            offsets = self.results.grid_offsets[largest_patch]
            im = ax5.imshow(offsets, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
            plt.colorbar(im, ax=ax5, label='Offset (px)')
            ax5.set_title(f'Spatial Grid ({largest_patch}px patches)')
            ax5.set_xlabel('X patch')
            ax5.set_ylabel('Y patch')

        # Row 3: X/Y profiles
        # 3.1 X profile
        ax6 = fig.add_subplot(gs[2, 0:2])
        if len(self.results.offsets_vs_x) > 0:
            ax6.plot(self.results.x_positions, self.results.offsets_vs_x, 'b-o', markersize=4)
            ax6.axhline(np.mean(self.results.offsets_vs_x), color='r', linestyle='--')
            # Linear fit
            if len(self.results.x_positions) > 2:
                coeffs = np.polyfit(self.results.x_positions, self.results.offsets_vs_x, 1)
                x_fit = np.array([self.results.x_positions.min(), self.results.x_positions.max()])
                ax6.plot(x_fit, coeffs[0] * x_fit + coeffs[1], 'g--',
                         label=f'slope={coeffs[0]:.2e}')
                ax6.legend()
        ax6.set_xlabel('X Position (px)')
        ax6.set_ylabel('Offset (px)')
        ax6.set_title('Offset vs X Position')
        ax6.grid(True, alpha=0.3)

        # 3.2 Y profile
        ax7 = fig.add_subplot(gs[2, 2:4])
        if len(self.results.offsets_vs_y) > 0:
            ax7.plot(self.results.y_positions, self.results.offsets_vs_y, 'b-o', markersize=4)
            ax7.axhline(np.mean(self.results.offsets_vs_y), color='r', linestyle='--')
            if len(self.results.y_positions) > 2:
                coeffs = np.polyfit(self.results.y_positions, self.results.offsets_vs_y, 1)
                y_fit = np.array([self.results.y_positions.min(), self.results.y_positions.max()])
                ax7.plot(y_fit, coeffs[0] * y_fit + coeffs[1], 'g--',
                         label=f'slope={coeffs[0]:.2e}')
                ax7.legend()
        ax7.set_xlabel('Y Position (px)')
        ax7.set_ylabel('Offset (px)')
        ax7.set_title('Offset vs Y Position')
        ax7.grid(True, alpha=0.3)

        # Row 4: Additional analyses
        # 4.1 Z-planes or rolling mean
        ax8 = fig.add_subplot(gs[3, 0:2])
        if self.num_planes > 1 and len(self.results.offsets_by_plane_fft) > 0:
            ax8.bar(self.results.plane_indices - 0.15, self.results.offsets_by_plane_fft,
                    width=0.3, label='FFT', color='blue', alpha=0.7)
            ax8.bar(self.results.plane_indices + 0.15, self.results.offsets_by_plane_int,
                    width=0.3, label='Integer', color='green', alpha=0.7)
            ax8.set_xlabel('Z-Plane')
            ax8.set_ylabel('Offset (px)')
            ax8.set_title('Z-Plane Variation')
            ax8.legend()
        elif self.results.rolling_offsets_fft:
            rs = sorted(self.results.rolling_offsets_fft.keys())
            means = [self.results.rolling_offsets_fft[r][0] for r in rs]
            stds = [self.results.rolling_offsets_fft[r][1] for r in rs]
            ax8.errorbar(rs, means, yerr=stds, fmt='b-o', capsize=3, label='FFT')
            ax8.set_xlabel('Rolling Window Size')
            ax8.set_ylabel('Offset (px)')
            ax8.set_title('Rolling Mean Subtraction')
            ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 4.2 Spatial stats per patch size
        ax9 = fig.add_subplot(gs[3, 2:4])
        if self.results.spatial_stats:
            patch_sizes = sorted(self.results.spatial_stats.keys())
            means = [self.results.spatial_stats[ps].get('mean', 0) for ps in patch_sizes]
            stds = [self.results.spatial_stats[ps].get('std', 0) for ps in patch_sizes]
            x = np.arange(len(patch_sizes))
            ax9.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
            ax9.set_xticks(x)
            ax9.set_xticklabels([f'{ps}px' for ps in patch_sizes])
            ax9.set_xlabel('Patch Size')
            ax9.set_ylabel('Mean Offset (px)')
            ax9.set_title('Spatial Analysis by Patch Size')
            ax9.grid(True, alpha=0.3, axis='y')

        return fig

    def _format_stats_table(self) -> str:
        """Format statistics as text table."""
        lines = ["TEMPORAL STATISTICS", "=" * 30]

        if len(self.results.offsets_by_frame_fft) > 0:
            stats = self.results.compute_statistics(self.results.offsets_by_frame_fft)
            lines.append("FFT Method:")
            lines.append(f"  Mean:   {stats.get('mean', 0):+.4f}")
            lines.append(f"  Median: {stats.get('median', 0):+.4f}")
            lines.append(f"  Std:    {stats.get('std', 0):.4f}")
            lines.append(f"  Range:  [{stats.get('min', 0):.3f}, {stats.get('max', 0):.3f}]")
            lines.append(f"  IQR:    {stats.get('iqr', 0):.4f}")
            lines.append(f"  MAD:    {stats.get('mad', 0):.4f}")

        if len(self.results.offsets_by_frame_int) > 0:
            stats = self.results.compute_statistics(self.results.offsets_by_frame_int)
            lines.append("\nInteger Method:")
            lines.append(f"  Mean:   {stats.get('mean', 0):+.4f}")
            lines.append(f"  Median: {stats.get('median', 0):+.4f}")
            lines.append(f"  Std:    {stats.get('std', 0):.4f}")

        lines.append("\n" + "=" * 30)
        lines.append("SPATIAL STATISTICS")
        for patch_size, stats in sorted(self.results.spatial_stats.items()):
            lines.append(f"\n{patch_size}px patches ({stats.get('n_patches', 0)} valid):")
            lines.append(f"  Mean:   {stats.get('mean', 0):+.4f}")
            lines.append(f"  Median: {stats.get('median', 0):+.4f}")
            lines.append(f"  Std:    {stats.get('std', 0):.4f}")

        return "\n".join(lines)

    def _create_temporal_distribution_figure(self):
        """Create detailed temporal distribution figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Temporal Offset Distribution Analysis', fontsize=14, fontweight='bold')

        # 1. FFT histogram with detailed stats
        ax1 = axes[0, 0]
        if len(self.results.offsets_by_frame_fft) > 0:
            data = self.results.offsets_by_frame_fft
            ax1.hist(data, bins=100, edgecolor='black', alpha=0.7, color='blue', density=True)

            mean_val = np.mean(data)
            median_val = np.median(data)
            p5, p95 = np.percentile(data, [5, 95])

            ax1.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax1.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
            ax1.axvline(p5, color='gray', linestyle=':', linewidth=1.5, label=f'5%: {p5:.4f}')
            ax1.axvline(p95, color='gray', linestyle=':', linewidth=1.5, label=f'95%: {p95:.4f}')
            ax1.legend(fontsize=8)
        ax1.set_xlabel('Offset (pixels)')
        ax1.set_ylabel('Density')
        ax1.set_title('FFT Method Distribution')

        # 2. Integer histogram
        ax2 = axes[0, 1]
        if len(self.results.offsets_by_frame_int) > 0:
            data = self.results.offsets_by_frame_int
            ax2.hist(data, bins=100, edgecolor='black', alpha=0.7, color='green', density=True)

            mean_val = np.mean(data)
            median_val = np.median(data)
            ax2.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax2.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
            ax2.legend(fontsize=8)
        ax2.set_xlabel('Offset (pixels)')
        ax2.set_ylabel('Density')
        ax2.set_title('Integer Method Distribution')

        # 3. Box plot comparison
        ax3 = axes[0, 2]
        box_data = []
        labels = []
        if len(self.results.offsets_by_frame_fft) > 0:
            box_data.append(self.results.offsets_by_frame_fft)
            labels.append('FFT')
        if len(self.results.offsets_by_frame_int) > 0:
            box_data.append(self.results.offsets_by_frame_int)
            labels.append('Integer')
        if box_data:
            bp = ax3.boxplot(box_data, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
        ax3.set_ylabel('Offset (pixels)')
        ax3.set_title('Method Comparison (Box Plot)')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Cumulative distribution
        ax4 = axes[1, 0]
        if len(self.results.offsets_by_frame_fft) > 0:
            sorted_data = np.sort(self.results.offsets_by_frame_fft)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax4.plot(sorted_data, cdf, 'b-', linewidth=1.5, label='FFT')
        if len(self.results.offsets_by_frame_int) > 0:
            sorted_data = np.sort(self.results.offsets_by_frame_int)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax4.plot(sorted_data, cdf, 'g-', linewidth=1.5, label='Integer')
        ax4.set_xlabel('Offset (pixels)')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Q-Q plot (FFT vs Integer)
        ax5 = axes[1, 1]
        if len(self.results.offsets_by_frame_fft) > 0 and len(self.results.offsets_by_frame_int) > 0:
            n = min(len(self.results.offsets_by_frame_fft), len(self.results.offsets_by_frame_int))
            fft_quantiles = np.percentile(self.results.offsets_by_frame_fft, np.linspace(0, 100, n))
            int_quantiles = np.percentile(self.results.offsets_by_frame_int, np.linspace(0, 100, n))
            ax5.scatter(fft_quantiles, int_quantiles, alpha=0.5, s=10)
            lims = [min(fft_quantiles.min(), int_quantiles.min()),
                    max(fft_quantiles.max(), int_quantiles.max())]
            ax5.plot(lims, lims, 'r--', linewidth=1.5, label='y=x')
            ax5.legend()
        ax5.set_xlabel('FFT Quantiles')
        ax5.set_ylabel('Integer Quantiles')
        ax5.set_title('Q-Q Plot: FFT vs Integer')
        ax5.grid(True, alpha=0.3)

        # 6. Difference histogram
        ax6 = axes[1, 2]
        if len(self.results.offsets_by_frame_fft) > 0 and len(self.results.offsets_by_frame_int) > 0:
            diff = self.results.offsets_by_frame_fft - self.results.offsets_by_frame_int
            ax6.hist(diff, bins=50, edgecolor='black', alpha=0.7, color='purple')
            ax6.axvline(np.mean(diff), color='red', linestyle='--', linewidth=2,
                        label=f'Mean diff: {np.mean(diff):.4f}')
            ax6.legend()
        ax6.set_xlabel('FFT - Integer (pixels)')
        ax6.set_ylabel('Count')
        ax6.set_title('Method Difference Distribution')

        plt.tight_layout()
        return fig

    def _create_temporal_timeseries_figure(self):
        """Create temporal time series figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Temporal Time Series Analysis', fontsize=14, fontweight='bold')

        frames = self.results.frame_indices

        # 1. FFT time series
        ax1 = axes[0, 0]
        if len(self.results.offsets_by_frame_fft) > 0:
            ax1.plot(frames, self.results.offsets_by_frame_fft, 'b-', linewidth=0.3, alpha=0.7)
            mean_val = np.mean(self.results.offsets_by_frame_fft)
            std_val = np.std(self.results.offsets_by_frame_fft)
            ax1.axhline(mean_val, color='red', linestyle='--', linewidth=1.5)
            ax1.fill_between(frames, mean_val - std_val, mean_val + std_val,
                             alpha=0.2, color='red')
            ax1.set_ylabel('Offset (px)')
            ax1.set_title(f'FFT: mean={mean_val:.4f} ± {std_val:.4f}')
        ax1.set_xlabel('Frame')
        ax1.grid(True, alpha=0.3)

        # 2. Integer time series
        ax2 = axes[0, 1]
        if len(self.results.offsets_by_frame_int) > 0:
            ax2.plot(frames, self.results.offsets_by_frame_int, 'g-', linewidth=0.3, alpha=0.7)
            mean_val = np.mean(self.results.offsets_by_frame_int)
            std_val = np.std(self.results.offsets_by_frame_int)
            ax2.axhline(mean_val, color='red', linestyle='--', linewidth=1.5)
            ax2.fill_between(frames, mean_val - std_val, mean_val + std_val,
                             alpha=0.2, color='red')
            ax2.set_ylabel('Offset (px)')
            ax2.set_title(f'Integer: mean={mean_val:.4f} ± {std_val:.4f}')
        ax2.set_xlabel('Frame')
        ax2.grid(True, alpha=0.3)

        # 3. Rolling mean (FFT)
        ax3 = axes[1, 0]
        if len(self.results.offsets_by_frame_fft) > 0:
            data = self.results.offsets_by_frame_fft
            for window in [10, 50, 200]:
                if window < len(data) // 2:
                    rolling = np.convolve(data, np.ones(window)/window, mode='valid')
                    ax3.plot(frames[:len(rolling)], rolling, linewidth=1.5,
                             label=f'Window={window}', alpha=0.8)
            ax3.axhline(np.mean(data), color='red', linestyle='--', linewidth=1)
            ax3.legend(fontsize=8)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Offset (px)')
        ax3.set_title('Rolling Mean (FFT)')
        ax3.grid(True, alpha=0.3)

        # 4. Rolling std
        ax4 = axes[1, 1]
        if len(self.results.offsets_by_frame_fft) > 0:
            data = self.results.offsets_by_frame_fft
            window = min(50, len(data) // 5)
            if window > 1:
                # Compute rolling std
                rolling_std = []
                for i in range(len(data) - window + 1):
                    rolling_std.append(np.std(data[i:i+window]))
                ax4.plot(frames[:len(rolling_std)], rolling_std, 'b-', linewidth=1)
            ax4.axhline(np.std(data), color='red', linestyle='--',
                        label=f'Overall std: {np.std(data):.4f}')
            ax4.legend()
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Rolling Std (px)')
        ax4.set_title(f'Temporal Stability (window={window})')
        ax4.grid(True, alpha=0.3)

        # 5. Autocorrelation
        ax5 = axes[2, 0]
        if len(self.results.offsets_by_frame_fft) > 10:
            data = self.results.offsets_by_frame_fft - np.mean(self.results.offsets_by_frame_fft)
            lags = min(200, len(data) // 2)
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + lags]
            autocorr = autocorr / autocorr[0]
            ax5.plot(range(lags), autocorr, 'b-', linewidth=1)
            ax5.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax5.axhline(0.05, color='red', linestyle='--', alpha=0.5)
            ax5.axhline(-0.05, color='red', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Lag (frames)')
        ax5.set_ylabel('Autocorrelation')
        ax5.set_title('Temporal Autocorrelation (FFT)')
        ax5.grid(True, alpha=0.3)

        # 6. Power spectrum
        ax6 = axes[2, 1]
        if len(self.results.offsets_by_frame_fft) > 10:
            data = self.results.offsets_by_frame_fft - np.mean(self.results.offsets_by_frame_fft)
            fft_result = np.fft.rfft(data)
            power = np.abs(fft_result) ** 2
            freqs = np.fft.rfftfreq(len(data))
            ax6.semilogy(freqs[1:], power[1:], 'b-', linewidth=0.5, alpha=0.7)
        ax6.set_xlabel('Frequency (cycles/frame)')
        ax6.set_ylabel('Power')
        ax6.set_title('Power Spectrum')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_window_size_figure(self):
        """Create window size analysis figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Temporal Window Size Analysis', fontsize=14, fontweight='bold')

        ws = sorted(self.results.offsets_by_window_fft.keys())
        offs_fft = [self.results.offsets_by_window_fft[w] for w in ws]
        offs_int = [self.results.offsets_by_window_int.get(w, np.nan) for w in ws]

        # 1. Main comparison (log scale)
        ax1 = axes[0, 0]
        ax1.semilogx(ws, offs_fft, 'b-o', label='FFT', markersize=5, linewidth=1.5)
        ax1.semilogx(ws, offs_int, 'g-s', label='Integer', markersize=5, linewidth=1.5)
        ax1.set_xlabel('Window Size (frames)')
        ax1.set_ylabel('Offset (pixels)')
        ax1.set_title('Offset vs Window Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Variance by window size
        ax2 = axes[0, 1]
        if self.results.window_samples_fft:
            stds_fft = [np.std(self.results.window_samples_fft[w]) for w in ws]
            stds_int = [np.std(self.results.window_samples_int.get(w, [0])) for w in ws]
            ax2.semilogx(ws, stds_fft, 'b-o', label='FFT', markersize=5)
            ax2.semilogx(ws, stds_int, 'g-s', label='Integer', markersize=5)
            ax2.set_xlabel('Window Size (frames)')
            ax2.set_ylabel('Std of Offset Estimates')
            ax2.set_title('Estimation Variance vs Window Size')
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Convergence plot
        ax3 = axes[1, 0]
        if len(ws) > 1:
            # Convergence: how much does offset change as window increases
            diff_fft = np.abs(np.diff(offs_fft))
            ax3.semilogx(ws[1:], diff_fft, 'b-o', label='FFT', markersize=5)
            if any(~np.isnan(offs_int)):
                diff_int = np.abs(np.diff([o for o in offs_int if not np.isnan(o)]))
                valid_ws = [w for w, o in zip(ws[1:], offs_int[1:]) if not np.isnan(o)]
                if len(valid_ws) == len(diff_int):
                    ax3.semilogx(valid_ws, diff_int, 'g-s', label='Integer', markersize=5)
        ax3.set_xlabel('Window Size (frames)')
        ax3.set_ylabel('|Δ Offset|')
        ax3.set_title('Convergence (change between sizes)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. FFT vs Integer difference
        ax4 = axes[1, 1]
        diff = [f - i for f, i in zip(offs_fft, offs_int) if not np.isnan(i)]
        valid_ws = [w for w, i in zip(ws, offs_int) if not np.isnan(i)]
        if diff:
            ax4.semilogx(valid_ws, diff, 'm-d', markersize=5, linewidth=1.5)
            ax4.axhline(0, color='gray', linestyle='--')
            ax4.set_xlabel('Window Size (frames)')
            ax4.set_ylabel('FFT - Integer (pixels)')
            ax4.set_title('Method Difference vs Window Size')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_spatial_grid_figure(self, patch_size: int):
        """Create spatial grid figure for a specific patch size."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Spatial Grid Analysis ({patch_size}px patches)',
                     fontsize=14, fontweight='bold')

        offsets = self.results.grid_offsets[patch_size]
        intensities = self.results.grid_intensities[patch_size]
        valid_mask = self.results.grid_valid_mask[patch_size]
        stats = self.results.spatial_stats[patch_size]

        # 1. Offset heatmap
        ax1 = axes[0, 0]
        vmax = max(2, np.abs(np.nanmax(offsets)), np.abs(np.nanmin(offsets)))
        im1 = ax1.imshow(offsets, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        plt.colorbar(im1, ax=ax1, label='Offset (px)')
        ax1.set_title(f'Offset Grid (mean={stats.get("mean", 0):.4f})')
        ax1.set_xlabel('X patch')
        ax1.set_ylabel('Y patch')

        # 2. Intensity map
        ax2 = axes[0, 1]
        im2 = ax2.imshow(intensities, cmap='viridis', aspect='auto')
        plt.colorbar(im2, ax=ax2, label='Mean Intensity')
        ax2.set_title('Intensity Map')
        ax2.set_xlabel('X patch')
        ax2.set_ylabel('Y patch')

        # 3. Valid mask
        ax3 = axes[0, 2]
        im3 = ax3.imshow(valid_mask, cmap='gray', aspect='auto')
        ax3.set_title(f'Valid Patches ({stats.get("n_patches", 0)}/{stats.get("n_total", 0)})')
        ax3.set_xlabel('X patch')
        ax3.set_ylabel('Y patch')

        # 4. Histogram of offsets
        ax4 = axes[1, 0]
        valid_offsets = offsets[valid_mask]
        if len(valid_offsets) > 0:
            ax4.hist(valid_offsets, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax4.axvline(stats.get('mean', 0), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {stats.get("mean", 0):.4f}')
            ax4.axvline(stats.get('median', 0), color='orange', linestyle='--', linewidth=2,
                        label=f'Median: {stats.get("median", 0):.4f}')
            ax4.legend()
        ax4.set_xlabel('Offset (pixels)')
        ax4.set_ylabel('Count')
        ax4.set_title('Offset Distribution')

        # 5. Row means (Y profile)
        ax5 = axes[1, 1]
        row_means = np.nanmean(offsets, axis=1)
        row_stds = np.nanstd(offsets, axis=1)
        rows = np.arange(len(row_means))
        ax5.errorbar(rows, row_means, yerr=row_stds, fmt='b-o', capsize=3, markersize=4)
        ax5.axhline(stats.get('mean', 0), color='red', linestyle='--')
        ax5.set_xlabel('Row (Y)')
        ax5.set_ylabel('Mean Offset (px)')
        ax5.set_title('Row-wise Mean ± Std')
        ax5.grid(True, alpha=0.3)

        # 6. Column means (X profile)
        ax6 = axes[1, 2]
        col_means = np.nanmean(offsets, axis=0)
        col_stds = np.nanstd(offsets, axis=0)
        cols = np.arange(len(col_means))
        ax6.errorbar(cols, col_means, yerr=col_stds, fmt='b-o', capsize=3, markersize=4)
        ax6.axhline(stats.get('mean', 0), color='red', linestyle='--')
        ax6.set_xlabel('Column (X)')
        ax6.set_ylabel('Mean Offset (px)')
        ax6.set_title('Column-wise Mean ± Std')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_spatial_profiles_figure(self):
        """Create X/Y distribution profiles figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Spatial Distribution Profiles', fontsize=14, fontweight='bold')

        # 1. X profile
        ax1 = axes[0, 0]
        if len(self.results.offsets_vs_x) > 0:
            ax1.plot(self.results.x_positions, self.results.offsets_vs_x, 'b-o', markersize=6)
            mean_val = np.mean(self.results.offsets_vs_x)
            ax1.axhline(mean_val, color='red', linestyle='--', linewidth=1.5,
                        label=f'Mean: {mean_val:.4f}')

            # Linear fit
            if len(self.results.x_positions) > 2:
                coeffs = np.polyfit(self.results.x_positions, self.results.offsets_vs_x, 1)
                x_fit = self.results.x_positions
                ax1.plot(x_fit, coeffs[0] * x_fit + coeffs[1], 'g--', linewidth=2,
                         label=f'Slope: {coeffs[0]:.2e} px/px')
            ax1.legend()
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Offset (pixels)')
        ax1.set_title('Offset vs X Position')
        ax1.grid(True, alpha=0.3)

        # 2. Y profile
        ax2 = axes[0, 1]
        if len(self.results.offsets_vs_y) > 0:
            ax2.plot(self.results.y_positions, self.results.offsets_vs_y, 'b-o', markersize=6)
            mean_val = np.mean(self.results.offsets_vs_y)
            ax2.axhline(mean_val, color='red', linestyle='--', linewidth=1.5,
                        label=f'Mean: {mean_val:.4f}')

            if len(self.results.y_positions) > 2:
                coeffs = np.polyfit(self.results.y_positions, self.results.offsets_vs_y, 1)
                y_fit = self.results.y_positions
                ax2.plot(y_fit, coeffs[0] * y_fit + coeffs[1], 'g--', linewidth=2,
                         label=f'Slope: {coeffs[0]:.2e} px/px')
            ax2.legend()
        ax2.set_xlabel('Y Position (pixels)')
        ax2.set_ylabel('Offset (pixels)')
        ax2.set_title('Offset vs Y Position')
        ax2.grid(True, alpha=0.3)

        # 3. Combined scatter from all patch sizes
        ax3 = axes[1, 0]
        for patch_size, offsets in self.results.grid_offsets.items():
            valid = offsets[self.results.grid_valid_mask[patch_size]].flatten()
            if len(valid) > 0:
                ax3.scatter([patch_size] * len(valid), valid, alpha=0.3, s=10, label=f'{patch_size}px')
        ax3.set_xlabel('Patch Size (pixels)')
        ax3.set_ylabel('Offset (pixels)')
        ax3.set_title('Offset Distribution by Patch Size')
        ax3.grid(True, alpha=0.3)

        # 4. Statistics comparison
        ax4 = axes[1, 1]
        if self.results.spatial_stats:
            patch_sizes = sorted(self.results.spatial_stats.keys())
            metrics = ['mean', 'median', 'std', 'iqr']
            x = np.arange(len(patch_sizes))
            width = 0.2

            for i, metric in enumerate(metrics):
                values = [self.results.spatial_stats[ps].get(metric, 0) for ps in patch_sizes]
                ax4.bar(x + i * width, values, width, label=metric, alpha=0.7)

            ax4.set_xticks(x + width * 1.5)
            ax4.set_xticklabels([f'{ps}px' for ps in patch_sizes])
            ax4.set_xlabel('Patch Size')
            ax4.set_ylabel('Value (pixels)')
            ax4.set_title('Spatial Statistics by Patch Size')
            ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _create_method_comparison_figure(self):
        """Create FFT vs Integer method comparison figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('FFT vs Integer Method Comparison', fontsize=14, fontweight='bold')

        fft_data = self.results.offsets_by_frame_fft
        int_data = self.results.offsets_by_frame_int

        # 1. Scatter plot
        ax1 = axes[0, 0]
        if len(fft_data) > 0 and len(int_data) > 0:
            ax1.scatter(fft_data, int_data, alpha=0.3, s=5)
            lims = [min(fft_data.min(), int_data.min()),
                    max(fft_data.max(), int_data.max())]
            ax1.plot(lims, lims, 'r--', linewidth=1.5, label='y=x')
            ax1.set_xlabel('FFT Offset (pixels)')
            ax1.set_ylabel('Integer Offset (pixels)')
            ax1.legend()
        ax1.set_title('Method Correlation')
        ax1.grid(True, alpha=0.3)

        # 2. Difference histogram
        ax2 = axes[0, 1]
        if len(fft_data) > 0 and len(int_data) > 0:
            diff = fft_data - int_data
            ax2.hist(diff, bins=50, edgecolor='black', alpha=0.7, color='purple')
            ax2.axvline(np.mean(diff), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(diff):.4f}')
            ax2.axvline(np.median(diff), color='orange', linestyle='--', linewidth=2,
                        label=f'Median: {np.median(diff):.4f}')
            ax2.legend()
        ax2.set_xlabel('FFT - Integer (pixels)')
        ax2.set_ylabel('Count')
        ax2.set_title('Difference Distribution')

        # 3. Difference over time
        ax3 = axes[1, 0]
        if len(fft_data) > 0 and len(int_data) > 0:
            diff = fft_data - int_data
            ax3.plot(self.results.frame_indices, diff, 'purple', linewidth=0.3, alpha=0.7)
            ax3.axhline(np.mean(diff), color='red', linestyle='--')
            # Rolling mean
            window = min(50, len(diff) // 5)
            if window > 1:
                rolling = np.convolve(diff, np.ones(window)/window, mode='valid')
                ax3.plot(self.results.frame_indices[:len(rolling)], rolling,
                         'orange', linewidth=1.5, label=f'Rolling mean (w={window})')
                ax3.legend()
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('FFT - Integer (pixels)')
        ax3.set_title('Difference Over Time')
        ax3.grid(True, alpha=0.3)

        # 4. Window size comparison
        ax4 = axes[1, 1]
        if self.results.offsets_by_window_fft:
            ws = sorted(self.results.offsets_by_window_fft.keys())
            diff = [self.results.offsets_by_window_fft[w] - self.results.offsets_by_window_int.get(w, np.nan)
                    for w in ws]
            valid_ws = [w for w, d in zip(ws, diff) if not np.isnan(d)]
            valid_diff = [d for d in diff if not np.isnan(d)]
            ax4.semilogx(valid_ws, valid_diff, 'm-d', markersize=5)
            ax4.axhline(0, color='gray', linestyle='--')
        ax4.set_xlabel('Window Size (frames)')
        ax4.set_ylabel('FFT - Integer (pixels)')
        ax4.set_title('Method Difference vs Window Size')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_temporal_figure(self):
        """Create temporal analysis figure with time series, histogram, and window size effect."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f'Scan-Phase Temporal Analysis\n'
            f'{self.num_frames} frames, {self.num_planes} planes, {self.frame_width}x{self.frame_height} px',
            fontsize=12
        )

        fft_data = self.results.offsets_by_frame_fft
        int_data = self.results.offsets_by_frame_int

        # 1. time series
        ax1 = axes[0, 0]
        if len(fft_data) > 0:
            ax1.plot(self.results.frame_indices, fft_data, 'b-', linewidth=0.5, alpha=0.7, label='FFT')
            mean_fft = np.mean(fft_data)
            ax1.axhline(mean_fft, color='blue', linestyle='--', linewidth=1.5)
        if len(int_data) > 0:
            ax1.plot(self.results.frame_indices, int_data, 'g-', linewidth=0.5, alpha=0.5, label='Integer')
        ax1.set_xlabel('frame')
        ax1.set_ylabel('offset (px)')
        ax1.set_title('per-frame offset')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. histogram
        ax2 = axes[0, 1]
        if len(fft_data) > 0:
            ax2.hist(fft_data, bins=50, alpha=0.7, label='FFT', color='blue', density=True)
            stats = self.results.compute_statistics(fft_data)
            stats_text = (
                f"mean: {stats['mean']:+.4f}\n"
                f"median: {stats['median']:+.4f}\n"
                f"std: {stats['std']:.4f}\n"
                f"range: [{stats['min']:.3f}, {stats['max']:.3f}]"
            )
            ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if len(int_data) > 0:
            ax2.hist(int_data, bins=50, alpha=0.5, label='Integer', color='green', density=True)
        ax2.set_xlabel('offset (px)')
        ax2.set_ylabel('density')
        ax2.set_title('distribution')
        ax2.legend()

        # 3. window size effect
        ax3 = axes[1, 0]
        if self.results.offsets_by_window_fft:
            ws = sorted(self.results.offsets_by_window_fft.keys())
            offs_fft = [self.results.offsets_by_window_fft[w] for w in ws]
            offs_int = [self.results.offsets_by_window_int.get(w, np.nan) for w in ws]
            ax3.semilogx(ws, offs_fft, 'b-o', label='FFT', markersize=4)
            ax3.semilogx(ws, offs_int, 'g-s', label='Integer', markersize=4)
            ax3.set_xlabel('window size (frames)')
            ax3.set_ylabel('offset (px)')
            ax3.set_title('window size effect')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'no window data', transform=ax3.transAxes,
                     ha='center', va='center')
            ax3.set_axis_off()

        # 4. window size variance (if samples available)
        ax4 = axes[1, 1]
        if self.results.window_samples_fft:
            ws = sorted(self.results.window_samples_fft.keys())
            means = [np.mean(self.results.window_samples_fft[w]) for w in ws]
            stds = [np.std(self.results.window_samples_fft[w]) for w in ws]
            ax4.errorbar(ws, means, yerr=stds, fmt='b-o', capsize=3, markersize=4)
            ax4.set_xscale('log')
            ax4.set_xlabel('window size (frames)')
            ax4.set_ylabel('offset ± std (px)')
            ax4.set_title('estimation variance')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'no variance data', transform=ax4.transAxes,
                     ha='center', va='center')
            ax4.set_axis_off()

        plt.tight_layout()
        return fig

    def _create_spatial_figure(self):
        """Create spatial analysis figure with grid heatmap and X/Y profiles."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Scan-Phase Spatial Analysis', fontsize=12)

        # 1. spatial grid heatmap (use largest patch size available)
        ax1 = fig.add_subplot(gs[0, :])
        if self.results.grid_offsets:
            # pick largest patch size for cleaner visualization
            patch_size = max(self.results.grid_offsets.keys())
            offsets = self.results.grid_offsets[patch_size]
            valid_mask = self.results.grid_valid_mask.get(patch_size)

            # mask invalid patches
            if valid_mask is not None:
                offsets = np.where(valid_mask, offsets, np.nan)

            vmax = max(0.5, np.nanmax(np.abs(offsets)))
            im = ax1.imshow(offsets, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            plt.colorbar(im, ax=ax1, label='offset (px)')
            ax1.set_xlabel('x patch')
            ax1.set_ylabel('y patch')
            ax1.set_title(f'spatial grid ({patch_size}px patches)')

            # add stats annotation
            valid_offsets = offsets[~np.isnan(offsets)]
            if len(valid_offsets) > 0:
                stats_text = f"mean: {np.mean(valid_offsets):+.4f}, std: {np.std(valid_offsets):.4f}"
                ax1.set_xlabel(f'x patch    [{stats_text}]')
        else:
            ax1.text(0.5, 0.5, 'no spatial grid data', transform=ax1.transAxes,
                     ha='center', va='center')
            ax1.set_axis_off()

        # 2. offset vs X position
        ax2 = fig.add_subplot(gs[1, 0])
        if len(self.results.offsets_vs_x) > 0:
            ax2.errorbar(self.results.x_positions, self.results.offsets_vs_x,
                         yerr=self.results.offsets_vs_x_std, fmt='b-o', capsize=3, markersize=4)
            ax2.axhline(np.mean(self.results.offsets_vs_x), color='r', linestyle='--', alpha=0.7)
            ax2.set_xlabel('x position (px)')
            ax2.set_ylabel('offset (px)')
            ax2.set_title('offset vs x')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'no x profile data', transform=ax2.transAxes,
                     ha='center', va='center')
            ax2.set_axis_off()

        # 3. offset vs Y position
        ax3 = fig.add_subplot(gs[1, 1])
        if len(self.results.offsets_vs_y) > 0:
            ax3.errorbar(self.results.y_positions, self.results.offsets_vs_y,
                         yerr=self.results.offsets_vs_y_std, fmt='b-o', capsize=3, markersize=4)
            ax3.axhline(np.mean(self.results.offsets_vs_y), color='r', linestyle='--', alpha=0.7)
            ax3.set_xlabel('y position (px)')
            ax3.set_ylabel('offset (px)')
            ax3.set_title('offset vs y')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'no y profile data', transform=ax3.transAxes,
                     ha='center', va='center')
            ax3.set_axis_off()

        plt.tight_layout()
        return fig

    def _create_zplane_figure(self):
        """Create z-plane analysis figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'Z-Plane Analysis ({self.num_planes} planes)', fontsize=12)

        planes = self.results.plane_indices
        fft_offs = self.results.offsets_by_plane_fft
        int_offs = self.results.offsets_by_plane_int

        # 1. Bar comparison
        ax1 = axes[0]
        x = np.arange(len(planes))
        width = 0.35
        ax1.bar(x - width/2, fft_offs, width, label='FFT', color='blue', alpha=0.7)
        ax1.bar(x + width/2, int_offs, width, label='Integer', color='green', alpha=0.7)
        ax1.axhline(np.mean(fft_offs), color='blue', linestyle='--', alpha=0.5)
        ax1.axhline(np.mean(int_offs), color='green', linestyle='--', alpha=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(p) for p in planes])
        ax1.set_xlabel('Z-Plane')
        ax1.set_ylabel('Offset (pixels)')
        ax1.set_title('Offset by Plane')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Deviation from mean
        ax2 = axes[1]
        fft_dev = fft_offs - np.mean(fft_offs)
        int_dev = int_offs - np.mean(int_offs)
        ax2.bar(x - width/2, fft_dev, width, label='FFT', color='blue', alpha=0.7)
        ax2.bar(x + width/2, int_dev, width, label='Integer', color='green', alpha=0.7)
        ax2.axhline(0, color='red', linestyle='-', linewidth=1.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(p) for p in planes])
        ax2.set_xlabel('Z-Plane')
        ax2.set_ylabel('Deviation from Mean (pixels)')
        ax2.set_title('Per-Plane Deviation')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. FFT vs Integer scatter
        ax3 = axes[2]
        ax3.scatter(fft_offs, int_offs, s=100, c=planes, cmap='viridis')
        lims = [min(fft_offs.min(), int_offs.min()) - 0.1,
                max(fft_offs.max(), int_offs.max()) + 0.1]
        ax3.plot(lims, lims, 'r--', linewidth=1.5, label='y=x')
        for i, (f, i_val) in enumerate(zip(fft_offs, int_offs)):
            ax3.annotate(str(planes[i]), (f, i_val), fontsize=8)
        ax3.set_xlabel('FFT Offset (pixels)')
        ax3.set_ylabel('Integer Offset (pixels)')
        ax3.set_title('Method Comparison by Plane')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_rolling_mean_figure(self):
        """Create rolling mean subtraction analysis figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Rolling Mean Subtraction Analysis', fontsize=14, fontweight='bold')

        rs = sorted(self.results.rolling_offsets_fft.keys())
        fft_means = [self.results.rolling_offsets_fft[r][0] for r in rs]
        fft_stds = [self.results.rolling_offsets_fft[r][1] for r in rs]
        fft_medians = [self.results.rolling_offsets_fft[r][2] for r in rs]

        int_means = [self.results.rolling_offsets_int.get(r, (0, 0, 0))[0] for r in rs]
        int_stds = [self.results.rolling_offsets_int.get(r, (0, 0, 0))[1] for r in rs]

        # 1. Mean offset vs rolling window
        ax1 = axes[0]
        ax1.errorbar(rs, fft_means, yerr=fft_stds, fmt='b-o', capsize=3, label='FFT', markersize=5)
        ax1.errorbar(rs, int_means, yerr=int_stds, fmt='g-s', capsize=3, label='Integer', markersize=5)
        ax1.set_xlabel('Rolling Window Size (frames)')
        ax1.set_ylabel('Offset (pixels)')
        ax1.set_title('Mean Offset (±std)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Std vs rolling window
        ax2 = axes[1]
        ax2.plot(rs, fft_stds, 'b-o', label='FFT', markersize=5)
        ax2.plot(rs, int_stds, 'g-s', label='Integer', markersize=5)
        ax2.set_xlabel('Rolling Window Size (frames)')
        ax2.set_ylabel('Std of Offset')
        ax2.set_title('Estimation Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Mean vs Median
        ax3 = axes[2]
        ax3.plot(rs, fft_means, 'b-o', label='FFT Mean', markersize=5)
        ax3.plot(rs, fft_medians, 'b--s', label='FFT Median', markersize=5, alpha=0.7)
        ax3.set_xlabel('Rolling Window Size (frames)')
        ax3.set_ylabel('Offset (pixels)')
        ax3.set_title('Mean vs Median (FFT)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_results(self, output_path: Union[str, Path]) -> Path:
        """Save results to NPZ file."""
        output_path = Path(output_path)

        # Collect all results
        data = {
            # Temporal
            'offsets_by_frame_fft': self.results.offsets_by_frame_fft,
            'offsets_by_frame_int': self.results.offsets_by_frame_int,
            'frame_indices': self.results.frame_indices,

            # Window sizes
            'window_sizes': self.results.window_sizes,

            # Spatial (save each patch size)
            'x_positions': self.results.x_positions,
            'y_positions': self.results.y_positions,
            'offsets_vs_x': self.results.offsets_vs_x,
            'offsets_vs_y': self.results.offsets_vs_y,

            # Z-planes
            'plane_indices': self.results.plane_indices,
            'offsets_by_plane_fft': self.results.offsets_by_plane_fft,
            'offsets_by_plane_int': self.results.offsets_by_plane_int,

            # Metadata
            'num_frames': self.results.num_frames,
            'num_planes': self.results.num_planes,
            'frame_shape': self.results.frame_shape,
            'analysis_time_seconds': self.results.analysis_time_seconds,
        }

        # Add window size dicts
        for ws in self.results.window_sizes:
            data[f'window_{ws}_fft'] = self.results.offsets_by_window_fft.get(ws, np.nan)
            data[f'window_{ws}_int'] = self.results.offsets_by_window_int.get(ws, np.nan)

        # Add spatial grids
        for patch_size in self.results.grid_offsets.keys():
            data[f'grid_{patch_size}_offsets'] = self.results.grid_offsets[patch_size]
            data[f'grid_{patch_size}_intensities'] = self.results.grid_intensities[patch_size]
            data[f'grid_{patch_size}_valid'] = self.results.grid_valid_mask[patch_size]

        np.savez_compressed(output_path, **data)
        logger.info(f"Results saved to: {output_path}")
        return output_path


def analyze_scanphase(
    data,
    output_dir: Optional[Union[str, Path]] = None,
    fft_method: str = "1d",
    image_format: str = "png",
    save_data: bool = True,
    show_plots: bool = False,
) -> ScanPhaseResults:
    """
    Run comprehensive scan-phase analysis.

    Parameters
    ----------
    data : array-like
        Input imaging data (2D, 3D, or 4D).
    output_dir : str or Path, optional
        Directory to save outputs. If None, no files are saved.
    fft_method : str
        FFT method: '1d' (fast) or '2d' (more accurate).
    image_format : str
        Format for saved images: 'png', 'pdf', 'svg', 'tiff'.
    save_data : bool
        Save numerical results as .npz file.
    show_plots : bool
        Display plots interactively.

    Returns
    -------
    ScanPhaseResults
        Comprehensive analysis results.
    """
    analyzer = ScanPhaseAnalysis(data)
    results = analyzer.run_full_analysis(fft_method=fft_method)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots
        analyzer.generate_plots(
            output_dir=output_dir,
            format=image_format,
            show=show_plots,
        )

        # Save data
        if save_data:
            analyzer.save_results(output_dir / "scanphase_results.npz")

    elif show_plots:
        analyzer.generate_plots(show=True)

    return results


def run_scanphase_analysis(
    data_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Optional[ScanPhaseResults]:
    """
    Run scan-phase analysis with file dialog support.

    Main entry point for CLI. Shows file dialog if no path provided.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to input data. If None, shows file dialog.
    output_dir : str or Path, optional
        Output directory. If None, creates alongside input.
    **kwargs
        Additional arguments passed to analyze_scanphase().

    Returns
    -------
    ScanPhaseResults or None
        Results if analysis completed, None if cancelled.
    """
    from mbo_utilities import imread

    # Handle file selection
    if data_path is None:
        try:
            from mbo_utilities.file_io import select_file
            data_path = select_file(
                title="Select data for scan-phase analysis",
                filetypes=[
                    ("TIFF files", "*.tif *.tiff"),
                    ("All files", "*.*"),
                ],
            )
            if data_path is None:
                logger.info("File selection cancelled")
                return None
        except ImportError:
            logger.error("No file path provided and file dialog not available")
            return None

    data_path = Path(data_path)

    # Set output directory
    if output_dir is None:
        output_dir = data_path.parent / f"{data_path.stem}_scanphase_analysis"

    logger.info(f"Loading data from: {data_path}")
    data = imread(data_path)

    logger.info(f"Running scan-phase analysis...")
    return analyze_scanphase(data, output_dir=output_dir, **kwargs)
