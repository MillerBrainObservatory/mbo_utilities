"""
Comprehensive scan-phase analysis for bidirectional scanning correction.

This module provides tools to analyze how the optimal phase offset varies as a function of:
- Time (across frames)
- Z-plane (across depth)
- X and Y spatial position
- Windowing (temporal averaging)
- FFT vs non-FFT methods

The analysis helps users understand their data characteristics and choose
appropriate phase correction parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np

from mbo_utilities import log
from mbo_utilities.phasecorr import _phase_corr_2d

logger = log.get("analysis.scanphase")

# Window sizes for temporal averaging analysis
DEFAULT_WINDOW_SIZES = [1, 10, 20, 100, 200, 500, 1000, 2000]


@dataclass
class ScanPhaseResults:
    """Results from a scan-phase analysis run.

    Attributes
    ----------
    offsets_by_frame : np.ndarray
        Phase offset computed for each individual frame. Shape: (num_frames,)
    offsets_by_plane : np.ndarray
        Mean phase offset per z-plane. Shape: (num_planes,)
    offsets_by_window : dict[int, float]
        Mean offset computed using different temporal window sizes.
        Keys are window sizes, values are computed offsets.
    spatial_offsets : np.ndarray
        2D array of offsets computed in spatial windows. Shape: (ny_windows, nx_windows)
    spatial_x_positions : np.ndarray
        X positions of spatial window centers.
    spatial_y_positions : np.ndarray
        Y positions of spatial window centers.
    method : str
        Method used ('fft' or 'integer').
    fft_method : str
        FFT sub-method if applicable ('1d' or '2d').
    """

    # Core results
    offsets_by_frame: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_by_plane: np.ndarray = field(default_factory=lambda: np.array([]))
    offsets_by_window: dict = field(default_factory=dict)
    offsets_by_window_fft: dict = field(default_factory=dict)
    offsets_by_window_nofft: dict = field(default_factory=dict)

    # Spatial analysis
    spatial_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    spatial_x_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    spatial_y_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    spatial_intensities: np.ndarray = field(default_factory=lambda: np.array([]))

    # Time series (for temporal drift)
    frame_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    plane_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    # Analysis parameters
    method: str = "fft"
    fft_method: str = "1d"
    window_sizes_analyzed: list = field(default_factory=list)

    # Metadata
    num_frames: int = 0
    num_planes: int = 0
    frame_shape: tuple = ()

    def get_summary_stats(self) -> dict:
        """Get summary statistics for the analysis."""
        stats = {
            "num_frames": self.num_frames,
            "num_planes": self.num_planes,
            "frame_shape": self.frame_shape,
            "method": self.method,
        }

        if len(self.offsets_by_frame) > 0:
            stats.update({
                "offset_mean": float(np.nanmean(self.offsets_by_frame)),
                "offset_std": float(np.nanstd(self.offsets_by_frame)),
                "offset_min": float(np.nanmin(self.offsets_by_frame)),
                "offset_max": float(np.nanmax(self.offsets_by_frame)),
                "offset_range": float(np.nanmax(self.offsets_by_frame) - np.nanmin(self.offsets_by_frame)),
            })

        if len(self.spatial_offsets) > 0:
            flat = self.spatial_offsets.flatten()
            valid = flat[~np.isnan(flat)]
            if len(valid) > 0:
                stats.update({
                    "spatial_offset_mean": float(np.mean(valid)),
                    "spatial_offset_std": float(np.std(valid)),
                    "spatial_offset_range": float(np.max(valid) - np.min(valid)),
                })

        return stats


class ScanPhaseAnalysis:
    """
    Comprehensive scan-phase analyzer for bidirectional scanning data.

    This class provides methods to analyze how the optimal phase correction
    offset varies across different dimensions of the data:

    1. **Temporal variation**: How does the offset change over time (frames)?
    2. **Z-plane variation**: Does the offset differ between planes?
    3. **Spatial variation**: Is the offset uniform across X and Y?
    4. **Window size effects**: How does temporal averaging affect the estimate?
    5. **Method comparison**: FFT vs integer-only methods

    Parameters
    ----------
    data : array-like
        Input data array. Can be 2D (YX), 3D (TYX or ZYX), or 4D (TZYX).
    num_planes : int, optional
        Number of z-planes for 3D/4D data. If None, inferred from data.

    Examples
    --------
    >>> from mbo_utilities import imread
    >>> from mbo_utilities.analysis import ScanPhaseAnalysis
    >>> data = imread("path/to/data.tiff")
    >>> analyzer = ScanPhaseAnalysis(data)
    >>> results = analyzer.run_full_analysis()
    >>> analyzer.save_plots("output_dir/")
    """

    def __init__(
        self,
        data,
        num_planes: Optional[int] = None,
    ):
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
            frame = np.asarray(self.data)
        elif self.ndim == 3:
            frame = np.asarray(self.data[frame_idx])
        elif self.ndim == 4:
            frame = np.asarray(self.data[frame_idx, plane_idx])
        else:
            raise ValueError(f"Unsupported data dimensionality: {self.ndim}")

        # Ensure we return a proper 2D array (squeeze any singleton dimensions)
        frame = np.squeeze(frame)
        if frame.ndim != 2:
            raise ValueError(
                f"Expected 2D frame after squeeze, got shape {frame.shape}. "
                f"Original data shape: {self.shape}, frame_idx={frame_idx}, plane_idx={plane_idx}"
            )
        return frame

    def _get_frames(self, frame_indices: list, plane_idx: int = 0) -> np.ndarray:
        """Get multiple frames and stack them."""
        frames = []
        for idx in frame_indices:
            frames.append(self._get_frame(idx, plane_idx))
        return np.stack(frames, axis=0)

    def analyze_temporal_variation(
        self,
        use_fft: bool = True,
        fft_method: str = "1d",
        upsample: int = 4,
        border: int = 4,
        max_offset: int = 10,
        use_gradient: bool = True,
        sample_frames: Optional[int] = None,
        plane_idx: int = 0,
        progress_callback=None,
    ) -> np.ndarray:
        """
        Analyze how phase offset varies over time (frames).

        Parameters
        ----------
        use_fft : bool
            Use FFT-based phase correlation (subpixel precision).
        fft_method : str
            FFT method: '1d' (fast) or '2d' (more accurate).
        upsample : int
            Upsampling factor for subpixel precision.
        border : int
            Border pixels to exclude from analysis.
        max_offset : int
            Maximum offset to search for.
        use_gradient : bool
            Use gradient enhancement for better edge detection.
        sample_frames : int, optional
            If set, sample this many frames evenly spaced. None = all frames.
        plane_idx : int
            Z-plane to analyze.
        progress_callback : callable, optional
            Function(current, total) for progress updates.

        Returns
        -------
        np.ndarray
            Array of offsets, one per analyzed frame.
        """
        if sample_frames and sample_frames < self.num_frames:
            frame_indices = np.linspace(0, self.num_frames - 1, sample_frames, dtype=int)
        else:
            frame_indices = np.arange(self.num_frames)

        offsets = []
        total = len(frame_indices)

        for i, fidx in enumerate(frame_indices):
            if progress_callback:
                progress_callback(i, total)

            frame = self._get_frame(fidx, plane_idx)
            offset = _phase_corr_2d(
                frame,
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=use_fft,
                fft_method=fft_method,
                use_gradient=use_gradient,
            )
            offsets.append(offset)

        self.results.offsets_by_frame = np.array(offsets)
        self.results.frame_indices = frame_indices
        self.results.method = "fft" if use_fft else "integer"
        self.results.fft_method = fft_method

        logger.info(
            f"Temporal analysis complete: {len(offsets)} frames, "
            f"mean offset = {np.mean(offsets):.3f} +/- {np.std(offsets):.3f}"
        )

        return self.results.offsets_by_frame

    def analyze_z_plane_variation(
        self,
        use_fft: bool = True,
        fft_method: str = "1d",
        upsample: int = 4,
        border: int = 4,
        max_offset: int = 10,
        use_gradient: bool = True,
        frames_per_plane: int = 10,
        progress_callback=None,
    ) -> np.ndarray:
        """
        Analyze how phase offset varies across z-planes.

        Averages multiple frames per plane for more robust estimates.

        Parameters
        ----------
        frames_per_plane : int
            Number of frames to average per plane.

        Returns
        -------
        np.ndarray
            Array of offsets, one per z-plane.
        """
        offsets_by_plane = []

        for plane_idx in range(self.num_planes):
            if progress_callback:
                progress_callback(plane_idx, self.num_planes)

            # Sample frames for this plane
            sample_indices = np.linspace(
                0, self.num_frames - 1,
                min(frames_per_plane, self.num_frames),
                dtype=int
            )

            # Get frames and compute mean
            frames = self._get_frames(sample_indices.tolist(), plane_idx)
            mean_frame = np.mean(frames, axis=0)

            offset = _phase_corr_2d(
                mean_frame,
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=use_fft,
                fft_method=fft_method,
                use_gradient=use_gradient,
            )
            offsets_by_plane.append(offset)

        self.results.offsets_by_plane = np.array(offsets_by_plane)
        self.results.plane_indices = np.arange(self.num_planes)

        logger.info(
            f"Z-plane analysis complete: {self.num_planes} planes, "
            f"range = {np.min(offsets_by_plane):.3f} to {np.max(offsets_by_plane):.3f}"
        )

        return self.results.offsets_by_plane

    def analyze_spatial_variation(
        self,
        window_size: int = 64,
        stride: int = 32,
        use_fft: bool = True,
        fft_method: str = "1d",
        upsample: int = 10,
        border: int = 0,
        max_offset: int = 10,
        min_intensity: float = 50.0,
        frame_idx: int = 0,
        plane_idx: int = 0,
        num_frames_to_average: int = 10,
        progress_callback=None,
    ) -> tuple:
        """
        Analyze spatial variation in phase offset across the image.

        This helps detect if the phase shift is non-uniform across the
        field of view, which could indicate scanner calibration issues.

        Parameters
        ----------
        window_size : int
            Size of analysis windows (pixels).
        stride : int
            Step size between windows.
        min_intensity : float
            Minimum window intensity to include in analysis.
        num_frames_to_average : int
            Number of frames to average for more robust spatial analysis.

        Returns
        -------
        tuple
            (offsets_2d, x_positions, y_positions, intensities)
        """
        # Get averaged frame for analysis
        sample_indices = np.linspace(
            0, self.num_frames - 1,
            min(num_frames_to_average, self.num_frames),
            dtype=int
        )
        frames = self._get_frames(sample_indices.tolist(), plane_idx)
        frame = np.mean(frames, axis=0)

        # Split into even and odd rows
        pre = frame[::2]
        post = frame[1::2]
        m = min(pre.shape[0], post.shape[0])
        even_rows = pre[:m]
        odd_rows = post[:m]

        h, w = even_rows.shape

        x_positions = []
        y_positions = []
        shifts = []
        intensities = []

        # Calculate total windows for progress
        n_y_windows = max(1, (h - window_size) // stride + 1)
        n_x_windows = max(1, (w - window_size) // stride + 1)
        total_windows = n_y_windows * n_x_windows
        current_window = 0

        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                if progress_callback:
                    progress_callback(current_window, total_windows)
                current_window += 1

                window_a = even_rows[y:y + window_size, x:x + window_size]
                window_b = odd_rows[y:y + window_size, x:x + window_size]

                # Check minimum intensity
                mean_intensity = (window_a.mean() + window_b.mean()) / 2
                if mean_intensity < min_intensity:
                    continue

                # Compute shift for this window
                try:
                    # Create a combined frame for _phase_corr_2d
                    combined = np.zeros((window_size * 2, window_size))
                    combined[::2] = window_a
                    combined[1::2] = window_b

                    shift = _phase_corr_2d(
                        combined,
                        upsample=upsample,
                        border=border,
                        max_offset=max_offset,
                        use_fft=use_fft,
                        fft_method=fft_method,
                        use_gradient=True,
                    )

                    x_positions.append(x + window_size // 2)
                    y_positions.append(y + window_size // 2)
                    shifts.append(float(shift))
                    intensities.append(mean_intensity)

                except Exception as e:
                    logger.debug(f"Window analysis failed at ({x}, {y}): {e}")
                    continue

        # Store results
        self.results.spatial_x_positions = np.array(x_positions)
        self.results.spatial_y_positions = np.array(y_positions)
        self.results.spatial_offsets = np.array(shifts)
        self.results.spatial_intensities = np.array(intensities)

        logger.info(
            f"Spatial analysis complete: {len(shifts)} windows analyzed, "
            f"range = {np.min(shifts):.3f} to {np.max(shifts):.3f}"
        )

        return (
            self.results.spatial_offsets,
            self.results.spatial_x_positions,
            self.results.spatial_y_positions,
            self.results.spatial_intensities,
        )

    def analyze_window_size_effects(
        self,
        window_sizes: Optional[list] = None,
        use_fft: bool = True,
        fft_method: str = "1d",
        upsample: int = 4,
        border: int = 4,
        max_offset: int = 10,
        use_gradient: bool = True,
        plane_idx: int = 0,
        compare_methods: bool = True,
        progress_callback=None,
    ) -> dict:
        """
        Analyze how temporal window size affects the offset estimate.

        Computes offsets using mean images from different numbers of frames:
        1, 10, 20, 100, 200, 500, 1000, 2000 frames.

        Parameters
        ----------
        window_sizes : list, optional
            Window sizes to test. Default: [1, 10, 20, 100, 200, 500, 1000, 2000]
        compare_methods : bool
            If True, compare both FFT and non-FFT methods.

        Returns
        -------
        dict
            {window_size: offset} for each tested window size.
        """
        if window_sizes is None:
            window_sizes = [ws for ws in DEFAULT_WINDOW_SIZES if ws <= self.num_frames]

        self.results.window_sizes_analyzed = window_sizes
        offsets_fft = {}
        offsets_nofft = {}

        total_steps = len(window_sizes) * (2 if compare_methods else 1)
        current_step = 0

        for ws in window_sizes:
            if progress_callback:
                progress_callback(current_step, total_steps)

            # Sample frames evenly across the recording
            sample_indices = np.linspace(0, self.num_frames - 1, ws, dtype=int)
            frames = self._get_frames(sample_indices.tolist(), plane_idx)
            mean_frame = np.mean(frames, axis=0)

            # FFT method
            offset_fft = _phase_corr_2d(
                mean_frame,
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=True,
                fft_method=fft_method,
                use_gradient=use_gradient,
            )
            offsets_fft[ws] = float(offset_fft)
            current_step += 1

            if compare_methods:
                if progress_callback:
                    progress_callback(current_step, total_steps)

                # Non-FFT method
                offset_nofft = _phase_corr_2d(
                    mean_frame,
                    upsample=upsample,
                    border=border,
                    max_offset=max_offset,
                    use_fft=False,
                    fft_method=fft_method,
                    use_gradient=use_gradient,
                )
                offsets_nofft[ws] = float(offset_nofft)
                current_step += 1

        self.results.offsets_by_window_fft = offsets_fft
        self.results.offsets_by_window_nofft = offsets_nofft
        self.results.offsets_by_window = offsets_fft  # Primary result

        logger.info(
            f"Window size analysis complete: {len(window_sizes)} sizes tested"
        )

        return offsets_fft

    def run_full_analysis(
        self,
        use_fft: bool = True,
        fft_method: str = "1d",
        upsample: int = 4,
        border: int = 4,
        max_offset: int = 10,
        use_gradient: bool = True,
        sample_frames: Optional[int] = 100,
        spatial_window_size: int = 64,
        spatial_stride: int = 32,
        progress_callback=None,
    ) -> ScanPhaseResults:
        """
        Run comprehensive scan-phase analysis.

        Performs all analysis types:
        - Temporal variation (per-frame offsets)
        - Z-plane variation
        - Spatial variation
        - Window size effects (both FFT and non-FFT)

        Parameters
        ----------
        sample_frames : int, optional
            Number of frames to sample for temporal analysis. None = all.
        spatial_window_size : int
            Window size for spatial analysis.
        spatial_stride : int
            Stride for spatial analysis.
        progress_callback : callable, optional
            Function(step_name, current, total) for progress updates.

        Returns
        -------
        ScanPhaseResults
            Complete analysis results.
        """
        def report_progress(step_name, current, total):
            if progress_callback:
                progress_callback(step_name, current, total)

        # 1. Temporal variation
        logger.info("Starting temporal variation analysis...")
        self.analyze_temporal_variation(
            use_fft=use_fft,
            fft_method=fft_method,
            upsample=upsample,
            border=border,
            max_offset=max_offset,
            use_gradient=use_gradient,
            sample_frames=sample_frames,
            progress_callback=lambda c, t: report_progress("Temporal", c, t),
        )

        # 2. Z-plane variation (if multi-plane)
        if self.num_planes > 1:
            logger.info("Starting z-plane variation analysis...")
            self.analyze_z_plane_variation(
                use_fft=use_fft,
                fft_method=fft_method,
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_gradient=use_gradient,
                progress_callback=lambda c, t: report_progress("Z-planes", c, t),
            )

        # 3. Spatial variation
        logger.info("Starting spatial variation analysis...")
        self.analyze_spatial_variation(
            window_size=spatial_window_size,
            stride=spatial_stride,
            use_fft=use_fft,
            fft_method=fft_method,
            upsample=10,  # Higher for spatial
            border=0,
            max_offset=max_offset,
            progress_callback=lambda c, t: report_progress("Spatial", c, t),
        )

        # 4. Window size effects
        logger.info("Starting window size analysis...")
        self.analyze_window_size_effects(
            use_fft=use_fft,
            fft_method=fft_method,
            upsample=upsample,
            border=border,
            max_offset=max_offset,
            use_gradient=use_gradient,
            compare_methods=True,
            progress_callback=lambda c, t: report_progress("Window sizes", c, t),
        )

        return self.results

    def generate_plots(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        format: str = "png",
        dpi: int = 150,
        show: bool = False,
    ) -> list:
        """
        Generate comprehensive visualization plots.

        Creates multiple figures:
        1. Temporal variation plot (offset vs frame)
        2. Z-plane variation plot (offset vs plane)
        3. Spatial variation map
        4. Window size comparison
        5. FFT vs non-FFT comparison
        6. Summary statistics figure

        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save plots. If None, plots are not saved.
        format : str
            Image format: 'png', 'pdf', 'svg', etc.
        dpi : int
            Resolution for raster formats.
        show : bool
            Whether to display plots interactively.

        Returns
        -------
        list
            List of paths to saved figures.
        """
        import matplotlib.pyplot as plt

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # 1. Comprehensive summary figure
        fig = self._create_summary_figure()
        if output_dir:
            path = output_dir / f"scanphase_summary.{format}"
            fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
            saved_files.append(path)
            logger.info(f"Saved: {path}")
        if show:
            plt.show()
        plt.close(fig)

        # 2. Temporal variation
        if len(self.results.offsets_by_frame) > 0:
            fig = self._create_temporal_figure()
            if output_dir:
                path = output_dir / f"scanphase_temporal.{format}"
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                saved_files.append(path)
            if show:
                plt.show()
            plt.close(fig)

        # 3. Z-plane variation
        if len(self.results.offsets_by_plane) > 1:
            fig = self._create_zplane_figure()
            if output_dir:
                path = output_dir / f"scanphase_zplanes.{format}"
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                saved_files.append(path)
            if show:
                plt.show()
            plt.close(fig)

        # 4. Spatial variation
        if len(self.results.spatial_offsets) > 0:
            fig = self._create_spatial_figure()
            if output_dir:
                path = output_dir / f"scanphase_spatial.{format}"
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                saved_files.append(path)
            if show:
                plt.show()
            plt.close(fig)

        # 5. Window size analysis
        if self.results.offsets_by_window:
            fig = self._create_window_figure()
            if output_dir:
                path = output_dir / f"scanphase_windows.{format}"
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                saved_files.append(path)
            if show:
                plt.show()
            plt.close(fig)

        return saved_files

    def _create_summary_figure(self):
        """Create comprehensive summary figure."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        stats = self.results.get_summary_stats()

        # Title with key info
        fig.suptitle(
            f"Scan-Phase Analysis Summary\n"
            f"Frames: {stats['num_frames']}, Planes: {stats['num_planes']}, "
            f"Size: {stats['frame_shape'][1]}x{stats['frame_shape'][0]}",
            fontsize=14, fontweight='bold'
        )

        # 1. Temporal variation (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.results.offsets_by_frame) > 0:
            ax1.plot(self.results.frame_indices, self.results.offsets_by_frame,
                     'b-', linewidth=0.5, alpha=0.7)
            ax1.axhline(stats.get('offset_mean', 0), color='r', linestyle='--',
                        label=f"Mean: {stats.get('offset_mean', 0):.3f}")
            ax1.fill_between(
                self.results.frame_indices,
                stats.get('offset_mean', 0) - stats.get('offset_std', 0),
                stats.get('offset_mean', 0) + stats.get('offset_std', 0),
                alpha=0.2, color='red'
            )
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Offset (pixels)')
            ax1.set_title('Temporal Variation')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No temporal data', ha='center', va='center')
            ax1.set_title('Temporal Variation')

        # 2. Z-plane variation (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.results.offsets_by_plane) > 1:
            ax2.bar(self.results.plane_indices, self.results.offsets_by_plane,
                    color='steelblue', edgecolor='black', alpha=0.7)
            ax2.axhline(np.mean(self.results.offsets_by_plane), color='r',
                        linestyle='--', label='Mean')
            ax2.set_xlabel('Z-Plane')
            ax2.set_ylabel('Offset (pixels)')
            ax2.set_title('Z-Plane Variation')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Single plane', ha='center', va='center')
            ax2.set_title('Z-Plane Variation')

        # 3. Histogram (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if len(self.results.offsets_by_frame) > 0:
            ax3.hist(self.results.offsets_by_frame, bins=50, edgecolor='black',
                     alpha=0.7, color='steelblue')
            ax3.axvline(stats.get('offset_mean', 0), color='r', linestyle='--',
                        linewidth=2, label=f"Mean: {stats.get('offset_mean', 0):.3f}")
            ax3.axvline(np.median(self.results.offsets_by_frame), color='orange',
                        linestyle='--', linewidth=2,
                        label=f"Median: {np.median(self.results.offsets_by_frame):.3f}")
            ax3.set_xlabel('Offset (pixels)')
            ax3.set_ylabel('Count')
            ax3.set_title('Offset Distribution')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax3.set_title('Offset Distribution')

        # 4. Spatial variation - scatter (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        if len(self.results.spatial_offsets) > 0:
            scatter = ax4.scatter(
                self.results.spatial_x_positions,
                self.results.spatial_y_positions,
                c=self.results.spatial_offsets,
                s=30, cmap='RdBu_r', vmin=-2, vmax=2,
                edgecolors='gray', linewidth=0.5, alpha=0.8
            )
            plt.colorbar(scatter, ax=ax4, label='Offset (px)')
            ax4.set_xlabel('X Position')
            ax4.set_ylabel('Y Position')
            ax4.set_title('Spatial Offset Map')
        else:
            ax4.text(0.5, 0.5, 'No spatial data', ha='center', va='center')
            ax4.set_title('Spatial Offset Map')

        # 5. Spatial variation - X trend (middle center)
        ax5 = fig.add_subplot(gs[1, 1])
        if len(self.results.spatial_offsets) > 0:
            ax5.scatter(self.results.spatial_x_positions, self.results.spatial_offsets,
                        c=self.results.spatial_intensities, s=20, alpha=0.6, cmap='viridis')
            ax5.axhline(0, color='red', linestyle='--', alpha=0.5)

            # Linear fit
            if len(self.results.spatial_x_positions) > 2:
                coeffs = np.polyfit(self.results.spatial_x_positions,
                                    self.results.spatial_offsets, 1)
                x_trend = np.array([0, self.frame_width])
                y_trend = coeffs[0] * x_trend + coeffs[1]
                ax5.plot(x_trend, y_trend, 'r-', linewidth=2, alpha=0.7,
                         label=f'slope={coeffs[0]:.5f}')
                ax5.legend()

            ax5.set_xlabel('X Position')
            ax5.set_ylabel('Offset (pixels)')
            ax5.set_title('Offset vs X Position')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No spatial data', ha='center', va='center')
            ax5.set_title('Offset vs X Position')

        # 6. Window size comparison (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        if self.results.offsets_by_window_fft:
            ws = list(self.results.offsets_by_window_fft.keys())
            offs_fft = list(self.results.offsets_by_window_fft.values())
            ax6.semilogx(ws, offs_fft, 'b-o', label='FFT', markersize=6)

            if self.results.offsets_by_window_nofft:
                offs_nofft = list(self.results.offsets_by_window_nofft.values())
                ax6.semilogx(ws, offs_nofft, 'g-s', label='Integer', markersize=6)

            ax6.set_xlabel('Window Size (frames)')
            ax6.set_ylabel('Offset (pixels)')
            ax6.set_title('Window Size Effect')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No window data', ha='center', va='center')
            ax6.set_title('Window Size Effect')

        # 7. Statistics text box (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.axis('off')
        stats_text = self._format_stats_text(stats)
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # 8. Recommendations (bottom middle + right)
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        recommendations = self._generate_recommendations(stats)
        ax8.text(0.05, 0.95, recommendations, transform=ax8.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        return fig

    def _create_temporal_figure(self):
        """Create detailed temporal variation figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Temporal Phase Offset Analysis', fontsize=14, fontweight='bold')

        offsets = self.results.offsets_by_frame
        frames = self.results.frame_indices

        # 1. Time series
        ax1 = axes[0, 0]
        ax1.plot(frames, offsets, 'b-', linewidth=0.5, alpha=0.7)
        ax1.axhline(np.mean(offsets), color='r', linestyle='--',
                    label=f'Mean: {np.mean(offsets):.3f}')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Offset (pixels)')
        ax1.set_title('Offset vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Rolling average
        ax2 = axes[0, 1]
        window = min(50, len(offsets) // 5)
        if window > 1:
            rolling_mean = np.convolve(offsets, np.ones(window)/window, mode='valid')
            ax2.plot(frames[:len(rolling_mean)], rolling_mean, 'b-', linewidth=1.5)
        ax2.axhline(np.mean(offsets), color='r', linestyle='--')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Offset (pixels)')
        ax2.set_title(f'Rolling Average (window={window})')
        ax2.grid(True, alpha=0.3)

        # 3. Histogram
        ax3 = axes[1, 0]
        ax3.hist(offsets, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.axvline(np.mean(offsets), color='r', linestyle='--', linewidth=2)
        ax3.axvline(np.median(offsets), color='orange', linestyle='--', linewidth=2)
        ax3.set_xlabel('Offset (pixels)')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution')

        # 4. Autocorrelation
        ax4 = axes[1, 1]
        if len(offsets) > 10:
            lags = min(100, len(offsets) // 2)
            autocorr = np.correlate(offsets - np.mean(offsets),
                                    offsets - np.mean(offsets), mode='full')
            autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + lags]
            autocorr = autocorr / autocorr[0]
            ax4.plot(range(lags), autocorr)
            ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax4.set_xlabel('Lag (frames)')
            ax4.set_ylabel('Autocorrelation')
            ax4.set_title('Temporal Autocorrelation')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_zplane_figure(self):
        """Create z-plane variation figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Z-Plane Phase Offset Analysis', fontsize=14, fontweight='bold')

        offsets = self.results.offsets_by_plane
        planes = self.results.plane_indices

        # 1. Bar chart
        ax1 = axes[0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(planes)))
        ax1.bar(planes, offsets, color=colors, edgecolor='black', alpha=0.8)
        ax1.axhline(np.mean(offsets), color='r', linestyle='--',
                    label=f'Mean: {np.mean(offsets):.3f}')
        ax1.set_xlabel('Z-Plane')
        ax1.set_ylabel('Offset (pixels)')
        ax1.set_title('Offset by Plane')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Deviation from mean
        ax2 = axes[1]
        mean_offset = np.mean(offsets)
        deviations = offsets - mean_offset
        ax2.bar(planes, deviations, color='steelblue', edgecolor='black', alpha=0.8)
        ax2.axhline(0, color='r', linestyle='-', linewidth=2)
        ax2.set_xlabel('Z-Plane')
        ax2.set_ylabel('Deviation from Mean (pixels)')
        ax2.set_title('Per-Plane Deviation')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _create_spatial_figure(self):
        """Create spatial variation figure."""
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Spatial Phase Offset Analysis', fontsize=14, fontweight='bold')

        x = self.results.spatial_x_positions
        y = self.results.spatial_y_positions
        shifts = self.results.spatial_offsets
        intensities = self.results.spatial_intensities

        # 1. Scatter with offset coloring
        ax1 = axes[0, 0]
        scatter = ax1.scatter(x, y, c=shifts, s=30, cmap='RdBu_r',
                              vmin=-2, vmax=2, edgecolors='gray', linewidth=0.3)
        plt.colorbar(scatter, ax=ax1, label='Offset (px)')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Offset Map')

        # 2. Interpolated map
        ax2 = axes[0, 1]
        if len(x) > 4:
            grid_x, grid_y = np.meshgrid(
                np.linspace(x.min(), x.max(), 100),
                np.linspace(y.min(), y.max(), 100)
            )
            try:
                grid_shifts = griddata((x, y), shifts, (grid_x, grid_y), method='cubic')
                im = ax2.imshow(grid_shifts, extent=[x.min(), x.max(), y.min(), y.max()],
                                origin='lower', cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
                plt.colorbar(im, ax=ax2, label='Offset (px)')
            except Exception:
                ax2.text(0.5, 0.5, 'Interpolation failed', ha='center', va='center',
                         transform=ax2.transAxes)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Interpolated Map')

        # 3. Offset vs X
        ax3 = axes[0, 2]
        ax3.scatter(x, shifts, c=intensities, s=20, alpha=0.6, cmap='viridis')
        ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
        if len(x) > 2:
            coeffs = np.polyfit(x, shifts, 1)
            x_fit = np.array([x.min(), x.max()])
            ax3.plot(x_fit, coeffs[0] * x_fit + coeffs[1], 'r-', linewidth=2,
                     label=f'slope={coeffs[0]:.5f}')
            ax3.legend()
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Offset (pixels)')
        ax3.set_title('Offset vs X')
        ax3.grid(True, alpha=0.3)

        # 4. Offset vs Y
        ax4 = axes[1, 0]
        ax4.scatter(y, shifts, c=intensities, s=20, alpha=0.6, cmap='viridis')
        ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Y Position')
        ax4.set_ylabel('Offset (pixels)')
        ax4.set_title('Offset vs Y')
        ax4.grid(True, alpha=0.3)

        # 5. Histogram
        ax5 = axes[1, 1]
        ax5.hist(shifts, bins=50, edgecolor='black', alpha=0.7)
        ax5.axvline(np.mean(shifts), color='r', linestyle='--',
                    label=f'Mean: {np.mean(shifts):.3f}')
        ax5.axvline(np.median(shifts), color='orange', linestyle='--',
                    label=f'Median: {np.median(shifts):.3f}')
        ax5.set_xlabel('Offset (pixels)')
        ax5.set_ylabel('Count')
        ax5.set_title('Distribution')
        ax5.legend()

        # 6. Binned X analysis
        ax6 = axes[1, 2]
        n_bins = 10
        x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
        bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        bin_means = []
        bin_stds = []
        for i in range(n_bins):
            mask = (x >= x_bins[i]) & (x < x_bins[i + 1])
            if mask.sum() > 0:
                bin_means.append(shifts[mask].mean())
                bin_stds.append(shifts[mask].std())
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        ax6.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5)
        ax6.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax6.set_xlabel('X Position (binned)')
        ax6.set_ylabel('Mean Offset (pixels)')
        ax6.set_title('Binned X Analysis')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_window_figure(self):
        """Create window size comparison figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Window Size Effect on Phase Estimation', fontsize=14, fontweight='bold')

        ws_fft = list(self.results.offsets_by_window_fft.keys())
        offs_fft = list(self.results.offsets_by_window_fft.values())

        # 1. Comparison plot
        ax1 = axes[0]
        ax1.semilogx(ws_fft, offs_fft, 'b-o', label='FFT (subpixel)', markersize=8, linewidth=2)

        if self.results.offsets_by_window_nofft:
            ws_nofft = list(self.results.offsets_by_window_nofft.keys())
            offs_nofft = list(self.results.offsets_by_window_nofft.values())
            ax1.semilogx(ws_nofft, offs_nofft, 'g-s', label='Integer', markersize=8, linewidth=2)

        ax1.set_xlabel('Window Size (frames)')
        ax1.set_ylabel('Estimated Offset (pixels)')
        ax1.set_title('FFT vs Integer Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')

        # 2. Convergence analysis
        ax2 = axes[1]
        if len(offs_fft) > 1:
            final_value = offs_fft[-1]
            deviations = [abs(v - final_value) for v in offs_fft]
            ax2.semilogx(ws_fft, deviations, 'b-o', markersize=8, linewidth=2)
            ax2.set_xlabel('Window Size (frames)')
            ax2.set_ylabel('Deviation from Final Estimate (pixels)')
            ax2.set_title('Convergence to Final Estimate')
            ax2.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        return fig

    def _format_stats_text(self, stats: dict) -> str:
        """Format statistics as text."""
        lines = [
            "ANALYSIS STATISTICS",
            "=" * 30,
            f"Frames analyzed: {stats['num_frames']}",
            f"Z-planes: {stats['num_planes']}",
            f"Frame size: {stats['frame_shape'][1]} x {stats['frame_shape'][0]}",
            "",
        ]

        if 'offset_mean' in stats:
            lines.extend([
                "Offset Statistics:",
                f"  Mean:  {stats['offset_mean']:.4f} px",
                f"  Std:   {stats['offset_std']:.4f} px",
                f"  Min:   {stats['offset_min']:.4f} px",
                f"  Max:   {stats['offset_max']:.4f} px",
                f"  Range: {stats['offset_range']:.4f} px",
            ])

        if 'spatial_offset_mean' in stats:
            lines.extend([
                "",
                "Spatial Variation:",
                f"  Mean:  {stats['spatial_offset_mean']:.4f} px",
                f"  Std:   {stats['spatial_offset_std']:.4f} px",
                f"  Range: {stats['spatial_offset_range']:.4f} px",
            ])

        return "\n".join(lines)

    def _generate_recommendations(self, stats: dict) -> str:
        """Generate recommendations based on analysis results."""
        lines = [
            "RECOMMENDATIONS",
            "=" * 60,
            "",
        ]

        # Check temporal stability
        if 'offset_std' in stats:
            if stats['offset_std'] < 0.1:
                lines.append("TEMPORAL: Offset is very stable. Use method='mean' for best results.")
            elif stats['offset_std'] < 0.3:
                lines.append("TEMPORAL: Offset is moderately stable. method='mean' should work well.")
            else:
                lines.append("TEMPORAL: High offset variation detected!")
                lines.append("   Consider using method='frame' for per-frame correction.")

        # Check spatial uniformity
        if 'spatial_offset_std' in stats:
            if stats['spatial_offset_std'] < 0.2:
                lines.append("SPATIAL: Offset is uniform across the field of view.")
            elif stats['spatial_offset_std'] < 0.5:
                lines.append("SPATIAL: Moderate spatial variation. Global correction should suffice.")
            else:
                lines.append("SPATIAL: High spatial variation detected!")
                lines.append("   This may indicate scanner calibration issues.")
                lines.append("   Consider scanner recalibration or spatially-varying correction.")

        # Window size recommendation
        if self.results.offsets_by_window_fft:
            ws = list(self.results.offsets_by_window_fft.keys())
            offs = list(self.results.offsets_by_window_fft.values())
            if len(offs) > 3:
                # Check convergence
                final = offs[-1]
                converged_at = None
                for i, (w, o) in enumerate(zip(ws, offs)):
                    if abs(o - final) < 0.05:
                        converged_at = w
                        break
                if converged_at:
                    lines.append(f"WINDOW: Estimate converges at ~{converged_at} frames.")
                    lines.append(f"   Using method='mean' with {converged_at}+ frames is optimal.")

        # FFT vs integer
        if self.results.offsets_by_window_fft and self.results.offsets_by_window_nofft:
            fft_final = list(self.results.offsets_by_window_fft.values())[-1]
            nofft_final = list(self.results.offsets_by_window_nofft.values())[-1]
            diff = abs(fft_final - nofft_final)
            if diff < 0.1:
                lines.append("FFT vs INTEGER: Results are similar. Integer method is faster.")
            else:
                lines.append(f"FFT vs INTEGER: {diff:.2f} px difference. FFT provides subpixel precision.")
                if abs(fft_final - round(fft_final)) > 0.2:
                    lines.append("   Subpixel correction recommended: use_fft=True")

        # Parameter suggestions
        lines.extend([
            "",
            "SUGGESTED PARAMETERS:",
        ])

        if 'offset_mean' in stats:
            suggested_offset = stats['offset_mean']
            lines.append(f"  fix_phase=True")
            lines.append(f"  use_fft={'True' if abs(suggested_offset - round(suggested_offset)) > 0.2 else 'False'}")
            lines.append(f"  phasecorr_method='mean'")

        return "\n".join(lines)

    def save_results(
        self,
        output_path: Union[str, Path],
        format: Literal["npz", "json"] = "npz"
    ) -> Path:
        """
        Save analysis results to file.

        Parameters
        ----------
        output_path : str or Path
            Output file path.
        format : str
            Output format: 'npz' (NumPy) or 'json'.

        Returns
        -------
        Path
            Path to saved file.
        """
        output_path = Path(output_path)

        if format == "npz":
            np.savez(
                output_path,
                offsets_by_frame=self.results.offsets_by_frame,
                offsets_by_plane=self.results.offsets_by_plane,
                spatial_offsets=self.results.spatial_offsets,
                spatial_x_positions=self.results.spatial_x_positions,
                spatial_y_positions=self.results.spatial_y_positions,
                frame_indices=self.results.frame_indices,
                plane_indices=self.results.plane_indices,
                window_sizes=np.array(self.results.window_sizes_analyzed),
                window_offsets_fft=np.array(list(self.results.offsets_by_window_fft.values())),
                window_offsets_nofft=np.array(list(self.results.offsets_by_window_nofft.values())),
            )
        elif format == "json":
            import json
            data = {
                "stats": self.results.get_summary_stats(),
                "offsets_by_frame": self.results.offsets_by_frame.tolist(),
                "offsets_by_plane": self.results.offsets_by_plane.tolist(),
                "offsets_by_window_fft": self.results.offsets_by_window_fft,
                "offsets_by_window_nofft": self.results.offsets_by_window_nofft,
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")
        return output_path


def analyze_scanphase(
    data,
    output_dir: Optional[Union[str, Path]] = None,
    use_fft: bool = True,
    fft_method: str = "1d",
    sample_frames: Optional[int] = 100,
    image_format: str = "png",
    save_data: bool = True,
    show_plots: bool = False,
) -> ScanPhaseResults:
    """
    Convenience function for complete scan-phase analysis.

    Parameters
    ----------
    data : array-like
        Input imaging data (2D, 3D, or 4D).
    output_dir : str or Path, optional
        Directory to save outputs. If None, no files are saved.
    use_fft : bool
        Use FFT-based phase correlation.
    fft_method : str
        FFT method: '1d' (fast) or '2d' (more accurate).
    sample_frames : int, optional
        Number of frames to sample for temporal analysis.
    image_format : str
        Format for saved images: 'png', 'pdf', 'svg', 'tiff'.
    save_data : bool
        Save numerical results as .npz file.
    show_plots : bool
        Display plots interactively.

    Returns
    -------
    ScanPhaseResults
        Analysis results.

    Examples
    --------
    >>> from mbo_utilities import imread
    >>> from mbo_utilities.analysis import analyze_scanphase
    >>> data = imread("path/to/data.tiff")
    >>> results = analyze_scanphase(data, output_dir="./analysis/")
    >>> print(results.get_summary_stats())
    """
    analyzer = ScanPhaseAnalysis(data)
    results = analyzer.run_full_analysis(
        use_fft=use_fft,
        fft_method=fft_method,
        sample_frames=sample_frames,
    )

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

    This is the main entry point for the CLI command. If no data_path
    is provided, shows a file selection dialog with detailed information
    about what the analysis does and how to interpret results.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to input data. If None, shows file dialog.
    output_dir : str or Path, optional
        Output directory. If None, saves alongside input.
    **kwargs
        Additional arguments passed to analyze_scanphase().

    Returns
    -------
    ScanPhaseResults or None
        Results if analysis completed, None if cancelled.
    """
    from mbo_utilities import imread

    # Handle file selection with custom dialog
    if data_path is None:
        from mbo_utilities.graphics.run_gui import _setup_qt_backend
        from mbo_utilities.analysis._scanphase_dialog import select_scanphase_file

        _setup_qt_backend()
        data_path = select_scanphase_file()
        if not data_path:
            print("No file selected, exiting.")
            return None

    data_path = Path(data_path) if isinstance(data_path, str) else data_path

    # Handle list of files (multi-select)
    if isinstance(data_path, list):
        data_path = data_path[0]

    # Determine output directory
    if output_dir is None:
        if isinstance(data_path, (str, Path)):
            data_path_obj = Path(data_path)
            output_dir = data_path_obj.parent / f"{data_path_obj.stem}_scanphase_analysis"
        else:
            output_dir = Path.cwd() / "scanphase_analysis"

    print(f"Loading data from: {data_path}")
    data = imread(data_path)
    print(f"Data shape: {data.shape}, dtype: {data.dtype}")

    print(f"Output directory: {output_dir}")
    print("Running scan-phase analysis...")

    results = analyze_scanphase(
        data,
        output_dir=output_dir,
        **kwargs
    )

    stats = results.get_summary_stats()
    print("\n" + "=" * 60)
    print("SCAN-PHASE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Mean offset: {stats.get('offset_mean', 'N/A'):.4f} pixels")
    print(f"Offset std:  {stats.get('offset_std', 'N/A'):.4f} pixels")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)

    return results
