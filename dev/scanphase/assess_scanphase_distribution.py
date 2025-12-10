"""
Comprehensive Scan-Phase Distribution Assessment Script.

Analyzes bi-directional scan phase shift distribution across multiple dimensions:
- Vertical (Y-axis): How phase shift varies across rows
- Horizontal (X-axis): How phase shift varies across columns
- Temporal: How phase shift varies over frames
- Mean-window analysis: Phase shift computed from increasing window sizes
- Rolling mean subtraction: Phase shift after removing slow drift
- Both FFT (subpixel) and non-FFT (integer) methods

Usage:
    python assess_scanphase_distribution.py <path_to_data>

Or import and use the ScanPhaseAssessment class directly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import warnings

# Import mbo_utilities for data loading and phase correlation
import mbo_utilities as mbo
from mbo_utilities.phasecorr import (
    _phase_corr_2d,
    _phase_corr_1d_fft,
    MBO_WINDOW_METHODS,
)
from skimage.registration import phase_cross_correlation


@dataclass
class PhaseAssessmentResult:
    """Results from a single phase assessment."""
    shift: float
    method: str
    window_type: str
    window_size: int = 1
    region: Optional[Tuple[int, int, int, int]] = None  # (y_start, y_end, x_start, x_end)
    frame_indices: Optional[Tuple[int, int]] = None  # (start, end)
    confidence: float = 1.0  # Correlation confidence if available


@dataclass
class ScanPhaseAssessment:
    """
    scan-phase distribution assessment.

    Analyzes phase shift distribution across spatial and temporal dimensions
    using both FFT and non-FFT methods.
    """
    data: np.ndarray  # (T, Z, Y, X) or (T, Y, X) or (Y, X)

    # Analysis parameters
    upsample: int = 10
    border: int = 4
    max_offset: int = 10

    # Results storage
    results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize data shape and validate."""
        self.original_shape = self.data.shape

        # Normalize to 3D (T, Y, X)
        if self.data.ndim == 2:
            self.data = self.data[np.newaxis, ...]
            self._is_single_frame = True
        elif self.data.ndim == 4:
            # (T, Z, Y, X) -> flatten to (T*Z, Y, X)
            t, z, h, w = self.data.shape
            self.data = self.data.reshape(t * z, h, w)
            self._is_single_frame = False
        elif self.data.ndim == 3:
            self._is_single_frame = False
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D data, got {self.data.ndim}D")

        self.n_frames, self.height, self.width = self.data.shape
        print(f"Data shape: {self.original_shape} -> {self.data.shape} (T, Y, X)")
        print(f"Frames: {self.n_frames}, Height: {self.height}, Width: {self.width}")

    def _compute_shift_2d(
        self,
        frame: np.ndarray,
        use_fft: bool = True,
        fft_method: str = "2d"
    ) -> Tuple[float, float]:
        """
        Compute phase shift for a single 2D frame.

        Returns (shift, confidence).
        """
        shift = _phase_corr_2d(
            frame,
            upsample=self.upsample,
            border=self.border,
            max_offset=self.max_offset,
            use_fft=use_fft,
            fft_method=fft_method,
        )
        return shift, 1.0

    def _compute_shift_region(
        self,
        frame: np.ndarray,
        y_start: int, y_end: int,
        x_start: int, x_end: int,
        use_fft: bool = True,
        fft_method: str = "2d"
    ) -> Tuple[float, float]:
        """Compute phase shift for a specific region."""
        region = frame[y_start:y_end, x_start:x_end]
        if region.shape[0] < 10 or region.shape[1] < 10:
            return np.nan, 0.0
        return self._compute_shift_2d(region, use_fft, fft_method)

    def analyze_vertical_distribution(
        self,
        n_bands: int = 8,
        use_fft: bool = True,
        fft_method: str = "2d",
        frame_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how phase shift varies vertically (across Y-axis).

        Divides the image into horizontal bands and computes shift for each.

        Parameters
        ----------
        n_bands : int
            Number of horizontal bands to analyze
        use_fft : bool
            Use FFT-based correlation
        fft_method : str
            '1d' or '2d' FFT method
        frame_idx : int, optional
            Specific frame to analyze (default: mean of all frames)

        Returns
        -------
        dict with 'y_centers', 'shifts', 'confidences'
        """
        method_key = f"fft_{fft_method}" if use_fft else "integer"

        if frame_idx is not None:
            frame = self.data[frame_idx]
        else:
            frame = np.mean(self.data, axis=0)

        band_height = self.height // n_bands
        y_centers = []
        shifts = []
        confidences = []

        for i in range(n_bands):
            y_start = i * band_height
            y_end = min((i + 1) * band_height, self.height)
            y_center = (y_start + y_end) // 2

            shift, conf = self._compute_shift_region(
                frame, y_start, y_end, 0, self.width, use_fft, fft_method
            )

            y_centers.append(y_center)
            shifts.append(shift)
            confidences.append(conf)

        result = {
            'y_centers': np.array(y_centers),
            'shifts': np.array(shifts),
            'confidences': np.array(confidences),
            'method': method_key,
        }

        self.results[f'vertical_{method_key}'] = result
        return result

    def analyze_horizontal_distribution(
        self,
        n_bands: int = 8,
        use_fft: bool = True,
        fft_method: str = "2d",
        frame_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how phase shift varies horizontally (across X-axis).

        Divides the image into vertical bands and computes shift for each.
        """
        method_key = f"fft_{fft_method}" if use_fft else "integer"

        if frame_idx is not None:
            frame = self.data[frame_idx]
        else:
            frame = np.mean(self.data, axis=0)

        band_width = self.width // n_bands
        x_centers = []
        shifts = []
        confidences = []

        for i in range(n_bands):
            x_start = i * band_width
            x_end = min((i + 1) * band_width, self.width)
            x_center = (x_start + x_end) // 2

            shift, conf = self._compute_shift_region(
                frame, 0, self.height, x_start, x_end, use_fft, fft_method
            )

            x_centers.append(x_center)
            shifts.append(shift)
            confidences.append(conf)

        result = {
            'x_centers': np.array(x_centers),
            'shifts': np.array(shifts),
            'confidences': np.array(confidences),
            'method': method_key,
        }

        self.results[f'horizontal_{method_key}'] = result
        return result

    def analyze_temporal_variation(
        self,
        use_fft: bool = True,
        fft_method: str = "2d",
        sample_every: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how phase shift varies over time (across frames).

        Parameters
        ----------
        sample_every : int
            Compute shift every N frames (for speed)
        """
        method_key = f"fft_{fft_method}" if use_fft else "integer"

        if self._is_single_frame:
            print("Warning: Single frame data, temporal analysis not meaningful")
            return {'frame_indices': np.array([0]), 'shifts': np.array([np.nan])}

        frame_indices = list(range(0, self.n_frames, sample_every))
        shifts = []
        confidences = []

        for idx in frame_indices:
            shift, conf = self._compute_shift_2d(
                self.data[idx], use_fft, fft_method
            )
            shifts.append(shift)
            confidences.append(conf)

        result = {
            'frame_indices': np.array(frame_indices),
            'shifts': np.array(shifts),
            'confidences': np.array(confidences),
            'method': method_key,
        }

        self.results[f'temporal_{method_key}'] = result
        return result

    def analyze_mean_windows(
        self,
        window_sizes: List[int] = None,
        use_fft: bool = True,
        fft_method: str = "2d",
    ) -> Dict[str, Any]:
        """
        Analyze phase shift computed from mean of increasing window sizes.

        This tests how stable the phase estimate is as more frames are averaged.

        Parameters
        ----------
        window_sizes : list of int
            Window sizes to test (default: [1, 2, 4, 8, 16, 32, 64, ...])
        """
        method_key = f"fft_{fft_method}" if use_fft else "integer"

        if window_sizes is None:
            # Generate window sizes up to n_frames
            window_sizes = []
            size = 1
            while size <= self.n_frames:
                window_sizes.append(size)
                size *= 2
            if self.n_frames not in window_sizes:
                window_sizes.append(self.n_frames)

        shifts = []
        stds = []

        for ws in window_sizes:
            if ws > self.n_frames:
                shifts.append(np.nan)
                stds.append(np.nan)
                continue

            # Compute shift for multiple non-overlapping windows
            n_windows = max(1, self.n_frames // ws)
            window_shifts = []

            for i in range(n_windows):
                start_idx = i * ws
                end_idx = min(start_idx + ws, self.n_frames)
                window_mean = np.mean(self.data[start_idx:end_idx], axis=0)

                shift, _ = self._compute_shift_2d(window_mean, use_fft, fft_method)
                window_shifts.append(shift)

            shifts.append(np.mean(window_shifts))
            stds.append(np.std(window_shifts) if len(window_shifts) > 1 else 0)

        result = {
            'window_sizes': np.array(window_sizes),
            'shifts': np.array(shifts),
            'stds': np.array(stds),
            'method': method_key,
        }

        self.results[f'mean_window_{method_key}'] = result
        return result

    def analyze_rolling_mean_subtracted(
        self,
        window_sizes: List[int] = None,
        use_fft: bool = True,
        fft_method: str = "2d",
    ) -> Dict[str, Any]:
        """
        Analyze phase shift after subtracting rolling mean (high-pass filter).

        This emphasizes fast temporal changes and removes slow drift,
        which can improve phase correlation for data with baseline drift.

        Parameters
        ----------
        window_sizes : list of int
            Rolling window sizes to test
        """
        method_key = f"fft_{fft_method}" if use_fft else "integer"

        if self._is_single_frame:
            print("Warning: Single frame, rolling mean subtraction not applicable")
            return {'window_sizes': np.array([1]), 'shifts': np.array([np.nan])}

        if window_sizes is None:
            window_sizes = [3, 5, 10, 20, 50, 100]
            window_sizes = [ws for ws in window_sizes if ws < self.n_frames]

        shifts = []
        stds = []

        for ws in window_sizes:
            if ws >= self.n_frames:
                shifts.append(np.nan)
                stds.append(np.nan)
                continue

            # Compute rolling mean
            kernel = np.ones(ws) / ws

            # Apply rolling mean subtraction frame by frame
            frame_shifts = []
            for idx in range(ws, self.n_frames, max(1, (self.n_frames - ws) // 10)):
                # Compute rolling mean for this position
                window_start = idx - ws
                rolling_mean = np.mean(self.data[window_start:idx], axis=0)

                # Subtract from current frame
                subtracted = self.data[idx].astype(float) - rolling_mean

                # Shift to positive values for correlation
                subtracted = subtracted - subtracted.min() + 1

                shift, _ = self._compute_shift_2d(subtracted, use_fft, fft_method)
                frame_shifts.append(shift)

            shifts.append(np.mean(frame_shifts))
            stds.append(np.std(frame_shifts) if len(frame_shifts) > 1 else 0)

        result = {
            'window_sizes': np.array(window_sizes),
            'shifts': np.array(shifts),
            'stds': np.array(stds),
            'method': method_key,
        }

        self.results[f'rolling_sub_{method_key}'] = result
        return result

    def analyze_method_comparison(self) -> Dict[str, float]:
        """
        Compare FFT vs non-FFT methods on mean image.
        """
        mean_frame = np.mean(self.data, axis=0)

        results = {}

        # Non-FFT (integer)
        shift_int, _ = self._compute_shift_2d(mean_frame, use_fft=False)
        results['integer'] = shift_int

        # FFT 1D
        shift_fft1d, _ = self._compute_shift_2d(mean_frame, use_fft=True, fft_method="1d")
        results['fft_1d'] = shift_fft1d

        # FFT 2D
        shift_fft2d, _ = self._compute_shift_2d(mean_frame, use_fft=True, fft_method="2d")
        results['fft_2d'] = shift_fft2d

        self.results['method_comparison'] = results
        return results

    def run_full_analysis(
        self,
        n_spatial_bands: int = 8,
        temporal_sample_every: int = 1,
    ) -> Dict[str, Any]:
        """
        Run complete analysis across all dimensions and methods.
        """
        print("\n" + "=" * 60)
        print("SCAN-PHASE DISTRIBUTION ASSESSMENT")
        print("=" * 60)

        # Method comparison
        print("\n[1/7] Method comparison (FFT vs non-FFT)...")
        self.analyze_method_comparison()

        # Vertical analysis
        print("[2/7] Vertical distribution (FFT 2D)...")
        self.analyze_vertical_distribution(n_spatial_bands, use_fft=True, fft_method="2d")
        print("[3/7] Vertical distribution (non-FFT)...")
        self.analyze_vertical_distribution(n_spatial_bands, use_fft=False)

        # Horizontal analysis
        print("[4/7] Horizontal distribution (FFT 2D)...")
        self.analyze_horizontal_distribution(n_spatial_bands, use_fft=True, fft_method="2d")
        print("[5/7] Horizontal distribution (non-FFT)...")
        self.analyze_horizontal_distribution(n_spatial_bands, use_fft=False)

        # Temporal analysis
        if not self._is_single_frame:
            print("[6/7] Temporal variation...")
            self.analyze_temporal_variation(use_fft=True, fft_method="2d", sample_every=temporal_sample_every)
        else:
            print("[6/7] Temporal variation (skipped - single frame)")

        # Mean window analysis
        print("[7/7] Mean window analysis...")
        self.analyze_mean_windows(use_fft=True, fft_method="2d")
        self.analyze_rolling_mean_subtracted(use_fft=True, fft_method="2d")

        print("\nAnalysis complete!")
        return self.results

    def plot_results(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive visualization of all results.
        """
        fig = plt.figure(figsize=(20, 16))

        # 1. Method comparison
        ax1 = plt.subplot(3, 3, 1)
        if 'method_comparison' in self.results:
            methods = list(self.results['method_comparison'].keys())
            values = list(self.results['method_comparison'].values())
            bars = ax1.bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax1.set_ylabel('Shift (pixels)')
            ax1.set_title('Method Comparison\n(on mean image)')
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 2. Vertical distribution
        ax2 = plt.subplot(3, 3, 2)
        for key in self.results:
            if key.startswith('vertical_'):
                r = self.results[key]
                label = key.replace('vertical_', '')
                ax2.plot(r['y_centers'], r['shifts'], 'o-', label=label, markersize=6)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Y position (pixels)')
        ax2.set_ylabel('Shift (pixels)')
        ax2.set_title('Vertical Distribution\n(phase shift vs Y)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Horizontal distribution
        ax3 = plt.subplot(3, 3, 3)
        for key in self.results:
            if key.startswith('horizontal_'):
                r = self.results[key]
                label = key.replace('horizontal_', '')
                ax3.plot(r['x_centers'], r['shifts'], 'o-', label=label, markersize=6)
        ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('X position (pixels)')
        ax3.set_ylabel('Shift (pixels)')
        ax3.set_title('Horizontal Distribution\n(phase shift vs X)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Temporal variation
        ax4 = plt.subplot(3, 3, 4)
        for key in self.results:
            if key.startswith('temporal_'):
                r = self.results[key]
                if len(r['shifts']) > 1:
                    ax4.plot(r['frame_indices'], r['shifts'], '-',
                            label=key.replace('temporal_', ''), alpha=0.8)
        ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Frame index')
        ax4.set_ylabel('Shift (pixels)')
        ax4.set_title('Temporal Variation\n(phase shift over time)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. Mean window analysis
        ax5 = plt.subplot(3, 3, 5)
        for key in self.results:
            if key.startswith('mean_window_'):
                r = self.results[key]
                label = key.replace('mean_window_', '')
                ax5.errorbar(r['window_sizes'], r['shifts'], yerr=r['stds'],
                            fmt='o-', label=label, capsize=3, markersize=6)
        ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Window size (frames)')
        ax5.set_ylabel('Shift (pixels)')
        ax5.set_title('Mean Window Analysis\n(shift stability vs averaging)')
        ax5.set_xscale('log')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. Rolling mean subtraction
        ax6 = plt.subplot(3, 3, 6)
        for key in self.results:
            if key.startswith('rolling_sub_'):
                r = self.results[key]
                label = key.replace('rolling_sub_', '')
                if len(r['shifts']) > 0 and not np.all(np.isnan(r['shifts'])):
                    ax6.errorbar(r['window_sizes'], r['shifts'], yerr=r['stds'],
                                fmt='s-', label=label, capsize=3, markersize=6)
        ax6.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Rolling window size (frames)')
        ax6.set_ylabel('Shift (pixels)')
        ax6.set_title('Rolling Mean Subtracted\n(high-pass filtered)')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        # 7. Example frame with correction preview
        ax7 = plt.subplot(3, 3, 7)
        mean_frame = np.mean(self.data, axis=0)
        vmin, vmax = np.percentile(mean_frame, [1, 99])
        ax7.imshow(mean_frame, cmap='gray', vmin=vmin, vmax=vmax)
        ax7.set_title('Mean Frame (uncorrected)')
        ax7.axis('off')

        # 8. Even/odd row comparison
        ax8 = plt.subplot(3, 3, 8)
        even_rows = mean_frame[::2, :]
        odd_rows = mean_frame[1::2, :]
        # Show difference
        min_rows = min(even_rows.shape[0], odd_rows.shape[0])
        diff = even_rows[:min_rows] - odd_rows[:min_rows]
        im = ax8.imshow(diff, cmap='RdBu_r', vmin=-np.percentile(np.abs(diff), 95),
                       vmax=np.percentile(np.abs(diff), 95))
        ax8.set_title('Even - Odd Row Difference')
        ax8.axis('off')
        plt.colorbar(im, ax=ax8, fraction=0.046)

        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = "SUMMARY\n" + "=" * 40 + "\n\n"

        if 'method_comparison' in self.results:
            mc = self.results['method_comparison']
            summary_text += "Method Comparison (mean frame):\n"
            for m, v in mc.items():
                summary_text += f"  {m}: {v:.3f} px\n"
            summary_text += "\n"

        # Add spatial variation summary
        for key in ['vertical_fft_2d', 'horizontal_fft_2d']:
            if key in self.results:
                r = self.results[key]
                valid = ~np.isnan(r['shifts'])
                if valid.any():
                    summary_text += f"{key.replace('_', ' ').title()}:\n"
                    summary_text += f"  Range: [{r['shifts'][valid].min():.2f}, {r['shifts'][valid].max():.2f}] px\n"
                    summary_text += f"  Std: {r['shifts'][valid].std():.3f} px\n"
                    summary_text += "\n"

        # Add temporal summary
        for key in self.results:
            if key.startswith('temporal_'):
                r = self.results[key]
                valid = ~np.isnan(r['shifts'])
                if valid.any() and valid.sum() > 1:
                    summary_text += f"Temporal Variation:\n"
                    summary_text += f"  Mean: {r['shifts'][valid].mean():.3f} px\n"
                    summary_text += f"  Std: {r['shifts'][valid].std():.3f} px\n"
                    summary_text += f"  Range: [{r['shifts'][valid].min():.2f}, {r['shifts'][valid].max():.2f}] px\n"
                break

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Scan-Phase Distribution Assessment\n'
                    f'Data shape: {self.original_shape}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
            print(f"Saved plot to: {save_path}")

        return fig

    def print_summary(self):
        """Print text summary of results."""
        print("\n" + "=" * 60)
        print("SCAN-PHASE ASSESSMENT SUMMARY")
        print("=" * 60)

        if 'method_comparison' in self.results:
            print("\nMethod Comparison (on mean frame):")
            for method, shift in self.results['method_comparison'].items():
                print(f"  {method:12s}: {shift:+.3f} pixels")

        print("\nSpatial Variation:")
        for key in ['vertical_fft_2d', 'horizontal_fft_2d']:
            if key in self.results:
                r = self.results[key]
                valid = ~np.isnan(r['shifts'])
                if valid.any():
                    direction = 'Vertical' if 'vertical' in key else 'Horizontal'
                    print(f"  {direction}:")
                    print(f"    Mean: {r['shifts'][valid].mean():.3f} px")
                    print(f"    Std:  {r['shifts'][valid].std():.3f} px")
                    print(f"    Range: [{r['shifts'][valid].min():.2f}, {r['shifts'][valid].max():.2f}]")

        for key in self.results:
            if key.startswith('temporal_'):
                r = self.results[key]
                valid = ~np.isnan(r['shifts'])
                if valid.any() and valid.sum() > 1:
                    print(f"\nTemporal Variation:")
                    print(f"  Mean: {r['shifts'][valid].mean():.3f} px")
                    print(f"  Std:  {r['shifts'][valid].std():.3f} px")
                    print(f"  Range: [{r['shifts'][valid].min():.2f}, {r['shifts'][valid].max():.2f}]")
                break

        print("\n" + "=" * 60)


def main():
    """Main entry point for command-line usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python assess_scanphase_distribution.py <path_to_data>")
        print("\nThis script performs comprehensive scan-phase analysis including:")
        print("  - Vertical/horizontal spatial variation")
        print("  - Temporal variation across frames")
        print("  - Mean-window stability analysis")
        print("  - Rolling mean subtraction analysis")
        print("  - FFT vs non-FFT method comparison")
        sys.exit(1)

    inpath = Path(sys.argv[1])

    if not inpath.exists():
        print(f"Error: Path does not exist: {inpath}")
        sys.exit(1)

    print(f"Loading data from: {inpath}")
    data = mbo.imread(inpath)

    # Handle lazy array
    if hasattr(data, '__array__'):
        # Load a subset if too large
        if hasattr(data, 'shape') and len(data.shape) >= 3 and data.shape[0] > 100:
            print(f"Large dataset ({data.shape[0]} frames), loading first 100 frames...")
            data = np.array(data[:100])
        else:
            data = np.array(data)

    print(f"Data shape: {data.shape}")

    # Run assessment
    assessment = ScanPhaseAssessment(data)
    assessment.run_full_analysis()
    assessment.print_summary()

    # Save plot
    output_path = inpath if inpath.is_dir() else inpath.parent
    plot_path = output_path / 'scanphase_assessment.png'
    assessment.plot_results(save_path=plot_path)

    plt.show()


if __name__ == "__main__":
    main()
