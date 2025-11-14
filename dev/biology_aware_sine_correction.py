"""
Biology-aware sine wave phase correction.
Focuses on minimizing even/odd row differences in bright biological structures.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, curve_fit
from skimage.registration import phase_cross_correlation
import tifffile


def sine_model(x, amplitude, frequency, phase, offset):
    """Sine wave model for phase shift pattern."""
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset


def apply_column_phase_correction(frame, shift_per_column):
    """Apply per-column phase correction using FFT."""
    corrected = frame.copy()
    h, w = frame.shape

    for col in range(w):
        shift = shift_per_column[col]
        odd_rows = corrected[1::2, col]

        if abs(shift) > 0.01:
            # FFT-based subpixel shift
            freq = np.fft.fftfreq(len(odd_rows))
            phase_shift = np.exp(-2j * np.pi * freq * shift)
            fft = np.fft.fft(odd_rows)
            shifted = np.fft.ifft(fft * phase_shift).real
            corrected[1::2, col] = shifted

    return corrected


def compute_biological_alignment_metric(frame, percentile_threshold=70):
    """
    Compute alignment quality focusing ONLY on bright biological structures.

    Measures the absolute difference between even and odd rows,
    but ONLY in regions with high intensity (cells/neurons).

    Lower is better (less misalignment).
    """
    # Split into even/odd rows
    even_rows = frame[::2, :]
    odd_rows = frame[1::2, :]

    # Match dimensions
    min_rows = min(even_rows.shape[0], odd_rows.shape[0])
    even_rows = even_rows[:min_rows, :]
    odd_rows = odd_rows[:min_rows, :]

    # Create mask of bright regions (biological structures)
    # Use percentile threshold to focus on cells
    intensity_threshold = np.percentile(frame, percentile_threshold)

    # Mask: high intensity in EITHER even or odd rows
    mask_even = even_rows > intensity_threshold
    mask_odd = odd_rows > intensity_threshold
    biological_mask = mask_even | mask_odd

    if biological_mask.sum() == 0:
        return 1e6  # Very bad score if no biological structures

    # Compute absolute difference in biological regions
    diff = np.abs(even_rows - odd_rows)

    # Mean difference in biological regions (weighted by intensity)
    intensity_weights = (even_rows + odd_rows) / 2
    intensity_weights = np.where(biological_mask, intensity_weights, 0)

    if intensity_weights.sum() == 0:
        return 1e6

    weighted_diff = (diff * intensity_weights * biological_mask).sum() / intensity_weights.sum()

    return weighted_diff


def measure_local_shifts(frame, window_size=64, stride=32, upsample=10,
                         min_intensity_percentile=60, max_shift=10):
    """
    Measure phase shifts in bright biological regions across the image.

    Returns measurements weighted by biological signal strength.
    """
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    a, b = pre[:m], post[:m]
    h, w = a.shape

    # Threshold for biological structures
    intensity_threshold = np.percentile(frame, min_intensity_percentile)

    x_positions = []
    shifts = []
    intensities = []

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            window_a = a[y:y+window_size, x:x+window_size]
            window_b = b[y:y+window_size, x:x+window_size]

            mean_intensity = (window_a.mean() + window_b.mean()) / 2

            # Only analyze windows with biological structures
            if mean_intensity < intensity_threshold:
                continue

            try:
                shift_2d, _, _ = phase_cross_correlation(
                    window_a, window_b,
                    upsample_factor=upsample,
                    normalization=None
                )

                horizontal_shift = float(shift_2d[1])

                # Allow large shifts (up to max_shift)
                if abs(horizontal_shift) <= max_shift:
                    x_positions.append(x + window_size // 2)
                    shifts.append(horizontal_shift)
                    intensities.append(mean_intensity)
            except:
                continue

    return np.array(x_positions), np.array(shifts), np.array(intensities)


def objective_function_biology(params, frame_original, image_width):
    """
    Objective function based on biological alignment.

    params = [amplitude, frequency, phase, offset]

    Returns: alignment error in biological regions (lower is better)
    """
    amplitude, frequency, phase, offset = params

    # Generate shift pattern
    x_norm = np.arange(image_width) / image_width
    shift_per_column = sine_model(x_norm, amplitude, frequency, phase, offset)

    # Apply correction
    corrected = apply_column_phase_correction(frame_original, shift_per_column)

    # Measure biological alignment (lower is better)
    alignment_error = compute_biological_alignment_metric(corrected,
                                                          percentile_threshold=70)

    return alignment_error


def optimize_biology_aware_correction(frame, x_positions, shifts, intensities,
                                      verbose=True):
    """
    Optimize sine wave correction focusing on biological structures.

    Uses differential evolution to find parameters that minimize
    even/odd row differences in bright regions (cells/neurons).
    """
    w = frame.shape[1]

    # Initial guess from measurements
    # Weight by intensity for better initial estimate
    weighted_mean = np.average(shifts, weights=intensities)
    weighted_std = np.sqrt(np.average((shifts - weighted_mean)**2, weights=intensities))

    amp_guess = weighted_std * 1.5  # Amplitude guess
    offset_guess = weighted_mean

    if verbose:
        print('='*70)
        print('BIOLOGY-AWARE SINE WAVE PHASE CORRECTION')
        print('='*70)
        print(f'Image size: {frame.shape[1]} x {frame.shape[0]} pixels')
        print(f'Measurements: {len(shifts)} biological windows')
        print(f'Shift range (measured): [{shifts.min():.2f}, {shifts.max():.2f}] px')
        print(f'Initial guess: amp~{amp_guess:.2f}, offset~{offset_guess:.2f}')
        print('='*70)
        print('Optimizing to minimize even/odd differences in bright regions...')
        print('(This may take a few minutes)')
        print('='*70)

    # Define bounds allowing large shifts
    bounds = [
        (-10, 10),       # amplitude: allow up to ±10 pixel variation
        (0.1, 2.0),      # frequency: 0.1 to 2 cycles across image
        (-np.pi, np.pi), # phase: full range
        (-10, 10)        # offset: allow shifts from -10 to +10 pixels
    ]

    # Run differential evolution
    result = differential_evolution(
        objective_function_biology,
        bounds,
        args=(frame, w),
        maxiter=150,       # More iterations for better convergence
        popsize=20,        # Larger population for better exploration
        tol=0.001,
        seed=42,
        workers=1,
        disp=verbose,
        polish=True,       # Polish with local optimizer at the end
        atol=0.01
    )

    best_params = result.x
    best_error = result.fun

    # Generate final correction
    x_norm = np.arange(w) / w
    shift_per_column = sine_model(x_norm, *best_params)
    corrected = apply_column_phase_correction(frame, shift_per_column)

    # Also compute initial error for comparison
    initial_error = compute_biological_alignment_metric(frame, percentile_threshold=70)

    if verbose:
        amp, freq, phase, offset = best_params
        print('='*70)
        print('OPTIMIZATION COMPLETE')
        print('='*70)
        print(f'Best sine parameters:')
        print(f'  Amplitude: {amp:.3f} pixels')
        print(f'  Frequency: {freq:.3f} cycles')
        print(f'  Phase:     {phase:.3f} rad ({np.degrees(phase):.1f}°)')
        print(f'  Offset:    {offset:.3f} pixels')
        print(f'')
        print(f'Resulting shift range: [{shift_per_column.min():.2f}, '
              f'{shift_per_column.max():.2f}] pixels')
        print(f'')
        print(f'Alignment error (biological regions):')
        print(f'  Before: {initial_error:.2f}')
        print(f'  After:  {best_error:.2f}')
        print(f'  Improvement: {((initial_error - best_error) / initial_error * 100):.1f}%')
        print('='*70)

    return best_params, corrected, shift_per_column


# Main execution
if __name__ == '__main__':
    # Load data
    inpath = Path.home() / 'Documents/data/yao'
    tif_files = sorted(inpath.glob('*.tif'))
    original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
    print(f'Loading: {original_tif.name}')
    data = tifffile.imread(original_tif)
    frame = data[0, 5, :, :]  # Z=0, T=5

    print(f'Frame shape: {frame.shape}\n')

    # Measure shifts in biological regions
    print('Measuring phase shifts in biological structures...')
    x_pos, shifts, intensities = measure_local_shifts(
        frame,
        window_size=64,
        stride=24,  # Denser sampling
        upsample=10,
        min_intensity_percentile=60,  # Focus on bright structures
        max_shift=10
    )

    print(f'Found {len(shifts)} measurements in bright regions')
    print(f'Shift range: [{shifts.min():.2f}, {shifts.max():.2f}] px')
    print(f'Mean shift: {shifts.mean():.2f} px')
    print(f'Std shift: {shifts.std():.2f} px\n')

    if len(shifts) < 10:
        print('ERROR: Not enough biological structures detected!')
        print('Try lowering min_intensity_percentile')
        exit(1)

    # Optimize correction
    best_params, corrected, shift_pattern = optimize_biology_aware_correction(
        frame, x_pos, shifts, intensities, verbose=True
    )

    # Visualize results
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    vmin, vmax = np.percentile(frame, [1, 99.5])

    # Row 1: Full images
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_title('Original', fontsize=14, weight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(corrected, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('Biology-Aware Corrected', fontsize=14, weight='bold')
    ax2.axis('off')

    # Row 2: Shift pattern and measurements
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(shift_pattern, linewidth=2.5, label='Optimized sine fit', color='blue', zorder=5)
    ax3.scatter(x_pos, shifts, c=intensities, s=40, alpha=0.6,
               cmap='hot', label='Measurements (colored by intensity)',
               edgecolors='black', linewidth=0.5, zorder=10)
    ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Column (X position)', fontsize=12)
    ax3.set_ylabel('Shift (pixels)', fontsize=12)
    ax3.set_title('Optimal Shift Pattern (Biology-Focused)', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Difference map
    ax4 = fig.add_subplot(gs[1, 2:4])
    diff = corrected - frame
    im = ax4.imshow(diff, cmap='RdBu_r', vmin=-300, vmax=300, aspect='auto')
    ax4.set_title('Difference (Corrected - Original)', fontsize=14, weight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, label='Intensity change')

    # Row 3: Zoomed regions (left, center, right)
    regions = [
        ('Left/Center Neuron\n(you said -0.8px worked)', 220, shift_pattern[220]),
        ('Center', 256, shift_pattern[256]),
        ('Right Edge\n(you said 7px worked)', 450, shift_pattern[450])
    ]

    y = 192
    size = 120

    for i, (label, x, shift_val) in enumerate(regions):
        ax = fig.add_subplot(gs[2, i])
        zoom_orig = frame[y:y+size, x:x+size]
        zoom_corr = corrected[y:y+size, x:x+size]
        combined = np.hstack([zoom_orig, zoom_corr])
        ax.imshow(combined, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{label}\nApplied shift: {shift_val:.2f}px',
                    fontsize=11, weight='bold')
        ax.axis('off')
        ax.axvline(size, color='yellow', linestyle='--', linewidth=2, alpha=0.7)

    # Parameter text
    ax_text = fig.add_subplot(gs[2, 3])
    ax_text.axis('off')
    amp, freq, phase, offset = best_params
    param_text = f"""OPTIMIZED PARAMETERS

Amplitude: {amp:.3f} px
Frequency: {freq:.3f} cycles
Phase: {phase:.3f} rad
        ({np.degrees(phase):.1f}°)
Offset: {offset:.3f} px

SHIFT RANGE
Min: {shift_pattern.min():.2f} px
Max: {shift_pattern.max():.2f} px
Range: {shift_pattern.max() - shift_pattern.min():.2f} px

At x=220: {shift_pattern[220]:.2f} px
At x=450: {shift_pattern[450]:.2f} px
"""
    ax_text.text(0.1, 0.5, param_text, fontsize=11, family='monospace',
                verticalalignment='center')

    plt.suptitle('Biology-Aware Sine Wave Phase Correction',
                fontsize=16, weight='bold', y=0.98)

    output_path = Path.home() / 'Documents/data/yao' / 'biology_aware_correction.png'
    plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'\nSaved visualization to: {output_path}')

    # Save results
    output_dir = Path.home() / 'Documents/data/yao'
    tifffile.imwrite(output_dir / 'corrected_biology_aware.tif',
                    corrected.astype(np.int16))
    np.save(output_dir / 'shift_pattern_biology.npy', shift_pattern)
    np.save(output_dir / 'sine_params_biology.npy', best_params)

    print(f'\nSaved files:')
    print('  - corrected_biology_aware.tif')
    print('  - shift_pattern_biology.npy')
    print('  - sine_params_biology.npy')
    print('  - biology_aware_correction.png')

    plt.close()
