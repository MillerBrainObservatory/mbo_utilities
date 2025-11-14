"""
Optimize sine wave parameters by grid search over correlation quality.
Find the best amplitude, frequency, phase, and offset that maximize alignment.
"""
from pathlib import Path
import mbo_utilities as mbo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from skimage.registration import phase_cross_correlation


def sine_model(x, amplitude, frequency, phase, offset):
    """Sine wave model for phase shift pattern."""
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset


def measure_correlation_quality(frame, window_size=64, stride=32, min_intensity=50):
    """
    Measure overall correlation quality between even/odd rows in bright regions.
    Returns mean correlation coefficient (higher is better).
    """
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    a, b = pre[:m], post[:m]
    h, w = a.shape

    correlations = []
    weights = []

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            window_a = a[y:y+window_size, x:x+window_size]
            window_b = b[y:y+window_size, x:x+window_size]

            mean_intensity = (window_a.mean() + window_b.mean()) / 2
            if mean_intensity < min_intensity:
                continue

            # Compute normalized cross-correlation
            corr = np.corrcoef(window_a.ravel(), window_b.ravel())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
                weights.append(mean_intensity)

    if len(correlations) == 0:
        return 0.0

    # Weight by intensity
    correlations = np.array(correlations)
    weights = np.array(weights)
    return np.average(correlations, weights=weights)


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


def get_initial_shift_measurements(frame, window_size=64, stride=32,
                                   upsample=10, min_intensity=50, max_shift=10):
    """Get initial phase shift measurements from the original frame."""
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    a, b = pre[:m], post[:m]
    h, w = a.shape

    x_positions = []
    shifts = []
    intensities = []

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            window_a = a[y:y+window_size, x:x+window_size]
            window_b = b[y:y+window_size, x:x+window_size]

            mean_intensity = (window_a.mean() + window_b.mean()) / 2
            if mean_intensity < min_intensity:
                continue

            try:
                shift_2d, _, _ = phase_cross_correlation(
                    window_a, window_b,
                    upsample_factor=upsample,
                    normalization=None
                )

                horizontal_shift = float(shift_2d[1])

                if abs(horizontal_shift) <= max_shift:
                    x_positions.append(x + window_size // 2)
                    shifts.append(horizontal_shift)
                    intensities.append(mean_intensity)
            except:
                continue

    return np.array(x_positions), np.array(shifts), np.array(intensities)


def objective_function(params, frame_original, image_width):
    """
    Objective function to minimize (negative correlation quality).
    params = [amplitude, frequency, phase, offset]
    """
    amplitude, frequency, phase, offset = params

    # Generate shift pattern
    x_norm = np.arange(image_width) / image_width
    shift_per_column = sine_model(x_norm, amplitude, frequency, phase, offset)

    # Apply correction
    corrected = apply_column_phase_correction(frame_original, shift_per_column)

    # Measure quality
    quality = measure_correlation_quality(corrected, window_size=64,
                                         stride=32, min_intensity=50)

    # Return negative (we're minimizing)
    return -quality


def optimize_sine_correction(frame, initial_measurements=None, method='evolution',
                            verbose=True):
    """
    Optimize sine wave parameters to maximize correlation quality.

    Parameters
    ----------
    frame : ndarray
        Original frame to correct
    initial_measurements : tuple or None
        (x_positions, shifts, intensities) from initial analysis
    method : str
        'evolution' for differential evolution or 'minimize' for local optimization
    verbose : bool
        Print progress

    Returns
    -------
    best_params : ndarray
        [amplitude, frequency, phase, offset]
    corrected : ndarray
        Best corrected frame
    shift_per_column : ndarray
        Optimal shift pattern
    """
    w = frame.shape[1]

    # Get initial guess from measurements if provided
    if initial_measurements is not None:
        x_pos, shifts, _ = initial_measurements
        amp_guess = (shifts.max() - shifts.min()) / 2
        offset_guess = shifts.mean()
        freq_guess = 0.7  # Start with ~0.7 cycles
        phase_guess = 0
    else:
        amp_guess = 5.0
        freq_guess = 0.7
        phase_guess = 0
        offset_guess = 3.0

    if verbose:
        print('='*70)
        print('OPTIMIZING SINE WAVE PHASE CORRECTION')
        print('='*70)
        print(f'Image size: {frame.shape[1]} x {frame.shape[0]} pixels')
        print(f'Optimization method: {method}')
        print(f'Initial guess: amp={amp_guess:.2f}, freq={freq_guess:.3f}, '
              f'phase={phase_guess:.2f}, offset={offset_guess:.2f}')
        print('='*70)

    if method == 'evolution':
        # Global optimization using differential evolution
        bounds = [
            (-10, 10),      # amplitude: can be negative for inverted sine
            (0.1, 2.0),     # frequency: 0.1 to 2 cycles across image
            (-np.pi, np.pi), # phase
            (-10, 10)       # offset
        ]

        if verbose:
            print('Running differential evolution...')

        result = differential_evolution(
            objective_function,
            bounds,
            args=(frame, w),
            maxiter=100,
            popsize=15,
            tol=0.0001,
            seed=42,
            workers=1,
            disp=verbose,
            polish=True
        )

        best_params = result.x
        best_quality = -result.fun

    else:
        # Local optimization
        x0 = [amp_guess, freq_guess, phase_guess, offset_guess]

        bounds = [
            (-10, 10),      # amplitude
            (0.1, 2.0),     # frequency
            (-np.pi, np.pi), # phase
            (-10, 10)       # offset
        ]

        if verbose:
            print('Running local optimization...')

        result = minimize(
            objective_function,
            x0,
            args=(frame, w),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'disp': verbose}
        )

        best_params = result.x
        best_quality = -result.fun

    # Generate final correction
    x_norm = np.arange(w) / w
    shift_per_column = sine_model(x_norm, *best_params)
    corrected = apply_column_phase_correction(frame, shift_per_column)

    if verbose:
        amp, freq, phase, offset = best_params
        print('='*70)
        print('OPTIMIZATION COMPLETE')
        print('='*70)
        print(f'Best parameters:')
        print(f'  Amplitude: {amp:.3f} pixels')
        print(f'  Frequency: {freq:.3f} cycles')
        print(f'  Phase:     {phase:.3f} radians ({np.degrees(phase):.1f}°)')
        print(f'  Offset:    {offset:.3f} pixels')
        print(f'Correlation quality: {best_quality:.4f}')
        print(f'Shift range: [{shift_per_column.min():.2f}, '
              f'{shift_per_column.max():.2f}] pixels')
        print('='*70)

    return best_params, corrected, shift_per_column


# Main execution
if __name__ == '__main__':
    import tifffile

    # Load data
    inpath = Path.home() / 'Documents/data/yao'
    tif_files = sorted(inpath.glob('*.tif'))
    original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
    print(f'Loading: {original_tif.name}')
    data = tifffile.imread(original_tif)
    frame = data[0, 5, :, :]  # Z=0, T=5

    print(f'Frame shape: {frame.shape}\n')

    # Get initial measurements
    print('Analyzing initial phase shifts...')
    x_pos, shifts, intensities = get_initial_shift_measurements(
        frame, window_size=64, stride=32, upsample=10,
        min_intensity=50, max_shift=10
    )
    print(f'Measurements: {len(shifts)}')
    print(f'Shift range: [{shifts.min():.2f}, {shifts.max():.2f}] px\n')

    # Optimize using differential evolution
    best_params, corrected, shift_pattern = optimize_sine_correction(
        frame,
        initial_measurements=(x_pos, shifts, intensities),
        method='evolution',
        verbose=True
    )

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    vmin, vmax = np.percentile(frame, [1, 99.5])

    # Original
    axes[0, 0].imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 0].set_title('Original', fontsize=14, weight='bold')
    axes[0, 0].axis('off')

    # Corrected
    axes[0, 1].imshow(corrected, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title('Optimized Sine Correction', fontsize=14, weight='bold')
    axes[0, 1].axis('off')

    # Difference
    diff = corrected - frame
    axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-200, vmax=200, aspect='auto')
    axes[0, 2].set_title('Difference', fontsize=14, weight='bold')
    axes[0, 2].axis('off')

    # Shift pattern with measurements
    axes[1, 0].plot(shift_pattern, linewidth=2, label='Optimized sine fit', color='blue')
    axes[1, 0].scatter(x_pos, shifts, c='red', s=20, alpha=0.5,
                      label='Initial measurements', zorder=10)
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Column', fontsize=12)
    axes[1, 0].set_ylabel('Shift (pixels)', fontsize=12)
    axes[1, 0].set_title('Optimal Shift Pattern', fontsize=14, weight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Zoomed comparison - center
    y_center, x_center = 192, 220
    size = 100
    zoom_orig_center = frame[y_center:y_center+size, x_center:x_center+size]
    zoom_corr_center = corrected[y_center:y_center+size, x_center:x_center+size]
    combined_center = np.hstack([zoom_orig_center, zoom_corr_center])
    axes[1, 1].imshow(combined_center, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'Center: Original | Corrected\n(shift≈{shift_pattern[x_center]:.2f}px)',
                        fontsize=12, weight='bold')
    axes[1, 1].axis('off')

    # Zoomed comparison - right edge
    x_right = 400
    zoom_orig_right = frame[y_center:y_center+size, x_right:x_right+size]
    zoom_corr_right = corrected[y_center:y_center+size, x_right:x_right+size]
    combined_right = np.hstack([zoom_orig_right, zoom_corr_right])
    axes[1, 2].imshow(combined_right, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title(f'Right Edge: Original | Corrected\n(shift≈{shift_pattern[x_right]:.2f}px)',
                        fontsize=12, weight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()

    output_path = Path.home() / 'Documents/data/yao' / 'optimized_sine_correction.png'
    plt.savefig(output_path, dpi=150, facecolor='white')
    print(f'\nSaved visualization to: {output_path}')

    # Save results
    output_dir = Path.home() / 'Documents/data/yao'
    tifffile.imwrite(output_dir / 'corrected_optimized_sine.tif',
                    corrected.astype(np.int16))
    np.save(output_dir / 'shift_pattern_optimized.npy', shift_pattern)
    np.save(output_dir / 'sine_params_optimized.npy', best_params)

    print(f'\nSaved files to {output_dir}/:')
    print('  - corrected_optimized_sine.tif')
    print('  - shift_pattern_optimized.npy')
    print('  - sine_params_optimized.npy')
    print('  - optimized_sine_correction.png')

    plt.close()
