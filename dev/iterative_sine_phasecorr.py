"""
Iterative sine wave phase correction with correlation-based optimization.
Searches up to ±10 pixels and refines fit by measuring correlation quality.
"""
from pathlib import Path
import mbo_utilities as mbo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from skimage.registration import phase_cross_correlation


def sine_model(x, amplitude, frequency, phase, offset):
    """Sine wave model for phase shift pattern."""
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset


def measure_correlation_quality(frame, window_size=64, stride=32, min_intensity=50):
    """
    Measure overall correlation quality between even/odd rows in bright regions.
    Higher is better (more aligned).
    """
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    a, b = pre[:m], post[:m]
    h, w = a.shape

    correlations = []

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

    if len(correlations) == 0:
        return 0.0

    # Return mean correlation weighted by number of windows
    return np.mean(correlations)


def analyze_phase_pattern(frame, window_size=64, stride=32, upsample=10,
                          min_intensity=50, max_shift=10):
    """
    Analyze phase shift pattern across image with extended search range.
    """
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
                # Use extended search range
                shift_2d, _, _ = phase_cross_correlation(
                    window_a, window_b,
                    upsample_factor=upsample,
                    normalization=None
                )

                horizontal_shift = float(shift_2d[1])

                # Only keep shifts within reasonable range
                if abs(horizontal_shift) <= max_shift:
                    x_positions.append(x + window_size // 2)
                    shifts.append(horizontal_shift)
                    intensities.append(mean_intensity)
            except:
                continue

    return np.array(x_positions), np.array(shifts), np.array(intensities)


def fit_sine_to_shifts(x_positions, shifts, image_width):
    """
    Fit sine wave to measured shifts.
    Returns shift_per_column array and fit parameters.
    """
    # Normalize x to [0, 1]
    x_norm = x_positions / image_width

    # Initial guess
    amp_guess = (shifts.max() - shifts.min()) / 2
    offset_guess = shifts.mean()
    freq_guess = 0.5  # Start with one cycle across image
    phase_guess = 0

    try:
        # Fit sine model
        params, _ = curve_fit(
            sine_model, x_norm, shifts,
            p0=[amp_guess, freq_guess, phase_guess, offset_guess],
            maxfev=10000,
            bounds=(
                [-15, 0.1, -2*np.pi, -15],  # Lower bounds
                [15, 2.0, 2*np.pi, 15]       # Upper bounds
            )
        )

        # Generate shift for each column
        x_columns_norm = np.arange(image_width) / image_width
        shift_per_column = sine_model(x_columns_norm, *params)

        return shift_per_column, params

    except:
        # If fit fails, use linear interpolation as fallback
        from scipy.interpolate import interp1d
        interp_func = interp1d(x_positions, shifts,
                              kind='linear',
                              fill_value='extrapolate',
                              bounds_error=False)
        shift_per_column = interp_func(np.arange(image_width))
        return shift_per_column, None


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


def iterative_sine_correction(frame, max_iterations=5, window_size=64,
                              stride=32, upsample=10, min_intensity=50,
                              max_shift=10, verbose=True):
    """
    Iteratively refine sine wave correction by measuring correlation quality.

    Parameters
    ----------
    frame : ndarray
        Input frame
    max_iterations : int
        Maximum refinement iterations
    window_size : int
        Analysis window size
    stride : int
        Window stride
    upsample : int
        Upsampling factor for subpixel accuracy
    min_intensity : float
        Minimum intensity for analysis
    max_shift : float
        Maximum allowed shift in pixels
    verbose : bool
        Print progress

    Returns
    -------
    corrected : ndarray
        Phase-corrected frame
    shift_per_column : ndarray
        Final shift pattern
    history : dict
        Optimization history
    """
    current_frame = frame.copy()
    history = {
        'iteration': [],
        'correlation': [],
        'num_measurements': [],
        'shift_range': [],
        'sine_params': []
    }

    if verbose:
        print('='*70)
        print('ITERATIVE SINE WAVE PHASE CORRECTION')
        print('='*70)
        print(f'Image size: {frame.shape[1]} x {frame.shape[0]} pixels')
        print(f'Search range: ±{max_shift} pixels')
        print(f'Max iterations: {max_iterations}')
        print('='*70)

    for iteration in range(max_iterations):
        if verbose:
            print(f'\nIteration {iteration + 1}/{max_iterations}:')

        # Analyze current frame
        x_pos, shifts, intensities = analyze_phase_pattern(
            current_frame, window_size, stride, upsample,
            min_intensity, max_shift
        )

        if len(shifts) < 10:
            if verbose:
                print(f'  WARNING: Only {len(shifts)} measurements, stopping.')
            break

        # Fit sine wave
        shift_per_column, sine_params = fit_sine_to_shifts(
            x_pos, shifts, frame.shape[1]
        )

        # Apply correction
        corrected = apply_column_phase_correction(current_frame, shift_per_column)

        # Measure quality
        corr_quality = measure_correlation_quality(
            corrected, window_size, stride, min_intensity
        )

        # Record history
        history['iteration'].append(iteration)
        history['correlation'].append(corr_quality)
        history['num_measurements'].append(len(shifts))
        history['shift_range'].append((shifts.min(), shifts.max()))
        history['sine_params'].append(sine_params)

        if verbose:
            print(f'  Measurements: {len(shifts)}')
            print(f'  Shift range: [{shifts.min():.2f}, {shifts.max():.2f}] px')
            print(f'  Shift std: {shifts.std():.2f} px')
            if sine_params is not None:
                amp, freq, phase, offset = sine_params
                print(f'  Sine fit: amp={amp:.2f}, freq={freq:.3f}, offset={offset:.2f}')
            print(f'  Correlation quality: {corr_quality:.4f}')

        # Check convergence
        if iteration > 0:
            improvement = corr_quality - history['correlation'][-2]
            if verbose:
                print(f'  Improvement: {improvement:+.4f}')

            # Stop if converged
            if improvement < 0.0001:
                if verbose:
                    print('  Converged!')
                break

        # Update for next iteration
        current_frame = corrected.copy()

    return corrected, shift_per_column, history


# Main execution
if __name__ == '__main__':
    # Load data
    import tifffile
    inpath = Path.home() / 'Documents/data/yao'
    # Find the original TIFF file (not corrected versions)
    tif_files = sorted(inpath.glob('*.tif'))
    original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
    print(f'Loading: {original_tif.name}')
    data = tifffile.imread(original_tif)
    print(f'Data shape: {data.shape}')
    # Structure: (Z, T, Y, X) = (2, 10, 512, 512)
    # Use first Z, middle T
    if data.ndim == 4:
        frame = data[0, 5, :, :]  # Z=0, T=5 (middle frame)
    elif data.ndim == 3:
        frame = data[0, :, :]
    else:
        frame = data

    print(f'Original frame shape: {frame.shape}\n')

    # Run iterative correction
    corrected, shifts, history = iterative_sine_correction(
        frame,
        max_iterations=5,
        window_size=64,
        stride=32,
        upsample=10,
        min_intensity=50,
        max_shift=10,
        verbose=True
    )

    print('\n' + '='*70)
    print('FINAL RESULTS')
    print('='*70)
    print(f'Iterations: {len(history["iteration"])}')
    print(f'Final correlation: {history["correlation"][-1]:.4f}')
    print(f'Improvement: {history["correlation"][-1] - history["correlation"][0]:+.4f}')
    print('='*70)

    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    vmin, vmax = np.percentile(frame, [1, 99.5])

    # Original
    axes[0, 0].imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 0].set_title('Original', fontsize=14, weight='bold')
    axes[0, 0].axis('off')

    # Corrected
    axes[0, 1].imshow(corrected, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title(f'Corrected (iter={len(history["iteration"])})',
                        fontsize=14, weight='bold')
    axes[0, 1].axis('off')

    # Difference
    diff = corrected - frame
    axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-200, vmax=200, aspect='auto')
    axes[0, 2].set_title('Difference', fontsize=14, weight='bold')
    axes[0, 2].axis('off')

    # Shift pattern
    axes[1, 0].plot(shifts, linewidth=2, label='Final shift pattern')
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Column', fontsize=12)
    axes[1, 0].set_ylabel('Shift (pixels)', fontsize=12)
    axes[1, 0].set_title('Final Shift Pattern', fontsize=14, weight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Convergence
    axes[1, 1].plot(history['iteration'], history['correlation'],
                   'o-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Correlation Quality', fontsize=12)
    axes[1, 1].set_title('Convergence History', fontsize=14, weight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Zoomed comparison (center, left edge, right edge)
    # Pick regions
    y = 192
    size = 100

    # Show right edge (which was problematic)
    x_right = 400
    zoom_orig = frame[y:y+size, x_right:x_right+size]
    zoom_corr = corrected[y:y+size, x_right:x_right+size]

    combined = np.hstack([zoom_orig, zoom_corr])
    axes[1, 2].imshow(combined, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title('Right Edge: Original | Corrected', fontsize=12, weight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()

    output_path = Path.home() / 'Documents/data/yao' / 'iterative_sine_correction.png'
    plt.savefig(output_path, dpi=150, facecolor='white')
    print(f'\nSaved visualization to: {output_path}')

    # Save corrected data
    import tifffile
    output_dir = Path.home() / 'Documents/data/yao'
    tifffile.imwrite(output_dir / 'corrected_iterative_sine.tif',
                    corrected.astype(np.int16))
    np.save(output_dir / 'shift_pattern_sine.npy', shifts)

    print(f'\nSaved files to {output_dir}/:')
    print('  - corrected_iterative_sine.tif')
    print('  - shift_pattern_sine.npy')
    print('  - iterative_sine_correction.png')

    plt.close()
