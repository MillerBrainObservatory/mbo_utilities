"""
Multi-harmonic sine wave correction using Fourier series.
Fits sum of multiple sine/cosine terms to capture complex spatial variation.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from skimage.registration import phase_cross_correlation
import tifffile


def fourier_series_model(x, *coeffs):
    """
    Fourier series: offset + sum of (a_n*sin + b_n*cos) terms.
    coeffs = [offset, a1, b1, a2, b2, ..., a_n, b_n]
    """
    offset = coeffs[0]
    result = np.full_like(x, offset)

    n_harmonics = (len(coeffs) - 1) // 2

    for n in range(1, n_harmonics + 1):
        a_n = coeffs[2*n - 1]
        b_n = coeffs[2*n]
        result += a_n * np.sin(2 * np.pi * n * x) + b_n * np.cos(2 * np.pi * n * x)

    return result


def polynomial_model(x, *coeffs):
    """Polynomial model: sum of c_i * x^i"""
    return np.polyval(coeffs[::-1], x)


def apply_column_phase_correction(frame, shift_per_column):
    """Apply per-column FFT-based phase correction."""
    corrected = frame.copy()
    h, w = frame.shape

    for col in range(w):
        shift = shift_per_column[col]
        odd_rows = corrected[1::2, col]

        if abs(shift) > 0.01:
            freq = np.fft.fftfreq(len(odd_rows))
            phase_shift = np.exp(-2j * np.pi * freq * shift)
            fft = np.fft.fft(odd_rows)
            shifted = np.fft.ifft(fft * phase_shift).real
            corrected[1::2, col] = shifted

    return corrected


def measure_local_shifts(frame, window_size=64, stride=20, upsample=10,
                         min_intensity_percentile=50, max_shift=10):
    """Measure phase shifts across the image."""
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    a, b = pre[:m], post[:m]
    h, w = a.shape

    intensity_threshold = np.percentile(frame, min_intensity_percentile)

    x_positions = []
    y_positions = []
    shifts = []
    intensities = []

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            window_a = a[y:y+window_size, x:x+window_size]
            window_b = b[y:y+window_size, x:x+window_size]

            mean_intensity = (window_a.mean() + window_b.mean()) / 2

            if mean_intensity < intensity_threshold:
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
                    y_positions.append(y + window_size // 2)
                    shifts.append(horizontal_shift)
                    intensities.append(mean_intensity)
            except:
                continue

    return (np.array(x_positions), np.array(y_positions),
            np.array(shifts), np.array(intensities))


# Load data
inpath = Path.home() / 'Documents/data/yao'
tif_files = sorted(inpath.glob('*.tif'))
original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
print(f'Loading: {original_tif.name}')
data = tifffile.imread(original_tif)
frame = data[0, 5, :, :]

print(f'Frame shape: {frame.shape}\n')

# Measure shifts with dense sampling
print('Measuring phase shifts (dense sampling for better fit)...')
x_pos, y_pos, shifts, intensities = measure_local_shifts(
    frame,
    window_size=64,
    stride=20,  # Very dense sampling
    upsample=10,
    min_intensity_percentile=50,
    max_shift=10
)

print(f'Measurements: {len(shifts)}')
print(f'Shift range: [{shifts.min():.2f}, {shifts.max():.2f}] px')
print(f'Mean: {shifts.mean():.2f} px, Std: {shifts.std():.2f} px\n')

# Normalize x to [0, 1]
x_norm = x_pos / frame.shape[1]

# Weight by intensity for fitting
weights = intensities / intensities.max()

print('='*70)
print('FITTING MULTIPLE MODELS')
print('='*70)

# Model 1: Polynomial (degree 3)
print('\n1. Polynomial (degree 3)...')
poly_coeffs = np.polyfit(x_norm, shifts, deg=3, w=weights)
x_all_norm = np.arange(frame.shape[1]) / frame.shape[1]
shift_poly = polynomial_model(x_all_norm, *poly_coeffs)
print(f'   Coeffs: {poly_coeffs}')
print(f'   Range: [{shift_poly.min():.2f}, {shift_poly.max():.2f}] px')

# Model 2: Polynomial (degree 4)
print('\n2. Polynomial (degree 4)...')
poly4_coeffs = np.polyfit(x_norm, shifts, deg=4, w=weights)
shift_poly4 = polynomial_model(x_all_norm, *poly4_coeffs)
print(f'   Coeffs: {poly4_coeffs}')
print(f'   Range: [{shift_poly4.min():.2f}, {shift_poly4.max():.2f}] px')

# Model 3: Fourier series (2 harmonics)
print('\n3. Fourier series (2 harmonics)...')
try:
    # Initial guess: offset + 2 harmonics (5 coeffs total)
    p0_fourier2 = [shifts.mean(), 2, 0, 1, 0]  # offset, a1, b1, a2, b2

    fourier2_coeffs, _ = curve_fit(
        fourier_series_model, x_norm, shifts,
        p0=p0_fourier2,
        sigma=1.0/weights,
        absolute_sigma=False,
        maxfev=20000
    )

    shift_fourier2 = fourier_series_model(x_all_norm, *fourier2_coeffs)
    print(f'   Coeffs: {fourier2_coeffs}')
    print(f'   Range: [{shift_fourier2.min():.2f}, {shift_fourier2.max():.2f}] px')
except Exception as e:
    print(f'   Failed: {e}')
    shift_fourier2 = None
    fourier2_coeffs = None

# Model 4: Fourier series (3 harmonics)
print('\n4. Fourier series (3 harmonics)...')
try:
    p0_fourier3 = [shifts.mean(), 2, 0, 1, 0, 0.5, 0]  # offset + 3 harmonics

    fourier3_coeffs, _ = curve_fit(
        fourier_series_model, x_norm, shifts,
        p0=p0_fourier3,
        sigma=1.0/weights,
        absolute_sigma=False,
        maxfev=20000
    )

    shift_fourier3 = fourier_series_model(x_all_norm, *fourier3_coeffs)
    print(f'   Coeffs: {fourier3_coeffs}')
    print(f'   Range: [{shift_fourier3.min():.2f}, {shift_fourier3.max():.2f}] px')
except Exception as e:
    print(f'   Failed: {e}')
    shift_fourier3 = None
    fourier3_coeffs = None

# Model 5: Piecewise linear (3 segments)
print('\n5. Piecewise linear (3 segments)...')
# Divide into 3 regions and fit each separately
thirds = frame.shape[1] // 3

segment_shifts = []
segment_bounds = [0, thirds, 2*thirds, frame.shape[1]]

for i in range(3):
    x_start = segment_bounds[i]
    x_end = segment_bounds[i+1]

    # Get measurements in this segment
    mask = (x_pos >= x_start) & (x_pos < x_end)
    if mask.sum() > 1:
        x_seg = x_pos[mask]
        shifts_seg = shifts[mask]
        weights_seg = weights[mask]

        # Linear fit
        coeffs_seg = np.polyfit(x_seg, shifts_seg, deg=1, w=weights_seg)

        # Generate shifts for this segment
        x_range = np.arange(x_start, x_end)
        shift_seg = np.polyval(coeffs_seg, x_range)
        segment_shifts.extend(shift_seg)

        print(f'   Segment {i+1} ({x_start}-{x_end}): slope={coeffs_seg[0]:.4f}, '
              f'range=[{shift_seg.min():.2f}, {shift_seg.max():.2f}] px')

shift_piecewise = np.array(segment_shifts)
print(f'   Overall range: [{shift_piecewise.min():.2f}, {shift_piecewise.max():.2f}] px')

print('='*70)

# Create test cases
test_cases = [
    ('Polynomial deg=3', shift_poly),
    ('Polynomial deg=4', shift_poly4),
    ('Piecewise Linear (3 segments)', shift_piecewise),
]

if shift_fourier2 is not None:
    test_cases.append(('Fourier 2 harmonics', shift_fourier2))
if shift_fourier3 is not None:
    test_cases.append(('Fourier 3 harmonics', shift_fourier3))

# Visualize
n_cases = len(test_cases)
fig = plt.figure(figsize=(24, 4 * n_cases))

vmin, vmax = np.percentile(frame, [1, 99.5])

for i, (label, shift_i) in enumerate(test_cases):
    # Apply correction
    corrected_i = apply_column_phase_correction(frame, shift_i)

    # Row i: 5 subplots
    # 1. Shift pattern
    ax1 = plt.subplot(n_cases, 5, i*5 + 1)
    ax1.plot(shift_i, linewidth=2.5, color='blue', label='Fit')
    ax1.scatter(x_pos, shifts, c='red', s=15, alpha=0.4, label='Measurements')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Shift (px)', fontsize=10)
    ax1.set_xlabel('X position', fontsize=10)
    ax1.set_title(f'{label}\nRange: [{shift_i.min():.1f}, {shift_i.max():.1f}] px',
                 fontsize=11, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # 2. Original
    ax2 = plt.subplot(n_cases, 5, i*5 + 2)
    ax2.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('Original', fontsize=10)
    ax2.axis('off')

    # 3. Corrected
    ax3 = plt.subplot(n_cases, 5, i*5 + 3)
    ax3.imshow(corrected_i, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax3.set_title('Corrected', fontsize=10)
    ax3.axis('off')

    # 4. Zoom neuron region (left side that needs ~3-4 more pixels)
    x_neuron_left = 180  # Left side of neuron
    y = 192
    size = 100
    zoom_neuron = corrected_i[y:y+size, x_neuron_left:x_neuron_left+size]
    ax4 = plt.subplot(n_cases, 5, i*5 + 4)
    ax4.imshow(zoom_neuron, cmap='gray', vmin=vmin, vmax=vmax)
    ax4.set_title(f'Neuron Left (x={x_neuron_left}\nshift={shift_i[x_neuron_left]:.2f}px)',
                 fontsize=9)
    ax4.axis('off')

    # 5. Zoom right edge
    x_right = 480
    zoom_right = corrected_i[y:y+size, x_right:x_right+size]
    ax5 = plt.subplot(n_cases, 5, i*5 + 5)
    ax5.imshow(zoom_right, cmap='gray', vmin=vmin, vmax=vmax)
    ax5.set_title(f'Right Edge (x={x_right}\nshift={shift_i[x_right]:.2f}px)',
                 fontsize=9)
    ax5.axis('off')

plt.suptitle('Multi-Model Phase Correction - Complex Spatial Variation\n'
            'USER: Which model best corrects the neuron left side AND right edge?',
            fontsize=14, weight='bold', y=0.995)

plt.tight_layout()

output_path = Path.home() / 'Documents/data/yao' / 'multi_model_correction.png'
plt.savefig(output_path, dpi=120, facecolor='white', bbox_inches='tight')
print(f'\nSaved to: {output_path}')

# Save best polynomial fit as default
corrected_poly4 = apply_column_phase_correction(frame, shift_poly4)
tifffile.imwrite(inpath / 'corrected_poly4.tif', corrected_poly4.astype(np.int16))
np.save(inpath / 'shift_pattern_poly4.npy', shift_poly4)
print(f'\nSaved polynomial (deg=4) correction:')
print('  - corrected_poly4.tif')
print('  - shift_pattern_poly4.npy')

print('\n' + '='*70)
print('REVIEW: multi_model_correction.png')
print('These models can capture more complex spatial variation than sine waves.')
print('Tell me which row works best!')
print('='*70)

plt.close()
