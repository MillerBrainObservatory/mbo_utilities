"""
Direct sine fit to measurements, then test variations.
Let the user visually assess which parameters work best.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
import tifffile


def sine_model(x, amplitude, frequency, phase, offset):
    """Sine wave model."""
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset


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


def measure_local_shifts(frame, window_size=64, stride=24, upsample=10,
                         min_intensity_percentile=55, max_shift=10):
    """Measure phase shifts across the image."""
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    a, b = pre[:m], post[:m]
    h, w = a.shape

    intensity_threshold = np.percentile(frame, min_intensity_percentile)

    x_positions = []
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
                    shifts.append(horizontal_shift)
                    intensities.append(mean_intensity)
            except:
                continue

    return np.array(x_positions), np.array(shifts), np.array(intensities)


# Load data
inpath = Path.home() / 'Documents/data/yao'
tif_files = sorted(inpath.glob('*.tif'))
original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
print(f'Loading: {original_tif.name}')
data = tifffile.imread(original_tif)
frame = data[0, 5, :, :]

print(f'Frame shape: {frame.shape}\n')

# Measure shifts
print('Measuring phase shifts across image...')
x_pos, shifts, intensities = measure_local_shifts(
    frame,
    window_size=64,
    stride=24,
    upsample=10,
    min_intensity_percentile=55,
    max_shift=10
)

print(f'Measurements: {len(shifts)}')
print(f'Shift range: [{shifts.min():.2f}, {shifts.max():.2f}] px')
print(f'Mean: {shifts.mean():.2f} px, Std: {shifts.std():.2f} px\n')

# Normalize x to [0, 1]
x_norm = x_pos / frame.shape[1]

# Fit sine wave to measurements (weighted by intensity)
print('Fitting sine wave to measurements...')

# Initial guess
amp_guess = (shifts.max() - shifts.min()) / 2
offset_guess = shifts.mean()
freq_guess = 0.7

try:
    # Weighted fit
    params, pcov = curve_fit(
        sine_model, x_norm, shifts,
        p0=[amp_guess, freq_guess, 0, offset_guess],
        sigma=1.0 / intensities,  # Weight by intensity
        absolute_sigma=False,
        maxfev=20000,
        bounds=(
            [-15, 0.1, -2*np.pi, -15],
            [15, 2.0, 2*np.pi, 15]
        )
    )

    amp_fit, freq_fit, phase_fit, offset_fit = params

    print('='*70)
    print('DIRECT SINE FIT TO MEASUREMENTS')
    print('='*70)
    print(f'Amplitude: {amp_fit:.3f} px')
    print(f'Frequency: {freq_fit:.3f} cycles')
    print(f'Phase:     {phase_fit:.3f} rad ({np.degrees(phase_fit):.1f} deg)')
    print(f'Offset:    {offset_fit:.3f} px')

    # Generate shift pattern
    x_all_norm = np.arange(frame.shape[1]) / frame.shape[1]
    shift_pattern_fit = sine_model(x_all_norm, *params)

    print(f'Resulting shift range: [{shift_pattern_fit.min():.2f}, '
          f'{shift_pattern_fit.max():.2f}] px')
    print('='*70)

except Exception as e:
    print(f'Sine fit failed: {e}')
    print('Using direct measurements only')
    shift_pattern_fit = None
    params = None

# Now create test variations with DIFFERENT amplitudes and offsets
# Test cases based on user's observations:
# - Left/center neuron: needs ~-0.8 px
# - Right edge: needs ~7 px
# This suggests we need amplitude ~4-5 px with appropriate phase/offset

print('\nCreating test variations...')

test_cases = []

# Case 1: Direct fit (if successful)
if shift_pattern_fit is not None:
    test_cases.append(('Direct Fit', params, shift_pattern_fit))

# Case 2-6: Manual parameter exploration
# Based on measurements showing -1.3 to 6.3 px range

manual_params = [
    # (label, amp, freq, phase, offset)
    ('Large Amp (5px)', 5.0, 0.7, 0.0, 2.0),
    ('Large Amp Inverted', -5.0, 0.7, 0.0, 2.0),
    ('Wider Range (8px)', 8.0, 0.7, np.pi, 3.0),
    ('Match User (4px, offset 3)', 4.0, 0.7, np.pi/2, 3.0),
    ('High Freq', 5.0, 1.0, 0.0, 2.0),
]

for label, amp, freq, phase, offset in manual_params:
    params_test = np.array([amp, freq, phase, offset])
    shift_test = sine_model(x_all_norm, *params_test)
    test_cases.append((label, params_test, shift_test))

# Create comprehensive visualization
n_cases = len(test_cases)
fig = plt.figure(figsize=(24, 4 * n_cases))

vmin, vmax = np.percentile(frame, [1, 99.5])

for i, (label, params_i, shift_i) in enumerate(test_cases):
    amp_i, freq_i, phase_i, offset_i = params_i

    # Apply correction
    corrected_i = apply_column_phase_correction(frame, shift_i)

    # Row i: 5 subplots
    # 1. Shift pattern
    ax1 = plt.subplot(n_cases, 5, i*5 + 1)
    ax1.plot(shift_i, linewidth=2, color='blue')
    ax1.scatter(x_pos, shifts, c='red', s=20, alpha=0.4)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Shift (px)', fontsize=10)
    ax1.set_title(f'{label}\nA={amp_i:.2f}, f={freq_i:.2f}, φ={np.degrees(phase_i):.0f}°, off={offset_i:.2f}',
                 fontsize=10, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Range: [{shift_i.min():.1f}, {shift_i.max():.1f}] px',
            transform=ax1.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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

    # 4. Zoom center/left (neuron at x~220)
    x_left = 200
    y = 192
    size = 100
    zoom_corr_left = corrected_i[y:y+size, x_left:x_left+size]
    ax4 = plt.subplot(n_cases, 5, i*5 + 4)
    ax4.imshow(zoom_corr_left, cmap='gray', vmin=vmin, vmax=vmax)
    ax4.set_title(f'Neuron (x={x_left}, shift={shift_i[x_left]:.2f}px)', fontsize=9)
    ax4.axis('off')

    # 5. Zoom right edge (x~450)
    x_right = 440
    zoom_corr_right = corrected_i[y:y+size, x_right:x_right+size]
    ax5 = plt.subplot(n_cases, 5, i*5 + 5)
    ax5.imshow(zoom_corr_right, cmap='gray', vmin=vmin, vmax=vmax)
    ax5.set_title(f'Right Edge (x={x_right}, shift={shift_i[x_right]:.2f}px)', fontsize=9)
    ax5.axis('off')

plt.suptitle('Sine Wave Phase Correction - Parameter Exploration\n'
            'USER: Find which row looks best for the neuron and right edge',
            fontsize=14, weight='bold', y=0.995)

plt.tight_layout()

output_path = Path.home() / 'Documents/data/yao' / 'sine_parameter_exploration.png'
plt.savefig(output_path, dpi=120, facecolor='white', bbox_inches='tight')
print(f'\nSaved exploration to: {output_path}')

# Save the best fit version
if shift_pattern_fit is not None:
    corrected_fit = apply_column_phase_correction(frame, shift_pattern_fit)
    tifffile.imwrite(inpath / 'corrected_sine_direct_fit.tif',
                    corrected_fit.astype(np.int16))
    np.save(inpath / 'shift_pattern_direct_fit.npy', shift_pattern_fit)
    np.save(inpath / 'sine_params_direct_fit.npy', params)
    print(f'\nSaved direct fit files:')
    print('  - corrected_sine_direct_fit.tif')
    print('  - shift_pattern_direct_fit.npy')
    print('  - sine_params_direct_fit.npy')

print('\n' + '='*70)
print('NEXT STEP: Review sine_parameter_exploration.png')
print('Find which row looks best visually for:')
print('  1. The neuron (left/center)')
print('  2. The right edge')
print('Then tell me which parameters to use!')
print('='*70)

plt.close()
