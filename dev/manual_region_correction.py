"""
Manual region-based phase correction.
User specifies shift values for key X positions, we interpolate between them.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
import tifffile


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


# Load data
inpath = Path.home() / 'Documents/data/yao'
tif_files = sorted(inpath.glob('*.tif'))
original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
print(f'Loading: {original_tif.name}')
data = tifffile.imread(original_tif)
frame = data[0, 5, :, :]

print(f'Frame shape: {frame.shape}\n')

# MANUAL CONTROL POINTS
# Based on your observations:
# - Neuron left side (x~180): needs ~3-4 more pixels than current
# - Neuron right side (x~260): looked okay with some previous attempts
# - Right edge (x~450-480): needs ~7 pixels
#
# Let's define control points (x_position, shift_in_pixels)

print('='*70)
print('MANUAL CONTROL POINT PHASE CORRECTION')
print('='*70)
print('Define shift values at key X positions.')
print('We will interpolate smoothly between them.')
print('='*70)

# Test multiple control point configurations
test_configs = [
    {
        'name': 'Config 1: Based on your feedback',
        'description': 'Left needs more shift, right edge ~7px',
        'points': [
            (0, 5.0),      # Far left edge
            (180, 3.5),    # Neuron left side
            (260, -0.8),   # Neuron right/center
            (450, 7.0),    # Right edge
            (511, 7.5),    # Far right
        ]
    },
    {
        'name': 'Config 2: Steeper left gradient',
        'description': 'More aggressive left side correction',
        'points': [
            (0, 8.0),      # Far left edge (aggressive)
            (180, 4.0),    # Neuron left side
            (260, -0.8),   # Neuron center
            (400, 5.0),    # Right-ish
            (511, 7.0),    # Far right
        ]
    },
    {
        'name': 'Config 3: Center-focused',
        'description': 'Keep center near 0, increase edges',
        'points': [
            (0, 6.0),      # Left edge
            (100, 2.0),
            (256, 0.0),    # Dead center at 0
            (400, 4.0),
            (511, 7.0),    # Right edge
        ]
    },
    {
        'name': 'Config 4: Symmetric V-shape',
        'description': 'High at edges, low in center',
        'points': [
            (0, 7.0),      # Left edge
            (128, 3.0),
            (256, -1.0),   # Center low
            (384, 3.0),
            (511, 7.0),    # Right edge
        ]
    },
    {
        'name': 'Config 5: Inverted (just in case)',
        'description': 'What if we need to shift the other direction?',
        'points': [
            (0, -5.0),     # Negative shifts
            (180, -3.5),
            (260, 0.8),
            (450, -7.0),
            (511, -7.5),
        ]
    },
    {
        'name': 'Config 6: Large range exploration',
        'description': 'Very large shifts to see if we need more',
        'points': [
            (0, 10.0),     # Very large
            (128, 6.0),
            (256, 0.0),
            (384, 5.0),
            (511, 9.0),
        ]
    },
]

# Interpolation methods to try
interp_methods = ['linear', 'pchip']

# Generate all test cases
all_test_cases = []

for config in test_configs:
    points = config['points']
    x_points = np.array([p[0] for p in points])
    shift_points = np.array([p[1] for p in points])

    for method in interp_methods:
        if method == 'linear':
            interp = interp1d(x_points, shift_points, kind='linear',
                            fill_value='extrapolate')
        elif method == 'pchip':
            # PCHIP: monotonic cubic interpolation
            interp = PchipInterpolator(x_points, shift_points)

        x_all = np.arange(frame.shape[1])
        shift_pattern = interp(x_all)

        all_test_cases.append({
            'name': f"{config['name']} ({method})",
            'description': config['description'],
            'points': points,
            'shift_pattern': shift_pattern,
            'method': method
        })

# Visualize
n_cases = len(all_test_cases)
fig = plt.figure(figsize=(24, 3.5 * n_cases))

vmin, vmax = np.percentile(frame, [1, 99.5])

for i, test_case in enumerate(all_test_cases):
    shift_i = test_case['shift_pattern']
    points = test_case['points']

    # Apply correction
    corrected_i = apply_column_phase_correction(frame, shift_i)

    # Row i: 5 subplots
    # 1. Shift pattern
    ax1 = plt.subplot(n_cases, 5, i*5 + 1)
    ax1.plot(shift_i, linewidth=2, color='blue', label='Interpolated')
    # Plot control points
    x_ctrl = [p[0] for p in points]
    y_ctrl = [p[1] for p in points]
    ax1.scatter(x_ctrl, y_ctrl, c='red', s=80, zorder=10,
               edgecolors='black', linewidth=1.5, label='Control points')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Shift (px)', fontsize=9)
    ax1.set_xlabel('X position', fontsize=9)
    ax1.set_title(f"{test_case['name']}\n{test_case['description']}\n"
                 f"Range: [{shift_i.min():.1f}, {shift_i.max():.1f}] px",
                 fontsize=9, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc='best')

    # 2. Original
    ax2 = plt.subplot(n_cases, 5, i*5 + 2)
    ax2.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('Original', fontsize=9)
    ax2.axis('off')

    # 3. Corrected full
    ax3 = plt.subplot(n_cases, 5, i*5 + 3)
    ax3.imshow(corrected_i, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax3.set_title('Corrected', fontsize=9)
    ax3.axis('off')

    # 4. Neuron left (x~180)
    x_neuron = 180
    y = 192
    size = 110
    zoom_neuron = corrected_i[y:y+size, x_neuron:x_neuron+size]
    ax4 = plt.subplot(n_cases, 5, i*5 + 4)
    ax4.imshow(zoom_neuron, cmap='gray', vmin=vmin, vmax=vmax)
    ax4.set_title(f'Neuron Left\nx={x_neuron}, shift={shift_i[x_neuron]:.2f}px',
                 fontsize=8)
    ax4.axis('off')

    # 5. Right edge (x~480)
    x_right = 480
    zoom_right = corrected_i[y:y+size, x_right-size:x_right]
    ax5 = plt.subplot(n_cases, 5, i*5 + 5)
    ax5.imshow(zoom_right, cmap='gray', vmin=vmin, vmax=vmax)
    ax5.set_title(f'Right Edge\nx={x_right}, shift={shift_i[x_right]:.2f}px',
                 fontsize=8)
    ax5.axis('off')

plt.suptitle('Manual Control Point Phase Correction\n'
            'USER: Find the row where BOTH neuron and right edge look best!\n'
            'Then tell me which config # and we can refine the control points.',
            fontsize=13, weight='bold', y=0.997)

plt.tight_layout()

output_path = Path.home() / 'Documents/data/yao' / 'manual_control_points.png'
plt.savefig(output_path, dpi=110, facecolor='white', bbox_inches='tight')
print(f'\nSaved to: {output_path}')

# Save a default version (Config 1 PCHIP)
config1_pchip = [tc for tc in all_test_cases if 'Config 1' in tc['name'] and 'pchip' in tc['name']][0]
corrected_default = apply_column_phase_correction(frame, config1_pchip['shift_pattern'])
tifffile.imwrite(inpath / 'corrected_manual_config1.tif', corrected_default.astype(np.int16))
np.save(inpath / 'shift_pattern_manual_config1.npy', config1_pchip['shift_pattern'])

print(f'\nSaved Config 1 (PCHIP) as default:')
print('  - corrected_manual_config1.tif')
print('  - shift_pattern_manual_config1.npy')

print('\n' + '='*70)
print('NEXT STEPS:')
print('1. Review manual_control_points.png')
print('2. Find which Config # looks best')
print('3. Tell me: "Config X looks closest, but needs adjustment at..."')
print('4. I will create refined versions with your feedback')
print('='*70)

plt.close()
