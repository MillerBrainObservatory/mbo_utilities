"""
Debug sine fitting - figure out why fit is failing
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
import tifffile

# Load data
inpath = Path.home() / 'Documents/data/yao'
tif_files = sorted(inpath.glob('*.tif'))
original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
data = tifffile.imread(original_tif)
frame = data[0, 5, :, :].astype(float)

print(f'Frame: {frame.shape}')

# Measure shifts (same as notebook)
def measure_shifts(frame, window_size=64, stride=24):
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    a, b = pre[:m], post[:m]
    h, w = a.shape

    thresh = np.percentile(frame, 55)

    x_pos, shifts, intensities = [], [], []

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            win_a = a[y:y+window_size, x:x+window_size]
            win_b = b[y:y+window_size, x:x+window_size]

            mean_int = (win_a.mean() + win_b.mean()) / 2
            if mean_int < thresh:
                continue

            try:
                shift_2d, _, _ = phase_cross_correlation(win_a, win_b,
                                                         upsample_factor=10,
                                                         normalization=None)
                h_shift = float(shift_2d[1])

                if abs(h_shift) <= 10:
                    x_pos.append(x + window_size // 2)
                    shifts.append(h_shift)
                    intensities.append(mean_int)
            except:
                continue

    return np.array(x_pos), np.array(shifts), np.array(intensities)

x_pos, shifts, intensities = measure_shifts(frame)

print(f'\nMeasurements: {len(shifts)}')
print(f'X range: [{x_pos.min()}, {x_pos.max()}]')
print(f'Shift range: [{shifts.min():.2f}, {shifts.max():.2f}]')
print(f'Shift mean: {shifts.mean():.2f}')
print(f'Shift std: {shifts.std():.2f}')

# Remove outliers
median_shift = np.median(shifts)
mad = np.median(np.abs(shifts - median_shift))
mask = np.abs(shifts - median_shift) < 3 * mad

print(f'\nOutliers removed: {(~mask).sum()}')
x_clean = x_pos[mask]
shifts_clean = shifts[mask]
weights = intensities[mask]

print(f'Clean shift range: [{shifts_clean.min():.2f}, {shifts_clean.max():.2f}]')

# Try different fitting approaches

print('\n' + '='*70)
print('ATTEMPT 1: Fit sine using pixel indices directly')
print('='*70)

def sine_model_v1(x_pixels, A, left_corr, right_corr):
    """This is what the notebook does - WRONG because indexing"""
    xn = np.linspace(left_corr * np.pi, right_corr * np.pi, frame.shape[1])
    return A * np.sin(xn[x_pixels.astype(int)])

try:
    params1, _ = curve_fit(sine_model_v1, x_clean, shifts_clean,
                          p0=[5, -0.4, 1.4],
                          sigma=1.0/weights,
                          maxfev=10000,
                          bounds=([-15, -1, 0], [15, 0, 2]))
    A1, left1, right1 = params1
    print(f'SUCCESS: A={A1:.3f}, left={left1:.3f}, right={right1:.3f}')
except Exception as e:
    print(f'FAILED: {e}')
    A1, left1, right1 = None, None, None

print('\n' + '='*70)
print('ATTEMPT 2: Fit sine with normalized x')
print('='*70)

def sine_model_v2(x_norm, A, left_corr, right_corr):
    """Use normalized x [0, 1]"""
    xn = left_corr * np.pi + x_norm * (right_corr - left_corr) * np.pi
    return A * np.sin(xn)

x_norm = x_clean / frame.shape[1]

try:
    params2, _ = curve_fit(sine_model_v2, x_norm, shifts_clean,
                          p0=[5, -0.4, 1.4],
                          sigma=1.0/weights,
                          maxfev=10000,
                          bounds=([-15, -1, 0], [15, 0, 2]))
    A2, left2, right2 = params2
    print(f'SUCCESS: A={A2:.3f}, left={left2:.3f}, right={right2:.3f}')

    # Generate full pattern
    x_all_norm = np.arange(frame.shape[1]) / frame.shape[1]
    dx_col_v2 = sine_model_v2(x_all_norm, *params2)
    print(f'Pattern range: [{dx_col_v2.min():.2f}, {dx_col_v2.max():.2f}]')
except Exception as e:
    print(f'FAILED: {e}')
    A2, left2, right2 = None, None, None

print('\n' + '='*70)
print('ATTEMPT 3: Use Yao exact method - linspace THEN evaluate')
print('='*70)

# Just use Yao's exact parameters
A3, left3, right3 = 9, -0.4, 1.4
xn3 = np.linspace(left3 * np.pi, right3 * np.pi, frame.shape[1])
dx_col_v3 = A3 * np.sin(xn3)
print(f'Yao params: A={A3}, left={left3}, right={right3}')
print(f'Pattern range: [{dx_col_v3.min():.2f}, {dx_col_v3.max():.2f}]')

# Plot all attempts
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Measurements
axes[0, 0].scatter(x_pos, shifts, c='red', s=20, alpha=0.4, label='All measurements')
axes[0, 0].scatter(x_clean, shifts_clean, c='blue', s=20, alpha=0.6, label='After outlier removal')
axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_xlabel('X position')
axes[0, 0].set_ylabel('Measured shift (px)')
axes[0, 0].set_title('Measurements')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Attempt 2 (normalized x)
if A2 is not None:
    axes[0, 1].plot(dx_col_v2, linewidth=2, label=f'Fit: A={A2:.2f}', color='blue')
    axes[0, 1].scatter(x_pos, shifts, c='red', s=15, alpha=0.4, label='Measurements')
    axes[0, 1].set_title(f'Attempt 2 (normalized x fit)\nA={A2:.2f}, left={left2:.2f}, right={right2:.2f}')
else:
    axes[0, 1].text(0.5, 0.5, 'Fit failed', ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Attempt 2 (FAILED)')
axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Attempt 3 (Yao exact)
axes[1, 0].plot(dx_col_v3, linewidth=2, label=f'Yao: A={A3}', color='green')
axes[1, 0].scatter(x_pos, shifts, c='red', s=15, alpha=0.4, label='Measurements')
axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_title(f'Attempt 3 (Yao exact params)\nA={A3}, left={left3}, right={right3}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Comparison
if A2 is not None:
    axes[1, 1].plot(dx_col_v2, linewidth=2, label='Fitted', color='blue', alpha=0.7)
axes[1, 1].plot(dx_col_v3, linewidth=2, label='Yao exact', color='green', alpha=0.7)
axes[1, 1].scatter(x_pos, shifts, c='red', s=15, alpha=0.4, label='Measurements')
axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_title('Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path.home() / 'Documents/data/yao' / 'debug_sine_fit.png'
plt.savefig(output_path, dpi=150, facecolor='white')
print(f'\nSaved debug plot: {output_path}')

print('\n' + '='*70)
print('CONCLUSION:')
print('='*70)
print('The notebook should use ATTEMPT 2 or just use Yao exact params.')
print('The indexing approach (ATTEMPT 1) is broken.')
print('='*70)

plt.show()
