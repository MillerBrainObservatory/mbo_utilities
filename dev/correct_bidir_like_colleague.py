"""
Bidirectional phase correction matching colleague's MATLAB approach.

KEY INSIGHTS FROM COLLEAGUE'S CODE:
1. Bidirectional scanning means EVEN rows are scanned in REVERSE direction
2. Must FLIP even rows before correction, then flip back
3. Shift EVEN rows (not odd)
4. Use sine pattern with zero correction in center region
5. Max shift ~9 pixels at edges
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tifffile


def apply_bidir_correction_colleague_method(frame, A=9, left_corr=0.4, right_corr=0.6,
                                           zero_center_start=130, zero_center_end=385):
    """
    Apply bidirectional correction following colleague's MATLAB code.

    Parameters
    ----------
    frame : ndarray (H, W)
        Input frame
    A : float
        Maximum shift in pixels at edges (default 9, matching colleague)
    left_corr : float
        Left edge parameter (default 0.4)
    right_corr : float
        Right edge parameter (default 0.6), should equal 1 - left_corr
    zero_center_start : int
        Start column for zero-correction region (default 130)
    zero_center_end : int
        End column for zero-correction region (default 385)

    Returns
    -------
    corrected : ndarray
        Corrected frame
    dx_col : ndarray
        Shift pattern applied (for visualization)
    """
    nr, nc = frame.shape

    # Build per-column shift field: zero in center, ±A at edges
    # Map columns to [left_corr*pi, right_corr*pi]
    xn = np.linspace(left_corr * np.pi, right_corr * np.pi, nc)
    dx_col = A * np.sin(xn)

    # Force central region to zero (no correction needed there)
    dx_col[zero_center_start:zero_center_end] = 0

    # Make a copy
    current_slice = frame.copy().astype(float)

    # Step 1: Flip reverse-scanned lines (even rows are backwards in bidirectional)
    current_slice[1::2, :] = np.fliplr(current_slice[1::2, :])

    # Step 2: Extract even rows (which are now flipped)
    even = current_slice[1::2, :]  # rows 1, 3, 5, ... (0-indexed, so these are "even" in 1-indexed)
    m = even.shape[0]
    even_corr = np.zeros_like(even)

    # Step 3: Apply variable horizontal shift to even rows via interpolation
    x = np.arange(nc)  # column indices
    x_src = x - dx_col  # where to sample from (per column)

    for r in range(m):
        # Linear interpolation with nearest-neighbor extrapolation at edges
        F = interp1d(x, even[r, :], kind='linear',
                    bounds_error=False, fill_value='extrapolate')
        even_corr[r, :] = F(x_src)

    # Step 4: Put corrected even rows back
    Icorr = current_slice.copy()
    Icorr[1::2, :] = even_corr

    # Step 5: Unflip both original and corrected (undo step 1)
    current_slice[1::2, :] = np.fliplr(current_slice[1::2, :])
    Icorr[1::2, :] = np.fliplr(Icorr[1::2, :])

    return Icorr, dx_col


# Load data
inpath = Path.home() / 'Documents/data/yao'
tif_files = sorted(inpath.glob('*.tif'))
original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
print(f'Loading: {original_tif.name}')
data = tifffile.imread(original_tif)
frame = data[0, 5, :, :]  # Z=0, T=5

print(f'Frame shape: {frame.shape}\n')

print('='*70)
print("COLLEAGUE'S METHOD: Bidirectional Correction")
print('='*70)
print('Key differences from our previous attempts:')
print('  1. FLIP even rows before correction (they are backwards!)')
print('  2. Shift EVEN rows, not odd rows')
print('  3. Zero correction in center columns (130-385)')
print('  4. Max shift A=9 pixels at edges')
print('  5. Flip back after correction')
print('='*70)

# Test multiple parameter combinations
test_cases = [
    {
        'name': "Colleague's exact params",
        'A': 9,
        'left_corr': 0.4,
        'right_corr': 0.6,
        'zero_start': 130,
        'zero_end': 385
    },
    {
        'name': 'A=7 (slightly less)',
        'A': 7,
        'left_corr': 0.4,
        'right_corr': 0.6,
        'zero_start': 130,
        'zero_end': 385
    },
    {
        'name': 'A=11 (slightly more)',
        'A': 11,
        'left_corr': 0.4,
        'right_corr': 0.6,
        'zero_start': 130,
        'zero_end': 385
    },
    {
        'name': 'Narrower zero region',
        'A': 9,
        'left_corr': 0.4,
        'right_corr': 0.6,
        'zero_start': 180,
        'zero_end': 332
    },
    {
        'name': 'Wider zero region',
        'A': 9,
        'left_corr': 0.4,
        'right_corr': 0.6,
        'zero_start': 100,
        'zero_end': 412
    },
    {
        'name': 'Asymmetric (left bias)',
        'A': 9,
        'left_corr': 0.3,
        'right_corr': 0.7,
        'zero_start': 130,
        'zero_end': 385
    },
]

# Visualize
n_cases = len(test_cases)
fig = plt.figure(figsize=(24, 4 * n_cases))

vmin, vmax = np.percentile(frame, [1, 99.5])

results = []

for i, params in enumerate(test_cases):
    print(f"\nTesting: {params['name']}")

    corrected, dx_col = apply_bidir_correction_colleague_method(
        frame,
        A=params['A'],
        left_corr=params['left_corr'],
        right_corr=params['right_corr'],
        zero_center_start=params['zero_start'],
        zero_center_end=params['zero_end']
    )

    results.append((params, corrected, dx_col))

    print(f"  Shift range: [{dx_col.min():.2f}, {dx_col.max():.2f}] px")

    # Row i: 5 subplots
    # 1. Shift pattern
    ax1 = plt.subplot(n_cases, 5, i*5 + 1)
    ax1.plot(dx_col, linewidth=2.5, color='blue')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axvspan(params['zero_start'], params['zero_end'],
               alpha=0.2, color='green', label='Zero region')
    ax1.set_ylabel('Shift (px)', fontsize=10)
    ax1.set_xlabel('Column (X)', fontsize=10)
    ax1.set_title(f"{params['name']}\nA={params['A']}, "
                 f"zero=[{params['zero_start']},{params['zero_end']}]\n"
                 f"Range: [{dx_col.min():.1f}, {dx_col.max():.1f}] px",
                 fontsize=10, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # 2. Original
    ax2 = plt.subplot(n_cases, 5, i*5 + 2)
    ax2.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('Original', fontsize=10)
    ax2.axis('off')

    # 3. Corrected
    ax3 = plt.subplot(n_cases, 5, i*5 + 3)
    ax3.imshow(corrected, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax3.set_title('Corrected', fontsize=10)
    ax3.axis('off')

    # 4. Neuron zoom (center-left, x~220)
    x_neuron = 220
    y = 192
    size = 110
    zoom_orig = frame[y:y+size, x_neuron:x_neuron+size]
    zoom_corr = corrected[y:y+size, x_neuron:x_neuron+size]
    combined = np.hstack([zoom_orig, zoom_corr])
    ax4 = plt.subplot(n_cases, 5, i*5 + 4)
    ax4.imshow(combined, cmap='gray', vmin=vmin, vmax=vmax)
    ax4.axvline(size, color='yellow', linestyle='--', linewidth=2)
    ax4.set_title(f'Neuron (x={x_neuron})\nOrig | Corr\nshift={dx_col[x_neuron]:.2f}px',
                 fontsize=9)
    ax4.axis('off')

    # 5. Right edge zoom
    x_right = 480
    zoom_orig_r = frame[y:y+size, x_right-size:x_right]
    zoom_corr_r = corrected[y:y+size, x_right-size:x_right]
    combined_r = np.hstack([zoom_orig_r, zoom_corr_r])
    ax5 = plt.subplot(n_cases, 5, i*5 + 5)
    ax5.imshow(combined_r, cmap='gray', vmin=vmin, vmax=vmax)
    ax5.axvline(size, color='yellow', linestyle='--', linewidth=2)
    ax5.set_title(f'Right Edge (x={x_right})\nOrig | Corr\nshift={dx_col[x_right]:.2f}px',
                 fontsize=9)
    ax5.axis('off')

plt.suptitle("COLLEAGUE'S METHOD: Proper Bidirectional Correction\n"
            "(Flip even rows → shift → flip back)\n"
            "Find which row looks best!",
            fontsize=14, weight='bold', y=0.995)

plt.tight_layout()

output_path = Path.home() / 'Documents/data/yao' / 'colleague_method_correction.png'
plt.savefig(output_path, dpi=120, facecolor='white', bbox_inches='tight')
print(f'\n\nSaved to: {output_path}')

# Save the exact colleague params version
colleague_exact = results[0]
tifffile.imwrite(inpath / 'corrected_colleague_method.tif',
                colleague_exact[1].astype(np.int16))
np.save(inpath / 'shift_pattern_colleague.npy', colleague_exact[2])

print(f"\nSaved colleague's exact method:")
print('  - corrected_colleague_method.tif')
print('  - shift_pattern_colleague.npy')

print('\n' + '='*70)
print('This should look MUCH better because we are now:')
print('  1. Flipping the backward-scanned rows')
print('  2. Shifting the correct rows (even, not odd)')
print('  3. Using zero correction in the center where it looks good')
print('='*70)

plt.close()
