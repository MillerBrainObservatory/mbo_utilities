"""
Exact reproduction of colleague's MATLAB code.
Testing step-by-step to find the bug.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tifffile


# Load data
inpath = Path.home() / 'Documents/data/yao'
tif_files = sorted(inpath.glob('*.tif'))
original_tif = [f for f in tif_files if 'corrected' not in f.name.lower()][0]
print(f'Loading: {original_tif.name}')
data = tifffile.imread(original_tif)
I = data[0, 5, :, :].astype(float)  # Use float like MATLAB does with double()

print(f'Frame shape: {I.shape}')
nr, nc = I.shape

# Parameters from MATLAB
A = 9
LeftCorr = -0.4
RightCorr = 1 - LeftCorr  # This gives 1.4, NOT 0.6!

print(f'\nParameters:')
print(f'  A = {A}')
print(f'  LeftCorr = {LeftCorr}')
print(f'  RightCorr = {RightCorr}')

# Build per-column shift field dx(x): zero in center, ±A at edges
xn = np.linspace(LeftCorr * np.pi, RightCorr * np.pi, nc)
dx_col = A * np.sin(xn)
dx_col[130:385] = 0  # No need to correct central columns

print(f'  dx_col range: [{dx_col.min():.3f}, {dx_col.max():.3f}]')
print(f'  dx_col shape: {dx_col.shape}')

# DEBUG: Check a few values
print(f'\n  dx_col[0] = {dx_col[0]:.3f}')
print(f'  dx_col[130] = {dx_col[130]:.3f}')
print(f'  dx_col[256] = {dx_col[256]:.3f}')
print(f'  dx_col[384] = {dx_col[384]:.3f}')
print(f'  dx_col[511] = {dx_col[511]:.3f}')

# Step 0: Flip reverse-scanned lines (bidirectional)
# MATLAB: currentSlice(2:2:end, :) = fliplr(currentSlice(2:2:end, :));
# In Python 0-indexed: rows 1,3,5,... are the "even" rows in 1-indexed MATLAB
currentSlice = I.copy()
currentSlice[1::2, :] = np.fliplr(currentSlice[1::2, :])

print(f'\nStep 0: Flipped even rows (1::2)')

# Apply variable horizontal shift to EVEN rows via interpolation
even = currentSlice[1::2, :]  # rows 1,3,5,...
m = even.shape[0]
even_corr = np.zeros_like(even)

print(f'Even rows shape: {even.shape}')

x = np.arange(nc)  # 0, 1, 2, ..., nc-1
x_src = x - dx_col  # where to sample from (per column)

print(f'\nSample x_src values:')
print(f'  x_src[0] = {x_src[0]:.3f} (shift by {dx_col[0]:.3f})')
print(f'  x_src[256] = {x_src[256]:.3f} (shift by {dx_col[256]:.3f})')
print(f'  x_src[511] = {x_src[511]:.3f} (shift by {dx_col[511]:.3f})')

# Interpolate row by row
for r in range(m):
    # MATLAB: F = griddedInterpolant(x, even(r, :), 'linear', 'nearest');
    # 'nearest' is the extrapolation method for out-of-bounds
    F = interp1d(x, even[r, :], kind='linear',
                bounds_error=False, fill_value='extrapolate')
    even_corr[r, :] = F(x_src)

print(f'\nApplied correction to {m} even rows')

# Put corrected even rows back and unflip
Icorr = currentSlice.copy()
Icorr[1::2, :] = even_corr

# Unflip step 0
currentSlice[1::2, :] = np.fliplr(currentSlice[1::2, :])
Icorr[1::2, :] = np.fliplr(Icorr[1::2, :])

print(f'Step 5: Unflipped rows')

# Visualize side-by-side
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

vmin, vmax = np.percentile(I, [1, 99.5])

# Row 1: Full images
axes[0, 0].imshow(I, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
axes[0, 0].set_title('Original', fontsize=14, weight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(Icorr, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
axes[0, 1].set_title('Corrected (Colleague Method)', fontsize=14, weight='bold')
axes[0, 1].axis('off')

diff = Icorr - I
axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-300, vmax=300, aspect='auto')
axes[0, 2].set_title('Difference', fontsize=14, weight='bold')
axes[0, 2].axis('off')

# Row 2: Shift pattern and zooms
axes[1, 0].plot(dx_col, linewidth=2, color='blue')
axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axvspan(130, 385, alpha=0.2, color='green', label='Zero region')
axes[1, 0].set_xlabel('Column', fontsize=12)
axes[1, 0].set_ylabel('Shift (pixels)', fontsize=12)
axes[1, 0].set_title('Shift Pattern', fontsize=14, weight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Zoom neuron (x~220)
x_neuron = 220
y = 192
size = 110
zoom_orig = I[y:y+size, x_neuron:x_neuron+size]
zoom_corr = Icorr[y:y+size, x_neuron:x_neuron+size]
combined = np.hstack([zoom_orig, zoom_corr])
axes[1, 1].imshow(combined, cmap='gray', vmin=vmin, vmax=vmax)
axes[1, 1].axvline(size, color='yellow', linestyle='--', linewidth=2)
axes[1, 1].set_title(f'Neuron (x={x_neuron}): Orig | Corr\nshift={dx_col[x_neuron]:.2f}px',
                    fontsize=12, weight='bold')
axes[1, 1].axis('off')

# Zoom right edge
x_right = 480
zoom_orig_r = I[y:y+size, x_right-size:x_right]
zoom_corr_r = Icorr[y:y+size, x_right-size:x_right]
combined_r = np.hstack([zoom_orig_r, zoom_corr_r])
axes[1, 2].imshow(combined_r, cmap='gray', vmin=vmin, vmax=vmax)
axes[1, 2].axvline(size, color='yellow', linestyle='--', linewidth=2)
axes[1, 2].set_title(f'Right Edge (x={x_right}): Orig | Corr\nshift={dx_col[x_right]:.2f}px',
                    fontsize=12, weight='bold')
axes[1, 2].axis('off')

plt.tight_layout()

output_path = Path.home() / 'Documents/data/yao' / 'colleague_exact_test.png'
plt.savefig(output_path, dpi=150, facecolor='white')
print(f'\nSaved to: {output_path}')

# Save
tifffile.imwrite(inpath / 'corrected_colleague_exact_test.tif', Icorr.astype(np.int16))

print('\n' + '='*70)
print('EXACT COLLEAGUE METHOD TEST')
print('='*70)
print('If this still looks wrong, the problem might be:')
print('  1. Different frame being used')
print('  2. Data already partially corrected')
print('  3. Different scanning direction convention')
print('='*70)

plt.show()
