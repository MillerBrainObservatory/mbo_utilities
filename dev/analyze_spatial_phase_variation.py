"""
Analyze spatial variation in bidirectional phase shift across the image.
This checks if the phase shift varies across the field of view, which would
indicate issues with scanner calibration or non-uniform scanning.
"""
from pathlib import Path
import mbo_utilities as mbo
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

# Load data
inpath = Path.home() / 'Documents/data/yao'
data = mbo.imread(inpath)
frame = data[14, 0, :, :]

print(f'Frame shape: {frame.shape}')
print(f'Frame range: [{frame.min()}, {frame.max()}]')

# Split into even and odd rows
pre = frame[::2]
post = frame[1::2]
m = min(pre.shape[0], post.shape[0])
a = pre[:m]
b = post[:m]

print(f'Even rows shape: {a.shape}')
print(f'Odd rows shape: {b.shape}')

# Analyze phase shift in sliding windows across the horizontal dimension
window_size = 64  # Window size
stride = 32  # Stride for sliding window
upsample = 10

h, w = a.shape
x_positions = []
y_positions = []
shifts = []
intensities = []

print(f'\nAnalyzing phase shift in {window_size}x{window_size} windows...')
print(f'Stride: {stride} pixels')

for y in range(0, h - window_size, stride):
    for x in range(0, w - window_size, stride):
        window_a = a[y:y+window_size, x:x+window_size]
        window_b = b[y:y+window_size, x:x+window_size]

        # Only analyze windows with sufficient signal
        mean_intensity = (window_a.mean() + window_b.mean()) / 2
        if mean_intensity < 50:  # Skip low-signal regions
            continue

        # Compute shift for this window
        try:
            shift_2d, error, phasediff = phase_cross_correlation(
                window_a, window_b,
                upsample_factor=upsample
            )

            # Store horizontal shift
            x_center = x + window_size // 2
            y_center = y + window_size // 2

            x_positions.append(x_center)
            y_positions.append(y_center)
            shifts.append(float(shift_2d[1]))  # Horizontal shift
            intensities.append(mean_intensity)
        except Exception as e:
            continue

x_positions = np.array(x_positions)
y_positions = np.array(y_positions)
shifts = np.array(shifts)
intensities = np.array(intensities)

print(f'\nAnalyzed {len(shifts)} windows')
print(f'Shift statistics:')
print(f'  Mean: {shifts.mean():.3f} pixels')
print(f'  Std:  {shifts.std():.3f} pixels')
print(f'  Min:  {shifts.min():.3f} pixels')
print(f'  Max:  {shifts.max():.3f} pixels')
print(f'  Range: {shifts.max() - shifts.min():.3f} pixels')

# Analyze trend across horizontal position
print(f'\nAnalyzing horizontal gradient...')

# Bin by X position
x_bins = np.linspace(0, w, 10)
x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
x_binned_shifts = []
x_binned_stds = []

for i in range(len(x_bins) - 1):
    mask = (x_positions >= x_bins[i]) & (x_positions < x_bins[i+1])
    if mask.sum() > 0:
        x_binned_shifts.append(shifts[mask].mean())
        x_binned_stds.append(shifts[mask].std())
    else:
        x_binned_shifts.append(np.nan)
        x_binned_stds.append(np.nan)

x_binned_shifts = np.array(x_binned_shifts)
x_binned_stds = np.array(x_binned_stds)

# Linear fit to see if there's a gradient
valid_mask = ~np.isnan(x_binned_shifts)
if valid_mask.sum() > 2:
    coeffs = np.polyfit(x_bin_centers[valid_mask], x_binned_shifts[valid_mask], 1)
    print(f'Linear fit: shift = {coeffs[0]:.6f} * x + {coeffs[1]:.3f}')
    print(f'Predicted shift change left-to-right: {coeffs[0] * w:.3f} pixels')

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Original frame with measurement locations
ax1 = plt.subplot(2, 3, 1)
vmin, vmax = np.percentile(frame, [1, 99.5])
ax1.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
ax1.scatter(x_positions, y_positions * 2, c=shifts, s=30, cmap='RdBu_r',
           vmin=-2, vmax=2, edgecolors='yellow', linewidth=0.5, alpha=0.8)
ax1.set_title('Measurement Locations\n(colored by shift)', fontsize=12)
ax1.set_xlabel('X position (pixels)')
ax1.set_ylabel('Y position (pixels)')
cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
cbar1.set_label('Shift (pixels)', rotation=270, labelpad=15)

# 2. Shift map (interpolated)
ax2 = plt.subplot(2, 3, 2)
# Create a grid for interpolation
from scipy.interpolate import griddata
grid_x, grid_y = np.meshgrid(np.linspace(0, w, 100), np.linspace(0, h//2, 100))
grid_shifts = griddata((x_positions, y_positions), shifts, (grid_x, grid_y), method='cubic')

im2 = ax2.imshow(grid_shifts, extent=[0, w, 0, h//2], origin='lower',
                cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
ax2.set_title('Interpolated Shift Map', fontsize=12)
ax2.set_xlabel('X position (pixels)')
ax2.set_ylabel('Y position (even rows)')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Shift (pixels)', rotation=270, labelpad=15)

# 3. Scatter plot: shift vs X position
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(x_positions, shifts, c=intensities, s=20, alpha=0.6, cmap='viridis')
ax3.axhline(0, color='red', linestyle='--', alpha=0.5, label='Zero shift')
ax3.set_xlabel('X position (pixels)')
ax3.set_ylabel('Shift (pixels)')
ax3.set_title('Shift vs Horizontal Position', fontsize=12)
ax3.grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter, ax=ax3)
cbar3.set_label('Window intensity', rotation=270, labelpad=15)

# Add trend line
if valid_mask.sum() > 2:
    x_trend = np.array([0, w])
    y_trend = coeffs[0] * x_trend + coeffs[1]
    ax3.plot(x_trend, y_trend, 'r-', linewidth=2, alpha=0.7,
            label=f'Linear fit: {coeffs[0]:.5f}*x + {coeffs[1]:.2f}')
    ax3.legend()

# 4. Binned shifts across X
ax4 = plt.subplot(2, 3, 4)
ax4.errorbar(x_bin_centers[valid_mask], x_binned_shifts[valid_mask],
            yerr=x_binned_stds[valid_mask], fmt='o-', capsize=5, markersize=8)
ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('X position (pixels)')
ax4.set_ylabel('Mean shift (pixels)')
ax4.set_title('Binned Shifts Across X-axis', fontsize=12)
ax4.grid(True, alpha=0.3)

# 5. Histogram of shifts
ax5 = plt.subplot(2, 3, 5)
ax5.hist(shifts, bins=50, edgecolor='black', alpha=0.7)
ax5.axvline(shifts.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {shifts.mean():.3f} px')
ax5.axvline(np.median(shifts), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {np.median(shifts):.3f} px')
ax5.set_xlabel('Shift (pixels)')
ax5.set_ylabel('Count')
ax5.set_title('Distribution of Shifts', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Shift vs Y position (check for vertical variation)
ax6 = plt.subplot(2, 3, 6)
scatter2 = ax6.scatter(y_positions, shifts, c=x_positions, s=20, alpha=0.6, cmap='plasma')
ax6.axhline(0, color='red', linestyle='--', alpha=0.5)
ax6.set_xlabel('Y position (even rows)')
ax6.set_ylabel('Shift (pixels)')
ax6.set_title('Shift vs Vertical Position', fontsize=12)
ax6.grid(True, alpha=0.3)
cbar6 = plt.colorbar(scatter2, ax=ax6)
cbar6.set_label('X position', rotation=270, labelpad=15)

plt.tight_layout()

output_path = Path.home() / 'Documents/data/yao' / 'spatial_phase_analysis.png'
plt.savefig(output_path, dpi=150, facecolor='white')
print(f'\nSaved analysis to: {output_path}')

# Create a detailed text report
print('\n' + '='*70)
print('SPATIAL PHASE SHIFT ANALYSIS')
print('='*70)

print(f'\nImage size: {frame.shape[1]} x {frame.shape[0]} pixels')
print(f'Windows analyzed: {len(shifts)}')

print(f'\nShift statistics across entire image:')
print(f'  Mean ± Std:  {shifts.mean():.3f} ± {shifts.std():.3f} pixels')
print(f'  Median:      {np.median(shifts):.3f} pixels')
print(f'  Range:       [{shifts.min():.3f}, {shifts.max():.3f}] pixels')
print(f'  Total range: {shifts.max() - shifts.min():.3f} pixels')

if valid_mask.sum() > 2:
    print(f'\nHorizontal gradient analysis:')
    print(f'  Slope:       {coeffs[0]:.6f} pixels/pixel')
    print(f'  Intercept:   {coeffs[1]:.3f} pixels')
    print(f'  Left edge (~x=0):     {coeffs[1]:.3f} pixels')
    print(f'  Right edge (~x={w}): {coeffs[0]*w + coeffs[1]:.3f} pixels')
    print(f'  TOTAL CHANGE:         {coeffs[0]*w:.3f} pixels')

# Check if this is problematic
if shifts.std() > 0.5:
    print(f'\nWARNING  WARNING: Large spatial variation detected!')
    print(f'   Standard deviation of {shifts.std():.3f} pixels indicates')
    print(f'   the phase shift is NOT uniform across the image.')

if valid_mask.sum() > 2 and abs(coeffs[0] * w) > 2:
    print(f'\nWARNING  CRITICAL: Strong horizontal gradient detected!')
    print(f'   Shift changes by {coeffs[0]*w:.3f} pixels from left to right.')
    print(f'   This suggests scanner calibration issues or non-uniform sampling.')

print('\n' + '='*70)
print('INTERPRETATION:')
print('='*70)
print('If the shift varies significantly across the image (>1-2 pixels),')
print('a single global shift correction will NOT work properly.')
print('You may need:')
print('  1. Per-column shift correction (different shift for each X position)')
print('  2. Scanner recalibration (if hardware issue)')
print('  3. Non-uniform sampling correction')
print('='*70)

# plt.show()  # Don't show interactively, just save
plt.close('all')
