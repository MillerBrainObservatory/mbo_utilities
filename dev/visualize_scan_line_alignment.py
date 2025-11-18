"""
Visualize scan line alignment by showing even/odd rows separately.
This makes the bidirectional phase offset clearly visible.
"""
from pathlib import Path
import mbo_utilities as mbo
import numpy as np
import matplotlib.pyplot as plt

# Load data
inpath = Path.home() / 'Documents/data/yao'
data = mbo.imread(inpath)
frame = data[14, 0, :, :]

print(f'Frame shape: {frame.shape}')

# Pick a region with good signal and structure
y_start, x_start = 180, 220
height, width = 100, 150

region = frame[y_start:y_start+height, x_start:x_start+width]

# Test different shift amounts
test_shifts = [0, -0.5, -0.8, -1.0, -1.2, -1.5]

fig, axes = plt.subplots(len(test_shifts), 3, figsize=(15, len(test_shifts)*4))

for i, shift in enumerate(test_shifts):
    # Apply global shift for testing
    if shift == 0:
        corrected = frame.copy()
    else:
        corrected = mbo.phasecorr.apply_scan_phase_offsets(frame, shift)

    # Extract region
    corrected_region = corrected[y_start:y_start+height, x_start:x_start+width]

    # Split into even and odd rows
    even_rows = corrected_region[::2, :]
    odd_rows = corrected_region[1::2, :]

    # Upscale for visualization (interleave with black rows)
    even_vis = np.zeros((corrected_region.shape[0], corrected_region.shape[1]))
    even_vis[::2, :] = even_rows

    odd_vis = np.zeros((corrected_region.shape[0], corrected_region.shape[1]))
    odd_vis[1::2, :] = odd_rows

    vmin, vmax = np.percentile(corrected_region, [5, 99])

    # Show even rows only (cyan)
    axes[i, 0].imshow(even_vis, cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f'Shift={shift:.1f}px: EVEN rows', fontsize=12, weight='bold')
    axes[i, 0].axis('off')

    # Show odd rows only (magenta)
    axes[i, 1].imshow(odd_vis, cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f'Shift={shift:.1f}px: ODD rows', fontsize=12, weight='bold')
    axes[i, 1].axis('off')

    # Show both overlaid (look for misalignment)
    axes[i, 2].imshow(corrected_region, cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, 2].set_title(f'Shift={shift:.1f}px: COMBINED', fontsize=12, weight='bold')
    axes[i, 2].axis('off')

    # Add horizontal lines to highlight scan lines
    if i == 0:  # Only on first row
        for row in range(0, height, 10):
            axes[i, 2].axhline(row, color='red', alpha=0.2, linewidth=0.5)

plt.tight_layout()
output_path = Path.home() / 'Documents/data/yao' / 'scan_line_alignment.png'
plt.savefig(output_path, dpi=150, facecolor='white')
print(f'Saved to: {output_path}')
# plt.close()  # Don't close, let it display

# Now create a more detailed view of just one horizontal line profile
# to see the jaggedness
fig, axes = plt.subplots(len(test_shifts), 1, figsize=(16, len(test_shifts)*2))

for i, shift in enumerate(test_shifts):
    if shift == 0:
        corrected = frame.copy()
    else:
        corrected = mbo.phasecorr.apply_scan_phase_offsets(frame, shift)

    # Pick a single column and plot even vs odd row values
    col = x_start + width // 2  # Middle of our region

    # Get a vertical slice
    even_profile = corrected[y_start:y_start+height:2, col]
    odd_profile = corrected[y_start+1:y_start+height:2, col]

    # Plot
    y_even = np.arange(0, height, 2)
    y_odd = np.arange(1, height, 2)

    axes[i].plot(y_even, even_profile, 'o-', label='Even rows', markersize=4, linewidth=1.5, alpha=0.7)
    axes[i].plot(y_odd, odd_profile, 's-', label='Odd rows', markersize=4, linewidth=1.5, alpha=0.7)
    axes[i].set_xlabel('Row position', fontsize=10)
    axes[i].set_ylabel('Intensity', fontsize=10)
    axes[i].set_title(f'Shift={shift:.1f}px: Vertical profile at x={col}', fontsize=12, weight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
profile_path = Path.home() / 'Documents/data/yao' / 'scan_line_profiles.png'
plt.savefig(profile_path, dpi=150, facecolor='white')
print(f'Saved to: {profile_path}')
# plt.close()  # Don't close

print("\n" + "="*70)
print("HOW TO READ THESE IMAGES:")
print("="*70)
print("1. scan_line_alignment.png:")
print("   - Left column: even scan lines only")
print("   - Middle column: odd scan lines only")
print("   - Right column: both combined")
print("   - Look for the shift where even/odd rows ALIGN BEST")
print("")
print("2. scan_line_profiles.png:")
print("   - Shows intensity along a vertical line")
print("   - Even and odd rows should track each other")
print("   - Look for minimum offset between the two curves")
print("="*70)
