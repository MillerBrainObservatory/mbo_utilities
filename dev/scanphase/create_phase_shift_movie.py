"""
Create a movie showing different bidirectional phase shifts for visual comparison.
Each frame shows the image with a different shift applied, with the shift value overlaid.
"""
from pathlib import Path
import mbo_utilities as mbo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
import tifffile

# Load data
inpath = Path.home() / 'Documents/data/yao'
data = mbo.imread(inpath)
frame = data[14, 0, :, :]

print(f'Frame shape: {frame.shape}')
print(f'Frame range: [{frame.min()}, {frame.max()}]')

# Generate shifts to test
shifts = np.linspace(-0.5, -1.5, num=21)  # 21 frames from -0.5 to -1.5
print(f'\nGenerating {len(shifts)} frames with shifts from {shifts[0]:.2f} to {shifts[-1]:.2f}')

# Apply each shift and collect frames
frames_with_shifts = []
for shift in shifts:
    shifted_frame = mbo.phasecorr.apply_scan_phase_offsets(frame, shift)
    frames_with_shifts.append((shift, shifted_frame))

# Create output directory
output_dir = Path.home() / 'Documents/data/yao'
output_dir.mkdir(parents=True, exist_ok=True)

# Method 1: Save as animated GIF
print('\nCreating animated GIF...')
fig, ax = plt.subplots(figsize=(10, 10))

# Use a good colormap for 2P data
vmin, vmax = np.percentile(frame, [1, 99.5])

im = ax.imshow(frames_with_shifts[0][1], cmap='gray', vmin=vmin, vmax=vmax)
ax.axis('off')

# Add text overlay for shift value
text = ax.text(0.05, 0.95, f'Shift: {frames_with_shifts[0][0]:.3f} px',
               transform=ax.transAxes,
               fontsize=20,
               color='yellow',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

def update(frame_idx):
    shift, shifted_frame = frames_with_shifts[frame_idx]
    im.set_array(shifted_frame)
    text.set_text(f'Shift: {shift:.3f} px')
    return [im, text]

ani = animation.FuncAnimation(fig, update, frames=len(frames_with_shifts),
                             interval=300, blit=True, repeat=True)

gif_path = output_dir / 'phase_shift_comparison.gif'
ani.save(gif_path, writer='pillow', fps=3, dpi=100)
print(f'Saved GIF to: {gif_path}')
plt.close()

# Method 2: Save as multi-page TIFF with text overlay
print('\nCreating annotated multi-page TIFF...')
annotated_frames = []
for shift, shifted_frame in frames_with_shifts:
    # Create figure for this frame
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(shifted_frame, cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.text(0.05, 0.95, f'Shift: {shift:.3f} px',
           transform=ax.transAxes,
           fontsize=20,
           color='yellow',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Convert to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)[:, :, :3]  # Drop alpha channel
    annotated_frames.append(img_array)
    plt.close(fig)

tiff_path = output_dir / 'phase_shift_comparison_annotated.tif'
tifffile.imwrite(tiff_path, np.array(annotated_frames), photometric='rgb')
print(f'Saved annotated TIFF to: {tiff_path}')

# Method 3: Save as multi-page TIFF (raw data, no annotation) for ImageJ
print('\nCreating raw multi-page TIFF (for ImageJ)...')
raw_frames = np.array([sf for _, sf in frames_with_shifts])
raw_tiff_path = output_dir / 'phase_shift_comparison_raw.tif'

# Create ImageJ-compatible metadata
metadata = {
    'axes': 'TYX',
    'fps': 3,
}

# Add shift values to description
description_lines = [f'Frame {i}: shift = {shift:.4f} px'
                    for i, (shift, _) in enumerate(frames_with_shifts)]
metadata['description'] = '\n'.join(description_lines)

tifffile.imwrite(raw_tiff_path, raw_frames.astype(np.int16),
                metadata=metadata, imagej=True)
print(f'Saved raw TIFF to: {raw_tiff_path}')

# Method 4: Create side-by-side comparison of a few key shifts
print('\nCreating side-by-side comparison figure...')
key_shifts = [-0.5, -0.7, -0.9, -1.1, -1.3, -1.5]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, target_shift in enumerate(key_shifts):
    # Find closest shift
    idx = np.argmin(np.abs(shifts - target_shift))
    shift, shifted_frame = frames_with_shifts[idx]

    axes[i].imshow(shifted_frame, cmap='gray', vmin=vmin, vmax=vmax)
    axes[i].set_title(f'Shift: {shift:.3f} px', fontsize=16, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    axes[i].axis('off')

plt.tight_layout()
comparison_path = output_dir / 'phase_shift_sidebyside.png'
plt.savefig(comparison_path, dpi=150, facecolor='black')
print(f'Saved side-by-side comparison to: {comparison_path}')
plt.close()

# Create a zoom-in comparison on a region with cells
print('\nCreating zoomed comparison of high-signal region...')

# Find a good region with signal
# Use the center or find brightest region
row_means = frame.mean(axis=1)
col_means = frame.mean(axis=0)

# Find region with highest signal
window_size = 128
best_signal = 0
best_y, best_x = 0, 0

for y in range(0, frame.shape[0] - window_size, 32):
    for x in range(0, frame.shape[1] - window_size, 32):
        window_signal = frame[y:y+window_size, x:x+window_size].mean()
        if window_signal > best_signal:
            best_signal = window_signal
            best_y, best_x = y, x

print(f'Best region found at y={best_y}, x={best_x} with mean signal {best_signal:.1f}')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, target_shift in enumerate(key_shifts):
    # Find closest shift
    idx = np.argmin(np.abs(shifts - target_shift))
    shift, shifted_frame = frames_with_shifts[idx]

    # Extract zoom region
    zoom = shifted_frame[best_y:best_y+window_size, best_x:best_x+window_size]

    axes[i].imshow(zoom, cmap='gray', vmin=vmin, vmax=vmax)
    axes[i].set_title(f'Shift: {shift:.3f} px', fontsize=16, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    axes[i].axis('off')

plt.tight_layout()
zoom_path = output_dir / 'phase_shift_zoomed_comparison.png'
plt.savefig(zoom_path, dpi=150, facecolor='black')
print(f'Saved zoomed comparison to: {zoom_path}')
plt.close()

print('\n' + '='*70)
print('SUMMARY - Files created:')
print('='*70)
print(f'1. Animated GIF: {gif_path}')
print(f'2. Annotated TIFF: {tiff_path}')
print(f'3. Raw TIFF (ImageJ): {raw_tiff_path}')
print(f'4. Side-by-side PNG: {comparison_path}')
print(f'5. Zoomed comparison PNG: {zoom_path}')
print('='*70)
print('\nYou can send any of these to your colleague.')
print('The GIF will play automatically in most viewers.')
print('The raw TIFF can be opened in ImageJ and played as a stack.')
print('='*70)
