# %% [markdown]
# ### 13.11 Validation: Visual Quality Check of Scan-Phase Correction

# %%
import matplotlib.pyplot as plt
from mbo_utilities.lazy_array import imread

print("Loading data for visual validation...")

# Load data with different correction methods
data_none = imread(test_path, roi=test_roi, fix_phase=False)
data_int = imread(test_path, roi=test_roi, fix_phase=True, use_fft=False)
data_fft2d = imread(test_path, roi=test_roi, fix_phase=True, use_fft=True, fft_method="2d")
data_fft1d = imread(test_path, roi=test_roi, fix_phase=True, use_fft=True, fft_method="1d")

# Get a single frame from middle of stack
frame_idx = data_none.shape[0] // 2
frame_none = data_none[frame_idx]
frame_int = data_int[frame_idx]
frame_fft2d = data_fft2d[frame_idx]
frame_fft1d = data_fft1d[frame_idx]

print(f"Loaded frame {frame_idx}, shape: {frame_none.shape}")

# %%
# Visual comparison: Full frames
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle(f"Scan-Phase Correction Comparison (Frame {frame_idx})", fontsize=16)

methods = [
    (frame_none, "No Correction"),
    (frame_int, "Integer Method"),
    (frame_fft2d, "FFT 2D Subpixel"),
    (frame_fft1d, "FFT 1D Subpixel (Optimized)")
]

for ax, (frame, title) in zip(axes.flat, methods):
    im = ax.imshow(frame, cmap='gray', vmin=0, vmax=np.percentile(frame, 99.5))
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %%
# Zoomed comparison: Look at specific region with clear structure
zoom_y, zoom_x = 200, 200
zoom_size = 100

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle(f"Zoomed Comparison (Y:{zoom_y}-{zoom_y+zoom_size}, X:{zoom_x}-{zoom_x+zoom_size})", fontsize=16)

for ax, (frame, title) in zip(axes.flat, methods):
    crop = frame[zoom_y:zoom_y+zoom_size, zoom_x:zoom_x+zoom_size]
    im = ax.imshow(crop, cmap='gray', vmin=0, vmax=np.percentile(crop, 99.5), interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %%
# Even/Odd row separation to visualize alignment
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle("Even/Odd Row Alignment Analysis", fontsize=16)

for i, (frame, method_name) in enumerate(methods):
    crop = frame[zoom_y:zoom_y+zoom_size, zoom_x:zoom_x+zoom_size]

    # Extract even and odd rows
    even_rows = crop[::2]
    odd_rows = crop[1::2]

    # Compute difference (should be minimal if well aligned)
    min_rows = min(even_rows.shape[0], odd_rows.shape[0])
    diff = np.abs(even_rows[:min_rows] - odd_rows[:min_rows])

    # Plot even rows
    ax = axes[i, 0]
    im = ax.imshow(even_rows, cmap='gray', aspect='auto', interpolation='nearest')
    ax.set_title(f"{method_name}\nEven Rows")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot odd rows
    ax = axes[i, 1]
    im = ax.imshow(odd_rows, cmap='gray', aspect='auto', interpolation='nearest')
    ax.set_title(f"Odd Rows")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot difference
    ax = axes[i, 2]
    im = ax.imshow(diff, cmap='hot', aspect='auto', interpolation='nearest')
    ax.set_title(f"|Even - Odd|\nMean: {np.mean(diff):.2f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %%
# Quantitative metrics
print("\n" + "="*60)
print("SCAN-PHASE CORRECTION ACCURACY METRICS")
print("="*60)

results = []
for frame, method_name in methods:
    crop = frame[zoom_y:zoom_y+zoom_size, zoom_x:zoom_x+zoom_size]

    even_rows = crop[::2]
    odd_rows = crop[1::2]
    min_rows = min(even_rows.shape[0], odd_rows.shape[0])

    # Mean absolute difference
    diff = np.abs(even_rows[:min_rows] - odd_rows[:min_rows])
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    std_diff = np.std(diff)

    results.append({
        'Method': method_name,
        'Mean |Even-Odd|': mean_diff,
        'Max |Even-Odd|': max_diff,
        'Std |Even-Odd|': std_diff,
    })

df_validation = pd.DataFrame(results)

# Compute improvement vs no correction
baseline = df_validation.loc[0, 'Mean |Even-Odd|']
df_validation['Improvement (%)'] = ((baseline - df_validation['Mean |Even-Odd|']) / baseline * 100).round(1)

print("\nAlignment Quality (Lower is Better):")
print(df_validation.to_string(index=False))
print("\n" + "="*60)

# %%
# Profile variation: Check horizontal profile across even/odd boundary
mid_row = zoom_size // 2
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f"Horizontal Profile at Row {mid_row} (Even vs Odd)", fontsize=16)

for ax, (frame, method_name) in zip(axes.flat, methods):
    crop = frame[zoom_y:zoom_y+zoom_size, zoom_x:zoom_x+zoom_size]

    # Get profiles from even and odd rows near middle
    even_profile = crop[mid_row if mid_row % 2 == 0 else mid_row-1]
    odd_profile = crop[mid_row if mid_row % 2 == 1 else mid_row+1]

    x = np.arange(len(even_profile))
    ax.plot(x, even_profile, label='Even row', alpha=0.7, linewidth=2)
    ax.plot(x, odd_profile, label='Odd row', alpha=0.7, linewidth=2)
    ax.set_title(method_name)
    ax.set_xlabel('Column')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Validation complete! Check the plots above to verify correction accuracy.")
