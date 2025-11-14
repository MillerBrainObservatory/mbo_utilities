"""
Test weighted phase correlation methods for bidirectional scan phase correction.
This emphasizes high-contrast regions (where cells are) and ignores background zeros.
"""
from pathlib import Path
import mbo_utilities as mbo
import numpy as np
from skimage.registration import phase_cross_correlation

inpath = Path.home() / 'Documents/data/yao'
data = mbo.imread(inpath)
frame = data[14, 0, :, :]

print(f'Frame shape: {frame.shape}')
print(f'Frame range: [{frame.min()}, {frame.max()}]')
print(f'Fraction of zeros: {(frame == 0).sum() / frame.size:.2%}')


def weighted_phase_corr_2d(frame, upsample=10, percentile_threshold=50, use_gradient=True):
    """
    Phase correlation weighted by high-contrast regions.

    Parameters
    ----------
    frame : ndarray (H, W)
        Input image
    upsample : int
        Upsampling factor for subpixel precision
    percentile_threshold : float
        Percentile threshold for selecting high-contrast regions (0-100)
        Higher = only brightest regions, Lower = more regions included
    use_gradient : bool
        If True, use gradient magnitude as weight
    """
    # Split into even and odd rows
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])

    a = pre[:m]
    b = post[:m]

    # Compute weights based on signal strength
    if use_gradient:
        # Use gradient magnitude to find edges/structure
        grad_y_a = np.abs(np.diff(a, axis=0, prepend=a[:1]))
        grad_x_a = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
        weight_a = np.sqrt(grad_y_a**2 + grad_x_a**2)

        grad_y_b = np.abs(np.diff(b, axis=0, prepend=b[:1]))
        grad_x_b = np.abs(np.diff(b, axis=1, prepend=b[:, :1]))
        weight_b = np.sqrt(grad_y_b**2 + grad_x_b**2)

        # Combine weights
        weight = (weight_a + weight_b) / 2
    else:
        # Use intensity as weight
        weight = (a + b) / 2

    # Threshold to only include high-contrast regions
    threshold = np.percentile(weight[weight > 0], percentile_threshold)
    mask = weight >= threshold

    print(f"  Percentile {percentile_threshold}: threshold={threshold:.1f}")
    print(f"  Pixels above threshold: {mask.sum()} ({mask.sum()/mask.size:.1%})")

    # Apply mask and weights
    a_weighted = a * mask
    b_weighted = b * mask

    # Use scikit-image phase_cross_correlation
    shift, error, phasediff = phase_cross_correlation(
        a_weighted, b_weighted,
        upsample_factor=upsample
    )

    return float(shift[1]), mask  # Return horizontal shift and mask


def intensity_weighted_1d(frame, upsample=10, percentile_threshold=50):
    """
    1D correlation weighted by row intensity.
    Only uses rows that have significant signal.
    """
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])

    a = pre[:m]
    b = post[:m]

    # Compute row weights (mean intensity per row)
    row_weights_a = a.mean(axis=1)
    row_weights_b = b.mean(axis=1)
    row_weights = (row_weights_a + row_weights_b) / 2

    # Threshold rows
    threshold = np.percentile(row_weights[row_weights > 0], percentile_threshold)
    good_rows = row_weights >= threshold

    print(f"  Using {good_rows.sum()}/{len(good_rows)} rows above threshold {threshold:.1f}")

    # Weight and average rows
    a_1d = np.average(a[good_rows], axis=0, weights=row_weights[good_rows])
    b_1d = np.average(b[good_rows], axis=0, weights=row_weights[good_rows])

    # Remove DC component
    a_1d = a_1d - np.mean(a_1d)
    b_1d = b_1d - np.mean(b_1d)

    # 1D FFT cross-correlation
    A = np.fft.fft(a_1d)
    B = np.fft.fft(b_1d)

    cross_power = A * np.conj(B)
    cross_power /= np.abs(cross_power) + 1e-10

    correlation = np.fft.ifft(cross_power).real

    # Find peak
    maxima = np.argmax(correlation)

    if upsample > 1:
        # Refine around peak
        width = 3
        shift_range = np.linspace(maxima - width, maxima + width, width * upsample * 2)

        upsampled_corr = np.zeros_like(shift_range)
        for i, shift in enumerate(shift_range):
            freq = np.fft.fftfreq(len(a_1d))
            phase_shift = np.exp(-2j * np.pi * freq * shift)
            shifted_B = np.fft.ifft(B * phase_shift).real
            upsampled_corr[i] = np.dot(a_1d, shifted_B)

        maxima = shift_range[np.argmax(upsampled_corr)]

    # Handle wrap-around
    shift = maxima
    if shift > len(a_1d) / 2:
        shift -= len(a_1d)

    return float(shift)


def local_correlation_voting(frame, window_size=64, stride=32, upsample=10, min_intensity=100):
    """
    Divide frame into windows, compute shift for each high-intensity window,
    then vote/average to get final shift. Ignores low-signal regions.
    """
    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])

    a = pre[:m]
    b = post[:m]

    h, w = a.shape
    shifts = []
    weights = []

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            window_a = a[y:y+window_size, x:x+window_size]
            window_b = b[y:y+window_size, x:x+window_size]

            # Only use windows with significant signal
            mean_intensity = (window_a.mean() + window_b.mean()) / 2
            if mean_intensity < min_intensity:
                continue

            # Compute shift for this window
            shift, error, phasediff = phase_cross_correlation(
                window_a, window_b,
                upsample_factor=upsample
            )

            shifts.append(float(shift[1]))
            weights.append(mean_intensity)

    if not shifts:
        print("  WARNING: No high-intensity windows found!")
        return 0.0

    # Weighted average (or could use median)
    shifts = np.array(shifts)
    weights = np.array(weights)

    print(f"  Found {len(shifts)} high-intensity windows")
    print(f"  Shift range: [{shifts.min():.2f}, {shifts.max():.2f}]")
    print(f"  Shift std: {shifts.std():.2f}")

    # Return weighted mean
    return np.average(shifts, weights=weights)


# Test all methods
print("\n" + "="*70)
print("METHOD 1: Weighted 2D Phase Correlation (gradient-based weight)")
print("="*70)
for percentile in [30, 50, 70, 90]:
    print(f"\nPercentile threshold: {percentile}")
    shift, mask = weighted_phase_corr_2d(frame, upsample=10, percentile_threshold=percentile, use_gradient=True)
    print(f"  -> Detected shift: {shift:.4f} pixels")

print("\n" + "="*70)
print("METHOD 2: Weighted 2D Phase Correlation (intensity-based weight)")
print("="*70)
for percentile in [30, 50, 70, 90]:
    print(f"\nPercentile threshold: {percentile}")
    shift, mask = weighted_phase_corr_2d(frame, upsample=10, percentile_threshold=percentile, use_gradient=False)
    print(f"  -> Detected shift: {shift:.4f} pixels")

print("\n" + "="*70)
print("METHOD 3: Intensity-Weighted 1D Correlation")
print("="*70)
for percentile in [30, 50, 70, 90]:
    print(f"\nRow percentile threshold: {percentile}")
    shift = intensity_weighted_1d(frame, upsample=10, percentile_threshold=percentile)
    print(f"  -> Detected shift: {shift:.4f} pixels")

print("\n" + "="*70)
print("METHOD 4: Local Window Voting")
print("="*70)
for min_int in [50, 100, 200, 400]:
    print(f"\nMinimum window intensity: {min_int}")
    shift = local_correlation_voting(frame, window_size=64, stride=32, upsample=10, min_intensity=min_int)
    print(f"  -> Detected shift: {shift:.4f} pixels")

print("\n" + "="*70)
print("SUMMARY - Most likely candidates for your data:")
print("="*70)
print("Based on your estimate of -0.6 to -1.2 pixels, look for methods")
print("that give shifts in that range above.")
print("="*70)
