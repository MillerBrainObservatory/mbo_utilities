from pathlib import Path
import mbo_utilities as mbo
import numpy as np

inpath = Path.home() / 'Documents/data/yao'
data = mbo.imread(inpath)
frame = data[14, 0, :, :]

print(f'Frame shape: {frame.shape}')

# Test with gradient-based method (new default) - 1D FFT
print("\n--- 1D FFT with gradient (new default, fast) ---")
data_shifted_grad, offsets_grad = mbo.phasecorr.bidir_phasecorr(
    frame, use_fft=True, use_gradient=True, fft_method='1d', upsample=10
)
print(f'Detected offset: {offsets_grad:.4f} pixels')

# Test with 2D FFT + gradient (more accurate but slower)
print("\n--- 2D FFT with gradient (more accurate) ---")
data_shifted_2d, offsets_2d = mbo.phasecorr.bidir_phasecorr(
    frame, use_fft=True, use_gradient=True, fft_method='2d', upsample=10
)
print(f'Detected offset: {offsets_2d:.4f} pixels')

# Test without gradient (old method)
print("\n--- 1D FFT without gradient (old method) ---")
data_shifted_no_grad, offsets_no_grad = mbo.phasecorr.bidir_phasecorr(
    frame, use_fft=True, use_gradient=False, fft_method='1d', upsample=10
)
print(f'Detected offset: {offsets_no_grad:.4f} pixels')

# Test with 2D without gradient
print("\n--- 2D FFT without gradient (old method) ---")
data_shifted_2d_no_grad, offsets_2d_no_grad = mbo.phasecorr.bidir_phasecorr(
    frame, use_fft=True, use_gradient=False, fft_method='2d', upsample=10
)
print(f'Detected offset: {offsets_2d_no_grad:.4f} pixels')

# Test with integer method + gradient
print("\n--- Integer method with gradient ---")
data_shifted_int, offsets_int = mbo.phasecorr.bidir_phasecorr(
    frame, use_fft=False, use_gradient=True
)
print(f'Detected offset: {offsets_int:.4f} pixels')

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"1D FFT + gradient:      {offsets_grad:.4f} pixels")
print(f"2D FFT + gradient:      {offsets_2d:.4f} pixels")
print(f"1D FFT no gradient:     {offsets_no_grad:.4f} pixels")
print(f"2D FFT no gradient:     {offsets_2d_no_grad:.4f} pixels")
print(f"Integer + gradient:     {offsets_int:.4f} pixels")
print("="*60)
