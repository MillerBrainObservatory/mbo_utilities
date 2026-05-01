"""
Prototype faster mbo phase correction. Compare:

A. current:  2D FFT/IFFT over (Y/2, X) per frame  -> scipy.fourier_shift
B. proposed: 1D FFT/IFFT along x only             -> manual phase ramp
C. proposed: 1D FFT estimation along x only       -> like suite2p but with subpixel
D. integer fallback when |shift| < 0.5 px

Run: uv run python dev/bench_phasecorr_v2.py
"""
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import fourier_shift

import mbo_utilities as mbo
from mbo_utilities.analysis.phasecorr import bidir_phasecorr, _apply_offset
from suite2p.registration import bidiphase as s2p_bidi


RAW_DIR = Path(r"D:/demo/raw")
PLANE = 7
N_FRAMES = 500
N_REPEATS = 3


def apply_offset_2d_fft(img, offset):
    """current implementation: 2D FFT on every chunk."""
    rows = img[..., 1::2, :]
    f = np.fft.fftn(rows, axes=(-2, -1))
    shift_vec = (0,) * (f.ndim - 1) + (offset,)
    rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec), axes=(-2, -1)).real
    return img


def apply_offset_1d_fft(img, offset):
    """proposed: 1D FFT along x only -- the shift is along x, so y FFT is wasted."""
    rows = img[..., 1::2, :]
    n = rows.shape[-1]
    f = np.fft.fft(rows, axis=-1)
    k = np.fft.fftfreq(n)
    phase = np.exp(-2j * np.pi * k * offset).astype(np.complex64)
    rows[:] = np.fft.ifft(f * phase, axis=-1).real
    return img


def apply_offset_1d_fft_rfft(img, offset):
    """proposed: real-valued 1D rFFT along x only -- about 2x faster than complex FFT."""
    rows = img[..., 1::2, :].astype(np.float32, copy=False)
    n = rows.shape[-1]
    f = np.fft.rfft(rows, axis=-1)
    k = np.fft.rfftfreq(n)
    phase = np.exp(-2j * np.pi * k * offset).astype(np.complex64)
    shifted = np.fft.irfft(f * phase, n=n, axis=-1)
    img[..., 1::2, :] = shifted.astype(img.dtype)
    return img


def apply_offset_smart(img, offset, subpixel_threshold=0.1):
    """proposed: if shift is sub-pixel and tiny, round; otherwise rFFT."""
    if abs(offset) < subpixel_threshold:
        return img  # noise, skip
    if abs(offset - round(offset)) < 0.05:
        rows = img[..., 1::2, :]
        rows[:] = np.roll(rows, shift=int(round(offset)), axis=-1)
        return img
    return apply_offset_1d_fft_rfft(img, offset)


def estimate_1d_fft(frames, max_offset=4, upsample=10):
    """1D phase correlation along x only, with parabolic peak refinement for subpixel.

    similar in spirit to suite2p.bidiphase.compute but with parabolic
    interpolation around the integer peak for ~1/upsample precision.
    """
    f_pre  = np.fft.rfft(frames[:, 0:-1:2, :].astype(np.float32), axis=-1)
    f_post = np.fft.rfft(frames[:, 1::2,   :].astype(np.float32), axis=-1)
    n = min(f_pre.shape[1], f_post.shape[1])
    f_pre, f_post = f_pre[:, :n], f_post[:, :n]
    f_pre /= np.abs(f_pre) + 1e-5
    f_post /= np.abs(f_post) + 1e-5

    Lx = frames.shape[-1]
    cc = np.fft.irfft(f_pre * np.conj(f_post), n=Lx, axis=-1).mean(axis=(0, 1))
    cc = np.fft.fftshift(cc)
    peak = np.argmax(cc[Lx // 2 - max_offset : Lx // 2 + max_offset + 1])
    int_shift = peak - max_offset

    # parabolic peak refinement around the integer maximum
    idx = Lx // 2 + int_shift
    if 0 < idx < Lx - 1:
        y0, y1, y2 = cc[idx - 1], cc[idx], cc[idx + 1]
        denom = (y0 - 2 * y1 + y2)
        sub = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        # clip to reasonable subpixel adjustment
        sub = np.clip(sub, -0.5, 0.5)
        return float(int_shift) + float(sub)
    return float(int_shift)


def time_call(fn, *args, n=N_REPEATS, **kwargs):
    out = None
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return out, min(times), float(np.mean(times))


def main():
    print("loading raw...")
    arr = mbo.imread(RAW_DIR)
    arr.fix_phase = False
    z = PLANE - 1
    chunk = np.asarray(arr[:N_FRAMES, 0, z]).astype(np.int16)
    print(f"chunk shape={chunk.shape} dtype={chunk.dtype}")

    # ground-truth offset from current mbo (use_fft=True, upsample=10)
    _, gt = bidir_phasecorr(chunk, method="mean", use_fft=True,
                            upsample=10, max_offset=4, border=4)
    print(f"\nground truth offset (mbo current): {gt:+.4f} px")

    print("\n" + "=" * 78)
    print(f"APPLY OFFSET on (T={N_FRAMES}, Y, X) chunk, offset={gt:+.3f}")
    print("=" * 78)

    test = chunk.copy().astype(np.float32)
    _, tmin, _ = time_call(apply_offset_2d_fft, test, gt)
    print(f"  A. current 2D FFT:                    min={tmin*1000:7.1f} ms")

    test = chunk.copy().astype(np.float32)
    _, tmin, _ = time_call(apply_offset_1d_fft, test, gt)
    print(f"  B. 1D complex FFT along x:            min={tmin*1000:7.1f} ms")

    test = chunk.copy().astype(np.float32)
    _, tmin, _ = time_call(apply_offset_1d_fft_rfft, test, gt)
    print(f"  C. 1D rFFT along x:                   min={tmin*1000:7.1f} ms")

    test = chunk.copy()
    _, tmin, _ = time_call(apply_offset_smart, test, gt, 0.1)
    print(f"  D. smart (rFFT, skip if tiny):        min={tmin*1000:7.1f} ms")

    test = chunk.copy()
    _, tmin, _ = time_call(_apply_offset, test, gt, use_fft=False)
    print(f"  E. integer np.roll (mbo current int): min={tmin*1000:7.1f} ms")

    print("\n" + "=" * 78)
    print("ESTIMATE on the same chunk")
    print("=" * 78)

    chunk_f = chunk.astype(np.float32)
    (_, off_curr), tmin, _ = time_call(
        bidir_phasecorr, chunk, method="mean", use_fft=True,
        upsample=10, max_offset=4, border=4,
    )
    print(f"  mbo current (use_fft=True, upsample=10): shift={off_curr:+.4f}  min={tmin*1000:7.1f} ms")

    off_1d, tmin, _ = time_call(estimate_1d_fft, chunk_f, max_offset=4, upsample=10)
    print(f"  proposed 1D rFFT + parabolic refine:     shift={off_1d:+.4f}  min={tmin*1000:7.1f} ms")

    nimg = min(200, N_FRAMES)
    sub = chunk_f[np.linspace(0, N_FRAMES, nimg + 1, dtype=int)[:-1]]
    off_1d_sub, tmin, _ = time_call(estimate_1d_fft, sub, max_offset=4, upsample=10)
    print(f"  proposed 1D rFFT (nimg=200, subset):     shift={off_1d_sub:+.4f}  min={tmin*1000:7.1f} ms")

    print("\n" + "=" * 78)
    print("END-TO-END: estimate-once + apply on full chunk")
    print("=" * 78)

    # current mbo full path
    test = chunk.copy()
    t0 = time.perf_counter()
    _, off = bidir_phasecorr(test, method="mean", use_fft=True,
                             upsample=10, max_offset=4, border=4)
    t_curr = time.perf_counter() - t0
    print(f"  CURRENT  mbo (use_fft=True):                {t_curr*1000:7.1f} ms  shift={off:+.3f}")

    # proposed: estimate on subset, apply rFFT to full chunk
    test = chunk.copy()
    t0 = time.perf_counter()
    bp = estimate_1d_fft(sub, max_offset=4, upsample=10)
    apply_offset_1d_fft_rfft(test, bp)
    t_new = time.perf_counter() - t0
    print(f"  PROPOSED rFFT + apply rFFT (subset est):   {t_new*1000:7.1f} ms  shift={bp:+.3f}")

    # proposed smart: skip subpixel if tiny
    test = chunk.copy()
    t0 = time.perf_counter()
    bp = estimate_1d_fft(sub, max_offset=4, upsample=10)
    apply_offset_smart(test, bp)
    t_smart = time.perf_counter() - t0
    print(f"  PROPOSED smart (skip if |shift|<0.1):       {t_smart*1000:7.1f} ms  shift={bp:+.3f}")

    print(f"\n  speedup over current: rFFT={t_curr/t_new:.1f}x   smart={t_curr/t_smart:.1f}x")

    print("\n" + "=" * 78)
    print("CORRECTNESS: result vs current implementation (max abs diff over chunk)")
    print("=" * 78)
    ref = chunk.copy().astype(np.float32)
    apply_offset_2d_fft(ref, gt)

    test = chunk.copy().astype(np.float32)
    apply_offset_1d_fft(test, gt)
    print(f"  1D complex FFT vs current:  max abs diff = {np.abs(ref - test).max():.3e}   "
          f"mean abs diff = {np.abs(ref - test).mean():.3e}")

    test = chunk.copy().astype(np.float32)
    apply_offset_1d_fft_rfft(test, gt)
    print(f"  1D rFFT vs current:          max abs diff = {np.abs(ref - test).max():.3e}   "
          f"mean abs diff = {np.abs(ref - test).mean():.3e}")


if __name__ == "__main__":
    main()
