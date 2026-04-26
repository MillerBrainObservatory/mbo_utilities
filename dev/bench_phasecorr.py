"""
Benchmark mbo phase correction vs suite2p's bidiphase on the same data.

What we're comparing:

1. mbo bidir_phasecorr(use_fft=True, upsample=10)         -> 2D phase_cross_correlation, subpixel
2. mbo bidir_phasecorr(use_fft=False)                     -> 2D phase corr at integer precision
3. suite2p bidiphase.compute                              -> 1D FFT along x-axis only, integer
4. mbo _apply_offset(use_fft=True)                        -> 2D FFT shift (subpixel)
5. mbo _apply_offset(use_fft=False)                       -> integer np.roll
6. suite2p bidiphase.shift                                -> integer slice copy

Run with: uv run python dev/bench_phasecorr.py
"""
from pathlib import Path
import time
import numpy as np

import mbo_utilities as mbo
from mbo_utilities.analysis.phasecorr import bidir_phasecorr, _apply_offset
from suite2p.registration import bidiphase as s2p_bidi


RAW_DIR = Path(r"D:/demo/raw")
PLANE = 7  # 1-based (so z=6 0-based)
N_FRAMES = 500
N_REPEATS = 3


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
    print(f"reading {N_FRAMES} frames of plane {PLANE} (raw, no correction)...")
    t0 = time.perf_counter()
    chunk = np.asarray(arr[:N_FRAMES, 0, z]).astype(np.int16)  # (T, Y, X)
    t_read = time.perf_counter() - t0
    print(f"  read shape={chunk.shape} dtype={chunk.dtype} in {t_read*1000:.1f} ms")

    print("\n" + "=" * 70)
    print("OFFSET ESTIMATION (lower is better; offsets should agree across methods)")
    print("=" * 70)

    # 1. mbo with FFT subpixel
    (_, off_mbo_fft), tmin, tavg = time_call(
        bidir_phasecorr, chunk, method="mean", use_fft=True,
        upsample=10, max_offset=4, border=4,
    )
    print(f"  mbo bidir_phasecorr (use_fft=True, upsample=10): "
          f"shift={off_mbo_fft:+.3f} px   min={tmin*1000:7.1f} ms   avg={tavg*1000:7.1f} ms")

    # 2. mbo integer
    (_, off_mbo_int), tmin, tavg = time_call(
        bidir_phasecorr, chunk, method="mean", use_fft=False,
        upsample=1, max_offset=4, border=4,
    )
    print(f"  mbo bidir_phasecorr (use_fft=False, integer):   "
          f"shift={off_mbo_int:+.3f} px   min={tmin*1000:7.1f} ms   avg={tavg*1000:7.1f} ms")

    # 3. suite2p
    chunk_f = chunk.astype(np.float32)
    off_s2p, tmin, tavg = time_call(s2p_bidi.compute, chunk_f)
    print(f"  suite2p bidiphase.compute (full {N_FRAMES} frames): "
          f"shift={off_s2p:+d} px   min={tmin*1000:7.1f} ms   avg={tavg*1000:7.1f} ms")

    # 3b. suite2p with nimg_init=200 (default in default_ops)
    nimg = min(200, N_FRAMES)
    sub = chunk_f[np.linspace(0, N_FRAMES, nimg + 1, dtype=int)[:-1]]
    off_s2p2, tmin, tavg = time_call(s2p_bidi.compute, sub)
    print(f"  suite2p bidiphase.compute (nimg_init={nimg}):    "
          f"shift={off_s2p2:+d} px   min={tmin*1000:7.1f} ms   avg={tavg*1000:7.1f} ms")

    print("\n" + "=" * 70)
    print(f"OFFSET APPLICATION on (T={N_FRAMES}, Y, X) chunk; offset={off_mbo_fft:+.3f}")
    print("=" * 70)

    # 4. mbo apply with FFT (subpixel)
    test = chunk.copy()
    _, tmin, tavg = time_call(_apply_offset, test, off_mbo_fft, use_fft=True)
    print(f"  mbo _apply_offset (use_fft=True, subpixel 2D FFT/IFFT per chunk): "
          f"min={tmin*1000:7.1f} ms   avg={tavg*1000:7.1f} ms")

    # 5. mbo apply integer
    test = chunk.copy()
    _, tmin, tavg = time_call(_apply_offset, test, off_mbo_fft, use_fft=False)
    print(f"  mbo _apply_offset (use_fft=False, np.roll integer):              "
          f"min={tmin*1000:7.1f} ms   avg={tavg*1000:7.1f} ms")

    # 6. suite2p apply
    test = chunk.copy()
    _, tmin, tavg = time_call(s2p_bidi.shift, test, int(round(off_mbo_fft)))
    print(f"  suite2p bidi.shift (integer slice-copy, in-place):                "
          f"min={tmin*1000:7.1f} ms   avg={tavg*1000:7.1f} ms")

    print("\n" + "=" * 70)
    print("END-TO-END: estimate + apply on the same chunk (single call)")
    print("=" * 70)

    # full mbo path with FFT subpixel
    test = chunk.copy()
    t0 = time.perf_counter()
    _, off = bidir_phasecorr(test, method="mean", use_fft=True,
                             upsample=10, max_offset=4, border=4)
    t_mbo_full = time.perf_counter() - t0
    print(f"  mbo (use_fft=True):  est+apply = {t_mbo_full*1000:.1f} ms (offset={off:+.3f})")

    # full mbo path integer
    test = chunk.copy()
    t0 = time.perf_counter()
    _, off = bidir_phasecorr(test, method="mean", use_fft=False,
                             upsample=1, max_offset=4, border=4)
    t_mbo_int = time.perf_counter() - t0
    print(f"  mbo (use_fft=False): est+apply = {t_mbo_int*1000:.1f} ms (offset={off:+.3f})")

    # suite2p style: estimate once on subset, apply integer to all
    test = chunk_f.copy()
    t0 = time.perf_counter()
    bp = s2p_bidi.compute(sub)
    if bp != 0:
        s2p_bidi.shift(test, bp)
    t_s2p_full = time.perf_counter() - t0
    print(f"  suite2p style:       est+apply = {t_s2p_full*1000:.1f} ms (offset={bp:+d})")

    speedup_fft  = t_mbo_full / t_s2p_full if t_s2p_full else float("inf")
    speedup_int  = t_mbo_int  / t_s2p_full if t_s2p_full else float("inf")
    print(f"\n  mbo(use_fft=True)  is {speedup_fft:.1f}x slower than suite2p style")
    print(f"  mbo(use_fft=False) is {speedup_int:.1f}x slower than suite2p style")

    print("\n" + "=" * 70)
    print("SCALING: how does mbo's _apply_offset(use_fft=True) cost grow with chunk size?")
    print("=" * 70)
    for n in (50, 100, 200, 500):
        test = chunk[:n].copy()
        _, tmin, _ = time_call(_apply_offset, test, off_mbo_fft, use_fft=True, n=2)
        per_frame = tmin * 1000 / n
        print(f"  T={n:4d}: {tmin*1000:7.1f} ms total ({per_frame:.2f} ms/frame)")


if __name__ == "__main__":
    main()
