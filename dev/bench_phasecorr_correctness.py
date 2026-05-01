"""
Correctness + on-the-fly latency check for phase-correction methods.

Two independent tests:

A. KNOWN-SHIFT RECOVERY:
   - take a clean frame, apply a known sub-pixel shift to odd rows
   - run each estimator
   - report estimated shift vs ground truth across [-2.0, +2.0] in 0.25 steps

B. PER-FRAME LATENCY (GUI on-the-fly viability):
   - time each apply path on a single 2D frame (the GUI scrub case)
   - 33 ms budget for 30 FPS, 16 ms for 60 FPS

Run: uv run python dev/bench_phasecorr_correctness.py
"""
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

import mbo_utilities as mbo
from mbo_utilities.analysis.phasecorr import bidir_phasecorr, _apply_offset
from suite2p.registration import bidiphase as s2p_bidi


RAW_DIR = Path(r"D:/demo/raw")
PLANE = 7


# ---------------------------------------------------------------------------
# proposed implementations under test
# ---------------------------------------------------------------------------

def estimate_skimage_2d(frames, upsample=10, max_offset=4, border=4):
    """current mbo implementation: skimage 2D phase_cross_correlation on mean image."""
    mean_img = frames.mean(axis=0)
    h, w = mean_img.shape
    pre = mean_img[::2][border:h//2 - border, border:w - border]
    post = mean_img[1::2][border:h//2 - border, border:w - border]
    shift, *_ = phase_cross_correlation(pre, post, upsample_factor=upsample)
    dx = float(shift[1])
    if max_offset:
        dx = np.sign(dx) * min(abs(dx), max_offset)
    return dx


def estimate_1d_rfft_parabolic(frames, max_offset=4):
    """proposed: 1D rFFT phase corr along x, parabolic peak refinement."""
    f_pre  = np.fft.rfft(frames[:, 0:-1:2, :].astype(np.float32), axis=-1)
    f_post = np.fft.rfft(frames[:, 1::2,   :].astype(np.float32), axis=-1)
    n = min(f_pre.shape[1], f_post.shape[1])
    f_pre, f_post = f_pre[:, :n], f_post[:, :n]
    f_pre  /= np.abs(f_pre)  + 1e-5
    f_post /= np.abs(f_post) + 1e-5

    Lx = frames.shape[-1]
    # cross-corr from EVEN to ODD: sign convention matches mbo (correction to apply)
    cc = np.fft.irfft(f_pre * np.conj(f_post), n=Lx, axis=-1).mean(axis=(0, 1))
    cc = np.fft.fftshift(cc)
    win_lo, win_hi = Lx // 2 - max_offset, Lx // 2 + max_offset + 1
    peak = np.argmax(cc[win_lo:win_hi])
    int_shift = peak - max_offset

    idx = Lx // 2 + int_shift
    if 0 < idx < Lx - 1:
        y0, y1, y2 = cc[idx - 1], cc[idx], cc[idx + 1]
        denom = (y0 - 2 * y1 + y2)
        sub = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        sub = float(np.clip(sub, -0.5, 0.5))
        return float(int_shift) + sub
    return float(int_shift)


def estimate_1d_rfft_dft_upsample(frames, max_offset=4, upsample=10):
    """proposed: 1D rFFT phase corr, DFT-based subpixel refinement (skimage-style).

    matches skimage's matrix-DFT upsampling but only along x.
    """
    f_pre  = np.fft.rfft(frames[:, 0:-1:2, :].astype(np.float32), axis=-1)
    f_post = np.fft.rfft(frames[:, 1::2,   :].astype(np.float32), axis=-1)
    n = min(f_pre.shape[1], f_post.shape[1])
    f_pre, f_post = f_pre[:, :n], f_post[:, :n]
    f_pre  /= np.abs(f_pre)  + 1e-5
    f_post /= np.abs(f_post) + 1e-5

    Lx = frames.shape[-1]
    cc = np.fft.irfft(f_pre * np.conj(f_post), n=Lx, axis=-1).mean(axis=(0, 1))
    cc_sh = np.fft.fftshift(cc)

    win_lo, win_hi = Lx // 2 - max_offset, Lx // 2 + max_offset + 1
    peak = np.argmax(cc_sh[win_lo:win_hi])
    int_shift = peak - max_offset

    # DFT-based subpixel refinement: oversample the cross-power spectrum
    # in a small window around the integer peak.
    oversample = 1.0 / upsample
    grid = np.arange(-1.5, 1.5 + oversample / 2, oversample)
    # need full FFT (not rFFT) for this — go to complex once
    cps = np.fft.fft(np.fft.irfft(f_pre * np.conj(f_post), n=Lx, axis=-1), axis=-1).mean(axis=(0, 1))
    k = np.fft.fftfreq(Lx)
    # evaluate cross-power at grid points around int_shift
    vals = np.array([np.real((cps * np.exp(2j * np.pi * k * (int_shift + g))).sum() / Lx)
                     for g in grid])
    sub = grid[np.argmax(vals)]
    return float(int_shift + sub)


def estimate_suite2p(frames):
    """suite2p's bidiphase.compute (integer only)."""
    return float(s2p_bidi.compute(frames.astype(np.float32)))


# apply paths

def apply_2d_fft(img, offset):
    """current mbo path."""
    rows = img[..., 1::2, :]
    f = np.fft.fftn(rows, axes=(-2, -1))
    shift_vec = (0,) * (f.ndim - 1) + (offset,)
    rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec), axes=(-2, -1)).real


def apply_1d_rfft(img, offset):
    """proposed: rFFT along x only."""
    rows = img[..., 1::2, :].astype(np.float32, copy=False)
    n = rows.shape[-1]
    f = np.fft.rfft(rows, axis=-1)
    k = np.fft.rfftfreq(n)
    phase = np.exp(-2j * np.pi * k * offset).astype(np.complex64)
    shifted = np.fft.irfft(f * phase, n=n, axis=-1)
    img[..., 1::2, :] = shifted.astype(img.dtype)


def apply_integer_roll(img, offset):
    """integer np.roll (mbo current with use_fft=False)."""
    rows = img[..., 1::2, :]
    rows[:] = np.roll(rows, shift=int(round(offset)), axis=-1)


def apply_suite2p(img, offset):
    """suite2p slice-copy. requires 3D; wrap 2D in a singleton axis."""
    if img.ndim == 2:
        s2p_bidi.shift(img[np.newaxis], int(round(offset)))
    else:
        s2p_bidi.shift(img, int(round(offset)))


# ---------------------------------------------------------------------------
# part A — known-shift recovery
# ---------------------------------------------------------------------------

def make_synthetic_with_shift(clean_3d, shift_px):
    """Apply a known sub-pixel shift to odd rows (in opposite direction of correction).

    If true shift is +s (odd rows at +s relative to even), the *correction*
    that estimators should return is -s.
    """
    # shift via 2D FFT for ground-truth subpixel accuracy
    out = clean_3d.astype(np.float32).copy()
    rows = out[..., 1::2, :]
    n = rows.shape[-1]
    f = np.fft.rfft(rows, axis=-1)
    k = np.fft.rfftfreq(n)
    # +shift_px applied to odd rows means odd is at +shift_px
    phase = np.exp(-2j * np.pi * k * shift_px).astype(np.complex64)
    rows[:] = np.fft.irfft(f * phase, n=n, axis=-1)
    return out


def part_a_known_shifts():
    print("=" * 78)
    print("PART A: known-shift recovery on a clean reference image")
    print("=" * 78)

    # use a clean mean image as the source (no native bidi; even/odd rows are aligned)
    arr = mbo.imread(RAW_DIR)
    arr.fix_phase = True  # use mbo's correction to get a (mostly) clean image
    arr.use_fft = True
    z = PLANE - 1
    print("loading 200-frame clean reference (mbo-corrected)...")
    clean = np.asarray(arr[:200, 0, z]).astype(np.float32)

    truths = np.arange(-2.0, 2.001, 0.25)
    methods = {
        "mbo current 2D (skimage, upsample=10)": lambda f: estimate_skimage_2d(f, upsample=10),
        "mbo current 2D (skimage, upsample=100)": lambda f: estimate_skimage_2d(f, upsample=100),
        "1D rFFT + parabolic":                   lambda f: estimate_1d_rfft_parabolic(f),
        "1D rFFT + DFT upsample (=10)":          lambda f: estimate_1d_rfft_dft_upsample(f, upsample=10),
        "suite2p (integer)":                     lambda f: estimate_suite2p(f),
    }

    # header
    print(f"\n{'true shift':>12s}  " + "  ".join(f"{name[:24]:>24s}" for name in methods))
    print("-" * (14 + 26 * len(methods)))

    rms_err = {name: [] for name in methods}
    int_err = {name: [] for name in methods}

    for true_shift in truths:
        synth = make_synthetic_with_shift(clean, true_shift)
        # estimators return the CORRECTION (negative of true_shift)
        expected = -true_shift
        row = f"{true_shift:+12.3f}  "
        for name, fn in methods.items():
            est = fn(synth)
            err = est - expected
            rms_err[name].append(err ** 2)
            int_err[name].append(round(est) - round(expected))
            row += f"{est:+10.3f} (e={err:+5.3f})    "
        print(row)

    print("\n--- summary ---")
    print(f"{'method':<42s}  {'RMS err (px)':>14s}  {'int mismatches':>16s}")
    for name in methods:
        rmse = float(np.sqrt(np.mean(rms_err[name])))
        n_int_bad = int(np.sum(np.array(int_err[name]) != 0))
        print(f"  {name:<40s}  {rmse:14.4f}  {n_int_bad:>16d}/{len(truths)}")


# ---------------------------------------------------------------------------
# part B — on-the-fly per-frame latency
# ---------------------------------------------------------------------------

def part_b_per_frame_latency():
    print("\n" + "=" * 78)
    print("PART B: per-frame apply latency (GUI on-the-fly)")
    print("=" * 78)
    print("Budget: 33 ms = 30 FPS, 16 ms = 60 FPS, <8 ms = 'instant'")

    arr = mbo.imread(RAW_DIR)
    arr.fix_phase = False
    z = PLANE - 1
    one_frame = np.asarray(arr[0, 0, z]).astype(np.int16)
    Y, X = one_frame.shape
    print(f"\nframe shape: ({Y}, {X}) dtype={one_frame.dtype}")

    methods = {
        "current 2D FFT (use_fft=True)": (apply_2d_fft, 0.5),
        "proposed 1D rFFT (use_fft=True)": (apply_1d_rfft, 0.5),
        "integer np.roll (use_fft=False)": (apply_integer_roll, 1.0),
        "suite2p slice-copy (integer)": (apply_suite2p, 1),
    }

    n_warmup = 5
    n_iter = 50

    print(f"\n{'method':<42s}  {'min (ms)':>10s}  {'mean (ms)':>10s}  {'p99 (ms)':>10s}  {'verdict':>15s}")
    print("-" * 100)
    for name, (fn, off) in methods.items():
        # warmup
        for _ in range(n_warmup):
            test = one_frame.copy()
            fn(test, off)
        # measure
        times = []
        for _ in range(n_iter):
            test = one_frame.copy()
            t0 = time.perf_counter()
            fn(test, off)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        verdict = "instant" if times.min() < 8 else "60 FPS ok" if times.min() < 16 else "30 FPS ok" if times.min() < 33 else "TOO SLOW"
        print(f"  {name:<40s}  {times.min():10.3f}  {times.mean():10.3f}  {np.percentile(times,99):10.3f}  {verdict:>15s}")

    # also test small chunk (typical GUI prefetch)
    print(f"\nsmall chunk (T=20) latency — typical for buffered GUI playback:")
    chunk = np.asarray(arr[:20, 0, z]).astype(np.int16)
    print(f"chunk shape: {chunk.shape}")
    print(f"\n{'method':<42s}  {'min (ms)':>10s}  {'mean (ms)':>10s}  {'per frame':>12s}")
    print("-" * 80)
    for name, (fn, off) in methods.items():
        for _ in range(n_warmup):
            test = chunk.copy()
            fn(test, off)
        times = []
        for _ in range(n_iter):
            test = chunk.copy()
            t0 = time.perf_counter()
            fn(test, off)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        per_frame = times.min() / chunk.shape[0]
        print(f"  {name:<40s}  {times.min():10.3f}  {times.mean():10.3f}  {per_frame:10.3f} ms")


# ---------------------------------------------------------------------------
# part C — applied-image equivalence (does 1D rFFT actually correct as well?)
# ---------------------------------------------------------------------------

def part_c_apply_equivalence():
    print("\n" + "=" * 78)
    print("PART C: corrected output equivalence — apply different methods, compare residual")
    print("=" * 78)

    arr = mbo.imread(RAW_DIR)
    arr.fix_phase = True
    arr.use_fft = True
    z = PLANE - 1
    print("loading 200-frame clean reference (mbo-corrected)...")
    clean = np.asarray(arr[:200, 0, z]).astype(np.float32)

    # inject a known shift, then correct with each method, measure residual vs clean
    print(f"\n{'true shift':>12s}  " + "  ".join(f"{n:<28s}" for n in
        ["2D FFT residual", "1D rFFT residual", "integer roll residual"]))
    print("-" * 110)

    for true_shift in [-1.5, -1.0, -0.7, -0.3, 0.3, 0.7, 1.0, 1.5]:
        synth = make_synthetic_with_shift(clean, true_shift)
        correction = -true_shift  # what an oracle would apply

        # baseline: residual without correction
        baseline_resid = np.abs(synth - clean).mean()

        a = synth.copy()
        apply_2d_fft(a, correction)
        r1 = np.abs(a - clean).mean()

        a = synth.copy()
        apply_1d_rfft(a, correction)
        r2 = np.abs(a - clean).mean()

        a = synth.copy()
        apply_integer_roll(a, correction)
        r3 = np.abs(a - clean).mean()

        print(f"{true_shift:+12.3f}  "
              f"{r1:8.4f} ({r1/baseline_resid*100:5.1f}%)         "
              f"{r2:8.4f} ({r2/baseline_resid*100:5.1f}%)         "
              f"{r3:8.4f} ({r3/baseline_resid*100:5.1f}%)        "
              f"   baseline={baseline_resid:.4f}")


if __name__ == "__main__":
    part_a_known_shifts()
    part_b_per_frame_latency()
    part_c_apply_equivalence()
