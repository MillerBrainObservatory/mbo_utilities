#!/usr/bin/env python3
"""Add phase correction benchmarks to the notebook."""

import json

# Read existing notebook
with open('dev/benchmark_loading.ipynb', 'r') as f:
    nb = json.load(f)

# New cells to add
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 13. Scan-Phase Correction Benchmarks\n",
            "\n",
            "Benchmark all aspects of the phase correction pipeline.\n",
            "\n",
            "The phase correction happens in `_read_pages` when `fix_phase=True`.\n",
            "It calls `bidir_phasecorr()` which:\n",
            "1. Computes a reference frame (via method: 'mean', 'max', 'std', 'mean-sub', or 'frame')\n",
            "2. Runs phase correlation to find the shift\n",
            "3. Applies the shift to odd rows\n",
            "\n",
            "Methods:\n",
            "- **'mean'**: Use mean projection as reference (default)\n",
            "- **'max'**: Use max projection\n",
            "- **'std'**: Use std projection  \n",
            "- **'mean-sub'**: First frame minus mean\n",
            "- **'frame'**: Per-frame correction (slowest, most accurate)\n",
            "- **None**: No correction\n",
            "\n",
            "FFT options:\n",
            "- **use_fft=False**: Fast integer-only correlation (default)\n",
            "- **use_fft=True**: Subpixel FFT-based correlation (slower, more precise)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from mbo_utilities.phasecorr import bidir_phasecorr, _phase_corr_2d, _apply_offset\n",
            "import numpy as np\n",
            "\n",
            "# Load a small chunk of data to benchmark\n",
            "arr_test = MboRawArray(files=files, fix_phase=False)  # Disable phase correction for now\n",
            "print(f\"Array shape: {arr_test.shape}\")\n",
            "print(f\"Array dtype: {arr_test.dtype}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 13.1. Baseline: Data Loading Without Phase Correction"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create arrays with phase correction disabled\n",
            "arr_no_fix = MboRawArray(files=files, fix_phase=False)\n",
            "\n",
            "print(\"\\n=== Load 1 frame (no phase correction) ===\")\n",
            "%timeit -n 10 -r 5 frame = arr_no_fix[0]\n",
            "\n",
            "print(\"\\n=== Load 10 frames (no phase correction) ===\")\n",
            "%timeit -n 5 -r 3 frames = arr_no_fix[:10]\n",
            "\n",
            "print(\"\\n=== Load 100 frames (no phase correction) ===\")\n",
            "%timeit -n 3 -r 3 frames = arr_no_fix[:100]"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 13.2. Phase Correction: Different Methods"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test different phasecorr methods\n",
            "methods = ['mean', 'max', 'std', 'mean-sub', 'frame']\n",
            "\n",
            "print(\"=\" * 70)\n",
            "print(\"PHASE CORRECTION METHOD COMPARISON (1 frame)\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "for method in methods:\n",
            "    arr_method = MboRawArray(files=files, fix_phase=True, phasecorr_method=method, use_fft=False)\n",
            "    print(f\"\\nMethod: {method}\")\n",
            "    %timeit -n 5 -r 3 frame = arr_method[0]\n",
            "    for tf in arr_method.tiff_files:\n",
            "        tf.close()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 70)\n",
            "print(\"PHASE CORRECTION METHOD COMPARISON (10 frames)\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "for method in methods:\n",
            "    arr_method = MboRawArray(files=files, fix_phase=True, phasecorr_method=method, use_fft=False)\n",
            "    print(f\"\\nMethod: {method}\")\n",
            "    %timeit -n 3 -r 3 frames = arr_method[:10]\n",
            "    for tf in arr_method.tiff_files:\n",
            "        tf.close()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 70)\n",
            "print(\"PHASE CORRECTION METHOD COMPARISON (100 frames)\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "for method in methods:\n",
            "    arr_method = MboRawArray(files=files, fix_phase=True, phasecorr_method=method, use_fft=False)\n",
            "    print(f\"\\nMethod: {method}\")\n",
            "    %timeit -n 2 -r 3 frames = arr_method[:100]\n",
            "    for tf in arr_method.tiff_files:\n",
            "        tf.close()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 13.3. FFT vs Integer Phase Correlation"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 70)\n",
            "print(\"FFT VS INTEGER CORRELATION (method='mean')\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "print(\"\\n=== use_fft=False (integer, fast) ===\")\n",
            "arr_int = MboRawArray(files=files, fix_phase=True, phasecorr_method='mean', use_fft=False)\n",
            "print(\"1 frame:\")\n",
            "%timeit -n 5 -r 3 frame = arr_int[0]\n",
            "print(\"10 frames:\")\n",
            "%timeit -n 3 -r 3 frames = arr_int[:10]\n",
            "\n",
            "print(\"\\n=== use_fft=True (subpixel, slow) ===\")\n",
            "arr_fft = MboRawArray(files=files, fix_phase=True, phasecorr_method='mean', use_fft=True)\n",
            "print(\"1 frame:\")\n",
            "%timeit -n 5 -r 3 frame = arr_fft[0]\n",
            "print(\"10 frames:\")\n",
            "%timeit -n 3 -r 3 frames = arr_fft[:10]\n",
            "\n",
            "for tf in arr_int.tiff_files:\n",
            "    tf.close()\n",
            "for tf in arr_fft.tiff_files:\n",
            "    tf.close()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 13.4. Component Breakdown: Phase Correction Pipeline"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load some data without phase correction to benchmark components\n",
            "arr_raw = MboRawArray(files=files, fix_phase=False)\n",
            "test_data = arr_raw[:10]  # 10 frames, 14 channels, 448x448\n",
            "print(f\"Test data shape: {test_data.shape}\")\n",
            "print(f\"Test data dtype: {test_data.dtype}\")\n",
            "\n",
            "# Flatten to (T*Z, Y, X) like bidir_phasecorr does\n",
            "flat_data = test_data.reshape(test_data.shape[0] * test_data.shape[1], *test_data.shape[-2:])\n",
            "print(f\"Flattened shape: {flat_data.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 70)\n",
            "print(\"COMPONENT BREAKDOWN: 10 frames (140 pages)\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "# Step 1: Compute reference frame\n",
            "print(\"\\n1. Compute reference frame (mean projection):\")\n",
            "%timeit -n 10 -r 5 ref_frame = np.mean(flat_data, axis=0)\n",
            "\n",
            "ref_frame = np.mean(flat_data, axis=0)\n",
            "\n",
            "# Step 2: Phase correlation on reference\n",
            "print(\"\\n2. Phase correlation (integer method):\")\n",
            "%timeit -n 10 -r 5 offset = _phase_corr_2d(ref_frame, upsample=5, border=3, max_offset=4, use_fft=False)\n",
            "\n",
            "offset = _phase_corr_2d(ref_frame, upsample=5, border=3, max_offset=4, use_fft=False)\n",
            "print(f\"   Computed offset: {offset:.2f} pixels\")\n",
            "\n",
            "# Step 3: Apply offset\n",
            "print(\"\\n3. Apply offset to all frames:\")\n",
            "%timeit -n 10 -r 5 corrected = _apply_offset(flat_data.copy(), offset, use_fft=False)\n",
            "\n",
            "# Full pipeline (for comparison)\n",
            "print(\"\\n4. Full bidir_phasecorr (method='mean'):\")\n",
            "%timeit -n 5 -r 3 corrected, offset = bidir_phasecorr(flat_data, method='mean', use_fft=False, upsample=5, max_offset=4, border=3)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare different reference methods\n",
            "print(\"=\" * 70)\n",
            "print(\"REFERENCE FRAME COMPUTATION (140 frames)\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "print(\"\\nmean projection:\")\n",
            "%timeit -n 10 -r 5 ref = np.mean(flat_data, axis=0)\n",
            "\n",
            "print(\"\\nmax projection:\")\n",
            "%timeit -n 10 -r 5 ref = np.max(flat_data, axis=0)\n",
            "\n",
            "print(\"\\nstd projection:\")\n",
            "%timeit -n 10 -r 5 ref = np.std(flat_data, axis=0)\n",
            "\n",
            "print(\"\\nmean-sub (first - mean):\")\n",
            "%timeit -n 10 -r 5 ref = flat_data[0] - np.mean(flat_data, axis=0)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 13.5. Detailed Integer Correlation Breakdown"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Break down the integer correlation method\n",
            "ref = np.mean(flat_data, axis=0)\n",
            "h, w = ref.shape\n",
            "border = 3\n",
            "\n",
            "# Separate even/odd rows\n",
            "pre = ref[::2]\n",
            "post = ref[1::2]\n",
            "m = min(pre.shape[0], post.shape[0])\n",
            "\n",
            "print(f\"Reference frame shape: {ref.shape}\")\n",
            "print(f\"Even rows shape: {pre.shape}\")\n",
            "print(f\"Odd rows shape: {post.shape}\")\n",
            "\n",
            "# Crop borders\n",
            "a = pre[border:m-border, border:w-border]\n",
            "b = post[border:m-border, border:w-border]\n",
            "\n",
            "print(f\"After border crop: {a.shape}\")\n",
            "\n",
            "print(\"\\n=\" * 70)\n",
            "print(\"INTEGER CORRELATION COMPONENTS\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "print(\"\\n1. Extract even/odd rows:\")\n",
            "%timeit -n 100 -r 5 pre = ref[::2]; post = ref[1::2]\n",
            "\n",
            "print(\"\\n2. Crop borders:\")\n",
            "%timeit -n 100 -r 5 a = pre[border:m-border, border:w-border]; b = post[border:m-border, border:w-border]\n",
            "\n",
            "print(\"\\n3. Compute row means:\")\n",
            "%timeit -n 100 -r 5 a_mean = a.mean(axis=0) - np.mean(a); b_mean = b.mean(axis=0) - np.mean(b)\n",
            "\n",
            "a_mean = a.mean(axis=0) - np.mean(a)\n",
            "b_mean = b.mean(axis=0) - np.mean(b)\n",
            "\n",
            "print(\"\\n4. Compute correlation scores (9 offsets):\")\n",
            "offsets = np.arange(-4, 5, 1)\n",
            "\n",
            "def compute_scores(a_mean, b_mean, offsets):\n",
            "    scores = np.empty_like(offsets, dtype=float)\n",
            "    for i, k in enumerate(offsets):\n",
            "        if k > 0:\n",
            "            aa = a_mean[:-k]\n",
            "            bb = b_mean[k:]\n",
            "        elif k < 0:\n",
            "            aa = a_mean[-k:]\n",
            "            bb = b_mean[:k]\n",
            "        else:\n",
            "            aa = a_mean\n",
            "            bb = b_mean\n",
            "        num = np.dot(aa, bb)\n",
            "        denom = np.linalg.norm(aa) * np.linalg.norm(bb)\n",
            "        scores[i] = num / denom if denom else 0.0\n",
            "    return scores\n",
            "\n",
            "%timeit -n 100 -r 5 scores = compute_scores(a_mean, b_mean, offsets)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 13.6. Apply Offset Breakdown"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 70)\n",
            "print(\"APPLY OFFSET COMPONENTS\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "offset_val = 2.5\n",
            "\n",
            "print(\"\\n1. Extract odd rows (view):\")\n",
            "%timeit -n 100 -r 5 rows = flat_data[..., 1::2, :]\n",
            "\n",
            "rows = flat_data[..., 1::2, :]\n",
            "print(f\"   Odd rows shape: {rows.shape}\")\n",
            "\n",
            "print(\"\\n2. Integer roll (use_fft=False):\")\n",
            "test_copy = flat_data.copy()\n",
            "%timeit -n 100 -r 5 test_copy[..., 1::2, :] = np.roll(test_copy[..., 1::2, :], shift=int(round(offset_val)), axis=-1)\n",
            "\n",
            "print(\"\\n3. FFT shift (use_fft=True):\")\n",
            "test_copy = flat_data.copy()\n",
            "def apply_fft_shift(data, offset):\n",
            "    rows = data[..., 1::2, :]\n",
            "    f = np.fft.fftn(rows, axes=(-2, -1))\n",
            "    from scipy.ndimage import fourier_shift\n",
            "    shift_vec = (0,) * (f.ndim - 1) + (offset,)\n",
            "    rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec), axes=(-2, -1)).real\n",
            "    return data\n",
            "\n",
            "%timeit -n 10 -r 5 result = apply_fft_shift(test_copy.copy(), offset_val)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 13.7. Per-Frame vs Single Reference Comparison"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 70)\n",
            "print(\"PER-FRAME VS SINGLE REFERENCE (100 frames = 1400 pages)\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "# Load 100 frames\n",
            "test_data_large = arr_raw[:100]\n",
            "flat_large = test_data_large.reshape(test_data_large.shape[0] * test_data_large.shape[1], *test_data_large.shape[-2:])\n",
            "print(f\"Data shape: {flat_large.shape}\")\n",
            "\n",
            "print(\"\\nSingle reference (method='mean'):\")\n",
            "%timeit -n 3 -r 3 corrected, offs = bidir_phasecorr(flat_large, method='mean', use_fft=False, upsample=5, max_offset=4, border=3)\n",
            "\n",
            "print(\"\\nPer-frame (method='frame'):\")\n",
            "%timeit -n 3 -r 3 corrected, offs = bidir_phasecorr(flat_large, method='frame', use_fft=False, upsample=5, max_offset=4, border=3)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13.8. Summary: Phase Correction Overhead\n",
            "\n",
            "This cell quantifies the overhead of phase correction."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import time\n",
            "\n",
            "results = []\n",
            "\n",
            "# Test configurations\n",
            "configs = [\n",
            "    ('No correction', {'fix_phase': False}),\n",
            "    (\"mean (int)\", {'fix_phase': True, 'phasecorr_method': 'mean', 'use_fft': False}),\n",
            "    (\"max (int)\", {'fix_phase': True, 'phasecorr_method': 'max', 'use_fft': False}),\n",
            "    (\"std (int)\", {'fix_phase': True, 'phasecorr_method': 'std', 'use_fft': False}),\n",
            "    (\"frame (int)\", {'fix_phase': True, 'phasecorr_method': 'frame', 'use_fft': False}),\n",
            "    (\"mean (FFT)\", {'fix_phase': True, 'phasecorr_method': 'mean', 'use_fft': True}),\n",
            "]\n",
            "\n",
            "n_frames = 100\n",
            "\n",
            "for name, config in configs:\n",
            "    arr_test_config = MboRawArray(files=files, **config)\n",
            "    \n",
            "    # Time it\n",
            "    times = []\n",
            "    for _ in range(3):\n",
            "        t0 = time.time()\n",
            "        _ = arr_test_config[:n_frames]\n",
            "        times.append(time.time() - t0)\n",
            "    \n",
            "    avg_time = np.mean(times)\n",
            "    results.append({'Configuration': name, 'Time (s)': avg_time})\n",
            "    \n",
            "    for tf in arr_test_config.tiff_files:\n",
            "        tf.close()\n",
            "\n",
            "# Create summary table\n",
            "df = pd.DataFrame(results)\n",
            "df['Overhead (%)'] = ((df['Time (s)'] / df.iloc[0]['Time (s)'] - 1) * 100).round(1)\n",
            "df['Speedup'] = (df.iloc[0]['Time (s)'] / df['Time (s)']).round(2)\n",
            "\n",
            "print(f\"\\n{'='*70}\")\n",
            "print(f\"PHASE CORRECTION OVERHEAD ({n_frames} frames)\")\n",
            "print(f\"{'='*70}\")\n",
            "print(df.to_string(index=False))\n",
            "print(f\"{'='*70}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13.9. Recommendations\n",
            "\n",
            "Based on the benchmarks above:\n",
            "\n",
            "1. **Fastest**: `fix_phase=False` - No correction overhead\n",
            "2. **Fast + Accurate**: `method='mean', use_fft=False` - Good balance (default)\n",
            "3. **Slow but precise**: `method='frame', use_fft=True` - Per-frame subpixel\n",
            "\n",
            "Potential optimizations:\n",
            "- Cache reference frames per file (if data is static)\n",
            "- Use numba JIT for correlation loop\n",
            "- Parallelize per-frame corrections\n",
            "- Pre-compute offsets and save to metadata"
        ]
    }
]

# Append to notebook
nb['cells'].extend(new_cells)

# Write back
with open('dev/benchmark_loading.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(new_cells)} new cells for phase correction benchmarking")
