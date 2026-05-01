# Quick Start

This tool was originally built for ScanImage TIFFs and has been extended to read any 2D-4D TIFF and OME-compatible TIFF / Zarr. Output goes through Fiji-readable TIFF/Zarr or directly into a Suite2p / Cellpose run.

## Overview

```
raw file (TIFF / Zarr / .bin / .h5)
    |
    +-->  Save As           ->  Fiji-compatible TIFF / Zarr / .bin / .h5
    |                           (preview, share, re-open)
    |
    +-->  Run -> Suite2p    ->  /save_path/zplane0N_tp00001-NNNNN/
                                (one folder per plane; reload by opening
                                 the parent folder)
```

1. **Open** a raw file (`o`) or a folder of files (`Shift+O`).
2. **Preview** by selecting a small range — e.g. `1:500` in the timepoint field — to run everything against just those frames.
3. **Save As** (`s`) for a Fiji-readable copy. The viewer keeps showing the original; reopen the saved file to work with it.
4. **Run** (Run tab) for full Suite2p / Cellpose processing. Run on the original raw data — Save As outputs aren't auto-fed in.
5. **Reload** Suite2p output: drop the *parent* directory (containing the `zplane0N_tp00001-NNNNN/` subdirs) onto the file dialog. Each plane loads as a registered binary; rerun detection/extraction without re-registering.

## Subset processing

Both Save As and Run respect the timepoint and z-plane selection from the Selection popup.

| Field | Example | Meaning |
| --- | --- | --- |
| Timepoints | `1:500` | first 500 frames |
| Timepoints | `1:5000:10` | every 10th frame, 1-5000 |
| Timepoints | `1:1000,500:600` | 1-1000 excluding 500-600 |
| Z-planes | `1:14:2` | every other plane |

Use a small range to dial in detection parameters in seconds; rerun on the full range once happy.

## Run tab

- **Main** — `tau`, `fs` (auto-filled from metadata), denoise, dF/F window
- **Registration** — Skip / Run / Force, non-rigid block size
- **ROI Detection** — anatomical_only, diameter (Y/X), Cellpose thresholds
- **Deconv/Classify** — neuropil coefficient, baseline, classifier path, accept-all-cells

Re-running on a Suite2p output that was loaded from `zplane0N_…/`: the pipeline links to the existing `data.bin` automatically when Registration is set to **Skip**, so detection sweeps don't copy gigabytes of binary per run.

Scan-phase correction is previewed live in the viewer (Fix Phase / Sub-Pixel checkboxes) and applies on Save As and Run.
