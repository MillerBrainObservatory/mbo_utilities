(scanphase_analysis)=
# Scan-Phase Analysis

Bidirectional resonant scanning causes alternating rows to be shifted horizontally. This tool measures that shift to help configure correction parameters.

## What It Does

1. **Per-frame offset** - measures the horizontal shift for each frame
2. **Window size analysis** - shows how many frames are needed for stable estimation
3. **Spatial heatmaps** - shows offset variation across the field of view
4. **Z-plane analysis** - checks if offset changes with depth
5. **Parameter guidance** - helps set `max_offset` based on signal intensity

## Running

```bash
uv run mbo scanphase                          # file dialog
uv run mbo scanphase /path/to/data.tiff       # specific file
uv run mbo scanphase ./raw_data/ -n 5         # first 5 tiffs from folder
uv run mbo scanphase data.tiff --show         # show plots
```

## Output Files

- `temporal.png` - offset over time + histogram
- `windows.png` - offset vs window size (convergence analysis)
- `spatial.png` - heatmaps at 32x32 and 64x64 patches
- `zplanes.png` - offset vs depth (if multi-plane)
- `parameters.png` - offset vs signal intensity
- `scanphase_results.npz` - numerical data

## What to Look For

### temporal.png

time series should be flat. large jumps indicate motion or hardware issues. typical offset is 0.5-2.0 px.

### windows.png

shows how estimate stabilizes with more frames. left plot: offset converges to stable value. right plot: variance decreases with window size. red line marks where std drops below 0.1 px.

use this to determine how many frames to average for correction.

### spatial.png

heatmaps show variation across fov. uniform = good. edges different from center is normal. gray = low signal (unreliable).

### zplanes.png

if offset varies with depth, may need per-plane correction. most data shows little variation.

### parameters.png

shows offset reliability vs signal. low signal = unreliable (high/variable offset). red line suggests intensity threshold below which measurements are noisy.

## Tips

- use `-n 2` or `-n 3` to run quickly on subset
- `--fft-method 1d` is faster and usually sufficient
- multi-ROI data: offsets are averaged across ROIs
