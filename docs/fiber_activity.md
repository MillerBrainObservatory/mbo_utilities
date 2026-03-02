# Fiber Activity Detection

The fiber activity pipeline detects and extracts fluorescence activity from neurite-like structures in timelapse calcium imaging data.
It is designed for single-plane timelapse stacks (2D + time) where the structures of interest are thin, elongated processes such as axons or dendrites rather than cell bodies.

## Overview

The pipeline performs the following steps:

1. **Load & normalise** — Read a single-plane timelapse TIFF and subtract the global minimum so that the background is near zero.

2. **Mean projection** — Collapse the time dimension into a single mean image, normalised to [0, 1]. This provides a structural reference for segmentation.

3. **Fibre enhancement** — Apply oriented gradient filters (horizontal and vertical) matched to the expected neurite thickness. Negative responses are rectified and the gradient magnitude is computed, producing an image where fibre-like structures are selectively amplified.

4. **Segmentation** — Threshold the enhanced image at a user-defined intensity percentile (default: 95th) to create a binary mask of candidate fibres. Small connected components below a minimum size are removed.

5. **ROI extraction** — Tile the binary mask into non-overlapping square regions (side length equal to the fibre thickness). For each tile whose centre pixel falls on the mask, compute the spatial mean across all frames to produce a 1-D temporal trace.

6. **Activity filtering** — Discard ROIs whose peak fluorescence (lightly smoothed) does not exceed the mean baseline by a minimum ratio. This removes regions with no detectable calcium transients.

7. **Output** — Return the surviving traces, their spatial coordinates, and intermediate images (mean, enhanced, mask). Optionally write traces and locations to CSV files.

## Usage

### As a library

```python
from mbo_utilities.analysis.fiber_activity import run

result = run(
    "timelapse.tif",
    fiber_thickness=3,
    frame_rate=17.58,
    activity_threshold=0.2,
    output_dir="./results",
)

# result.traces  — (n_frames, n_rois) array
# result.locations — (n_rois, 2) array of [x, y] centres
```

### From the command line

```bash
python -m mbo_utilities.analysis.fiber_activity timelapse.tif \
    --frame-rate 17.58 \
    --activity-threshold 0.2 \
    --output-dir ./results
```

## Parameters

| Parameter               | Default | Description                                                        |
|-------------------------|---------|--------------------------------------------------------------------|
| `fiber_thickness`       | 3       | Approximate neurite width in pixels; sets filter and ROI size.     |
| `percentile_threshold`  | 95.0    | Intensity percentile for binarisation of the enhanced image.       |
| `min_object_size`       | 4       | Connected components smaller than this (pixels) are discarded.     |
| `activity_threshold`    | 0.2     | Minimum (peak − mean) / mean ratio to keep an ROI.                |
| `frame_rate`            | 1.0     | Acquisition rate in Hz (used for time-axis labelling).             |
