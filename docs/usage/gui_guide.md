---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# GUI User Guide

```{contents}
:local:
:depth: 2
```

## Overview

The MBO GUI provides interactive data preview and processing tools for calcium imaging data.

**Key Features:**
- Time and Z-plane sliders for navigation
- Window functions (mean, max, std, mean-subtracted)
- Scan-phase correction preview
- Multi-ROI, multi-Z-plane statistics
- V-Min/V-Max contrast controls
- Suite2p processing integration
- Export to .tiff, .zarr, .bin, .h5

```{tip}
Hover over `(?)` icons for tooltips. Many buttons also show tooltips on hover.
```

## Installation

1. Create a virtual environment with uv:

```bash
uv venv --python 3.12.9
```

2. Install mbo_utilities (includes cellpose, suite2p, suite3d, graphical utilities):

```bash
uv pip install mbo_utilities>=2.0.1
```

3. Start the GUI:

```bash
uv run mbo
```

```{note}
If you encounter an `io.ImFontAtlas` error, run:
`uv pip install git+https://github.com/pygfx/pygfx.git@main`
```

## Data Selection

The data selection dialog opens when you run `mbo` without arguments.

**Supported filetypes:**
- `.tiff` - Raw ScanImage tiffs, BigTIFF, OME-TIFF
- `.zarr` - Zarr v3 arrays
- `.bin` - Suite2p binary format
- `.h5` - HDF5 files

```{important}
The **Data Preview widget** is only available for raw ScanImage tiffs. Non-ScanImage tiffs open as generic 3D/4D arrays.
```

### Open File vs Select Folder

**Open File(s):**
- Shows a file dialog where you can select one or multiple files
- Useful for selecting specific tiff files

**Select Folder:**
- Shows a folder-only dialog (files are hidden)
- All supported files in that folder will be loaded
- Useful for large datasets split across many files

### Load Options

These options only affect visualization (no data is modified on disk).

#### Separate ScanImage mROIs

When checked:
- Opens multi-ROI acquisitions as separate arrays
- Each mROI gets its own scan-phase offset calculation
- Useful for comparing phase offsets between ROIs

When unchecked:
- mROIs are vertically concatenated (stitched)
- Single phase-offset value calculated for the combined image

#### Enable Threading

Enables parallel loading for faster data access.

#### Enable Data Preview Widget

When checked, opens the full preview widget with window functions and scan-phase controls.

#### Metadata Preview Only

Opens a metadata viewer showing:
- Consolidated ScanImage metadata
- Raw metadata from tiff headers
- Suite2p `ops.npy` parameters (if applicable)

Metadata is shown in collapsible subgroups for easy navigation.

## Preview Data Widget

```{note}
This widget is only available for raw ScanImage tiffs.
```

### Loading Performance

- Large tiffs (e.g., 60,000 x 14 x 400 x 400) may take ~30s on first load
- Subsequent loads use caching and complete in <5s

### Window Functions

View temporal projections over a sliding window:

| Function | Description |
|----------|-------------|
| `mean` | Average intensity over window |
| `max` | Maximum intensity projection |
| `std` | Standard deviation |
| `mean-sub` | Mean-subtracted (highlights changes) |

**Parameters:**
- **Window Size**: Number of frames to include (3-20 recommended)
- **sigma**: Spatial gaussian filter (matches suite2p `smooth_sigma`)

```{warning}
Windows > 20 frames will slow rendering significantly.
```

### Contrast Controls

Dynamic range varies between projections. Two auto-contrast options:

- **Single frame**: Calculate vmin/vmax from current frame (fast)
- **Full dataset**: Calculate from all frames (slow for large data)

You can also manually adjust vmin/vmax using the histogram widget on the right side of the image.

### Scan-Phase Correction

Preview bidirectional scan-phase correction before saving.

**Controls:**

| Parameter | Description |
|-----------|-------------|
| **Fix Phase** | Enable/disable correction |
| **Sub-Pixel** | FFT-based sub-pixel correction (slower, more precise) |
| **Upsample** | Sub-pixel precision (1/N pixel). Higher = more precise, slower |
| **Exclude border-px** | Pixels to exclude from edges (galvo nonlinearity) |
| **max-offset** | Limit allowed offset (prevents outliers in noisy frames) |

**Recommended workflow:**

1. View a mean-subtracted projection (window size 3-15)
2. Toggle Fix Phase on/off to compare
3. Adjust border-px and max-offset for stable correction
4. Toggle Sub-Pixel to see if it improves results
5. If Sub-Pixel helps, adjust Upsample (typically 2-3 is sufficient)

```{tip}
Offsets rarely exceed 2-3 pixels. Use max-offset to prevent noisy frames from producing large spurious offsets.
```

### Summary Stats

Click the **Summary Stats** tab to compute statistics per z-plane:

| Metric | Description |
|--------|-------------|
| Mean | Average intensity |
| Std | Standard deviation |
| SNR | Signal-to-noise ratio (mean / std) |

Statistics are computed on every 10th frame. Views available:
- Per-array (individual mROIs)
- Combined (mean across all mROIs)

**Use cases:**
- Evaluate z-plane quality vs depth
- Compare mROI quality across the FOV

```{note}
These mean values are used for the mean-subtracted image preview.
```

## Saving Data

Access via **File → Save As** or the **Process** tab.

### Save Dialog Options

**Output formats:**
- `.zarr` - Zarr v3 (recommended for large data)
- `.tiff` - BigTIFF
- `.bin` - Suite2p binary format
- `.h5` - HDF5

**Options:**

| Option | Description |
|--------|-------------|
| Save mROI Separately | Export each mROI as a separate file |
| Overwrite | Replace existing output files |
| Register Z-Planes Axially | Apply Suite3D axial registration |
| Fix Scan Phase | Apply phase correction on export |
| Subpixel Phase Correction | Use FFT-based sub-pixel correction |
| Debug | Verbose logging |
| Chunk Size (MB) | Memory chunk size for streaming writes |

**Selection:**
- Choose specific mROIs to export
- Choose specific z-planes to export

```{note}
I/O speed depends on file size, number of source tiffs, output format, and FFT/upsampling settings.
```

## Reading Extracted Data

To open previously exported data (zarr, tiff, bin, h5):

1. **Disable** the "Enable Data Preview widget" option
2. Select the file or folder
3. Opens as a generic array viewer

```{warning}
Leaving Data Preview enabled for non-raw-ScanImage data will cause an error.
```

## Suite2p Processing

Access via the **Process** tab.

**Features:**
- Run Suite2p on the currently selected z-plane
- All suite2p parameters exposed with descriptions
- Use the crop selector to process a spatial subset

### Spatial Crop Selection

1. Click "Add Crop Selector" in the Process tab
2. Drag the yellow rectangle on the image to define the ROI
3. Only the cropped region will be processed

### Registration Settings

| Parameter | Description |
|-----------|-------------|
| Do Registration | Enable motion correction |
| Align by Channel | Channel to use for alignment |
| Initial Frames | Frames for reference image |
| Batch Size | Frames per batch |
| Max Shift Fraction | Maximum allowed shift |
| Smooth Sigma | Spatial smoothing for registration |
| Smooth Sigma Time | Temporal smoothing |
| Two-Step Registration | Coarse then fine registration |
| Subpixel Precision | Sub-pixel registration accuracy |
| Bad Frame Threshold | Threshold for rejecting frames |
| Normalize Frames | Normalize intensity before registration |

### Output Options

| Option | Description |
|--------|-------------|
| Keep Raw Movie | Retain unregistered data |
| Export Registered TIFF | Save registered movie as tiff |
| Export Chan2 TIFF | Export second channel |
| Force refImg | Force reference image recalculation |
| Pad FFT | Pad for FFT efficiency |
