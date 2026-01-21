(gui_guide)=

# GUI User Guide

Interactive data preview and processing tools for calcium imaging data.

```{image} /_images/gui/readme/02_step_data_view.png
:width: 100%
:alt: MBO Utilities GUI
```

## Quick Start

```bash
uv pip install mbo_utilities
mbo                    # opens file dialog
mbo /path/to/data      # opens specific file
mbo /path --metadata   # metadata only
```

## Features

- time and z-plane sliders
- window functions (mean, max, std, mean-subtracted)
- scan-phase correction preview
- multi-ROI statistics
- contrast controls (vmin/vmax)
- suite2p processing integration
- export to .tiff, .zarr, .bin, .h5

## Supported Formats

| Format | Description |
|--------|-------------|
| `.tiff` | raw scanimage, bigtiff, ome-tiff |
| `.zarr` | zarr v3 arrays |
| `.bin` | suite2p binary format |
| `.h5` | hdf5 files |

```{note}
The full **Data Preview widget** is only available for raw ScanImage tiffs.
```

## Data Selection Dialog

```{image} /_images/gui/readme/01_step_file_dialog.png
:width: 80%
:alt: File Selection Dialog
```

### Open File vs Select Folder

- **Open File(s)**: select specific tiff files
- **Select Folder**: load all supported files in folder

### Load Options

| Option | Description |
|--------|-------------|
| Separate ScanImage mROIs | split multi-ROI acquisitions |
| Enable Threading | parallel loading |
| Enable Data Preview Widget | full preview with window functions |
| Metadata Preview Only | show only metadata |

## Preview Widget

### Window Functions

| Function | Description |
|----------|-------------|
| mean | average intensity over window |
| max | maximum intensity projection |
| std | standard deviation |
| mean-sub | mean-subtracted (highlights changes) |

**Parameters:**

- **Window Size**: frames to include (3-20 recommended)
- **sigma**: spatial gaussian filter

### Scan-Phase Correction

Preview bidirectional phase correction before saving.

| Parameter | Description |
|-----------|-------------|
| Fix Phase | enable/disable correction |
| Sub-Pixel | FFT-based sub-pixel correction |
| Upsample | sub-pixel precision (1/N pixel) |
| Exclude border-px | exclude edge pixels |
| max-offset | limit allowed offset |

**Workflow:**

1. view mean-subtracted projection (window 3-15)
2. toggle Fix Phase on/off to compare
3. adjust border-px and max-offset
4. toggle Sub-Pixel for improvement
5. adjust Upsample if needed (2-3 typical)

### Summary Stats

Per z-plane statistics (computed on every 10th frame):

| Metric | Description |
|--------|-------------|
| Mean | average intensity |
| Std | standard deviation |
| SNR | signal-to-noise (mean/std) |

## Saving Data

Access via **File → Save As** or **Process** tab.

```{image} /_images/gui/readme/04_save_as_dialog.png
:width: 80%
:alt: Save As Dialog
```

### Output Formats

| Format | Description |
|--------|-------------|
| `.zarr` | recommended for large data |
| `.tiff` | bigtiff |
| `.bin` | suite2p binary format |
| `.h5` | hdf5 |

### Save Options

| Option | Description |
|--------|-------------|
| Overwrite | replace existing files |
| Register Z-Planes | suite3d axial registration |
| Fix Scan Phase | apply phase correction |
| Subpixel Phase Correction | FFT-based correction |
| Chunk Size (MB) | memory chunk size |

## Suite2p Processing

Access via **Process** tab.

```{image} /_images/gui/readme/06_suite2p_settings.png
:width: 80%
:alt: Suite2p Processing Settings
```

- run suite2p on selected z-plane
- all parameters exposed with descriptions
- crop selector for spatial subset

### Spatial Crop

1. click "Add Crop Selector"
2. drag yellow rectangle on image
3. only cropped region is processed

## Python API

```python
from mbo_utilities.gui import run_gui

# from file
run_gui("/path/to/data")

# from numpy array
import numpy as np
data = np.random.rand(100, 512, 512)
run_gui(data)
```

If no input provided and Qt available, opens file dialog.
