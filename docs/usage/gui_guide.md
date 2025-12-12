(gui_guide)=

# GUI User Guide

Interactive data preview and processing tools for calcium imaging data.

## Quick Start

```bash
uv pip install mbo_utilities
mbo                    # opens file dialog
mbo /path/to/data      # opens specific file
mbo /path/to/data --metadata   # metadata only
```

## Data Selection Dialog

### Open File vs Select Folder

- **Open File(s)**: select specific tiff files
- **Select Folder**: load all supported files in folder

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

### Summary Stats

Per z-plane statistics (computed on every 10th frame):

| Metric | Description |
|--------|-------------|
| Mean | average intensity |
| Std | standard deviation |
| SNR | signal-to-noise (mean/std) |

## Saving Data

Access via **File → Save As** or **Process** tab.

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
| Save mROI Separately | separate file per mROI |
| Overwrite | replace existing files |
| Register Z-Planes | suite3d axial registration |
| Chunk Size (MB) | memory chunk size |

## Suite2p Processing

Access via **Run** tab.

- run suite2p on selected z-plane or multi-zplane
- all parameters exposed with descriptions

## Python API

```python
from mbo_utilities.graphics import run_gui

# from file
run_gui("/path/to/data")

# from numpy array
import numpy as np
data = np.random.rand(100, 512, 512)
run_gui(data)
```

If no input provided and Qt available, opens file dialog.
