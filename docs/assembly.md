---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: lsp
  language: python
  name: python3
---

(assembly)=
# Image Assembly

Converting raw scanimage-tiff files into fused z-planes.

## Quickstart

All you need is a path to your .tiffs and a place to save them.

```python
import mbo_utilities as mbo

scan = mbo.imread(r"path/to/tiffs*")  # glob or list of filepaths

# Save options:
scan.roi = 1        # save just mROI 1
scan.roi = [1, 2]   # save mROI 1 and 2
scan.roi = 0        # save all mROIs separately
scan.roi = None     # stitch/fuse all mROIs

mbo.imwrite(scan, "/path/to/save", planes=[1, 7, 14])
# Creates: plane_01.tiff, plane_07.tiff, plane_14.tiff
```

```{code-cell} ipython3
from pathlib import Path
import numpy as np

import fastplotlib as fpl
import mbo_utilities as mbo
```

## Input data: Path to your raw .tiff file(s)

Make sure your `data_path` contains only `.tiff` files for this imaging session. If there are other `.tiff` files, such as from another session or a processed file for this session, those files will be included in the scan and lead to errors.

## Initialize a scanreader object

Pass a list of files, or a wildcard string (e.g. "/path/to/files/*" matches all files in that directory) to `mbo.imread()`.

**Tip:** `mbo.get_files()` is useful to easily get all files of the same filetype.

```{code-cell} ipython3
files = mbo.get_files("/path/to/data/raw", 'tif')
len(files)
```

```{code-cell} ipython3
scan = mbo.imread(files)
```

```{code-cell} ipython3
print(f'Planes: {scan.num_channels}')
print(f'Frames: {scan.num_frames}')
print(f'ROIs: {scan.num_rois}')
print(f'Shape (T, Z, Y, X): {scan.shape}')
```

## Accessing data in the scan

Numpy-like indexing:

```python
frame = scan[0, 0, :, :]   # first frame, plane 1
zplane7 = scan[:, 6, :, :] # all frames from z-plane 7
```

```{code-cell} ipython3
# Visualize the data
iw = scan.imshow()
iw.show()
```

```{code-cell} ipython3
iw.close()
```

## Save assembled files

The currently supported file extensions are `.tiff`, `.bin`, `.zarr`, and `.h5`.

### Basic Usage: Stitch ROIs and Save All Planes

```{code-cell} ipython3
save_path = Path("/path/to/save")
save_path.mkdir(exist_ok=True)

# Stitch all ROIs together (default behavior)
scan.roi = None
mbo.imwrite(scan, save_path, ext='.tiff')
# Creates: plane01_stitched.tiff, plane02_stitched.tiff, ..., plane14_stitched.tiff
```

### Save Specific Planes

```{code-cell} ipython3
# Save only first, middle, and last planes (for 14-plane volume)
mbo.imwrite(
    scan,
    save_path,
    planes=[1, 7, 14],
    overwrite=False,
    ext='.tiff'
)
# Creates: plane01_stitched.tiff, plane07_stitched.tiff, plane14_stitched.tiff
```

### ROI Handling Options

```{code-cell} ipython3
# Option 1: Stitch all ROIs together (default)
scan.roi = None
mbo.imwrite(scan, save_path / "stitched")
# Creates: plane01_stitched.tiff, plane02_stitched.tiff, ...

# Option 2: Save all ROIs as separate files
scan.roi = 0
mbo.imwrite(scan, save_path / "split_rois")
# Creates: plane01_roi1.tiff, plane01_roi2.tiff, ..., plane14_roi1.tiff, plane14_roi2.tiff

# Option 3: Save specific ROI only
scan.roi = 1
mbo.imwrite(scan, save_path / "roi1_only")
# Creates: plane01_roi1.tiff, plane02_roi1.tiff, ..., plane14_roi1.tiff

# Option 4: Save multiple specific ROIs
scan.roi = [1, 3]
mbo.imwrite(scan, save_path / "roi1_and_roi3")
# Creates: plane01_roi1.tiff, plane01_roi3.tiff, ..., plane14_roi1.tiff, plane14_roi3.tiff
```

### Enable Phase Correction

Fix bidirectional scan phase artifacts before saving:

```{code-cell} ipython3
scan.fix_phase = True
scan.phasecorr_method = "mean"  # Options: "mean", "median", "max"
scan.use_fft = True  # Use FFT-based correction (faster)
mbo.imwrite(scan, save_path / "phase_corrected")
```

### Z-Plane Registration

Align z-planes spatially using Suite3D registration:

```{code-cell} ipython3
# Automatic registration with Suite3D (requires Suite3D + CuPy installed)
mbo.imwrite(
    scan,
    save_path / "registered",
    register_z=True,
    roi=None
)
# Computes rigid shifts and applies them during write
# Validates registration results before proceeding
```

### Export Subset of Frames

Useful for testing or creating demos:

```{code-cell} ipython3
# Export only first 1000 frames
mbo.imwrite(scan, save_path / "test_data", num_frames=1000, planes=[1, 7, 14])
```

### Convert to Suite2p Binary Format

```{code-cell} ipython3
# Save as Suite2p-compatible binary files
mbo.imwrite(scan, save_path / "suite2p", ext='.bin', roi=0)
# Creates: plane01_roi1/data_raw.bin + ops.npy, plane01_roi2/data_raw.bin + ops.npy, ...
```

### Save to Zarr Format

```{code-cell} ipython3
# Save as Zarr v3 stores (efficient for large datasets)
mbo.imwrite(scan, save_path / "zarr_data", ext='.zarr', roi=0)
# Creates: plane01_roi1.zarr, plane01_roi2.zarr, ...
```

### Use Pre-Computed Registration Shifts

```{code-cell} ipython3
# Load previously computed shifts
import numpy as np
summary = np.load("previous_job/summary/summary.npy", allow_pickle=True).item()
shift_vectors = summary['plane_shifts']  # shape: (n_planes, 2)

# Apply shifts without re-running registration
mbo.imwrite(scan, save_path / "registered", shift_vectors=shift_vectors)
```

### Example Output

```text
Saving plane01_stitched.tiff: 100%|██████████| 108/108 [02:43<00:00,  1.51s/it]
Saving plane07_stitched.tiff: 100%|██████████| 108/108 [02:42<00:00,  1.51s/it]
Saving plane14_stitched.tiff: 100%|██████████| 108/108 [02:41<00:00,  1.50s/it]
```

## Vizualize data with [fastplotlib](https://www.fastplotlib.org/user_guide/guide.html#what-is-fastplotlib)

To get a rough idea of the quality of your extracted timeseries, we can create a fastplotlib visualization to preview traces of individual pixels.

Here, we simply click on any pixel in the movie, and we get a 2D trace (or "temporal component" as used in this field) of the pixel through the course of the movie:

More advanced visualizations can be easily created, i.e. adding a baseline subtracted element to the trace, or passing the trace through a frequency filter.

```{code-cell} ipython3
import tifffile
from ipywidgets import VBox

img = mbo.imread("path/to/assembled/plane07_stitched.tiff")
iw_movie = fpl.ImageWidget(img, cmap="viridis")

tfig = fpl.Figure()

raw_trace = tfig[0, 0].add_line(np.zeros(img.shape[0]))

@iw_movie.managed_graphics[0].add_event_handler("click")
def pixel_clicked(ev):
    col, row = ev.pick_info["index"]
    raw_trace.data[:, 1] = iw_movie.data[0][:, row, col]
    tfig[0, 0].auto_scale(maintain_aspect=False)

VBox([iw_movie.show(), tfig.show()])
```

