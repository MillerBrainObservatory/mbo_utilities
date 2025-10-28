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

`````{admonition} TLDR
:class: dropdown

If all you want is code to get started, this will 
``` python
import mbo_utilities as mbo
scan = mbo.imread("path/to/data/*.tiff") # or list of full filepaths
mbo.imwrite(scan, "path/to/assembled_data")
```

```{figure} ./_images/progress_bar.png
A progress bar will track the current file progress.
```
``````

## Overview

This section covers the steps to convert raw scanimage-tiff files into assembled, planar timeseries.

```{figure}  ./_images/assembly_1.png
:align: center

Overview of pre-processing steps that convert the raw scanimage tiffs into planar timeseries.
Starting with a raw, multi-page ScanImage Tiff, frames are {ref}`deinterleaved <ex_deinterleave>`, optionally pre-processed to eliminate scan-phase artifacts,
and fused to create an assembled timeseries. 
```

----

```{code-cell} ipython3
# imports
from pathlib import Path
import numpy as np

import fastplotlib as fpl
import mbo_utilities as mbo
```

## Input data: Path to your raw .tiff file(s)

```{admonition} One session per folder
:class: dropdown

Make sure your `data_path` contains only `.tiff` files for this imaging session.
If there are other `.tiff` files, such as from another session or a processed file for this session, those files will be included in the scan and lead to errors.

```

+++

## Ini

Pass a list of files, or a wildcard "/path/to/files/*" to `mbo.read_scan()`.

``` {tip}
{func}`mbo_utilities.get_files` is useful to easily get all files of the same filetype.
By default, this will ensure your raw tiffs are numerically sorted in order of acquisition time.
```

```{code-cell} ipython3
files = mbo.get_files("path//to//animal_01//raw", 'tif')
files[:2]

['/home/flynn/lbm_data/raw/mk301_03_01_2025_2roi_17p07hz_224x448px_2umpx_180mw_green_00001.tif',
 '/home/flynn/lbm_data/raw/mk301_03_01_2025_2roi_17p07hz_224x448px_2umpx_180mw_green_00002.tif']
```

```{code-cell} ipython3
scan = mbo.imread(files)

# T, Z, X, Y
scan.shape
```

```{code-cell} ipython3
print(f'Planes: {scan.num_channels}')
print(f'Planes: {scan.num_planes}')  # same as num_channels for MBO recordings
print(f'Frames: {scan.num_frames}')
print(f'ROIs: {scan.num_rois}')
print(f'Frame-Rate: {scan.frame_rate}')
print(f'Spatial Resolution: {scan.frame_rate}')
```

## Accessing data in the scan

The scan can be indexed like a numpy array, data will be loaded lazily as only the data you access here is loaded in memory.

```{code-cell} ipython3
# load the first 6 frames (0 to 5), the first z-plane, and all X/Y pixels
array = scan[:5, 0, :, :]
print(f'[T, X, Y]: {array.shape}')

# load a z stack, with a single frame for each Z
array = scan[0, :, :, :]
print(f'[Z, X, Y]: {array.shape}')
```

```{admonition} A note on performance
:class: dropdown

When you initialize a scan with `read_scan`, [tifffile](https://github.com/cgohlke/tifffile/blob/master/tifffile/tifffile.py) is going to iterate through every page in your tiff to "count" how many pages there are.
. Only a single page of data is held in memory, and using that information we can lazily load the scan (this is what the scanreader does).

For a single 35 Gb file, this process takes ~10 seconds.
For 216 files totaling 231 GB, ~ 2 minutes.

This only occurs once, and is cached by your operating system. So the next time you read the same scan, a 35GB file will be nearly instant, and a series of 216 files ~8 seconds.

```

```{code-cell} ipython3
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(np.mean(scan[:100, 6, :, :], axis=0))
plt.title('Mean Image for first 100 frames')
plt.show()
```

This will display a widget allowing you to scroll in time and in z.

```{code-cell} ipython3
image_widget = mbo.run_gui(scan)
image_widget.show()
```

```{code-cell} ipython3
image_widget.close()
```

## Save assembled files

The currently supported file extensions are `.tiff`, `.bin`, `.zarr`, and `.h5`.

### Basic Usage: Stitch ROIs and Save All Planes

```{code-cell} ipython3
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

img = mbo.imread("path/to/assembled/plane_07.tiff")
iw_movie = fpl.ImageWidget(img, cmap="viridis")

tfig = fpl.Figure()

raw_trace = tfig[0, 0].add_line(np.zeros(img.shape[0]))

@iw_movie.managed_graphics[0].add_event_handler("click")
def pixel_clicked(ev):
    col, row = ev.pick_info["index"]
    raw_trace.data[:, 1] =  iw_movie.data[0][:, row, col]
    tfig[0, 0].auto_scale(maintain_aspect=False)

VBox([iw_movie.show(), tfig.show()])
```

## Under the hood

```{figure}  ./_images/ex_diagram.png
```

