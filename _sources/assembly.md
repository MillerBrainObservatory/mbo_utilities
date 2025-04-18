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

`````{admonition} TL;DR
:class: dropdown

If all you want is code to get started, this will 
``` python
import mbo_utilities as mbo
scan = mbo.read_scan("path/to/data/*.tiff")
mbo.save_as(scan, "path/to/assembled_data", ext=".tiff")
```
``````

## Overview 

This section covers the steps to convert raw scanimage-tiff files into assembled, planar timeseries.

```{figure}  ./_images/ex_diagram.png
:align: center

Overview of pre-processing steps that convert the raw scanimage tiffs into planar timeseries.
Starting with a raw, multi-page ScanImage Tiff, frames are deinterleaved, optionally pre-processed to eliminate scan-phase artifacts,
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

## Initialize a scanreader object

Pass a list of files, or a wildcard "/path/to/files/*" to `mbo.read_scan()`.

``` {tip}
{func}`mbo_utilities.get_files` is useful to easily get all files of the same filetype
```

```{code-cell} ipython3
files = mbo.get_files("/home/flynn/lbm_data/mk301", 'tif')
files[:2]
```

```{code-cell} ipython3
scan = mbo.read_scan(files)

# T, Z, X, Y
scan.shape
```

```{code-cell} ipython3
print(f'Planes: {scan.num_channels}')
print(f'Frames: {scan.num_frames}')
print(f'ROIs: {scan.num_rois}')
print(f'frame-rate: {scan.fps}')
```

## Accessing data in the scan

The scan can be indexed like a numpy array, data will be loaded lazily as only the data you access here is loaded in memory.

```{code-cell} ipython3
# load the first 6 frames (0 to 5), the first z-plane, and all X/Y pixels
array = scan[:5, 0, :, :]
print(f'[T, X, Y]: {array.shape}')

# load a z stack, with a single frame for each Z
array = scan[0, :, :, :]
# 4 fields (join_contiguous=False), one for each ROI
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

The currently supported file extensions are `.tiff`.

```{code-cell} ipython3
save_path = Path("/home/flynn/lbm_data/mk301/assembled")
save_path.mkdir(exist_ok=True)

mbo.save_as(
    scan,
    save_path,
    planes=[0, 6, 13],      # for 14 z-planes, first, middle, last 
    overwrite=True,
    ext = '.tiff',
    trim_edge=[2, 2, 2, 2], # post-assembly pixels to trim [left, right, top, bottom]
    append_str="_demo",      # this will be appended to the filename, 'plane_N_demo.ext'
    fix_phase=False          # fix bi-directional scan phase offset
)
```

## Vizualize data with [fastplotlib](https://www.fastplotlib.org/user_guide/guide.html#what-is-fastplotlib)

To get a rough idea of the quality of your extracted timeseries, we can create a fastplotlib visualization to preview traces of individual pixels.

Here, we simply click on any pixel in the movie, and we get a 2D trace (or "temporal component" as used in this field) of the pixel through the course of the movie:

```{figure} ./_images/ex_diagram.png 
:align: center
```

More advanced visualizations can be easily created, i.e. adding a baseline subtracted element to the trace, or passing the trace through a frequency filter.

```{code-cell} ipython3
import tifffile
from ipywidgets import VBox

img = tifffile.memmap("/home/flynn/lbm_data/mk301/assembled/plane_07_demo.tiff")
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

```{code-cell} ipython3

```
