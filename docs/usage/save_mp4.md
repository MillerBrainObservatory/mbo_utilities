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
# `save_mp4`

{func}`mbo_utilities.save_mp4`

`save_mp4` converts a 3D numpy array or TIFF stack (`[T, Y, X]`) into an `.mp4` video.  

It supports optional temporal smoothing, playback speed adjustment, and colormaps.

```{code-cell} ipython3
from pathlib import Path
import mbo_utilities as mbo
import tifffile
```

## Load Data

```{code-cell} ipython3
save_path = Path().home().joinpath("dev")
files = mbo.get_files(save_path, 'tif', 4)
data = tifffile.imread(files[0])
data.shape  # should be [T, Y, X]
```

## Example Usage

::::{tab-set}

:::{tab-item} Default

```python
import mbo_utilities.plot_util

mbo_utilities.plot_util.save_mp4(save_path / "default.mp4", data)
```
:::

:::{tab-item} 2× Speedup

```python
import mbo_utilities.plot_util

mbo_utilities.plot_util.save_mp4(save_path / "speedup_2x.mp4", data, speedup=2)
```
:::

:::{tab-item} 4× Speedup

```python
import mbo_utilities.plot_util

mbo_utilities.plot_util.save_mp4(save_path / "speedup_4x.mp4", data, speedup=4)
```
:::

:::{tab-item} Smoothing + Speedup

```python
import mbo_utilities.plot_util

mbo_utilities.plot_util.save_mp4(save_path / "windowed_5frames_4x.mp4", data, speedup=4, win=5)
```
:::

::::  

---

## Parameters

```{code-block} python
mbo.save_mp4(
    fname,         # output filename
    images,        # 3D array or TIFF path
    framerate=17,
    speedup=1,
    chunk_size=100,
    cmap="gray",
    win=7,
    vcodec="libx264",
    normalize=True
)
```

