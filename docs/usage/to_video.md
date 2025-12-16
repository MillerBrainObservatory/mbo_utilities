---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Video Export

Export calcium imaging data to video files for presentations, websites, and sharing.

```{note}
The `to_video` function creates high-quality MP4 videos optimized for web playback.
It supports both 3D `(T, Y, X)` and 4D `(T, Z, Y, X)` data, including lazy arrays.
```

## Basic Usage

```python
from mbo_utilities import imread, to_video

# load data
arr = imread("data.tif")

# basic export
to_video(arr, "output.mp4")
```

## Quick Preview (Speed Factor)

Use `speed_factor` to create fast previews - great for checking cell stability:

```python
# 10x playback speed (all frames, just faster)
to_video(arr, "preview.mp4", speed_factor=10)
```

```{tip}
`speed_factor=10` with `fps=30` results in 300 fps effective playback.
All frames are included, just displayed faster.
```

## High-Quality Export

For presentations and websites, use enhancement options:

```python
to_video(
    arr,
    "movie.mp4",
    fps=30,
    speed_factor=5,
    temporal_smooth=3,  # reduce frame-to-frame flicker
    gamma=0.8,          # brighten midtones
    quality=10,         # highest quality
)
```

### Enhancement Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temporal_smooth` | 0 | Rolling average window (frames). 3-7 = subtle, 10+ = heavy |
| `spatial_smooth` | 0 | Gaussian blur sigma (pixels). 0.5-1.0 = subtle |
| `gamma` | 1.0 | Gamma correction. <1 = brighter, >1 = darker |
| `quality` | 9 | Video quality (1-10). 9-10 recommended for web |

## Intensity Scaling

Control the intensity range for better contrast:

```python
# auto percentile (default)
to_video(arr, "auto.mp4", vmin_percentile=1, vmax_percentile=99.5)

# manual range
to_video(arr, "manual.mp4", vmin=100, vmax=2000)
```

## Colormaps

Apply matplotlib colormaps for visualization:

```python
to_video(arr, "viridis.mp4", cmap="viridis")
to_video(arr, "hot.mp4", cmap="hot")
```

```{figure} ../_images/to_video/frame_viridis.png
:width: 400px
:align: center

Example frame with viridis colormap
```

## 4D Data (Multi-plane)

For 4D data `(T, Z, Y, X)`, select which z-plane to export:

```python
# export plane 0 (default)
to_video(arr_4d, "plane0.mp4")

# export specific plane
to_video(arr_4d, "plane5.mp4", plane=5)
```

## API Reference

```{eval-rst}
.. autofunction:: mbo_utilities.to_video
   :no-index:
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array | required | 3D `(T,Y,X)` or 4D `(T,Z,Y,X)` array |
| `output_path` | str/Path | required | Output video path (.mp4, .avi, .mov) |
| `fps` | int | 30 | Base frame rate |
| `speed_factor` | float | 1.0 | Playback speed multiplier |
| `plane` | int | None | Z-plane to export (for 4D data) |
| `vmin` | float | None | Min intensity (or use percentile) |
| `vmax` | float | None | Max intensity (or use percentile) |
| `vmin_percentile` | float | 1.0 | Percentile for auto vmin |
| `vmax_percentile` | float | 99.5 | Percentile for auto vmax |
| `temporal_smooth` | int | 0 | Rolling average window size |
| `spatial_smooth` | float | 0 | Gaussian blur sigma |
| `gamma` | float | 1.0 | Gamma correction |
| `cmap` | str | None | Matplotlib colormap name |
| `quality` | int | 9 | Video quality (1-10) |
| `codec` | str | "libx264" | Video codec |
| `max_frames` | int | None | Limit frames to export |

## Examples Gallery

### Raw vs Enhanced

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Raw Export
```python
to_video(arr, "raw.mp4")
```
Default settings, no enhancement.
:::

:::{grid-item-card} Enhanced Export
```python
to_video(arr, "enhanced.mp4",
         temporal_smooth=3,
         gamma=0.8)
```
Reduced flicker, brighter midtones.
:::
::::

### Speed Comparison

::::{grid} 1 3 3 3
:gutter: 2

:::{grid-item-card} 1x Speed
```python
to_video(arr, "1x.mp4",
         speed_factor=1)
```
Real-time playback.
:::

:::{grid-item-card} 5x Speed
```python
to_video(arr, "5x.mp4",
         speed_factor=5)
```
Good for presentations.
:::

:::{grid-item-card} 10x Speed
```python
to_video(arr, "10x.mp4",
         speed_factor=10)
```
Quick stability check.
:::
::::

## Gamma Correction

Gamma < 1 brightens midtones (useful for dim calcium signals):

```{figure} ../_images/to_video/gamma_comparison.png
:width: 600px
:align: center

Gamma comparison: 1.0 (default), 0.7 (brighter), 1.3 (darker)
```

```{seealso}
- {func}`mbo_utilities.save_mp4` - Legacy function (use `to_video` instead)
- {func}`mbo_utilities.imwrite` - Save to file formats (TIFF, Zarr, HDF5)
```
