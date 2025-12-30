(cli_usage)=

# Command Line Interface

The `mbo` command provides tools for viewing, converting, and analyzing imaging data.

| Command | Description |
|---------|-------------|
| `mbo` | Launch GUI with file dialog |
| `mbo view` | View data in GUI |
| `mbo convert` | Convert between formats |
| `mbo info` | Show array info |
| `mbo scanphase` | Analyze scan-phase offset |
| `mbo download` | Download files from GitHub |
| `mbo formats` | List supported formats |

## GUI Mode

```bash
mbo                            # file dialog
mbo /path/to/data              # open specific file
mbo /path/to/data --metadata   # show only metadata
mbo view /data --roi 0 --roi 2 # view specific ROIs
```

```{image} /_images/gui/readme/01_step_file_dialog.png
:width: 340px
:align: center
```

## Convert

Convert between formats with optional processing.

```bash
mbo convert input.tiff output/ -e .zarr           # tiff to zarr
mbo convert input.tiff output/ -e .bin            # tiff to suite2p binary
mbo convert input.zarr output/ -e .tiff           # zarr to tiff
mbo convert input.tiff output/ -e .zarr -p 1 -p 7 # specific planes
mbo convert input.tiff output/ --fix-phase        # with phase correction
mbo convert input.tiff output/ -n 1000            # first 1000 frames
```

| Option | Description |
|--------|-------------|
| `-e, --ext` | Output format: `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy` |
| `-p, --planes` | Z-planes to export (1-based), repeatable |
| `-n, --num-frames` | Limit number of frames |
| `--roi` | ROI selection: `None`, `0`, `N`, or `"1,3"` |
| `--fix-phase` | Bidirectional phase correction |
| `--overwrite` | Replace existing files |

<details>
<summary><b>All Convert Options</b></summary>

| Option | Description |
|--------|-------------|
| `--output-suffix` | Custom filename suffix (default: `_stitched` for multi-roi) |
| `--phasecorr-method` | `mean`, `median`, or `max` |
| `--register-z` | Z-plane registration via suite3d |
| `--ome/--no-ome` | OME-zarr metadata (zarr only) |
| `--chunk-mb` | Streaming chunk size (default: 100) |
| `--debug` | Verbose logging |

</details>

## Info

Display array shape, dtype, chunk info, and metadata without loading data.

```bash
mbo info /data/raw.tiff
mbo info /data/volume.zarr
mbo info /data/suite2p/plane0
```

## Scan-Phase Analysis

Bidirectional resonant scanning causes alternating rows to be shifted horizontally. This tool measures that shift to help configure correction parameters.

```bash
mbo scanphase                        # file dialog
mbo scanphase /path/to/data.tiff     # analyze file
mbo scanphase ./folder/ -n 5         # first 5 tiffs
mbo scanphase data.tiff -o ./results # custom output
mbo scanphase data.tiff --show       # show plots
mbo scanphase data.tiff --format pdf # output as pdf
```

**Output Files:**

::::{card-carousel} 2

:::{card} temporal.png
Offset time series + histogram. Should be flat; large jumps indicate motion/hardware issues. Typical offset: 0.5-2.0 px.

```{image} /_images/scanphase/temporal.png
:width: 100%
```
:::

:::{card} windows.png
Offset vs window size. Shows how estimate stabilizes with more frames. Red line marks where std drops below 0.1 px.

```{image} /_images/scanphase/windows.png
:width: 100%
```
:::

:::{card} spatial.png
Spatial heatmaps (32x32 and 64x64 patches). Edges different from center is normal. Gray = low signal.

```{image} /_images/scanphase/spatial.png
:width: 100%
```
:::

:::{card} zplanes.png
Offset vs depth (if multi-plane). Assess if offset varies with z due to resonant scanner angle.

```{image} /_images/scanphase/zplanes.png
:width: 100%
```
:::

:::{card} parameters.png
Offset vs signal intensity. Red line suggests threshold below which measurements are noisy.

```{image} /_images/scanphase/parameters.png
:width: 100%
```
:::

::::

**Tips:** Use `-n 2` or `-n 3` to run quickly on a subset of frames. Multi-ROI data: offsets are averaged across ROIs.

## Download

Download files from GitHub (auto-converts blob to raw URLs).

```bash
mbo download https://github.com/user/repo/blob/main/notebook.ipynb
mbo download https://github.com/user/repo/blob/main/data.npy -o ./data/
```

```
Downloading from:
  https://raw.githubusercontent.com/.../quickstart.ipynb
Saving to:
  C:\Users\...\quickstart.ipynb

Successfully downloaded: quickstart.ipynb
```

```{image} /_images/cli/jupyter_lab.png
:width: 80%
:align: center
```

## Utilities

```bash
mbo --check-install      # verify installation and GPU config
mbo --download-notebook  # download user guide notebook
mbo --download-file URL  # download any file
```

```
mbo_utilities v2.4.3 | Python 3.12.12
==================================================

CUDA Environment:
  Driver CUDA:         12.6

Features:
  [✓] PyTorch
  [✓] CuPy
  [ ] Suite2p (not installed)
  [ ] Suite3D (not installed)
  [ ] Rastermap (not installed)
```

## Formats

```bash
mbo formats
```

**Input:** `.tif`, `.tiff`, `.zarr`, `.bin`, `.h5`, `.hdf5`, `.npy`, `.json`
**Output:** `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy`

## Upgrade

| Method | Command |
|--------|---------|
| Install script | Re-run install script |
| CLI only | `uv tool upgrade mbo_utilities` |
| Virtual env | `uv pip install -U mbo_utilities` |
