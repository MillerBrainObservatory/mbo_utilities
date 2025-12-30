(cli_usage)=

# Command Line Interface

The `mbo` command provides tools for viewing, converting, and analyzing imaging data.

## Commands

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

````{list-table}
:widths: 50 50

* - ```bash
    mbo                            # file dialog
    mbo /path/to/data              # open specific file
    mbo /path/to/data --metadata   # show only metadata
    mbo view /data --roi 0 --roi 2 # view specific ROIs
    ```
  - ```{image} /_images/gui/readme/01_step_file_dialog.png
    :height: 280px
    ```
````

## Convert

Convert between formats with optional processing.

````{list-table}
:widths: 50 50

* - ```bash
    mbo convert input.tiff output/ -e .zarr           # tiff to zarr
    mbo convert input.tiff output/ -e .bin            # tiff to suite2p binary
    mbo convert input.zarr output/ -e .tiff           # zarr to tiff
    mbo convert input.tiff output/ -e .zarr -p 1 -p 7 # specific planes
    mbo convert input.tiff output/ --fix-phase        # with phase correction
    mbo convert input.tiff output/ -n 1000            # first 1000 frames
    mbo convert input.tiff output/ --output-suffix _session1
    ```
  - | Option | Description |
    |--------|-------------|
    | `-e, --ext` | Output format: `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy` |
    | `-p, --planes` | Z-planes to export (1-based), repeatable |
    | `-n, --num-frames` | Limit number of frames |
    | `--roi` | ROI selection: `None`, `0`, `N`, or `"1,3"` |
    | `--output-suffix` | Custom filename suffix |
    | `--fix-phase` | Bidirectional phase correction |
    | `--overwrite` | Replace existing files |
````

<details>
<summary><b>All Convert Options</b></summary>

| Option | Description |
|--------|-------------|
| `-e, --ext` | Output format: `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy` |
| `-p, --planes` | Z-planes to export (1-based), repeatable |
| `-n, --num-frames` | Limit number of frames |
| `--roi` | ROI selection: `None`, `0`, `N`, or `"1,3"` |
| `--output-suffix` | Custom filename suffix (default: `_stitched` for multi-roi) |
| `--fix-phase/--no-fix-phase` | Bidirectional phase correction |
| `--phasecorr-method` | `mean`, `median`, or `max` |
| `--register-z` | Z-plane registration via suite3d |
| `--ome/--no-ome` | OME-zarr metadata (zarr only) |
| `--overwrite` | Replace existing files |
| `--chunk-mb` | Streaming chunk size (default: 100) |
| `--debug` | Verbose logging |

</details>

## Info

Display array information without loading data into memory.

````{list-table}
:widths: 50 50

* - ```bash
    mbo info /data/raw.tiff
    mbo info /data/volume.zarr
    mbo info /data/suite2p/plane0
    ```
  - Shows shape, dtype, chunk info, and metadata for any supported format.
````

## Scan-Phase Analysis

Bidirectional resonant scanning causes alternating rows to be shifted horizontally. This tool measures that shift to help configure correction parameters.

````{list-table}
:widths: 50 50

* - ```bash
    mbo scanphase                        # file dialog
    mbo scanphase /path/to/data.tiff     # analyze file
    mbo scanphase ./folder/ -n 5         # first 5 tiffs
    mbo scanphase data.tiff -o ./results # custom output
    mbo scanphase data.tiff --show       # show plots
    mbo scanphase data.tiff --format pdf # output as pdf
    ```
  - **Output Files:**

    | File | Description |
    |------|-------------|
    | `temporal.png` | Offset time series + histogram |
    | `windows.png` | Offset vs window size (convergence) |
    | `spatial.png` | Spatial heatmaps at 32x32 and 64x64 patches |
    | `zplanes.png` | Offset vs depth (if multi-plane) |
    | `parameters.png` | Offset vs signal intensity |
    | `scanphase_results.npz` | Numerical data |
````

### Interpreting Results

**temporal.png**: Time series should be flat. Large jumps indicate motion or hardware issues. Typical offset is 0.5-2.0 px.

**windows.png**: Shows how estimate stabilizes with more frames. Left plot: offset converges to stable value. Right plot: variance decreases with window size. Red line marks where std drops below 0.1 px. Use this to determine how many frames to average for correction.

**spatial.png**: Heatmaps show variation across FOV. Edges different from center is normal. Gray = low signal (unreliable).

**zplanes.png**: Assess if offset varies with depth, owing to the angle on the resonant scanner.

**parameters.png**: Shows offset reliability vs signal. Low signal = unreliable (high/variable offset). Red line suggests intensity threshold below which measurements are noisy.

### Tips

- Use `-n 2` or `-n 3` to run quickly on subset of frames
- Multi-ROI data: offsets are averaged across ROIs

## Download

Download files from GitHub (auto-converts blob to raw URLs).

````{list-table}
:widths: 50 50

* - ```bash
    mbo download https://github.com/.../notebook.ipynb
    mbo download https://github.com/.../data.npy -o ./data/
    ```
  - ```
    Downloading from:
      https://raw.githubusercontent.com/.../notebook.ipynb
    Saving to:
      C:\Users\...\quickstart.ipynb

    Successfully downloaded: quickstart.ipynb
    ```
````

```{image} /_images/cli/jupyter_lab.png
:width: 80%
:align: center
```
<p align="center"><em>Downloaded quickstart notebook opened in JupyterLab</em></p>

## Utilities

````{list-table}
:widths: 50 50

* - ```bash
    mbo --check-install
    ```
  - ```
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
````

```bash
mbo --download-notebook  # download user guide notebook
mbo --download-file URL  # download any file
```

## Formats

Show supported file formats:

```bash
mbo formats
```

**Input formats:** `.tif`, `.tiff`, `.zarr`, `.bin`, `.h5`, `.hdf5`, `.npy`, `.json`
**Output formats:** `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy`

## Upgrade

| Method | Command |
|--------|---------|
| Install script | Re-run install script |
| CLI only | `uv tool upgrade mbo_utilities` |
| Virtual env | `uv pip install -U mbo_utilities` |
