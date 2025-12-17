(cli_usage)=

# Command Line Interface

The `mbo` command provides tools for viewing, converting, and analyzing imaging data.

## Commands

| Command | Description |
|---------|-------------|
| `mbo` | launch gui with file dialog |
| `mbo view` | view data in gui |
| `mbo convert` | convert between formats |
| `mbo info` | show array info |
| `mbo scanphase` | analyze scan-phase offset |
| `mbo download` | download files from github |
| `mbo formats` | list supported formats |

## GUI Mode

```bash
mbo                            # file dialog
mbo /path/to/data              # open specific file
mbo /path/to/data --metadata   # show only metadata
mbo view /data --roi 0 --roi 2 # view specific ROIs
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
mbo convert input.tiff output/ --output-suffix _session1  # plane01_session1.zarr
```

**Options:**

| Option | Description |
|--------|-------------|
| `-e, --ext` | output format: .tiff, .zarr, .bin, .h5, .npy |
| `-p, --planes` | z-planes to export (1-based), repeatable |
| `-n, --num-frames` | limit number of frames |
| `--roi` | roi selection: None, 0, N, or "1,3" |
| `--output-suffix` | custom filename suffix (default: _stitched for multi-roi) |
| `--fix-phase/--no-fix-phase` | bidirectional phase correction |
| `--phasecorr-method` | mean, median, or max |
| `--register-z` | z-plane registration via suite3d |
| `--ome/--no-ome` | ome-zarr metadata (zarr only) |
| `--overwrite` | replace existing files |
| `--chunk-mb` | streaming chunk size (default: 100) |
| `--debug` | verbose logging |

## Info

Display array information without loading data.

```bash
mbo info /data/raw.tiff
mbo info /data/volume.zarr
mbo info /data/suite2p/plane0
```

## Scan-Phase Analysis

Bidirectional resonant scanning causes alternating rows to be shifted horizontally. This tool measures that shift to help configure correction parameters.

### Usage

```bash
mbo scanphase                             # file dialog
mbo scanphase /path/to/data.tiff          # analyze file
mbo scanphase ./folder/ -n 5              # first 5 tiffs
mbo scanphase data.tiff -o ./results/     # custom output
mbo scanphase data.tiff --show            # show plots
mbo scanphase data.tiff --format pdf      # output as pdf
```

### Output Files

| File | Description |
|------|-------------|
| `temporal.png` | offset time series + histogram |
| `windows.png` | offset vs window size (convergence) |
| `spatial.png` | spatial heatmaps at 32x32 and 64x64 patches |
| `zplanes.png` | offset vs depth (if multi-plane) |
| `parameters.png` | offset vs signal intensity |
| `scanphase_results.npz` | numerical data |

### Interpreting Results

**temporal.png**: time series should be flat. large jumps indicate motion or hardware issues. typical offset is 0.5-2.0 px.

**windows.png**: shows how estimate stabilizes with more frames. left plot: offset converges to stable value. right plot: variance decreases with window size. red line marks where std drops below 0.1 px. use this to determine how many frames to average for correction.

**spatial.png**: heatmaps show variation across fov. edges different from center is normal. gray = low signal (unreliable).

**zplanes.png**: assess if offset varies with depth, owing to the angle on the resonant scanner.

**parameters.png**: shows offset reliability vs signal. low signal = unreliable (high/variable offset). red line suggests intensity threshold below which measurements are noisy.

### Tips

- use `-n 2` or `-n 3` to run quickly on subset of frames
- multi-ROI data: offsets are averaged across ROIs

## Download

Download files from github (auto-converts blob to raw urls).

```bash
mbo download https://github.com/user/repo/blob/main/notebook.ipynb
mbo download https://github.com/user/repo/blob/main/data.npy -o ./data/
```

## Utilities

```bash
mbo --download-notebook     # download user guide notebook
mbo --download-file URL     # download any file
mbo --check-install         # verify installation and gpu config
```

## Formats

Show supported file formats:

```bash
mbo formats
```

**Input formats:** .tif, .tiff, .zarr, .bin, .h5, .hdf5, .npy, .json
**Output formats:** .tiff, .zarr, .bin, .h5, .npy
