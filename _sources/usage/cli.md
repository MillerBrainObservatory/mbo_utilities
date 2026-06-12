(cli_usage)=

# Command Line Interface

The `mbo` command provides tools for viewing, converting, and analyzing imaging data.

| Command | Description |
|---------|-------------|
| `mbo` | Launch GUI with file dialog |
| `mbo convert` | Convert between formats |
| `mbo info` | Show array info |
| `mbo init` | Create starter notebooks |
| `mbo formats` | List supported formats |
| `mbo shortcut` | Create a desktop icon |
| `mbo gpu` | Show render/compute GPU and memory |

## GUI Mode

```bash
mbo                          # file dialog
mbo /path/to/data            # open specific file
mbo /path/to/data --metadata # show only metadata
```

<p align="center">
  <img src="/_images/gui/readme/01_step_file_dialog.png" height="360" alt="File Selection" />
  <img src="/_images/gui/readme/02_step_data_view.png" height="360" alt="Data Viewer" />
</p>

## Convert

Convert between formats, optionally selecting planes/frames and applying phase correction.

```bash
mbo convert /data/raw output/ -e .zarr            # tiff to zarr
mbo convert /data/raw output/ -e .bin             # tiff to suite2p binary
mbo convert /data/volume.zarr output/ -e .tiff    # zarr to tiff
mbo convert /data/raw output/ -p 1 -p 7           # specific planes (repeat -p)
mbo convert /data/raw output/ -n 500              # first 500 frames
mbo convert /data/raw output/ --fix-phase         # with phase correction
```

| Option | Description |
|--------|-------------|
| `-e, --ext` | Output format: `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy` (leading dot required) |
| `-p, --planes` | Z-plane to export (1-based); repeat for multiple: `-p 4 -p 5` |
| `-n, --num-frames` | Limit number of frames |
| `--roi` | ROI: `None`=stitch, `0`=split, `N`=specific, `"1,3"`=multiple |
| `--fix-phase` | Bidirectional phase correction |
| `--overwrite` | Replace existing files |

<details>
<summary><b>All Convert Options</b></summary>

| Option | Description |
|--------|-------------|
| `--output-name` | Output filename (binary format) |
| `--phasecorr-method` | `mean`, `median`, or `max` |
| `--register-z` | Z-plane registration (Suite3D) |
| `--ome/--no-ome` | OME-zarr metadata (zarr only, default on) |
| `--chunk-mb` | Streaming chunk size in MB (default: 100) |
| `--debug` | Verbose logging |

</details>

Output is named from the timepoint, channel, and plane ranges.

```
$ mbo convert E:/demo/mk355/raw E:/demo/mk355/convert -n 500 -p 4
Reading: E:/demo/mk355/raw
  Shape: (1574, 1, 14, 550, 448), dtype: int16
Writing: E:/demo/mk355/convert (format: .tiff)
Writing TIFF: 100%|███████████████████████████████| 500/500 [00:01<00:00, 394.76pg/s]

Done! Output saved to: E:/demo/mk355/convert/tp00001-00500_ch01_zplane04.tif
```

**Note:**
- `-e/--ext` needs the leading dot: `.zarr`, not `zarr`.
- `-p/--planes` is repeatable; pass each plane separately (`-p 4 -p 5`), not `4 5` or `4,5,6`.
- The frame limit is `-n/--num-frames` (not `--frames` or `--timepoints`).

## Info

Display shape, dtype, imaging metadata, and any Suite2p results found alongside the data. Nothing is loaded into memory.

```bash
mbo info /data/raw.tiff
mbo info /data/volume.zarr
mbo info /data/suite2p/plane0
mbo info /data/raw --all       # also dump the full raw metadata dict
```

```
$ mbo info E:/demo/mk301/raw
Loading: E:/demo/mk301/raw

E:/demo/mk301/raw
  Type            LBMArray
  Shape           (500, 1, 14, 448, 448)  [T, C, Z, Y, X]
  Dtype           int16

Imaging
  Frame rate      17.07 Hz
  Pixel size      2 x 2 um
  Frame size      448 x 448 px (Y x X)
  FOV             896 x 896 um

Acquisition
  Stack type      lbm
  Timepoints      500
  Z-planes        14
  Color channels  1
  mROIs           2
  Duration        29.3 s
  Value range     [-324, 4511]

Files (2)
  - mk301_03_01_2025_2roi_..._00000.tif
  - mk301_03_01_2025_2roi_..._00001.tif

Results
  none found
```

| Option | Description |
|--------|-------------|
| `--all` | Also dump the full raw metadata dict |
| `--no-metadata` | Skip imaging/acquisition sections |

## Init

Create starter notebooks (mbo + LBM-Suite2p user guides).

```bash
mbo init                       # notebooks in current directory
mbo init /path/to/raw          # notebooks in /path/to/scripts, data path filled in
mbo init /path/to/raw -o ./nb  # custom destination directory
```

| Option | Description |
|--------|-------------|
| `-o, --output` | Destination directory (overrides default location) |
| `--overwrite` | Overwrite existing notebooks |

With a `DATA_PATH` argument, notebooks are written to a `scripts/` directory beside the data and the data path is pre-filled. Without it, notebooks go in the current directory with default paths.

```{image} /_images/cli/jupyter_lab.png
:width: 80%
:align: center
```

## Shortcut

Create a desktop icon that opens the GUI.

```bash
mbo shortcut                  # "Miller Brain Studio"
mbo shortcut --name "MBO"     # custom name
```

```
Created: C:/Users/RBO/Desktop/MBO.lnk
```

Windows creates a `.lnk` (no console window); Linux creates a `.desktop` entry.

## GPU

Show which GPU renders the viewer and which runs compute (suite2p / cellpose / cupy), plus device memory.

```bash
mbo gpu               # render GPU, compute GPU, device memory
mbo gpu --processes   # also per-process VRAM
mbo gpu --watch 2     # refresh every 2s
mbo gpu --json        # machine-readable
```

```
Render GPU  (fastplotlib): NVIDIA RTX A4000 (DiscreteGPU) via Vulkan (wgpu default)
Compute GPU (suite2p/cellpose/cupy): NVIDIA RTX A4000  (cuda:0)

Device memory:
  GPU 0: NVIDIA RTX A4000 - 941/16376 MB used (6%), 15229 MB free, util 3%, 35C
```

## Utilities

```bash
mbo --check-install      # verify installation and GPU config
```

```
mbo_utilities v3.2.0 | Python 3.12.9
==================================================

CUDA Environment:
  Driver CUDA:         12.6
  GPU:                 NVIDIA RTX A4000

Features:
  [OK] PyTorch
  [OK] CuPy
  [ -] Suite2p (not installed)
  [ -] Suite3D (not installed)
  [ -] Rastermap (not installed)

Installation OK
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
