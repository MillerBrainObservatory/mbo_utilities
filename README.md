<p align="center">
<img src="mbo_utilities/assets/static/logo_utilities.png" height="220" alt="MBO Utilities logo">
</p>

<p align="center">
<a href="https://github.com/MillerBrainObservatory/mbo_utilities/actions/workflows/test_python.yml"><img src="https://github.com/MillerBrainObservatory/mbo_utilities/actions/workflows/test_python.yml/badge.svg" alt="CI"></a>
<a href="https://badge.fury.io/py/mbo-utilities"><img src="https://badge.fury.io/py/mbo-utilities.svg" alt="PyPI version"></a>
<a href="https://millerbrainobservatory.github.io/mbo_utilities/"><img src="https://img.shields.io/badge/docs-online-green" alt="Documentation"></a>
</p>

<p align="center">
<a href="#installation"><b>Installation</b></a> ·
<a href="https://millerbrainobservatory.github.io/mbo_utilities/"><b>Documentation</b></a> ·
<a href="https://millerbrainobservatory.github.io/mbo_utilities/user_guide.html"><b>User Guide</b></a> ·
<a href="https://millerbrainobservatory.github.io/mbo_utilities/file_formats.html"><b>Supported Formats</b></a> ·
<a href="https://github.com/MillerBrainObservatory/mbo_utilities/issues"><b>Issues</b></a>
</p>

Image processing utilities for the [Miller Brain Observatory](https://github.com/MillerBrainObservatory) (MBO).

- **Read and write imaging data** with `imread`/`imwrite` - fast, lazy I/O for ScanImage TIFFs, generic TIFFs, Suite2p binaries, Zarr, and HDF5
- **Run processing pipelines** for calcium imaging - motion correction, cell extraction, and signal analysis
- **Visualize data interactively** with a GPU-accelerated GUI for exploring large datasets

<p align="center">
  <img src="docs/_images/gui/readme/02_step_data_view.png" width="80%" alt="Data Viewer" />
  <br/>
  <em>GPU-accelerated visualization for large imaging datasets</em>
</p>

> **Note:**
> `mbo_utilities` is in **late-beta** stage of active development. There will be bugs that can be addressed quickly, file an [issue](https://github.com/MillerBrainObservatory/mbo_utilities/issues) or reach out on slack.

## Installation

`mbo_utilities` is available in [pypi](https://pypi.org/project/mbo_utilities/):

`pip install mbo_utilities`

> For help setting up a virtual environment, see [the MBO guide on virtual environments](https://millerbrainobservatory.github.io/guides/venvs.html).

### Optional Dependencies

```bash
# with lbm_suite2p_python, suite2p, cellpose
pip install "mbo_utilities[suite2p]"

# all suite2p deps + rastermap
pip install "mbo_utilities[rastermap]"

# suite3D for axial (z-plane) registration
pip install "mbo_utilities[suite3d]"

# all of the above
pip install "mbo_utilities[all]"
```

### With [UV](https://docs.astral.sh/uv/getting-started/features/) (Recommended)

The install script will either create a virtual environment with `mbo_utilities` installed, install the `mbo` CLI globally, or both:

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.ps1 | iex
```

```bash
# Linux/macOS
curl -sSL https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.sh | bash
```

> **Note:** The `mbo` command is available globally thanks to [uv tools](https://docs.astral.sh/uv/concepts/tools/). Update with the install script or manually with `uv tool upgrade mbo_utilities`.


## Command Line Interface

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

### GUI Mode

<table>
<tr><td>

```bash
mbo                            # file dialog
mbo /path/to/data              # open specific file
mbo /path/to/data --metadata   # show only metadata
mbo view /data --roi 0 --roi 2 # view specific ROIs
```

</td><td>
<img src="docs/_images/gui/readme/01_step_file_dialog.png" height="280" />
</td></tr>
</table>

### Convert

Convert between formats with optional processing.

<table>
<tr><td>

```bash
mbo convert input.tiff output/ -e .zarr           # tiff to zarr
mbo convert input.tiff output/ -e .bin            # tiff to suite2p binary
mbo convert input.zarr output/ -e .tiff           # zarr to tiff
mbo convert input.tiff output/ -e .zarr -p 1 -p 7 # specific planes
mbo convert input.tiff output/ --fix-phase        # with phase correction
mbo convert input.tiff output/ -n 1000            # first 1000 frames
```

</td><td>

| Option | Description |
|--------|-------------|
| `-e, --ext` | Output format: `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy` |
| `-p, --planes` | Z-planes to export (1-based), repeatable |
| `-n, --num-frames` | Limit number of frames |
| `--roi` | ROI selection: `None`, `0`, `N`, or `"1,3"` |
| `--fix-phase` | Bidirectional phase correction |
| `--overwrite` | Replace existing files |

</td></tr>
</table>

### Info

<table>
<tr><td>

```bash
mbo info /data/raw.tiff
mbo info /data/volume.zarr
mbo info /data/suite2p/plane0
```

</td><td>

Display array information without loading data into memory.

</td></tr>
</table>

### Scan-Phase Analysis

Bidirectional resonant scanning causes alternating rows to be shifted horizontally. This tool measures that shift to help configure correction parameters.

<table>
<tr><td>

```bash
mbo scanphase                        # file dialog
mbo scanphase /path/to/data.tiff     # analyze file
mbo scanphase ./folder/ -n 5         # first 5 tiffs
mbo scanphase data.tiff -o ./results # custom output
mbo scanphase data.tiff --show       # show plots
```

</td><td>

**Output Files:**

| File | Description |
|------|-------------|
| `temporal.png` | Offset time series + histogram |
| `windows.png` | Offset vs window size (convergence) |
| `spatial.png` | Spatial heatmaps at 32x32 and 64x64 patches |
| `zplanes.png` | Offset vs depth (if multi-plane) |

</td></tr>
</table>

### Download

Download files from GitHub (auto-converts blob to raw URLs).

<table>
<tr><td>

```bash
mbo download https://github.com/.../notebook.ipynb
mbo download https://github.com/.../data.npy -o ./data/
```

</td><td>

```
Downloading from:
  https://raw.githubusercontent.com/.../notebook.ipynb
Saving to:
  C:\Users\...\quickstart.ipynb

Successfully downloaded: quickstart.ipynb
```

</td></tr>
</table>

<p align="center">
  <img src="docs/_images/cli/jupyter_lab.png" width="80%" alt="Downloaded notebook in JupyterLab" />
  <br/>
  <em>Downloaded quickstart notebook opened in JupyterLab</em>
</p>

### Utilities

<table>
<tr><td>

```bash
mbo --check-install
```

</td><td>

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

</td></tr>
</table>

```bash
mbo formats  # list supported formats
```

**Input formats:** `.tif`, `.tiff`, `.zarr`, `.bin`, `.h5`, `.hdf5`, `.npy`, `.json`
**Output formats:** `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy`

### Upgrade

| Method | Command |
|--------|---------|
| Install script | Re-run install script |
| CLI only | `uv tool upgrade mbo_utilities` |
| Virtual env | `uv pip install -U mbo_utilities` |

## Supported ScanImage Configurations

`mbo_utilities` automatically detects and parses metadata from these ScanImage acquisition modes:

| Configuration | Detection | Result |
|---------------|-----------|--------|
| LBM single channel | `channelSave=[1..N]`, AI0 only | `lbm=True`, `colors=1` |
| LBM dual channel | `channelSave=[1..N]`, AI0+AI1 | `lbm=True`, `colors=2` |
| Piezo (single frame/slice) | `enable=True`, `framesPerSlice=1` | `piezo=True` |
| Piezo multi-frame (with avg) | `enable=True`, `logAvgFactor>1` | `piezo=True`, averaged frames |
| Piezo multi-frame (no avg) | `enable=True`, `framesPerSlice>1`, `logAvg=1` | `piezo=True`, raw frames |
| Single plane | `enable=False` | `zplanes=1` |

> **Note:** Frame-averaging (`logAverageFactor > 1`) is only available for non-LBM acquisitions.

## Uninstall

**If installed via quick install script:**

```powershell
# Windows
uv tool uninstall mbo_utilities
Remove-Item -Recurse -Force "$env:USERPROFILE\.mbo"
Remove-Item "$env:USERPROFILE\Desktop\MBO Utilities.lnk" -ErrorAction SilentlyContinue
```

```bash
# Linux/macOS
uv tool uninstall mbo_utilities
rm -rf ~/mbo
```

**If installed in a project venv:**

```bash
uv pip uninstall mbo_utilities
```

## Troubleshooting

<details>
<summary><b>GPU/CUDA Errors</b></summary>

**Error: "Failed to auto-detect CUDA root directory"**

This occurs when using GPU-accelerated features and CuPy cannot find your CUDA Toolkit.

**Check if CUDA is installed:**

```powershell
# Windows
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue
$env:CUDA_PATH
```

```bash
# Linux/macOS
nvcc --version
echo $CUDA_PATH
```

**Set CUDA_PATH:**

```powershell
# Windows (replace v12.6 with your version)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
[System.Environment]::SetEnvironmentVariable('CUDA_PATH', $env:CUDA_PATH, 'User')
```

```bash
# Linux/macOS (add to ~/.bashrc or ~/.zshrc)
export CUDA_PATH=/usr/local/cuda-12.6
```

If CUDA is not installed, download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

</details>

<details>
<summary><b>Git LFS Download Errors</b></summary>

There is a [bug in fastplotlib](https://github.com/fastplotlib/fastplotlib/issues/861) causing `git lfs` errors when installed from a git branch.

Set `GIT_LFS_SKIP_SMUDGE=1` and restart your terminal:

```powershell
# Windows
[System.Environment]::SetEnvironmentVariable('GIT_LFS_SKIP_SMUDGE', '1', 'User')
```

```bash
# Linux/macOS
echo 'export GIT_LFS_SKIP_SMUDGE=1' >> ~/.bashrc
source ~/.bashrc
```

</details>

## Built With

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Integration support
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Visualization
- **[Suite3D](https://github.com/alihaydaroglu/suite3d)** - Volumetric processing

## Issues & Support

- **Bug reports:** [GitHub Issues](https://github.com/MillerBrainObservatory/mbo_utilities/issues)
- **Questions:** See [documentation](https://millerbrainobservatory.github.io/mbo_utilities/) or open a discussion
