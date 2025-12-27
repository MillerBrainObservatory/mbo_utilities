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
<a href="https://millerbrainobservatory.github.io/mbo_utilities/array_types.html"><b>Array Types</b></a> ·
<a href="https://github.com/MillerBrainObservatory/mbo_utilities/issues"><b>Issues</b></a>
</p>

Image processing utilities for the [Miller Brain Observatory](https://github.com/MillerBrainObservatory) (MBO). Fast, lazy I/O with `imread`/`imwrite` for multiple array types (ScanImage and generic TIFFs, Suite2p binaries, Zarr and HDF5), an interactive GUI for data visualization, and processing pipelines for calcium imaging data.

<div align="center">
  <img src="docs/_images/gui/2_mbo_main_full.png" width="80%" />
</div>

> **Note:**
> `mbo_utilities` is in **late-beta** stage of active development. There will be bugs that can be addressed quickly, file an [issue](https://github.com/MillerBrainObservatory/mbo_utilities/issues) or reach out on slack.

## Installation

`mbo_utilities` is available in [pypi](https://pypi.org/project/mbo_utilities/):

`pip install mbo_utilities`

> For help setting up a virtual environment, see [the MBO guide](https://millerbrainobservatory.github.io/guides/venvs.html).

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

After [installing `uv`](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer):

**For CLI usage only** (global `mbo` command):

```bash
uv tool install mbo_utilities
```

This installs the CLI globally. Not for use as a library in your own code.

**For library/development use** (import in your code):

```bash
uv venv --python 3.12
uv pip install mbo_utilities
```

This creates a project-specific virtual environment.

**One-line install scripts** (CLI + desktop shortcut):

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.ps1 | iex
```

```bash
# Linux/macOS
curl -sSL https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.sh | bash
```

These scripts install to `~/mbo/envs/mbo_utilities` and add `mbo` to your PATH.

```bash
$ mbo --help
Usage: mbo [OPTIONS] COMMAND [ARGS]...

  MBO Utilities CLI - data preview and processing tools.

  GUI Mode:
    mbo                            Open file selection dialog
    mbo /path/to/data              Open specific file in GUI
    mbo /path/to/data --metadata   Show only metadata

  Commands:
    mbo convert INPUT OUTPUT       Convert between formats
    mbo info INPUT                 Show array information (CLI)
    mbo download URL               Download file from GitHub
    mbo formats                    List supported formats

  Utilities:
    mbo --download-notebook             Download user guide notebook
    mbo --check-install                 Verify installation

Options:
  --download-notebook   Download the mbo_utiltities user guide notebook.
  --notebook-url TEXT   URL of notebook to download.
  --download-file TEXT  Download a file from URL (e.g. GitHub).
  -o, --output TEXT     Output path for --download-file or --download-
                        notebook.
  --check-install       Verify the installation of mbo_utilities and
                        dependencies.
  --help                Show this message and exit.

Commands:
  convert    Convert imaging data between formats.
  download   Download a file from a URL (supports GitHub).
  formats    List supported file formats.
  info       Show information about an imaging dataset.
  scanphase  Scan-phase analysis for bidirectional scanning data.
  view       Open imaging data in the GUI viewer.
```

### Environment Locations

| Method | Use Case | Run Command |
|--------|----------|-------------|
| `uv tool install` | CLI only | `mbo` |
| `uv pip install` | Scientific Data Processing | `uv run mbo` or activate venv |
| Install script | CLI + desktop shortcut | `mbo` |

For library use, activate the venv or use `uv run`:

```bash
# option 1: activate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
mbo --help

# option 2: uv run (no activation needed)
uv run mbo --help
```

## Usage

**CLI Commands:**

| Command | Description |
|---------|-------------|
| `uv run mbo` | Launch interactive GUI |
| `uv run mbo --check-install` | Verify installation and GPU configuration |
| `uv run mbo /path/to/data.tiff` | View a supported file/folder |
| `uv run mbo /path/to/data.tiff --metadata` | View metadata for a supported file/folder |
| `uv run mbo --download-notebook` | Download user guide notebook |
| `uv run mbo info /path/to/data.tiff` | Show file info |
| `uv run mbo convert input.tiff output.zarr` | Convert file formats |
| `uv run mbo scanphase /path/to/data.tiff` | Scan-phase analysis |
| `uv run mbo formats` | List supported formats |
| `uv run pollen` | Pollen calibration tool (WIP) |

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
