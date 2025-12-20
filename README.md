<p align="center">
<img src="mbo_utilities/assets/static/logo_utilities.png" height="220" alt="MBO Utilities logo">
</p>

---

[![CI](https://github.com/MillerBrainObservatory/mbo_utilities/actions/workflows/test_python.yml/badge.svg)](https://github.com/MillerBrainObservatory/mbo_utilities/actions/workflows/test_python.yml)
[![PyPI version](https://badge.fury.io/py/mbo-utilities.svg)](https://badge.fury.io/py/mbo-utilities)
[![Documentation](https://img.shields.io/badge/docs-online-green)](https://millerbrainobservatory.github.io/mbo_utilities/)

[**Installation**](#installation) |
[**Documentation**](https://millerbrainobservatory.github.io/mbo_utilities/) |
[**User Guide**](https://millerbrainobservatory.github.io/mbo_utilities/user_guide.html) |
[**Array Types**](https://millerbrainobservatory.github.io/mbo_utilities/array_types.html) |
[**Issues**](https://github.com/MillerBrainObservatory/mbo_utilities/issues)

Image processing utilities for the [Miller Brain Observatory](https://github.com/MillerBrainObservatory) (MBO). Fast, lazy I/O with `imread`/`imwrite` for multiple array types (ScanImage and generic TIFFs, Suite2p binaries, Zarr and HDF5), an interactive GUI for data visualization, and processing pipelines for calcium imaging data.

<div align="center">
  <img src="docs/_images/GUI_Slide1.png" width="45%" />
  <img src="docs/_images/GUI_Slide2.png" width="45%" />
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

After [installing `uv`](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer), you can install `mbo_utilities` as a tool:

```python
# latest version - base
uv tool install "mbo_utilities"

# older release with all dependencies
uv tool install "mbo_utilities[all]==2.4.3"
```

Or with one of the provided install scripts:

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.ps1 | iex
```

**Linux/macOS:**

```bash
curl -sSL https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.sh | bash
```

Either of these methods allow you to call `mbo_utilities` from any terminal.

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
  --download-notebook   Download the user guide notebook and exit.
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

| Method | Location | Use Case |
|--------|----------|----------|
| `uv pip install` in project | `project/.venv/` | Project-specific, use with `uv run mbo` |
| `uv tool install mbo_utilities` | `~/.local/bin/` | Global `mbo` command |

## Usage

**Launch GUI**

```bash
# uv: in a project with .venv
uv run mbo

# tool / script installation
mbo
```

**Launch Metadata Viewer**

```bash
# uv: in a project with .venv
uv run mbo /path/to/data

# tool / script installation
mbo
```

### Commands

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
| `uv run pollen` | Pollen calibration tool |

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
