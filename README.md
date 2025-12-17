# MBO Utilities

> **Status:** Late-beta stage of active development. There will be bugs that can be addressed quickly, file an [issue](https://github.com/MillerBrainObservatory/mbo_utilities/issues) or reach out on slack.

Image processing utilities for the [Miller Brain Observatory](https://github.com/MillerBrainObservatory) (MBO).

## Features

- **Fast, lazy I/O** with `imread`/`imwrite` for [multiple array types](https://millerbrainobservatory.github.io/mbo_utilities/array_types.html) (ScanImage and generic TIFFs, Suite2p binaries, Zarr and HDF5)
- **Interactive GUI** via `uv run mbo` for visualization and processing
- **(WIP) Pollen calibration** via `uv run pollen`

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/mbo_utilities/)

## Installation

### Quick Install (Recommended)

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.ps1 | iex
```

**Linux/macOS:**

```bash
curl -sSL https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.sh | bash
```

### Manual Install

For help setting up a virtual environment, see the [Virtual Environments guide](https://millerbrainobservatory.github.io/guides/venvs.html).

```bash
uv venv --python 3.12.9
uv pip install mbo_utilities

# install all optional dependencies
uv pip install "mbo_utilities[all]"
```

### Installation Methods

| Method | Location | Use Case |
|--------|----------|----------|
| `uv pip install` in project | `project/.venv/` | Project-specific, use with `uv run mbo` |
| `uv tool install mbo_utilities` | `~/.local/bin/` | Global `mbo` command |

These are isolated - a global tool install won't affect project environments.

The GUI allows registration/segmentation for users to quickly process subsets of their datasets.

These pipelines need to be installed separately.

Currently, the only supported pipeline is LBM-Suite2p-Python.

A few exciting future prospects include [masknmf](https://github.com/apasarkar/masknmf-toolbox).

```bash
uv pip install mbo_utilities[suite2p]
```

## Usage

We encourage users to start with the [user_guide](https://millerbrainobservatory.github.io/mbo_utilities/user_guide.html).

You can run this code as a jupyter notebook or rendered in the [docs](./demos/user_guide.ipynb) and can be downloaded on the top right of the page.

See [array types](https://millerbrainobservatory.github.io/mbo_utilities/array_types.html) for additional information about each file-type and it's associated lazy array.

### Commands

**Launch GUI:**

```bash
uv run mbo
```

<p align="center">
  <img src="docs/_images/GUI_Slide1.png" alt="GUI Screenshot" width="45%">
  <img src="docs/_images/GUI_Slide2.png" alt="GUI Screenshot" width="45%">
</p>

**Download User Guide Notebook:**

```bash
uv run mbo --download-notebook
```

**Verify Installation:**

Check that all dependencies are correctly installed, including CUDA/GPU configuration for Suite3D z-registration:

```bash
uv run mbo --check-install
```

**Pollen Calibration:**

```bash
uv run pollen
```

**Download files from GitHub:**

```bash
# download a notebook from github
mbo download https://github.com/user/repo/blob/main/notebook.ipynb

# download to specific location
mbo download https://github.com/user/repo/blob/main/data.npy 
```

**Scan-phase analysis:**

```bash
# open file dialog
uv run mbo scanphase

# analyze specific file
uv run mbo scanphase /path/to/data.tiff

# use first 5 tiffs from a folder
uv run mbo scanphase ./folder/ -n 5

# custom output directory
uv run mbo scanphase data.tiff -o ./results/
```

**File format conversion:**

```bash
# convert tiff to zarr
uv run mbo convert input.tiff output.zarr

# convert to suite2p binary
uv run mbo convert input.tiff output.bin
```

**Show file info:**

```bash
uv run mbo info /path/to/data.tiff
```

**List supported formats:**

```bash
uv run mbo formats
```

## Installation Troubleshooting


### GPU/CUDA Errors

**Error: "Failed to auto-detect CUDA root directory"**

This error occurs when using GPU-accelerated features (like Suite3D registration) and CuPy cannot find your CUDA Toolkit installation.

#### Solution 1: Check if CUDA is installed

**PowerShell (Windows):**
```powershell
# Check for CUDA installations
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue

# Check if CUDA_PATH is set
$env:CUDA_PATH
```

**Linux/macOS:**
```bash
# Check for CUDA
nvcc --version

# Check CUDA_PATH
echo $CUDA_PATH
```

#### Solution 2: Set CUDA_PATH if CUDA is installed

**PowerShell (Windows):**
```powershell
# Find your CUDA version folder (e.g., v12.1, v11.8, v12.6)
# Replace with your actual version from the directory listing above
$cuda_version = "v12.6"

# Temporary (current session)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cuda_version"

# Permanent (restart terminal after)
[System.Environment]::SetEnvironmentVariable('CUDA_PATH', "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cuda_version", 'User')
```

**Linux/macOS:**
```bash
# Find CUDA installation (common locations)
ls /usr/local/cuda* 2>/dev/null

# Set CUDA_PATH (add to ~/.bashrc or ~/.zshrc for persistence)
export CUDA_PATH=/usr/local/cuda-12.6  # Replace with your version
```

#### Solution 3: Install CUDA Toolkit if not present

1. **Check your GPU compatibility**: Ensure you have an NVIDIA GPU
2. **Download CUDA Toolkit**: Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. **Install**: Follow the installer instructions
4. **Restart your terminal** to pick up the new environment variables

**Note:** After installing CUDA Toolkit, `CUDA_PATH` should be set automatically by the installer.

#### Alternative: Use CPU-only processing

If you don't have an NVIDIA GPU or prefer not to install CUDA, you can disable GPU processing in Suite3D by modifying the registration parameters (though this will be significantly slower).

### Git LFS Download Errors (when installing directly from Github)

There is a [bug in fastplotlib](https://github.com/fastplotlib/fastplotlib/issues/861) causing `git lfs` errors when installed from a git branch.

This should be fixed soon.

- In your terminal of choice, set the `GIT_LFS_SKIP_SMUDGE` environment variable to 1
- **restart your terminal**.

**PowerShell (Windows):**
```powershell
# current session only
$env:GIT_LFS_SKIP_SMUDGE="1"

# permanent for the current user)
[System.Environment]::SetEnvironmentVariable('GIT_LFS_SKIP_SMUDGE', '1', 'User')
```

**Command Prompt (Windows):**
```cmd
REM Temporary (current session only)
set GIT_LFS_SKIP_SMUDGE=1
  GIT_LFS_SKIP_SMUDGE=1

REM Permanent (system-wide, requires admin)
setx GIT_LFS_SKIP_SMUDGE 1

  SUCCESS: Specified value was saved.
```

**Linux/macOS (bash/zsh):**
```bash
# current session only
export GIT_LFS_SKIP_SMUDGE=1

# add to .bashrc/.zshrc to be permanent for the current user 
echo 'export GIT_LFS_SKIP_SMUDGE=1' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc  # or ~/.zshrc
```

After setting the variable, restart your terminal and retry the installation.

## Issues & Support

- **Bug reports:** [GitHub Issues](https://github.com/MillerBrainObservatory/mbo_utilities/issues)
- **Questions:** See [documentation](https://millerbrainobservatory.github.io/mbo_utilities/) or open a discussion

## Built With

This package integrates several open-source tools:

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Integration support
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Visualization
- **[Suite3D](https://github.com/alihaydaroglu/suite3d)** - Volumetric processing
