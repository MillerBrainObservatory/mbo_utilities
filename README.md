# MBO Utilities

General Python and shell utilities developed for the Miller Brain Observatory (MBO) workflows.

This package is still in a *late-beta* stage of development. As such, you may encounter bugs or unexpected behavior.

Please report any issues on the [GitHub Issues page](This package is still in a *late-beta* stage of development.)

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/mbo_utilities/)

Most functions have examples in docstrings.

Converting scanimage tiffs into intermediate filetypes for preprocessing or to use with Suite2p is covered [here](https://millerbrainobservatory.github.io/mbo_utilities/assembly.html).

Function examples [here](https://millerbrainobservatory.github.io/mbo_utilities/api/usage.html) are a work in progress.

---

## Installation

This package is fully installable with `pip`.

`conda` can still be used for the virtual environment, but be mindful to only install packages with `conda install` when absolutely necessary.

Make sure your environment is activated, be that conda, venv, or uv.

See our documentation on virtual environments [here](https://millerbrainobservatory.github.io/mbo_utilities/venvs.html).

To get the latest stable version:

```bash
# make a new environment in a location of your choosing
# preferably on your C: drive. e.g. C:\Users\YourName\project

uv venv --python 3.12.9 
uv pip install mbo_utilities
```

To get the latest version from github:

```bash
uv venv --python 3.12.9 
uv pip install git+https://github.com/MillerBrainObservatory/mbo_utilities.git@master
```

Using `UV` to install from github allows us to specify dependencies that are not yet released to `pypi`.
If *not* using `uv`, simply replace `uv pip` with `pip`, and install the latest pygfx (this is likely to change in the future).

``` bash
# into an environment with python 3.12.7-3.12.9 (tested)
pip install mbo_utilities
pip install git+https://github.com/pygfx/pygfx.git@main
```

To utilize the GPU, you will need CUDA and an appropriate [cupy](https://docs.cupy.dev/en/stable/install.html) installation.

By default, cupy for `CUDA 12.x` is installed.

Check which version of CUDA you have with `nvcc --version`.

```bash
nvcc --version
PS C:\Users\MBO-User\code> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jul_16_20:06:48_Pacific_Daylight_Time_2025
Cuda compilation tools, release 13.0, V13.0.48
Build cuda_13.0.r13.0/compiler.36260728_0
```

For CUDA 11.x and 13.x, you first need to uninstall 12x:

`uv pip uninstall cupy-cuda12x`

And replace `12` with the major CUDA version number, in this case `13`:

`uv pip install cupy-cuda13x`

## Troubleshooting

### Wrong PyTorch or CuPy version

The below error means you have the wrong version of pytorch install for your CUDA version.

``` bash
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "path\to\.venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

You can run `uv pip uninstall torch` and `uv pip install torch --torch-backend=auto`.

If not using `uv`, follow instructions here: https://pytorch.org/get-started/locally/.

Having the wrong `cupy` version will lead to the following error message.

``` bash
RuntimeError: CuPy failed to load nvrtc64_120_0.dll: FileNotFoundError: Could not find module 'nvrtc64_120_0.dll' (or one of its dependencies). Try using the full path with constructor syntax.
```

Uninstall cupy and reinstall the correct version for your CUDA version. Find your CUDA version with `nvcc --version` and replace `12` with your major CUDA version number:

```bash
uv pip uninstall cupy-cuda12x
uv pip install cupy-cuda13x
```

---

## Acknowledgements

This pipeline makes use of several open-source libraries:

- [suite2p](https://github.com/MouseLand/suite2p)
- [rastermap](https://github.com/MouseLand/rastermap)
- [Suite3D](https://github.com/alihaydaroglu/suite3d)
- [scanreader](https://github.com/atlab/scanreader)
