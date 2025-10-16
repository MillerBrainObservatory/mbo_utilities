# MBO Utilities

General Python and shell utilities developed for the Miller Brain Observatory (MBO) workflows.

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/mbo_utilities/)

Most functions have examples in docstrings.

Converting scanimage tiffs into intermediate filetypes for preprocessing or to use with Suite2p is covered [here](https://millerbrainobservatory.github.io/mbo_utilities/assembly.html).

Function examples [here](https://millerbrainobservatory.github.io/mbo_utilities/api/usage.html) are a work in progress.

---

## Installation

This package is fully installable with `pip`.

`conda` can still be used for the virtual environment, but be mindful to only install packages with `conda install` when absolutely necessary.

Make sure your environment is activated, be that conda, venv, or uv (recommended, just pre-pend uv to all below pip commands).

To get the minimal mbo_utilities, without 3D axial correction or GUI functionality:

``` bash
pip install mbo_utilities
```

To get the latest version:

```bash
pip install git+https://github.com/MillerBrainObservatory/mbo_utilities.git@master
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

`pip uninstall cupy-cuda12x`

And replace `12` with the major CUDA version number, in this case `13`:

`pip install cupy-cuda13x`

## Troubleshooting

### Wrong PyTorch or CuPy version

``` bash
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "path\to\.venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

This error means you have the wrong version of pytorch install for your CUDA version.

You can run `uv pip uninstall torch` and `uv pip install torch --torch-backend=auto`.
If not using `uv`, follow instructions here: https://pytorch.org/get-started/locally/.

``` bash
RuntimeError: CuPy failed to load nvrtc64_120_0.dll: FileNotFoundError: Could not find module 'nvrtc64_120_0.dll' (or one of its dependencies). Try using the full path with constructor syntax.
```

Having the wrong `cupy` version will lead to the following error message.

---

## Acknowledgements

This pipeline makes use of several open-source libraries:

- [suite2p](https://github.com/MouseLand/suite2p)
- [rastermap](https://github.com/MouseLand/rastermap)
- [Suite3D](https://github.com/alihaydaroglu/suite3d)
- [scanreader](https://github.com/atlab/scanreader)
