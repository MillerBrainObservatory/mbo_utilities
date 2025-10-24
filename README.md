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

For use with the GUI, install [LBM-Suite2p-Python](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/tree/master):

```bash
uv pip install lbm_suite2p_python
```

To get the latest mbo_utilities from github:

```bash
uv venv --python 3.12.9 
uv pip install git+https://github.com/MillerBrainObservatory/mbo_utilities.git@master
```

By default, cupy for `CUDA 12.x` is installed and Pytorch for `CUDA 12.9`.

To fully utilize the GPU, you will verify CUDA version and an appropriate [cupy](https://docs.cupy.dev/en/stable/install.html) installation.

Check which version of CUDA you have with `nvcc --version` (here, 13.0):

```bash
nvcc --version
PS C:\Users\User\code> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jul_16_20:06:48_Pacific_Daylight_Time_2025
Cuda compilation tools, release 13.0, V13.0.48
Build cuda_13.0.r13.0/compiler.36260728_0
```

You first need to uninstall 12x:

`uv pip uninstall cupy-cuda12x`

And replace `12` with the major CUDA version number, in this case `13`:

`uv pip install cupy-cuda13x`

For pytorch, you can run `uv pip uninstall torch` and `uv pip install torch --torch-backend=auto`.
This sometimes installs the wrong Torch-enabled cuda version.
If this happens, you should uninstall and reinstall following instruction
from the [pytorch getting-started page]( https://pytorch.org/get-started/locally/.)

The below error means you have the wrong version of pytorch install for your CUDA version.

``` bash
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "path\to\.venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

Having the wrong `cupy` version will lead to the following error message.

``` bash
RuntimeError: CuPy failed to load nvrtc64_120_0.dll: FileNotFoundError: Could not find module 'nvrtc64_120_0.dll' (or one of its dependencies). Try using the full path with constructor syntax.
```

Follow instructions above for getting the correct CUDA/Pytorch packages.

## Graphical User Interface

To start the gui:

```bash
uv run mbo
```

See the [GUI User Guide](./mbo_gui_user_guide.pdf) attached here.

<p align="center">
  <img src="docs/_images/GUI_Slide1.png" alt="GUI Slide 1" width="45%">
  <img src="docs/_images/GUI_Slide2.png" alt="GUI Slide 2" width="45%">
</p>


## Acknowledgements

This pipeline makes use of several open-source libraries:

- [suite2p](https://github.com/MouseLand/suite2p)
- [rastermap](https://github.com/MouseLand/rastermap)
- [Suite3D](https://github.com/alihaydaroglu/suite3d)
- [scanreader](https://github.com/atlab/scanreader)
