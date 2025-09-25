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

``` bash
# make sure your environment is activated, be that conda, venv, uv (recommended)
pip install mbo_utilities
```

To get the latest version:

```bash
pip install git+https://github.com/MillerBrainObservatory/mbo_utilities.git@master
```

To utilize the GPU, you will need CUDA and an appropriate [cupy](https://docs.cupy.dev/en/stable/install.html) installation.

Check which version of CUDA you have with `nvcc --version`.

Then, install either 11.x or 12.x with:

```bash
pip install cupy-cuda11x  # for CUDA 11.x
pip install cupy-cuda12x  # for CUDA 12.x
```

### RuntimeError with `cupy`

If you install the wrong version of cupy, you will see an error like:

`RuntimeError: CuPy failed to load nvrtc64_120_0.dll: FileNotFoundError: Could not find module 'nvrtc64_120_0.dll' (or one of its dependencies). Try using the full path with constructor syntax.`

You will need to uninstall cupy and reinstall the correct version.

```bash
pip uninstall cupy-cuda12x  # or cupy-cuda12x
pip install cupy-cuda11x  # or cupy-cuda12x
```

---

## Acknowledgements

Thank you to the developers of [scanreader](https://github.com/atlab/scanreader) and the broader open-source community.
