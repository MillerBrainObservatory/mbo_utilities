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
# make sure your environment is activated, be that conda, venv, uv venv (stick to these)
pip install mbo_utilities
```

To get the latest version:

```bash
pip install git+https://github.com/MillerBrainObservatory/mbo_utilities.git@master
```

---

## Acknowledgements

Thank you to the developers of [scanreader](https://github.com/atlab/scanreader) and the broader open-source community.
