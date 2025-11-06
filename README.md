# MBO Utilities

> **Status:** Late-beta stage of active development. There will be bugs that can be addressed quickly, file an [issue](https://github.com/MillerBrainObservatory/mbo_utilities/issues) or reach out on slack.

Image processing utilities for the [Miller Brain Observatory](https://github.com/MillerBrainObservatory/mbo_utilities/issues) (MBO).

## Features

- **Fast, lazy I/O** with `imread`/`imwrite` for [multiple array types](https://millerbrainobservatory.github.io/mbo_utilities/array_types.html) (ScanImage and generic TIFFs, Suite2p binaries, Zarr and HDF5)
- **Interactive GUI** via `uv run mbo` for visualization and processing
- **Pollen calibration** via `uv run pollen`
- **Multiple array wrappers** for seamless data handling

---

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/mbo_utilities/)

## Installation

```bash
uv pip install mbo_utilities
```

The GUI allows registration/segmentation for users to quickly process subsets of their datasets.

These pipelines need to be installed separately.

Currently, the only supported pipeline is LBM-Suite2p-Python. A few exciting future prospects include [masknmf](https://github.com/apasarkar/masknmf-toolbox).

```bash

uv pip install lbm_suite2p_python

```

## Usage

We encourage users to start with the [user_guide](https://millerbrainobservatory.github.io/mbo_utilities/user_guide.html)

You can run this code by available as a jupyter notebook or rendered in the [docs](./demos/user_guide.ipynb) and can be downloaded on the top right of the page.

See [array types](https://millerbrainobservatory.github.io/mbo_utilities/array_types.html) for additional information about each file-type and it's associated lazy array.

**Launch GUI:**

```bash
uv run mbo
```

<p align="center">
  <img src="docs/_images/GUI_Slide1.png" alt="GUI Screenshot" width="45%">
  <img src="docs/_images/GUI_Slide2.png" alt="GUI Screenshot" width="45%">
</p>

## Built With

This package integrates several open-source tools:

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Integration support
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Visualization
- **[Suite3D](https://github.com/alihaydaroglu/suite3d)** - Volumetric processing

## Issues & Support

- **Bug reports:** [GitHub Issues](https://github.com/MillerBrainObservatory/mbo_utilities/issues)
- **Questions:** See [documentation](https://millerbrainobservatory.github.io/mbo_utilities/) or open a discussion
