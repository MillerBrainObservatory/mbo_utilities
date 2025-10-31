# MBO Utilities

> **Status:** Late-beta stage of development


Microscopy image processing utilities for the Miller Brain Observatory (MBO).

Provides efficient, lazy I/O for ScanImage and generic TIFF, Zarr v3, HDF5, and Suite2p binary formats with multi-ROI support, phase correction, and a GUI for volumetric visualization.

A Suite2p processing pipeline is available via [LBM-Suite2p-Python](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python).

---

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/mbo_utilities/)

## Installation

```bash
uv pip install mbo_utilities
```

The GUI allows registration/segmentation for users to quickly process subsets of their datasets. These pipelines need to be installed separately.

Currently, the only supported pipeline is LBM-Suite2p-Python.

```bash

uv pip install lbm_suite2p_python

```

## Usage

We encourage users to start with the demo notebook [user_guide](./demos/user_guide.ipynb), available as a jupyter notebook or rendered in the [docs](https://millerbrainobservatory.github.io/mbo_utilities/user_guide.html) and can be downloaded on the top right of the page.

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
