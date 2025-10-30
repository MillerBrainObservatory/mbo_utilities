# MBO Utilities

> **Status:** Late-beta stage of development

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/mbo_utilities/)

Microscopy image processing utilities for the Miller Brain Observatory (MBO). Provides lazy-loading I/O for ScanImage TIFF, Zarr, HDF5, and Suite2p binary formats with multi-ROI support, phase correction, and a GUI for volumetric visualization.

A Suite2p processing pipeline is available via [LBM-Suite2p-Python](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python).

## Quick Start

See the [installation documentation](https://millerbrainobservatory.github.io/mbo_utilities/venvs.html) for environment setup and GPU troubleshooting.

```bash
uv pip install mbo_utilities
```

**With GUI support:**
```bash
uv pip install lbm_suite2p_python
```

## Basic Usage

**Read formats:**
```python
from mbo_utilities import imread

# Multi-ROI ScanImage with phase correction
arr = imread("scanimage_mROI.tif")           # Stitched FOV
arr = imread("scanimage_mROI.tif", roi=0)    # Single ROI

# Other formats
arr = imread("suite2p/data.bin")             # Suite2p binary
arr = imread("volume.zarr")                  # Zarr array
arr = imread("/path/to/planes/")             # Multi-file volumes
```

**Convert formats:**
```python
from mbo_utilities import imread, imwrite

data = imread("scanimage_file.tif")
imwrite(data, "output_dir", ext=".zarr")     # To Zarr
imwrite(data, "output_dir", ext=".tiff", roi=0)  # Split ROIs to TIFF
```

**Launch GUI:**
```bash
uv run mbo
```

<p align="center">
  <img src="docs/_images/GUI_Slide1.png" alt="GUI Screenshot" width="45%">
  <img src="docs/_images/GUI_Slide2.png" alt="GUI Screenshot" width="45%">
</p>

## Documentation

- **[Installation & GPU Setup](https://millerbrainobservatory.github.io/mbo_utilities/venvs.html)**
- **[User Guide](https://millerbrainobservatory.github.io/mbo_utilities/user_guide.html)**
- **[Array Types](https://millerbrainobservatory.github.io/mbo_utilities/array_types.html)** 
- **[API Reference](https://millerbrainobservatory.github.io/mbo_utilities/api/index.html)**
- **[GUI User Guide](./mbo_gui_user_guide.pdf)**

## Built With

This package integrates several open-source tools:

- **[scanreader](https://github.com/atlab/scanreader)** - ScanImage TIFF parsing
- **[Suite2p](https://github.com/MouseLand/suite2p)** - Integration support
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Visualization
- **[Suite3D](https://github.com/alihaydaroglu/suite3d)** - Volumetric processing

## Issues & Support

- **Bug reports:** [GitHub Issues](https://github.com/MillerBrainObservatory/mbo_utilities/issues)
- **Questions:** See [documentation](https://millerbrainobservatory.github.io/mbo_utilities/) or open a discussion
