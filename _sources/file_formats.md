---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(file_formats)=
# File Formats

`imread()` and the GUI file dialogs are primarily designed to read [ScanImage](https://docs.scanimage.org/) TIFFs
and future raw filetypes supported at the [Miller Brain Observatory](https://millerbrainobservatory.github.io/).

The raw metadata is made [OME](https://www.openmicroscopy.org/ome-xml/) and ImageJ/Fiji compatible when writing to disk, ensuring downstream tools can interpret volumetric and multi-channel data correctly.

Additional formats are made as-needed for specific tasks, e.g. Suite2p `.bin` and `.h5` for microscope calibrations.

## Quick Reference

| Input | Returns | Shape | Description |
|-------|---------|-------|-------------|
| **`.tiff`** | | | |
| ↳ ScanImage raw | `LBMArray` | (T, Z, Y, X) | LBM with z-planes as channels |
| | `PiezoArray` | (T, Z, Y, X) | Piezo z-stacks, optional averaging |
| | `LBMPiezoArray` | (Z, C, Y, X) | LBM + piezo (pollen calibration) |
| | `SinglePlaneArray` | (T, C, Y, X) | Single-plane time series |
| ↳ Standard/ImageJ | `TiffArray` | (T, Z, Y, X) | All TIFFs including ImageJ hyperstacks |
| **`.bin`** | `BinArray` | (T, Y, X) | Suite2p binary (requires shape) |
| **`.h5`** | `H5Array` | varies | HDF5 datasets |
| **`.zarr`** | `ZarrArray` | (T, Z, Y, X) | Zarr v3 / OME-Zarr |
| **`.npy`** | `NumpyArray` | varies | Memory-mapped numpy |
| **`np.ndarray`** | `NumpyArray` | varies | In-memory wrapper |
| **Directory** | | | |
| ↳ `ops.npy` | `Suite2pArray` | (T, Y, X) | Suite2p single-plane |
| ↳ `planeXX/ops.npy` | `Suite2pArray` | (T, Z, Y, X) | Suite2p volumetric |
| ↳ `planeXX.tiff` | `TiffArray` | (T, Z, Y, X) | Multi-plane TIFF volume |
| ↳ Isoview structure | `IsoviewArray` | (T, Z, V, Y, X) | Multi-view lightsheet |

### Detection Logic

```
imread(path)
│
├── np.ndarray ───────────────────────────► NumpyArray (in-memory)
├── .npy ─────────────────────────────────► NumpyArray (mmap)
├── .h5 / .hdf5 ──────────────────────────► H5Array
├── .zarr ────────────────────────────────► ZarrArray
├── .bin (with ops.npy nearby) ───────────► Suite2pArray
├── .bin (no ops.npy) ────────────────────► BinArray (shape required)
│
├── .tif / .tiff
│   ├── ScanImage metadata?
│   │   ├── stack_type == "lbm" ──────────► LBMArray
│   │   ├── stack_type == "piezo" ────────► PiezoArray
│   │   ├── stack_type == "pollen" ───────► LBMPiezoArray
│   │   └── stack_type == "single_plane" ─► SinglePlaneArray
│   └── else ─────────────────────────────► TiffArray 
│
└── Directory
    ├── Isoview zarr structure ───────────► IsoviewArray
    ├── *.zarr files ─────────────────────► ZarrArray
    ├── ops.npy ──────────────────────────► Suite2pArray
    ├── planeXX/ with ops.npy ────────────► Suite2pArray (volumetric)
    ├── planeXX.tiff files ───────────────► TiffArray (volumetric)
    └── ScanImage TIFFs ──────────────────► ScanImageArray subclass
```

## Array Types

(scanimage-arrays)=
### ScanImage Arrays

Returned when reading raw ScanImage TIFF files. `imread()` auto-detects the stack type:

This folder containing all `tiffs` for a single session can be passed to `imread`, as well as a single files or a list of files.

```python
import mbo_utilities as mbo

arr = mbo.imread("/path/to/raw/*.tif")
print(type(arr).__name__)  # LBMArray, PiezoArray, SinglePlaneArray, or LBMPiezoArray
print(arr.stack_type)      # 'lbm', 'piezo', 'single_plane', or 'pollen'
```

All ScanImage arrays support:

- **ROI handling**: `arr.roi = None` (stitch all), `arr.roi = 1` (specific ROI), `arr.roi = [1,2]` (multiple)
- **Phase correction**: `arr.fix_phase = True/False`
- **Metadata**: `arr.metadata["si"]` contains raw ScanImage headers
- **Axial Registration**: `suite3d`-based z-plane registration

(lbmarray)=
#### LBMArray

Light Beads Microscopy with z-planes interleaved as ScanImage channels.

```python
arr = mbo.imread("/path/to/lbm_data.tif")
print(arr.shape)      # (T, Z, Y, X)
print(arr.num_planes) # number of z-planes
# note: dz must be user-supplied (not in ScanImage metadata for LBM)
```

(piezoarray)=
#### PiezoArray

Aquisitions using the ScanImage Piezo `hStackManager` produce z-stacks with optional frame averaging.

```python
arr = mbo.imread("/path/to/piezo_data.tif")
print(arr.shape)            # (T, Z, Y, X)
print(arr.frames_per_slice) # frames per z-position
print(arr.can_average)      # True if averaging possible

arr.average_frames = True   # toggle averaging based on `scanimage.logAverageFactor` 
```

(lbmpiezoarray)=

#### LBMPiezoArray

Combined LBM + piezo, typically for pollen calibration.

```python
arr = mbo.imread("/path/to/pollen_calibration.tif")
print(arr.stack_type)       # 'pollen'
print(arr.shape)            # (Z, C, Y, X) - z-piezo positions × beamlets
```

(singleplanearray)=
#### SinglePlaneArray

Single-plane time series (no z-stack).

```python
arr = mbo.imread("/path/to/single_plane.tif")
print(arr.shape)  # (T, C, Y, X) where C=1
```

(tiffarray)=
### TiffArray

Universal TIFF reader for non-ScanImage files. Automatically handles standard TIFF stacks,
ImageJ hyperstacks (interleaved TZYX), and multi-plane volumes (planeXX.tiff directories).

```python
arr = mbo.imread("/path/to/processed.tif")
print(arr.shape)  # (T, Z, Y, X) - Z=1 for single-file stacks

# ImageJ hyperstacks are auto-detected
arr = mbo.imread("/path/to/imagej_stack.tif")
print(arr.shape)         # (T, Z, Y, X) with Z > 1
print(arr.is_volumetric) # True

# volumetric from directory
vol = mbo.imread("/path/to/tiff_output/")  # detects planeXX.tiff pattern
print(vol.shape)         # (T, Z, Y, X)
print(vol.is_volumetric) # True
```

(suite2parray)=
### Suite2pArray

Suite2p binary files with full ops.npy context.

```python
arr = mbo.imread("/path/to/suite2p/plane0")
print(arr.shape)       # (T, Y, X) single plane
print(arr.raw_file)    # path to data_raw.bin
print(arr.reg_file)    # path to data.bin

arr.switch_channel(use_raw=True)  # toggle raw/registered

# volumetric
vol = mbo.imread("/path/to/suite2p_output/")  # detects planeXX/ subdirs
print(vol.shape)  # (T, Z, Y, X)
```

Note: frame count is computed from actual file size, not ops.npy (which may be stale).

(binarray)=
### BinArray

Direct binary file access when no ops.npy context is available.

```python
from mbo_utilities.arrays import BinArray

# requires explicit shape
arr = BinArray("/path/to/data.bin", shape=(1000, 512, 512))
print(arr.shape)  # (T, Y, X)

# read/write via memmap
arr[0] = new_frame
arr.close()
```

(h5array)=
### H5Array

HDF5 datasets with auto-detection of common dataset names.

```python
arr = mbo.imread("/path/to/data.h5")
print(arr.dataset_name)  # 'mov', 'data', or first available

# specify dataset explicitly
arr = mbo.imread("/path/to/data.h5", dataset="imaging_data")
```

(zarrarray)=
### ZarrArray

Zarr v3 stores including OME-Zarr.

```python
arr = mbo.imread("/path/to/data.zarr")
print(arr.shape)     # (T, Z, Y, X)
print(arr.metadata)  # OME-NGFF attributes if present

# multiple zarr stores stacked as z-planes
arr = mbo.imread(["/path/plane01.zarr", "/path/plane02.zarr"])
```

(numpyarray)=
### NumpyArray

Wraps `.npy` files (memory-mapped) or in-memory numpy arrays.

```python
# from file
arr = mbo.imread("/path/to/data.npy")

# from in-memory array
import numpy as np
data = np.random.randn(100, 512, 512).astype(np.float32)
arr = mbo.imread(data)

print(arr.dims)  # 'TYX' or 'TZYX' (auto-inferred from shape)

# enables imwrite to any format
mbo.imwrite(arr, "output", ext=".zarr")
```

(isoviewarray)=
### IsoviewArray

Isoview lightsheet microscopy data (multi-view, multi-timepoint).

```python
from mbo_utilities.arrays import IsoviewArray

arr = IsoviewArray("/path/to/output")
print(arr.shape)     # (T, Z, V, Y, X) or (Z, V, Y, X) single timepoint
print(arr.views)     # [(camera, channel), ...] pairs
print(arr.num_views) # number of views
```

## Common Properties

All array types provide:

| Property    | Description                |
|-------------|----------------------------|
| `.shape`    | array dimensions           |
| `.dtype`    | data type                  |
| `.ndim`     | number of dimensions       |
| `.metadata` | file/array metadata dict   |

Most array types also provide:

| Property      | Description                       |
|---------------|-----------------------------------|
| `.dims`       | dimension labels (e.g., 'TZYX')   |
| `.num_planes` | number of z-planes                |
| `.close()`    | release file handles              |

ScanImage-specific:

| Property      | Description                                 |
|---------------|---------------------------------------------|
| `.stack_type` | 'lbm', 'piezo', 'single_plane', or 'pollen' |
| `.num_rois`   | number of ROIs                              |
| `.roi`        | ROI selection (None, int, or list)          |
| `.fix_phase`  | enable/disable phase correction             |

PiezoArray-specific:

| Property            | Description                |
|---------------------|----------------------------|
| `.frames_per_slice` | frames per z-position      |
| `.can_average`      | True if averaging possible |
| `.average_frames`   | toggle frame averaging     |

## Writing Data

All array types support `imwrite()`:

```python
import mbo_utilities as mbo

arr = mbo.imread("/path/to/data.tif")

# write to different formats
mbo.imwrite(arr, "output", ext=".zarr")   # OME-Zarr v3
mbo.imwrite(arr, "output", ext=".tiff")   # BigTIFF
mbo.imwrite(arr, "output", ext=".h5")     # HDF5
mbo.imwrite(arr, "output", ext=".npy")    # NumPy
mbo.imwrite(arr, "output", ext=".bin")    # Suite2p binary

# subset selection
mbo.imwrite(arr, "output", ext=".zarr", frames=range(100))
mbo.imwrite(arr, "output", ext=".zarr", planes=[0, 2, 4])

# zarr options
mbo.imwrite(arr, "output", ext=".zarr", sharded=True, compression_level=1)
```

Metadata is automatically adjusted when subsetting (e.g., `dz` doubles when selecting every 2nd plane).

## API Reference

- `mbo_utilities.imread()` - unified file reader
- `mbo_utilities.imwrite()` - unified file writer
- `mbo_utilities.arrays` - direct access to array classes
