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

## Dimensions

`imread()` returns 5D **TCZYX** arrays. `.shape` is always length 5 and the
order is fixed:

| Axis | Index | Meaning             |
|------|-------|---------------------|
| T    | 0     | timepoints (frames) |
| C    | 1     | channels            |
| Z    | 2     | z-planes            |
| Y    | 3     | image rows          |
| X    | 4     | image columns       |

A typical LBM volumetric scan (1574 frames, 14 z-planes, 550×448) reports
`shape == (1574, 1, 14, 550, 448)` — one channel, fourteen z-planes.

Size-1 axes are kept, not dropped. To drop them for inspection or display,
use `arr.squeeze()` or `imread(path, squeeze=True)` — a view that still holds
the 5D array underneath for writers and the viewer.

`BinArray` is the one exception: it reports the rank you pass in
(see [BinArray](#binarray)).

## Quick Reference

| Input | Returns | `shape` | Description |
|-------|---------|---------|-------------|
| **`.tiff`** | | | |
| ↳ ScanImage raw | `LBMArray` | `(T, C, Z, Y, X)` | LBM with z-planes as channels |
| | `PiezoArray` | `(T, C, Z, Y, X)` | Piezo z-stacks, optional averaging |
| | `LBMPiezoArray` | `(T, C, Z, Y, X)` | LBM + piezo (pollen calibration) |
| | `SinglePlaneArray` | `(T, C, Z, Y, X)` (Z=1) | Single-plane time series |
| ↳ Standard/ImageJ | `TiffArray` | `(T, C, Z, Y, X)` | All TIFFs including ImageJ hyperstacks |
| **`.bin`** | `BinArray` | as-passed, e.g. `(T, Y, X)` | Suite2p binary (requires shape) |
| **`.h5`** | `H5Array` | `(T, C, Z, Y, X)` | HDF5 datasets |
| **`.zarr`** | `ZarrArray` | `(T, C, Z, Y, X)` | Zarr v3 / OME-Zarr |
| **`.npy`** | `NumpyArray` | `(T, C, Z, Y, X)` | Memory-mapped numpy |
| **`np.ndarray`** | `NumpyArray` | `(T, C, Z, Y, X)` | In-memory wrapper |
| **Directory** | | | |
| ↳ `ops.npy` | `Suite2pArray` | `(T, C, Z, Y, X)` (Z=1) | Suite2p single-plane |
| ↳ `planeXX/ops.npy` | `Suite2pArray` | `(T, C, Z, Y, X)` | Suite2p volumetric |
| ↳ `planeXX.tiff` | `TiffArray` | `(T, C, Z, Y, X)` | Multi-plane TIFF volume |

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
- **Axial Registration**: phase-correlation z-plane registration (see `compute_axial_shifts`)

(lbmarray)=
#### LBMArray

Light Beads Microscopy with z-planes interleaved as ScanImage channels.

```python
arr = mbo.imread("/path/to/lbm_data.tif")
print(arr.shape)    # (T, C, Z, Y, X), e.g. (1574, 1, 14, 550, 448)
print(arr.num_planes) # number of z-planes (= shape[2])
# note: dz must be user-supplied (not in ScanImage metadata for LBM)
```

(piezoarray)=
#### PiezoArray

Aquisitions using the ScanImage Piezo `hStackManager` produce z-stacks with optional frame averaging.

```python
arr = mbo.imread("/path/to/piezo_data.tif")
print(arr.shape)          # (T, C, Z, Y, X)
print(arr.frames_per_slice) # frames per z-position
print(arr.can_average)      # True if averaging possible

arr.average_frames = True   # toggle averaging based on `scanimage.logAverageFactor`
```

(lbmpiezoarray)=

#### LBMPiezoArray

Combined LBM + piezo, typically for pollen calibration. Each piezo step
ends up as a z-plane in the canonical layout, and the LBM beamlets land on
the channel axis.

```python
arr = mbo.imread("/path/to/pollen_calibration.tif")
print(arr.stack_type)  # 'pollen'
print(arr.shape)     # (T, C, Z, Y, X) — C = beamlets, Z = piezo positions
```

(singleplanearray)=
#### SinglePlaneArray

Single-plane time series (no z-stack).

```python
arr = mbo.imread("/path/to/single_plane.tif")
print(arr.shape)  # (T, C, Z=1, Y, X)
```

(tiffarray)=
### TiffArray

Universal TIFF reader for non-ScanImage files. Automatically handles standard TIFF stacks,
ImageJ hyperstacks (interleaved TZYX), and multi-plane volumes (planeXX.tiff directories).

Whatever the file's on-disk rank, `.shape` is 5D TCZYX with singleton T/C/Z
filled in:

```python
# 2D image
arr = mbo.imread("/path/to/single_image.tif")
print(arr.shape)  # (1, 1, 1, Y, X)

# 3D time series
arr = mbo.imread("/path/to/tyx_stack.tif")
print(arr.shape)  # (T, 1, 1, Y, X)

# ImageJ hyperstack (auto-detected)
arr = mbo.imread("/path/to/imagej_hyperstack.tif")
print(arr.shape)          # (T, 1, Z, Y, X)
print(arr.is_volumetric)  # True

# volumetric from directory of planeXX.tif files
vol = mbo.imread("/path/to/tiff_output/")
print(vol.shape)           # (T, 1, Z, Y, X)
print(vol.is_volumetric)   # True
```

(suite2parray)=
### Suite2pArray

Suite2p binary files with full ops.npy context.

```python
arr = mbo.imread("/path/to/suite2p/plane0")
print(arr.shape)     # (T, 1, 1, Y, X) — single plane
print(arr.raw_file)    # path to data_raw.bin
print(arr.reg_file)    # path to data.bin

arr.switch_channel(use_raw=True)  # toggle raw/registered

# volumetric (planeXX/ subdirs each with ops.npy)
vol = mbo.imread("/path/to/suite2p_output/")
print(vol.shape)  # (T, 1, Z, Y, X)
```

Note: frame count is computed from actual file size, not ops.npy (which may be stale).

(binarray)=
### BinArray

Direct binary file access when no ops.npy context is available. The user
supplies the shape explicitly, and the array reports exactly that rank as
`.shape` — it is the one array type whose `.shape` is not 5D.

```python
from mbo_utilities.arrays import BinArray

# requires explicit shape — any rank up to 5D
arr = BinArray("/path/to/data.bin", shape=(1000, 512, 512))
print(arr.shape)  # (1000, 512, 512) — exactly what you passed in
print(arr.nz)     # 1               — TCZYX sizes are still available

# read/write via memmap
arr[0] = new_frame
arr.close()
```

(h5array)=
### H5Array

HDF5 datasets with auto-detection of common dataset names. Reads from
`/mov` by default — same name `mbo.imwrite(..., ext=".h5")` writes to —
falling back to `/data` or the first available dataset.

```python
arr = mbo.imread("/path/to/data.h5")
print(arr.dataset_name)  # 'mov', 'data', or first available
print(arr.shape)       # (T, C, Z, Y, X)

# specify dataset explicitly
arr = mbo.imread("/path/to/data.h5", dataset="imaging_data")
```

(zarrarray)=
### ZarrArray

Zarr v3 stores including OME-Zarr.

```python
arr = mbo.imread("/path/to/data.zarr")
print(arr.shape)   # (T, C, Z, Y, X)
print(arr.metadata)  # OME-NGFF attributes if present

# multiple zarr stores stacked as z-planes
arr = mbo.imread(["/path/plane01.zarr", "/path/plane02.zarr"])
```

You can also pass a path to the inner `zarr.json` (e.g. from a file picker)
and it will resolve to the parent `.zarr` store automatically.

(numpyarray)=
### NumpyArray

Wraps `.npy` files (memory-mapped) or in-memory numpy arrays. Numpy input
of any rank up to 5D is accepted; the missing dims are inferred from shape
heuristics and filled into the 5D `.shape`.

```python
# from file
arr = mbo.imread("/path/to/data.npy")

# from in-memory array
import numpy as np
data = np.random.randn(100, 512, 512).astype(np.float32)
arr = mbo.imread(data)

print(arr.shape)  # (100, 1, 1, 512, 512)

# enables imwrite to any format
mbo.imwrite(arr, "output", ext=".zarr")
```

## Common Properties

All array types provide:

| Property      | Description                                          |
|---------------|------------------------------------------------------|
| `.shape`      | 5D `(T, C, Z, Y, X)` (`BinArray`: the rank you passed in) |
| `.dtype`      | data type                                            |
| `.ndim`       | number of dims in `.shape` (5, except `BinArray`)    |
| `.dims`       | dim labels, e.g. `('T', 'C', 'Z', 'Y', 'X')`         |
| `.nt` `.nc` `.nz` `.ny` `.nx` | individual TCZYX sizes               |
| `.metadata`   | file/array metadata dict                             |
| `.num_planes` | number of z-planes (= `.nz`)                         |

The `.nt`/`.nc`/`.nz`/`.ny`/`.nx` accessors give individual sizes and are
correct for every array type, including `BinArray`.

Most array types also provide:

| Property   | Description           |
|------------|-----------------------|
| `.close()` | release file handles  |

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
