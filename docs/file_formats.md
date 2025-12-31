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

`imread` and `imwrite` designed for 3D and 4D image data in `.tiff`, `.zarr`, `.h5` formats, but has been extended to include several other formats including suite2p `.bin`/`.ops` files, `.hdf5` and `.npy`.

## Quick Reference

| Input | Returns | Shape | Description | |
|-------|---------|-------|-------------|:-:|
| **`.tiff`** | | | TIFF image stacks | |
| ↳ ScanImage | `ScanImageArray` | (T, Z, Y, X) | Multi-ROI volumetric with phase correction | [<i class="fa-solid fa-book"></i>](#scanimage-arrays) |
| &emsp;↳ LBM | `LBMArray` | (T, Z, Y, X) | Z-planes as ScanImage channels | [<i class="fa-solid fa-book"></i>](#lbmarray) |
| &emsp;↳ Piezo | `PiezoArray` | (T, Z, Y, X) | Piezo z-stacks, optional averaging | [<i class="fa-solid fa-book"></i>](#piezoarray) |
| &emsp;↳ Single-plane | `SinglePlaneArray` | (T, 1, Y, X) | Single-plane time series | [<i class="fa-solid fa-book"></i>](#singleplanearray) |
| ↳ Standard | `TiffArray` | (T, Z, Y, X) | Lazy page access, auto-detects volumetric | [<i class="fa-solid fa-book"></i>](#tiffarray) |
| ↳ MBO metadata | `MBOTiffArray` | (T, Z, Y, X) | Dask-backed with MBO metadata | [<i class="fa-solid fa-book"></i>](#mbotiffarray) |
| **`.bin`** | `BinArray` | (T, Y, X) | Direct binary file manipulation | [<i class="fa-solid fa-book"></i>](#binarray) |
| **`.h5` / `.hdf5`** | `H5Array` | varies | HDF5 datasets | [<i class="fa-solid fa-book"></i>](#h5array) |
| **`.zarr`** | `ZarrArray` | (T, Z, Y, X) | Zarr v3 / OME-Zarr stores | [<i class="fa-solid fa-book"></i>](#zarrarray) |
| **`.npy` / `np.ndarray`** | `NumpyArray` | varies | Memory-mapped or in-memory arrays | [<i class="fa-solid fa-book"></i>](#numpyarray) |
| **`.nwb`** | `NWBArray` | varies | Neurodata Without Borders files | [<i class="fa-solid fa-book"></i>](#nwbarray) |
| **Directory** | | | | |
| ↳ `ops.npy` | `Suite2pArray` | (T, Y, X) | Suite2p workflow integration | [<i class="fa-solid fa-book"></i>](#suite2parray) |
| ↳ `planeXX/ops.npy` | `Suite2pArray` | (T, Z, Y, X) | Multi-plane Suite2p output | [<i class="fa-solid fa-book"></i>](#suite2parray) |
| ↳ `planeXX.tiff` | `TiffArray` | (T, Z, Y, X) | Multi-plane TIFF volume | [<i class="fa-solid fa-book"></i>](#tiffarray) |
| ↳ Isoview lightsheet | `IsoviewArray` | (T, Z, V, Y, X) | Multi-view lightsheet data | [<i class="fa-solid fa-book"></i>](#isoviewarray) |

### Decision Tree

```
imread(path)
│
├── np.ndarray? ──────────────────────────────► NumpyArray
│
├── .npy ─────────────────────────────────────► NumpyArray (memory-mapped)
├── .nwb ─────────────────────────────────────► NWBArray
├── .h5 / .hdf5 ──────────────────────────────► H5Array
├── .zarr ────────────────────────────────────► ZarrArray
├── .bin ─────────────────────────────────────► BinArray
│
├── .tif / .tiff
│   ├── Has ScanImage ROI metadata?
│   │   ├── LBM acquisition? ─────────────────► LBMArray
│   │   ├── Piezo enabled? ───────────────────► PiezoArray
│   │   └── Single-plane? ────────────────────► SinglePlaneArray
│   ├── Has MBO metadata? ────────────────────► MBOTiffArray
│   └── Standard TIFF ────────────────────────► TiffArray
│
├── Directory
│   ├── Contains planeXX.tiff? ───────────────► TiffArray (volumetric)
│   ├── Contains planeXX/ with ops.npy? ──────► Suite2pArray (volumetric)
│   ├── Contains ops.npy? ────────────────────► Suite2pArray
│   └── Contains raw ScanImage TIFFs? ────────► ScanImageArray
│
└── List of paths
    ├── All .tif? ────────────────────────────► TiffArray or MBOTiffArray
    └── All .zarr? ───────────────────────────► ZarrArray (stacked along Z)
```

**Tip:** Use `open_scanimage()` for automatic ScanImage subclass detection:
```python
from mbo_utilities.arrays import open_scanimage
arr = open_scanimage("/path/to/data.tif")  # Returns LBMArray, PiezoArray, or SinglePlaneArray
```

## Array Type Details

(scanimage-arrays)=
### ScanImage Arrays

**Returned when:** Reading raw ScanImage TIFF files

The ScanImage array hierarchy handles different acquisition types from ScanImage:

```text
ScanImageArray (base class)
    ├── LBMArray        # Light Beads Microscopy (z-planes as channels)
    ├── PiezoArray      # Piezo z-stacks with optional frame averaging
    └── SinglePlaneArray # Single-plane time series
```

Use `open_scanimage()` for automatic stack type detection, or instantiate the specific class directly.

#### ScanImageArray (Base Class)

```python
import mbo_utilities as mbo
from mbo_utilities.arrays import open_scanimage, ScanImageArray

# Auto-detect stack type
scan = open_scanimage("/path/to/raw/*.tif")
print(type(scan).__name__)  # 'LBMArray', 'PiezoArray', or 'SinglePlaneArray'
print(scan.stack_type)      # 'lbm', 'piezo', or 'single_plane'

# Or use imread (returns ScanImageArray base class)
scan = mbo.imread("/path/to/raw/*.tif")

print(scan.shape)      # (T, Z, Y, X) - e.g., (10000, 14, 456, 896)
print(scan.num_rois)   # Number of ROIs
print(scan.num_planes) # Alias for num_channels (Z planes)

# ROI handling
scan.roi = None      # Stitch all ROIs horizontally (default)
scan.roi = 0         # Split into separate ROIs (returns tuple)
scan.roi = 1         # Use only ROI 1 (1-indexed)
scan.roi = [1, 2]    # Select specific ROIs

# Phase correction settings
scan.fix_phase = True           # Enable bidirectional scan-phase correction
scan.use_fft = True             # Use FFT-based phase correction
scan.phasecorr_method = "mean"  # "mean", "median", "max"
scan.border = 3                 # Border pixels to exclude
scan.upsample = 5               # Subpixel upsampling factor
scan.max_offset = 4             # Maximum phase offset to search
```

(lbmarray)=
#### LBMArray

For Light Beads Microscopy stacks where z-planes are interleaved as ScanImage channels.

```python
from mbo_utilities.arrays import LBMArray, open_scanimage

# Auto-detect and get LBMArray
arr = open_scanimage("/path/to/lbm_data.tif")
# arr is LBMArray if detected as LBM

# Key metadata context (shown in GUI tooltips):
# - Ly: Total height including fly-to lines between mROIs
# - dz: Must be user-supplied (not in ScanImage metadata for LBM)
# - fs: Volume rate (frame rate / num_zplanes)
```

(piezoarray)=
#### PiezoArray

For piezo z-stacks with optional frame averaging.

```python
from mbo_utilities.arrays import PiezoArray, open_scanimage

# With frame averaging enabled
arr = open_scanimage("/path/to/piezo_data.tif", average_frames=True)

# Piezo-specific properties
print(arr.frames_per_slice)    # Frames acquired per z-slice
print(arr.log_average_factor)  # >1 if pre-averaged at acquisition
print(arr.can_average)         # True if averaging is possible

# Toggle averaging (changes shape)
arr.average_frames = True
print(arr.shape)  # Reduced T dimension when averaging

# Key metadata context:
# - frames_per_slice: Frames at each z-position before piezo moves
# - log_average_factor: If >1, frames were pre-averaged during acquisition
# - dz: Z-step size from hStackManager.stackZStepSize
```

(singleplanearray)=
#### SinglePlaneArray

For single-plane time series without z-stack.

```python
from mbo_utilities.arrays import SinglePlaneArray, open_scanimage

arr = open_scanimage("/path/to/single_plane.tif")
# arr is SinglePlaneArray if no z-stack detected
```

**Key Features (all ScanImage arrays):**

- Automatic ROI stitching/splitting via `roi` property
- Bidirectional scan-phase correction (configurable methods)
- Stack type detection via `stack_type` property
- ROI position extraction from ScanImage metadata
- Contextual metadata descriptions via `get_param_description()`
- Stores all ScanImage-metadata in `array.metadata["si"]`

**Legacy alias:** `MboRawArray` is an alias for `ScanImageArray` for backwards compatibility.

---

(tiffarray)=
### TiffArray

**Returned when:** Reading processed TIFF file(s) without ScanImage metadata, or a directory with `planeXX.tiff` files

```python
# Single or multiple standard TIFF files
arr = mbo.imread("/path/to/processed.tif")
arr = mbo.imread(["/path/file1.tif", "/path/file2.tif"])
# Returns: TiffArray

print(type(arr))   # <class 'TiffArray'>
print(arr.shape)   # (T, 1, Y, X) - 4D with Z=1 for single files
print(arr.dtype)   # Data type from TIFF

# Directory with planeXX.tiff files (auto-detected volumetric)
vol = mbo.imread("/path/to/tiff_output/")
# Detects plane01.tiff, plane02.tiff, etc.
print(vol.shape)         # (T, Z, Y, X) - e.g., (10000, 14, 512, 512)
print(vol.is_volumetric) # True
print(vol.num_planes)    # 14

# Lazy frame reading
frame = arr[0]        # Read first frame
subset = arr[10:20]   # Read frames 10-19 (only those pages are loaded)

# Dtype conversion
arr32 = arr.astype(np.float32)
```

**Key Features:**

- Uses `TiffFile` handles for lazy page access
- Auto-detects volumetric structure from `planeXX.tiff` filename patterns
- Multi-file support (concatenated along time axis)
- Thread-safe page reading
- Always outputs 4D format: (T, Z, Y, X) where Z=1 for single files

---

(mbotiffarray)=
### MBOTiffArray

**Returned when:** Reading TIFFs with MBO-specific metadata (uses Dask backend)

```python
# MBO-processed TIFFs with metadata
arr = mbo.imread("/path/to/processed/*.tif")
# Returns: MBOTiffArray if MBO metadata detected

print(type(arr))   # <class 'MBOTiffArray'>
print(arr.shape)   # (T, Z, Y, X) - Dask infers shape
print(arr.dask)    # Access underlying dask.Array

# Lazy operations via Dask
mean_proj = arr[:100].mean(axis=0).compute()
```

**Key Features:**

- Dask-backed for truly lazy, chunked access
- Uses `tifffile.imread(aszarr=True)` for memory-mapped access
- Automatic dimension handling (2D → TZYX, 3D → TZYX, 4D passthrough)
- Preserves file tags from filenames

---

(binarray)=
### BinArray

**Returned when:** Explicitly reading a `.bin` file path

```python
# Reading a specific binary file
arr = mbo.imread("path/to/data_raw.bin")
# Returns: BinArray

print(type(arr))   # <class 'BinArray'>
print(arr.shape)   # (nframes, Ly, Lx)
print(arr.nframes) # Number of frames
print(arr.Ly, arr.Lx)  # Spatial dimensions

# Access data like numpy array
frame = arr[0]      # First frame
subset = arr[0:100] # First 100 frames

# Write access (if opened for writing)
arr[0] = new_frame

# Context manager support
with BinArray("data.bin", shape=(100, 512, 512)) as arr:
    arr[:] = data

# Close when done
arr.close()
```

**Key Features:**

- Direct binary file access via `np.memmap`
- Auto-infers shape from adjacent `ops.npy` if present
- Can provide shape manually: `BinArray("file.bin", shape=(1000, 512, 512))`
- Read/write access (supports `__setitem__`)
- Useful for creating new binary files from scratch

**When to use:**

- Reading/writing specific binary files in a Suite2p workflow
- Creating new binary files from scratch
- When you want to work with the file directly, not through Suite2p's abstraction

---

(suite2parray)=
### Suite2pArray

**Returned when:** Reading a directory containing `ops.npy`, or `ops.npy` directly, or a directory with multiple `planeXX/` subdirectories

```python
# Reading a Suite2p single-plane directory
arr = mbo.imread("/path/to/suite2p/plane0")
arr = mbo.imread("/path/to/suite2p/plane0/ops.npy")
# Returns: Suite2pArray

print(type(arr))   # <class 'Suite2pArray'>
print(arr.shape)   # (nframes, Ly, Lx) - 3D for single plane
print(arr.metadata)  # Full ops.npy contents

# Multi-plane Suite2p output (auto-detected volumetric)
vol = mbo.imread("/path/to/suite2p_output/")
# Detects plane01_stitched/, plane02_stitched/, etc.
print(vol.shape)         # (T, Z, Y, X) - e.g., (10000, 14, 512, 512)
print(vol.is_volumetric) # True
print(vol.num_planes)    # 14

# File paths (single plane)
print(arr.raw_file)    # Path to data_raw.bin (unregistered)
print(arr.reg_file)    # Path to data.bin (registered)
print(arr.active_file) # Currently active file

# Switch between raw and registered
arr.switch_channel(use_raw=True)   # Use data_raw.bin
arr.switch_channel(use_raw=False)  # Use data.bin (default)

# Visualization with both channels
iw = arr.imshow()  # Shows raw and registered side-by-side if both exist
```

**Key Features:**

- Full Suite2p context (metadata from `ops.npy`)
- Auto-detects volumetric structure from `planeXX/` subdirectory patterns
- Access to both raw (`data_raw.bin`) and registered (`data.bin`) data
- Memory-mapped via `np.memmap` for lazy loading
- File size validation against ops metadata
- For volumes: `switch_channel()` applies to all planes

---

(h5array)=
### H5Array

**Returned when:** Reading HDF5 files (`.h5`, `.hdf5`)

```python
# HDF5 dataset
arr = mbo.imread("/path/to/data.h5")
# Returns: H5Array

print(type(arr))        # <class 'H5Array'>
print(arr.shape)        # Dataset shape
print(arr.dataset_name) # Auto-detected: 'mov', 'data', or first available

# Optionally specify dataset name
arr = mbo.imread("/path/to/data.h5", dataset="imaging_data")

# Access data
frame = arr[0]
subset = arr[10:20, :, 100:200]

# File-level metadata
print(arr.metadata)  # HDF5 file attributes

# Close file handle
arr.close()
```

**Key Features:**

- Auto-detects common dataset names: `'mov'`, `'data'`, `'scan_corrections'`
- Lazy loading via `h5py.Dataset`
- Supports ellipsis indexing (`arr[..., 100:200]`)
- File-level attributes exposed via `.metadata`

---

(zarrarray)=
### ZarrArray

**Returned when:** Reading Zarr stores (`.zarr` directories)

```python
# Zarr store (standard or OME-Zarr)
arr = mbo.imread("/path/to/data.zarr")
# Returns: ZarrArray

print(type(arr))   # <class 'ZarrArray'>
print(arr.shape)   # (T, Z, Y, X) - always 4D

# Read multiple zarr stores as z-planes
arr = mbo.imread(["/path/plane01.zarr", "/path/plane02.zarr"])

# Access pre-computed statistics (if available in OME metadata)
print(arr.zstats)  # {'mean': [...], 'std': [...], 'snr': [...]}

# Access metadata (OME-NGFF attributes if present)
print(arr.metadata)
```

**Key Features:**

- Supports both standard Zarr arrays and OME-Zarr groups
- Auto-detects OME-Zarr structure (looks for `"0"` subarray in groups)
- Multi-store support (stacked along Z axis)
- Zarr v3 compatible
- Exposes OME-NGFF metadata via `.metadata`

---

(numpyarray)=
### NumpyArray

**Returned when:** Reading `.npy` files OR passing an in-memory numpy array to `imread()`

This is the most versatile array type - it wraps any numpy array and provides full `imwrite()` support.

#### From .npy Files (Memory-Mapped)

```python
# Read .npy file - memory-mapped for lazy loading
arr = mbo.imread("/path/to/data.npy")
# Returns: NumpyArray

print(type(arr))   # <class 'NumpyArray'>
print(arr.shape)   # (T, Y, X) or (T, Z, Y, X)
print(arr.dims)    # 'TYX' or 'TZYX' (auto-inferred)
```

#### From In-Memory Numpy Arrays

```python
import numpy as np
import mbo_utilities as mbo

# Create or load a numpy array from anywhere
data = np.random.randn(100, 512, 512).astype(np.float32)

# Wrap with imread - returns NumpyArray
arr = mbo.imread(data)

print(arr)
# NumpyArray(shape=(100, 512, 512), dtype=float32, dims='TYX' (in-memory))

# Now you have full imwrite support with all features
mbo.imwrite(arr, "output", ext=".zarr")   # Zarr v3 with chunking/sharding
mbo.imwrite(arr, "output", ext=".tiff")   # BigTIFF
mbo.imwrite(arr, "output", ext=".bin")    # Suite2p binary + ops.npy
mbo.imwrite(arr, "output", ext=".h5")     # HDF5
mbo.imwrite(arr, "output", ext=".npy")    # NumPy format
```

#### 4D Volumetric Data

```python
# 4D arrays are automatically detected as (T, Z, Y, X)
volume = np.random.randn(100, 15, 512, 512).astype(np.float32)
arr = mbo.imread(volume)

print(arr.dims)        # 'TZYX'
print(arr.num_planes)  # 15

# Write specific planes
mbo.imwrite(arr, "output", ext=".zarr", planes=[1, 7, 14])
```

**Key Features:**

- Automatic dimension inference (`TYX`, `TZYX`, `YX`, etc.)
- Memory-mapped for `.npy` files (lazy loading)
- Full `imwrite()` support with all output formats
- Chunked reduction operations (`mean`, `std`, `max`, `min`)
- Metadata auto-generation from array shape

---

(nwbarray)=
### NWBArray

**Returned when:** Reading NWB (Neurodata Without Borders) files

```python
# NWB file with TwoPhotonSeries
arr = mbo.imread("/path/to/experiment.nwb")
# Returns: NWBArray

print(type(arr))   # <class 'NWBArray'>
print(arr.shape)   # Shape from TwoPhotonSeries data

# Access data
frame = arr[0]
```

**Key Features:**

- Reads `TwoPhotonSeries` acquisition data from NWB files
- Requires `pynwb` package (`pip install pynwb`)
- Exposes underlying NWB data object

---

(isoviewarray)=
### IsoviewArray

**Returned when:** Manually instantiated for isoview lightsheet microscopy data

```python
from mbo_utilities.arrays import IsoviewArray

# Isoview lightsheet data (multi-timepoint)
arr = IsoviewArray("/path/to/output")
# Shape: (T, Z, Views, Y, X) - 5D

print(arr.shape)       # (10, 543, 4, 2048, 2048)
print(arr.views)       # [(0, 0), (1, 0), (2, 1), (3, 1)] - (camera, channel) pairs
print(arr.num_views)   # 4

# Access specific view
frame = arr[0, 100, 0]  # timepoint 0, z=100, view 0 (camera 0, channel 0)

# Get view index for camera/channel
idx = arr.view_index(camera=1, channel=0)

# Access labels and projections (consolidated structure only)
labels = arr.get_labels(timepoint=0, camera=0, label_type='segmentation')
proj = arr.get_projection(timepoint=0, camera=0, proj_type='xy')
```

**Key Features:**

- Supports two data structures:
  - **Consolidated**: `data_TM000000_SPM00.zarr/camera_0/0/`
  - **Separate**: `SPM00_TM000000_CM00_CHN01.zarr`
- Multi-view (camera/channel combinations)
- 5D shape: `(T, Z, Views, Y, X)` or 4D `(Z, Views, Y, X)` for single timepoint
- Access to segmentation labels and projections
- Lazy loading via Zarr

---

## Common Properties Across All Array Types

All lazy array types provide these standard properties:

| Property | Type | Description |
|----------|------|-------------|
| `.shape` | `tuple[int, ...]` | Array dimensions |
| `.dtype` | `np.dtype` | Data type |
| `.ndim` | `int` | Number of dimensions |
| `.metadata` | `dict` | Array/file metadata |
| `.filenames` | `list[Path]` | Source file paths |
| `._imwrite()` | method | Write to any output format |

Most array types also provide:

| Property | Type | Description |
|----------|------|-------------|
| `.num_planes` | `int` | Number of Z-planes |
| `.num_rois` | `int` | Number of ROIs (ScanImage arrays) |
| `.stack_type` | `str` | Stack type: 'lbm', 'piezo', or 'single_plane' (ScanImage arrays) |
| `.close()` | method | Release file handles |

PiezoArray-specific properties:

| Property | Type | Description |
|----------|------|-------------|
| `.frames_per_slice` | `int` | Frames acquired per z-slice |
| `.log_average_factor` | `int` | Averaging factor from acquisition |
| `.can_average` | `bool` | True if frame averaging is possible |
| `.average_frames` | `bool` | Toggle frame averaging on/off |

## API Reference

- `mbo_utilities.imread()` - Smart file reader
- `mbo_utilities.imwrite()` - Universal file writer
- `mbo_utilities.arrays` - Direct access to array classes
