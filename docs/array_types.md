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

(array_types)=
# Lazy Array Types

Understanding what `imread()` returns and when to use each array type.

## Overview

`mbo_utilities.imread()` is a smart file reader that automatically detects the file type and returns the appropriate lazy array class. All array types provide:

- **Lazy loading**: Data is read on-demand, not loaded entirely into memory
- **NumPy-like indexing**: Standard slicing syntax (`arr[0]`, `arr[10:20, :, 100:200]`)
- **`_imwrite()` support**: All arrays can be written to any output format via `imwrite()`
- **Metadata**: Accessible via `.metadata` property

## Quick Reference

| Input | Returns | Shape | Use Case |
|-------|---------|-------|----------|
| `.tif` (raw ScanImage) | `MboRawArray` | (T, Z, Y, X) | Multi-ROI volumetric data with phase correction |
| `.tif` (processed, single file) | `TiffArray` | (T, 1, Y, X) | Standard TIFF files, lazy page access |
| `.tif` (processed, multiple files) | `MBOTiffArray` | (T, Z, Y, X) | Dask-backed multi-file TIFF |
| Directory with `planeXX.tiff` files | `TiffArray` | (T, Z, Y, X) | Multi-plane TIFF volume (auto-detected) |
| `.bin` (direct path) | `BinArray` | (T, Y, X) | Direct binary file manipulation |
| Directory with `ops.npy` | `Suite2pArray` | (T, Y, X) | Suite2p workflow integration |
| Directory with `planeXX/` subdirs | `Suite2pArray` | (T, Z, Y, X) | Multi-plane Suite2p output (auto-detected) |
| `.h5` / `.hdf5` | `H5Array` | varies | HDF5 datasets |
| `.zarr` | `ZarrArray` | (T, Z, Y, X) | Zarr v3 / OME-Zarr stores |
| `.npy` | `NumpyArray` | varies | NumPy memory-mapped files |
| `.nwb` | `NWBArray` | varies | Neurodata Without Borders files |
| `np.ndarray` (in-memory) | `NumpyArray` | varies | Wrap numpy arrays for imwrite support |
| Isoview lightsheet directory | `IsoviewArray` | (T, Z, V, Y, X) | Multi-view lightsheet data |

## Array Type Details

### MboRawArray

**Returned when:** Reading raw ScanImage TIFF files with multi-ROI metadata

```python
import mbo_utilities as mbo

# Raw ScanImage TIFFs
scan = mbo.imread("/path/to/raw/*.tif")
# Returns: MboRawArray

print(type(scan))     # <class 'MboRawArray'>
print(scan.shape)     # (T, Z, Y, X) - e.g., (10000, 14, 456, 896)
print(scan.num_rois)  # Number of ROIs
print(scan.num_planes)  # Alias for num_channels (Z planes)

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

**Key Features:**

- Automatic ROI stitching/splitting via `roi` property
- Bidirectional scan-phase correction (configurable methods)
- Multi-plane volumetric data support
- ROI position extraction from ScanImage metadata
- Stores all ScanImage-metadata in array.metadata["si"]

---

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

## Decision Tree: What Will imread() Return?

```text
imread(input)
  │
  ├─ isinstance(input, np.ndarray)?
  │   └─ Yes → NumpyArray (in-memory wrapper)
  │
  ├─ Is input a Path or string?
  │   │
  │   ├─ .npy file?
  │   │   └─ NumpyArray (memory-mapped)
  │   │
  │   ├─ .nwb file?
  │   │   └─ NWBArray
  │   │
  │   ├─ .h5 / .hdf5 file?
  │   │   └─ H5Array
  │   │
  │   ├─ .zarr directory?
  │   │   └─ ZarrArray
  │   │
  │   ├─ .bin file (direct path)?
  │   │   └─ BinArray
  │   │
  │   ├─ .tif / .tiff file(s)?
  │   │   ├─ Has ScanImage ROI metadata? → MboRawArray
  │   │   ├─ Has MBO metadata (multiple files)? → MBOTiffArray
  │   │   └─ Standard TIFF → TiffArray
  │   │
  │   ├─ Directory?
  │   │   ├─ Contains planeXX.tiff files? → TiffArray (volumetric)
  │   │   ├─ Contains planeXX/ subdirs with ops.npy? → Suite2pArray (volumetric)
  │   │   ├─ Contains ops.npy? → Suite2pArray
  │   │   └─ Contains raw ScanImage TIFFs? → MboRawArray
  │   │
  │   └─ ops.npy file directly?
  │       └─ Suite2pArray
  │
  └─ List of paths?
      ├─ All .tif files → TiffArray or MBOTiffArray
      └─ All .zarr stores → ZarrArray (stacked along Z)
```

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
| `.num_rois` | `int` | Number of ROIs (MboRawArray) |
| `.close()` | method | Release file handles |

## API Reference

- `mbo_utilities.imread()` - Smart file reader
- `mbo_utilities.imwrite()` - Universal file writer
- `mbo_utilities.arrays` - Direct access to array classes
