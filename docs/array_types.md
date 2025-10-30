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

`mbo_utilities.imread()` is a smart file reader that automatically detects the file type and returns the appropriate lazy array class. This guide explains what to expect when reading different file formats.

## Quick Reference

| Input | Returns | Use Case |
|-------|---------|----------|
| `.tif` (raw ScanImage) | `MboRawArray` | Multi-ROI volumetric data with phase correction |
| `.tif` (processed) | `TiffArray` or `MBOTiffArray` | Standard TIFF files |
| `.bin` (direct path) | `BinArray` | Direct binary file manipulation |
| Directory with `ops.npy` | `Suite2pArray` | Suite2p workflow integration |
| `.h5` / `.hdf5` | `H5Array` | HDF5 datasets |
| `.zarr` | `ZarrArray` | Zarr v3 stores |
| `.npy` | `NpyArray` | NumPy memory-mapped files |

## Array Type Details

### MboRawArray

**Returned when:** Reading raw ScanImage TIFF files with multi-ROI metadata

```python
import mbo_utilities as mbo

# Raw ScanImage TIFFs
scan = mbo.imread("/path/to/raw/*.tif")
# Returns: MboRawArray

print(type(scan))  # <class 'MboRawArray'>
print(scan.shape)  # (T, Z, Y, X)
print(scan.num_rois)  # Number of ROIs

# Configuration options
scan.roi = None      # Stitch all ROIs (default)
scan.roi = 0         # Split into separate ROIs
scan.roi = 1         # Use only ROI 1
scan.fix_phase = True  # Enable scan-phase correction
scan.use_fft = True    # Use FFT-based phase correction
```

**Key Features:**

- Automatic ROI stitching/splitting
- Bidirectional scan-phase correction
- Lazy loading (doesn't load entire dataset into memory)
- Multi-plane volumetric data support

### BinArray

**Returned when:** Explicitly reading a `.bin` file path

```python
# Reading a specific binary file
arr = mbo.imread("path/to/data_raw.bin")
# Returns: BinArray

print(type(arr))   # <class 'BinArray'>
print(arr.shape)   # (nframes, Ly, Lx)
print(arr.nframes) # Number of frames

# Access data like numpy array
frame = arr[0]      # First frame
subset = arr[0:100] # First 100 frames

# Close when done
arr.close()
```

**Key Features:**

- Direct binary file access without full Suite2p context
- Automatically infers shape from adjacent `ops.npy` if present
- Can provide shape manually: `imread("file.bin", shape=(1000, 512, 512))`
- Read/write access to the binary file
- Useful for workflows that manipulate individual binaries (e.g., `data_raw.bin` vs `data.bin`)

**When to use:**

- Reading/writing specific binary files in a Suite2p workflow
- Creating new binary files from scratch
- When you want to work with the file directly, not through Suite2p's abstraction

### Suite2pArray

**Returned when:** Reading a directory containing `ops.npy`

```python
# Reading a Suite2p directory
arr = mbo.imread("/path/to/suite2p/plane0")
# Returns: Suite2pArray (reads ops.npy automatically)

print(type(arr))   # <class 'Suite2pArray'>
print(arr.shape)   # (nframes, Ly, Lx)

# Suite2pArray handles both data_raw.bin and data.bin
print(arr.raw_file)  # Path to data_raw.bin
print(arr.reg_file)  # Path to data.bin
print(arr.active_file)  # Currently active file

# Switch between raw and registered
arr.switch_channel(use_raw=True)   # Use data_raw.bin
arr.switch_channel(use_raw=False)  # Use data.bin
```

**Key Features:**

- Full Suite2p context (metadata from `ops.npy`)
- Access to both raw and registered data
- Integrates with Suite2p's processing pipeline
- Preserves all Suite2p metadata

**When to use:**

- Working with Suite2p output directories
- Need access to both raw and registered data
- Want full Suite2p metadata and context

### TiffArray & MBOTiffArray

**Returned when:** Reading processed TIFF files (not raw ScanImage)

```python
# Standard TIFF files
arr = mbo.imread("/path/to/processed.tif")
# Returns: TiffArray or MBOTiffArray (depending on metadata)

print(type(arr))   # <class 'TiffArray'> or <class 'MBOTiffArray'>
```

**Difference:**

- `MBOTiffArray`: Has MBO-specific metadata, uses Dask for lazy loading
- `TiffArray`: Standard TIFF, basic lazy loading

### H5Array

**Returned when:** Reading HDF5 files

```python
# HDF5 dataset
arr = mbo.imread("/path/to/data.h5")
# Returns: H5Array

# Optionally specify dataset name
arr = mbo.imread("/path/to/data.h5", dataset="imaging_data")
```

### ZarrArray

**Returned when:** Reading Zarr stores

```python
# Zarr directory or collection
arr = mbo.imread("/path/to/data.zarr")
# Returns: ZarrArray

# Can read nested zarr stores (one per plane)
arr = mbo.imread("/path/with/plane01.zarr", "/path/with/plane02.zarr")
```

### NpyArray

**Returned when:** Reading `.npy` memory-mapped files

```python
# NumPy file
arr = mbo.imread("/path/to/data.npy")
# Returns: NpyArray
```

## Decision Tree: What Will imread() Return?

```bash

Input
  │
  ├─ .tif/.tiff file?
  │   ├─ Has ScanImage metadata? → MboRawArray
  │   ├─ Has MBO metadata? → MBOTiffArray
  │   └─ Standard TIFF → TiffArray
  │
  ├─ .bin file?
  │   ├─ Explicit file path (e.g., "data_raw.bin")? → BinArray
  │   └─ Directory with ops.npy? → Suite2pArray
  │
  ├─ Directory with ops.npy? → Suite2pArray
  │
  ├─ .h5/.hdf5 file? → H5Array
  │
  ├─ .zarr directory? → ZarrArray
  │
  └─ .npy file? → NpyArray
```

- {ref}`user_guide` - Data extraction and assembly workflows
- {ref}`glossary` - Terminology and definitions
- API Reference: `mbo_utilities.imread()`, `mbo_utilities.imwrite()`
