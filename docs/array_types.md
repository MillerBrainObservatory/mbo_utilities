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
# Array Types and imread() Guide

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

## Common Workflows

### Workflow 1: ScanImage → Suite2p Binary

```python
import mbo_utilities as mbo

# Step 1: Read raw ScanImage TIFFs
scan = mbo.imread("/data/raw/*.tif")
# Returns: MboRawArray

# Step 2: Configure and write to binary
scan.roi = None  # Stitch ROIs
scan.fix_phase = True  # Enable phase correction

mbo.imwrite(
    scan,
    "/data/processed",
    ext=".bin",
    planes=[7],  # Just plane 7
    register_z=False  # No z-registration yet
)
# Creates: plane07_stitched/data_raw.bin + ops.npy
```

### Workflow 2: Read Binary for Processing

```python
# Read specific binary file
data = mbo.imread("/data/processed/plane07_stitched/data_raw.bin")
# Returns: BinArray (NOT Suite2pArray!)

# Process the data...
processed = run_my_registration(data)

# Write registered data with explicit name
mbo.imwrite(
    processed,
    "/data/processed/plane07_stitched",
    ext=".bin",
    output_name="data.bin"  # Explicit: this is registered data
)
```

### Workflow 3: Suite2p Integration

```python
# Read Suite2p directory
plane = mbo.imread("/data/suite2p/plane0")
# Returns: Suite2pArray

# Access raw data
plane.switch_channel(use_raw=True)
raw_frames = plane[0:100]

# Access registered data
plane.switch_channel(use_raw=False)
reg_frames = plane[0:100]
```

### Workflow 4: Format Conversion

```python
# Read from any format
data = mbo.imread("/data/input.h5")

# Convert to another format
mbo.imwrite(data, "/data/output", ext=".zarr")
mbo.imwrite(data, "/data/output", ext=".tiff")
mbo.imwrite(data, "/data/output", ext=".bin")
```

## Binary File Naming: data_raw.bin vs data.bin

In Suite2p workflows, there are typically two binary files:

- **`data_raw.bin`**: Raw, unregistered imaging data
- **`data.bin`**: Motion-corrected (registered) imaging data

### Controlling Binary Output Names

Use the `output_name` parameter in `imwrite()`:

```python
# Write raw data
mbo.imwrite(
    raw_data,
    output_dir,
    ext=".bin",
    output_name="data_raw.bin"  # Explicit
)

# Write registered data
mbo.imwrite(
    registered_data,
    output_dir,
    ext=".bin",
    output_name="data.bin"  # Explicit
)
```

### Example: Complete Suite2p Preprocessing Workflow

```python
import mbo_utilities as mbo

# 1. Read raw ScanImage TIFFs
scan = mbo.imread("/raw/*.tif")  # MboRawArray
scan.roi = None
scan.fix_phase = True

# 2. Write raw binary for Suite2p
output_dir = "/processed/plane07"
mbo.imwrite(
    scan,
    output_dir,
    ext=".bin",
    planes=[7],
    output_name="data_raw.bin",  # Explicit: raw data
    metadata={"notes": "Preprocessed, stitched, phase-corrected"}
)

# 3. Read the raw binary for registration
raw_data = mbo.imread(f"{output_dir}/data_raw.bin")  # BinArray
print(type(raw_data))  # <class 'BinArray'>

# 4. Run registration (example)
from suite2p import registration
ops = {**default_ops, "Ly": raw_data.Ly, "Lx": raw_data.Lx}
registered_data, rigid_offsets = registration.register_binary(raw_data, ops)

# 5. Write registered binary
mbo.imwrite(
    registered_data,
    output_dir,
    ext=".bin",
    output_name="data.bin",  # Explicit: registered data
    metadata=ops
)

# 6. Now read as Suite2pArray for analysis
plane = mbo.imread(output_dir)  # Suite2pArray (directory with ops.npy)
plane.switch_channel(use_raw=False)  # Use registered data
```

## Best Practices

### 1. Reading Binary Files: Be Explicit

```python
# ✅ GOOD: Explicit about what you want
raw_bin = mbo.imread("/data/plane0/data_raw.bin")  # BinArray
reg_bin = mbo.imread("/data/plane0/data.bin")      # BinArray

# ❌ POTENTIALLY CONFUSING: Depends on what's in the directory
data = mbo.imread("/data/plane0")  # Suite2pArray - which file is active?
```

### 2. Writing Binary Files: Use output_name

```python
mbo.imwrite(raw_data, output_dir, ext=".bin", output_name="data_raw.bin")
mbo.imwrite(reg_data, output_dir, ext=".bin", output_name="data.bin")

# UNCLEAR: Which binary is this?
mbo.imwrite(data, output_dir, ext=".bin")  # Defaults to data_raw.bin
```

### 3. BinArray vs Suite2pArray

**Use BinArray when:**

- Manipulating individual binary files
- Creating/reading specific .bin files
- Don't need full Suite2p context

**Use Suite2pArray when:**

- Working with complete Suite2p output
- Need to switch between raw/registered
- Want full metadata and context

```python
# BinArray: Direct file manipulation
raw = mbo.imread("plane0/data_raw.bin")  # Just this file
raw[0] = processed_frame  # Can write to it

# Suite2pArray: Full Suite2p context
plane = mbo.imread("plane0")  # Whole directory
plane.switch_channel(use_raw=True)  # Access either file
print(plane.metadata)  # Full ops.npy metadata
```

### 4. Shape Inference

BinArray will try to infer shape from `ops.npy`:

```python
# If ops.npy exists, shape is inferred
arr = mbo.imread("data_raw.bin")  # Reads shape from ops.npy

# Can provide shape explicitly
arr = mbo.imread("data_raw.bin", shape=(1000, 512, 512))

# Will raise error if ops.npy missing and shape not provided
arr = mbo.imread("standalone.bin")  # ValueError!
```

## Troubleshooting

### "Cannot infer shape for file.bin"

```python
# Problem: No ops.npy and no shape provided
arr = mbo.imread("file.bin")  # ERROR!

# Solution 1: Provide shape
arr = mbo.imread("file.bin", shape=(nframes, Ly, Lx))

# Solution 2: Ensure ops.npy exists
# (Created automatically when writing with ext=".bin")
```

### "Getting Suite2pArray when I want BinArray"

```python
# Problem: Passing directory, not file
arr = mbo.imread("/data/plane0")  # Suite2pArray

# Solution: Pass explicit .bin path
arr = mbo.imread("/data/plane0/data_raw.bin")  # BinArray
```

### "Which file is Suite2pArray reading?"

```python
plane = mbo.imread("/data/plane0")

# Check active file
print(plane.active_file)  # Shows which .bin is being used

# Suite2pArray prefers data.bin if it exists, falls back to data_raw.bin
# Use switch_channel() to control:
plane.switch_channel(use_raw=True)   # Force data_raw.bin
plane.switch_channel(use_raw=False)  # Force data.bin
```

## See Also

- {ref}`assembly` - Data extraction and assembly workflows
- {ref}`glossary` - Terminology and definitions
- API Reference: `mbo_utilities.imread()`, `mbo_utilities.imwrite()`
