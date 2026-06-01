# Metadata Handling Deep Dive

This document provides an in-depth breakdown of how metadata, options, and planar/volumetric data are handled in `mbo_utilities`.

The metadata system is centralized in `mbo_utilities.metadata` but heavily integrated into the Array classes (`Suite2pArray`, `TiffArray`, `ScanImageArray`). The system is designed to normalize disparate inputs (ScanImage headers, Suite2p `ops.npy`, ImageJ tags) into a minimal standard set of "canonical" keys.

## 1. Core Concepts & Variables

### The `ops` Variable
While `plane_ops` is often referenced in user scripts, in the codebase this conceptually maps to the **`ops` dictionary** loaded from `ops.npy`.
*   **In Suite2p**: Use `ops.npy`.
*   **In ScanImage**: Use `SI` header tags -> converted to a dictionary structure similar to `ops`.

### Important Variables
| Variable (Canonical) | Source Names (Aliases) | Description | Important for... |
| :--- | :--- | :--- | :--- |
| **`num_zplanes`** | `nplanes`, `num_slices`, `Z` | Number of Z-planes in the volume. **Critical volume flag.** | Distinguishing Planar vs Volumetric. |
| **`num_timepoints`** | `nframes`, `T`, `num_frames` | Total timepoints (volumes or frames). | Array shaping `(T, ...)`. |
| **`stack_type`** | `stackType` (derived) | `single_plane`, `lbm`, `piezo`, `pollen`. | Determining how Z is parsed from TIFFs. |
| **`fs`** | `frame_rate`, `scanFrameRate` | Sampling frequency (Hz) **per plane**. | Timing & Analysis. |
| **`dx`, `dy`, `dz`** | `micronsPerPixel`, `z_step` | Spatial resolution in µm. | Scale bars & physical measurements. |
| **`roi` / `fov`** | `linesPerFrame`, `pixelsPerLine` | Dimensions of the acquired window. | Reshaping raw data streams. |

## 2. Planar vs. Volumetric Handling

The system automatically detects whether data is 2D (Planar) or 3D+T (Volumetric) based on file structure and metadata tags.

### A. Detection Logic
| Context | **Planar (Single Plane)** | **Volumetric (Multi-plane)** |
| :--- | :--- | :--- |
| **Suite2p** | `ops.npy` exists in the target folder OR only `plane0/` exists. | Multiple `planeX/` subdirectories found containing `ops.npy`. |
| **ScanImage** | Metadata `num_zplanes` == 1. | Metadata `num_zplanes` > 1 OR `stack_type` is `lbm`/`piezo`. |
| **TIFF Files** | Single file with no special hierarchy. | Directory of files matching `plane\d+.*` naming pattern. |

### B. Internal Handling (The "Plane Ops")
In `Suite2pArray`, the distinction is explicit in the initialization:

1.  **`_init_volume`**:
    *   Iterates through sorted `plane_dirs`.
    *   Creates a `_SinglePlaneReader` for **each plane**.
    *   **"Plane Ops"**: Each `_SinglePlaneReader` loads its own `ops.npy` into `self.metadata`.
    *   **Aggregation**: The main array's `metadata` is a copy of the *first plane's* ops, but with `nplanes` updated to match the number of directories.
2.  **`_init_single_plane`**:
    *   Creates only one `_SinglePlaneReader`.
    *   `_is_volumetric` is set to `False`.

**Key Implication**: `Suite2pArray` acts as a facade. If volumetric, `array[t, z, y, x]` routes the request to the specific `_SinglePlaneReader` for appropriate `z`.

## 3. Metadata Extraction Options (`get_metadata`)

When you call `get_metadata(file)`, the following decision tree is executed:

1.  **Is it a Raw ScanImage TIFF?** (`is_raw_scanimage`)
    *   **Yes**: Parse `scanimage_metadata` attribute.
        *   Run `clean_scanimage_metadata`: Nests `SI.hChannels` -> `si['hChannels']`.
        *   Run `detect_stack_type`: Checks for LBM (channels=z) or Piezo usage.
        *   Compute `num_timepoints`: Adjusts for averaging and slices.
2.  **Is it a Processed TIFF?**
    *   **Yes** (has `shaped_metadata`): Read directly from ImageJ/MBO tags.
3.  **Fallback**:
    *   Look for a sibling `ops.npy`.
    *   Load it and map Suite2p keys (`nplanes`, `Lx`) to canonical keys.

## 4. In-Depth Reference Table

| Variable / Option | Canonical Key | Expected Value | Logic / Notes |
| :--- | :--- | :--- | :--- |
| **Plane Operations** | `ops` (internal) | Dictionary | In `Suite2pArray`, each Z-slice manages its own `ops` dict. High-level metadata matches Plane 0. |
| **Z-Plane Count** | `num_zplanes` | `int` (e.g., 1 or 30) | **LBM**: Count of `SI.hChannels.channelSave`. <br> **Piezo**: `SI.hStackManager.numSlices`. |
| **Stack Mode** | `stack_type` | `String` | **`lbm`**: Multi-channel acquisition where channels map to depths. <br> **`piezo`**: Slow Z-stage stepping. <br> **`pollen`**: LBM + Piezo calibration. |
| **Frame Rate** | `fs` | `float` (Hz) | Extracted from `SI.hRoiManager.scanFrameRate`. Valid for the *plane*, not the volume (usually). |
| **Timepoints** | `num_timepoints` | `int` | **Single**: Total frames in file. <br> **Volume**: `Total Frames // (Slices * AvgFactor)`. |
| **ROI / Grid** | `num_mrois` | `int` | Number of distinct scan fields (multi-ROI) stitched into the image. |
| **Pixel Size** | `dx`, `dy` | `float` (µm) | Pixel resolution. Crucial for scale bars. Note: `dz` (Z-step) is manually passed for LBM often. |

### Summary of `plane_ops`
If you are looking for a variable strictly named `plane_ops`, it does not exist as a global class. It likely refers to:
1.  The `ops` dictionary loaded inside `_SinglePlaneReader` in `arrays/suite2p.py`.
2.  The `ops` variable commonly used in users' analysis scripts when iterating over planes.

## 5. Output Metadata (`OutputMetadata`)

When writing subsets of data (e.g., every Nth z-plane, specific frame ranges), metadata needs to be adjusted accordingly. The `OutputMetadata` class handles this transformation automatically.

### A. Reactive Metadata Adjustments

| Selection Type | Metadata Adjustment |
| :--- | :--- |
| **Every Nth z-plane** | `dz` multiplied by step factor |
| **Contiguous frames** | `fs` adjusted for frame step |
| **Non-contiguous frames** | `fs` invalidated (set to `None`) |
| **Frame subsampling** | `fs` divided by frame step |

### B. Usage Example

```python
from mbo_utilities.metadata import OutputMetadata

# source metadata
source = {"dz": 5.0, "fs": 30.0, "dx": 0.5, "dy": 0.5}

# write every 2nd z-plane
out = OutputMetadata(source, plane_indices=[0, 2, 4, 6, 8, 10])
print(out.dz)           # 10.0 (doubled)
print(out.z_step_factor) # 2

# write non-contiguous frames (fs becomes invalid)
out = OutputMetadata(source, frame_indices=[0, 50, 200, 500])
print(out.fs)           # None
print(out.is_contiguous) # False

# write every 3rd frame (contiguous with step)
out = OutputMetadata(source, frame_indices=[0, 3, 6, 9, 12])
print(out.fs)           # 10.0 (30/3)
print(out.is_contiguous) # True
```

### C. Format-Specific Output

`OutputMetadata` provides builders for different output formats:

| Method | Output Format | Description |
| :--- | :--- | :--- |
| `to_imagej(shape)` | ImageJ TIFF | Returns `(ij_metadata, resolution)` tuple |
| `to_ome_ngff(dims)` | OME-NGFF v0.5 | Returns multiscales metadata dict |
| `to_napari_scale(dims)` | napari | Returns scale tuple for visualization |
| `to_dict()` | Generic | Returns flat metadata dict with all aliases |

### D. Key Properties

| Property | Type | Description |
| :--- | :--- | :--- |
| `dz` | `float \| None` | Adjusted z-step (source × z_step_factor) |
| `dx`, `dy` | `float` | Unchanged pixel sizes |
| `fs` | `float \| None` | Adjusted frame rate (None if non-contiguous) |
| `finterval` | `float \| None` | Frame interval in seconds (1/fs) |
| `is_contiguous` | `bool` | Whether frame selection is contiguous |
| `z_step_factor` | `int` | Multiplication factor for z-step |
| `frame_step` | `int` | Step between selected frames |
| `voxel_size` | `VoxelSize` | Adjusted voxel size dataclass |
