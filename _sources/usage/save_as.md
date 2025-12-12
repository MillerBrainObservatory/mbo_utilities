---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: lsp
  language: python
  name: python3
---

# Save Imaging Data with `imwrite`

{func}`mbo_utilities.imwrite` is the primary function for writing imaging data to disk in various formats.

It handles:
- Multi-dimensional imaging data (TZYX)
- Multiple ROI configurations
- Z-plane registration
- Phase correction
- Format conversion between TIFF, Zarr, HDF5, and Suite2p binary
- Chunked streaming for large datasets

## Supported File Formats

- `.tiff`, `.tif` : Multi-page TIFF (BigTIFF for >4GB)
- `.bin` : Suite2p-compatible binary format with ops.npy metadata
- `.zarr` : Zarr v3 array store
- `.h5`, `.hdf5` : HDF5 format

---

```{code-cell} ipython3
from pathlib import Path
import numpy as np
import mbo_utilities as mbo
```

## Basic Usage Examples

### Stitch All ROIs and Save as TIFF

The default behavior stitches/fuses all ROIs horizontally into a single field of view:

```{code-cell} ipython3
# Load raw data
data = mbo.imread("path/to/raw/*.tiff")

# Stitch all ROIs together
data.roi = None  # or just omit setting roi
mbo.imwrite(data, "output/session1")
# Creates: plane01_stitched.tiff, plane02_stitched.tiff, ..., plane14_stitched.tiff
```

### Save Specific Planes Only

Export only specific z-planes (1-based indexing):

```{code-cell} ipython3
# Save first, middle, and last planes (for 14-plane volume)
mbo.imwrite(data, "output/session1", planes=[1, 7, 14])
# Creates: plane01_stitched.tiff, plane07_stitched.tiff, plane14_stitched.tiff
```

Single plane:

```{code-cell} ipython3
# Save only plane 7
mbo.imwrite(data, "output/session1", planes=7)
# Creates: plane07_stitched.tiff
```

---

## ROI Handling Options

### Split All ROIs into Separate Files

```{code-cell} ipython3
data.roi = 0  # Special value: save all ROIs separately
mbo.imwrite(data, "output/split_rois")
# Creates: plane01_roi1.tiff, plane01_roi2.tiff, ..., plane14_roi1.tiff, plane14_roi2.tiff
```

### Save Specific ROI Only

```{code-cell} ipython3
data.roi = 1  # Save only ROI 1
mbo.imwrite(data, "output/roi1_only")
# Creates: plane01_roi1.tiff, plane02_roi1.tiff, ..., plane14_roi1.tiff
```

### Save Multiple Specific ROIs

```{code-cell} ipython3
data.roi = [1, 3]  # Save ROIs 1 and 3
mbo.imwrite(data, "output/roi1_and_roi3")
# Creates: plane01_roi1.tiff, plane01_roi3.tiff, ..., plane14_roi1.tiff, plane14_roi3.tiff
```

---

## Format Conversion

### Convert to Suite2p Binary Format

```{code-cell} ipython3
# Save as Suite2p-compatible binary files
data.roi = 0  # Must split ROIs for Suite2p format
mbo.imwrite(data, "output/suite2p", ext=".bin")
# Creates: plane01_roi1/data_raw.bin + ops.npy, plane01_roi2/data_raw.bin + ops.npy, ...
```

Each plane/ROI gets its own directory with:
- `data_raw.bin`: Raw imaging data
- `ops.npy`: Metadata file compatible with Suite2p

### Save to Zarr Format

Zarr is efficient for large datasets and supports compression:

```{code-cell} ipython3
# Save as Zarr v3 stores
data.roi = 0
mbo.imwrite(data, "output/zarr_data", ext=".zarr")
# Creates: plane01_roi1.zarr, plane01_roi2.zarr, ...
```

### Save to HDF5 Format

```{code-cell} ipython3
# Save as HDF5 files
mbo.imwrite(data, "output/hdf5_data", ext=".h5")
# Creates: plane01_stitched.h5, plane02_stitched.h5, ...
```

---

## Advanced Features

### Enable Phase Correction

Fix bidirectional scan phase artifacts before saving (for raw ScanImage data):

```{code-cell} ipython3
data = mbo.imread("path/to/raw/*.tiff")
data.fix_phase = True
data.phasecorr_method = "mean"  # Options: "mean", "median", "max"
data.use_fft = True  # Use FFT-based correction (faster)
mbo.imwrite(data, "output/phase_corrected")
```

**Phase correction methods:**
- `"mean"`: Use mean intensity for phase estimation (default)
- `"median"`: Use median intensity (more robust to outliers)
- `"max"`: Use maximum intensity projection

### Z-Plane Registration

Align z-planes spatially using Suite3D registration:

```{code-cell} ipython3
# Automatic registration with Suite3D (requires Suite3D + CuPy installed)
data = mbo.imread("path/to/raw/*.tiff")
mbo.imwrite(
    data,
    "output/registered",
    register_z=True,
    roi=None  # Usually stitch ROIs for registration
)
# Computes rigid shifts between planes
# Validates registration results
# Applies shifts during write to align planes
```

**Requirements:**
```bash
pip install mbo_utilities[suite3d,cuda12]
```

**What happens:**
1. Creates/reuses Suite3D job directory in output path
2. Computes rigid shifts between z-planes
3. Validates registration (checks `summary.npy` for valid `plane_shifts`)
4. Applies shifts during write
5. Output files are padded to accommodate all shifts

### Use Pre-Computed Registration Shifts

If you already have registration shifts from a previous run:

```{code-cell} ipython3
# Load previously computed shifts
summary = np.load("previous_job/summary/summary.npy", allow_pickle=True).item()
shift_vectors = summary['plane_shifts']  # shape: (n_planes, 2) for [dy, dx]

# Apply shifts without re-running registration
mbo.imwrite(data, "output/registered", shift_vectors=shift_vectors)
```

---

## Data Subsetting

### Export Subset of Frames

Useful for testing, creating demos, or extracting specific time windows:

```{code-cell} ipython3
# Export only first 1000 frames
mbo.imwrite(data, "output/test_data", num_frames=1000, planes=[1, 7, 14])
```

### Combine Frame Subsetting with ROI Selection

```{code-cell} ipython3
# First 500 frames, only ROI 1, planes 1 and 7
data.roi = 1
mbo.imwrite(data, "output/roi1_subset", num_frames=500, planes=[1, 7])
```

---

## File Management

### Overwrite Existing Files

By default, `imwrite` skips existing files. To overwrite:

```{code-cell} ipython3
mbo.imwrite(data, "output/session1", planes=[1, 2, 3], overwrite=True)
```

### Custom Metadata

Add custom metadata to output file headers/attributes:

```{code-cell} ipython3
custom_meta = {
    "experimenter": "MBO-User",
    "date": "2025-01-15",
    "experiment_id": "EXP-12345",
    "notes": "High laser power session"
}
mbo.imwrite(data, "output/session1", metadata=custom_meta)
```

---

## Performance Tuning

### Adjust Chunk Size

Control memory usage and write performance:

```{code-cell} ipython3
# Larger chunks = faster but more memory
mbo.imwrite(data, "output/session1", target_chunk_mb=50)

# Smaller chunks = less memory but slower
mbo.imwrite(data, "output/session1", target_chunk_mb=10)
```

Default is 20 MB. Adjust based on available RAM.

### Progress Callback

Monitor write progress in custom UIs:

```{code-cell} ipython3
def progress_handler(progress, plane):
    print(f"Plane {plane}: {progress*100:.1f}% complete")

mbo.imwrite(data, "output/session1", progress_callback=progress_handler)
```

**Example output:**
```
Plane 1: 25.0% complete
Plane 1: 50.0% complete
Plane 1: 75.0% complete
Plane 1: 100.0% complete
Plane 7: 25.0% complete
...
```

---

## Advanced Use Cases

### Reorder Planes During Write

```{code-cell} ipython3
# Write planes in custom order
mbo.imwrite(
    data,
    "output/session1",
    planes=[3, 2, 1],
    order=[2, 1, 0]  # Reverses the order
)
# Writes plane 3 first, then plane 2, then plane 1
```

### Debug Mode

Enable verbose logging for troubleshooting:

```{code-cell} ipython3
mbo.imwrite(data, "output/session1", debug=True)
```

---

## Complete Example Workflows

### Workflow 1: Raw ScanImage to Stitched TIFF

```{code-cell} ipython3
# Load raw data
data = mbo.imread("path/to/raw/*.tiff")

# Configure phase correction
data.fix_phase = True
data.phasecorr_method = "mean"

# Stitch all ROIs and save all planes
data.roi = None
mbo.imwrite(data, "output/stitched_corrected")
```

### Workflow 2: Multi-ROI Data to Suite2p Format

```{code-cell} ipython3
# Load data
data = mbo.imread("path/to/raw/*.tiff")

# Save each ROI separately in Suite2p binary format
data.roi = 0
mbo.imwrite(data, "output/suite2p_input", ext=".bin")
```

### Workflow 3: Registered Z-Stack with Subset Export

```{code-cell} ipython3
# Load data
data = mbo.imread("path/to/raw/*.tiff")

# Register z-planes and export first 2000 frames of key planes
mbo.imwrite(
    data,
    "output/registered_subset",
    register_z=True,
    roi=None,
    planes=[1, 5, 9, 14],
    num_frames=2000,
    overwrite=True
)
```

### Workflow 4: High-Performance Zarr Export

```{code-cell} ipython3
# Load data
data = mbo.imread("path/to/raw/*.tiff")

# Save to Zarr with large chunks for performance
data.roi = 0
mbo.imwrite(
    data,
    "output/zarr_optimized",
    ext=".zarr",
    target_chunk_mb=100,  # Large chunks for network storage
    planes=list(range(1, 15))  # All 14 planes
)
```

---

## File Naming Convention

Understanding output file names:

**Single ROI or stitched:**
```
plane01_stitched.tiff
plane02_stitched.tiff
...
```

**Multiple ROIs:**
```
plane01_roi1.tiff
plane01_roi2.tiff
plane02_roi1.tiff
plane02_roi2.tiff
...
```

**Binary format:**
```
plane01_roi1/
  ├── data_raw.bin
  └── ops.npy
plane01_roi2/
  ├── data_raw.bin
  └── ops.npy
...
```

---

## See Also

- {func}`mbo_utilities.imread` - Load imaging data from various formats
- {func}`mbo_utilities.run_gui` - Visualize data interactively
- {func}`mbo_utilities.save_mp4` - Export data as video
