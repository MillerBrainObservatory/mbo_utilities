# v2.5.0 Changelog

Major release with new array features system, comprehensive metadata refactor, performance benchmarking tools, and significant module reorganization for cleaner imports.

## Features

Arrays now support composable "features" that provide standardized access to common properties.

Instead of each array class implementing its own voxel size, frame rate, or statistics logic, features provide a consistent interface across all array types.

Features attach to arrays and expose properties like `arr.voxel_size`, `imwrite`. So the `save_as` GUI file menu is only accessible if the loaded array has the proper `imwrite` feature.

| Feature | Description |
|---------|-------------|
| `DimLabels` / `DimLabelsMixin` | dimension labeling (T, Z, Y, X) |
| `VoxelSizeFeature` | physical pixel/voxel dimensions |
| `FrameRateFeature` | temporal sampling frequency |
| `DisplayRangeFeature` | min/max for display scaling |
| `ROIFeature` | multi-ROI handling |
| `DataTypeFeature` | dtype with lazy conversion |
| `CompressionFeature` | codec settings |
| `ChunkSizeFeature` | chunking configuration |
| `StatsFeature` | per-slice statistics for z-planes, cameras, ROIs |
| `PhaseCorrectionFeature` | bidirectional scan correction |

See commits: [8542735](../../commit/8542735), [cf19fd0](../../commit/cf19fd0), [ae4845d](../../commit/ae4845d), [4d6ed61](../../commit/4d6ed61), [e07e51a](../../commit/e07e51a), [b91800f](../../commit/b91800f), [965e08f](../../commit/965e08f), [c6c4d6f](../../commit/c6c4d6f), [88772dd](../../commit/88772dd), [488ca62](../../commit/488ca62)

### Metadata

ScanImage TIFF files contain rich metadata about acquisition settings, but extracting and interpreting this data was scattered across multiple functions. The new `metadata` package consolidates all metadata handling into one place with clear separation between raw parsing, parameter access, and stack type detection.

The package automatically detects whether a file is from an LBM (light beads microscopy), piezo-stack, or single-plane acquisition and provides appropriate defaults for each.

- `StackType` enum for LBM, PIEZO, SINGLE_PLANE detection
- `detect_stack_type()`, `is_lbm_stack()`, `is_piezo_stack()` for automatic classification
- `get_stack_info()`, `extract_roi_slices()` for ScanImage multi-ROI parsing
- `VoxelSize` dataclass with proper unit handling (micrometers)
- Standardized parameter names with alias mapping for backwards compatibility
- Comprehensive test coverage in `tests/test_metadata_module.py`

See commit: [7be9ec2](../../commit/7be9ec2)

### Benchmarking System

Run benchmarks on any system to compare performance across different computers and track how the pipeline performs on different hardware. Results include system information (CPU, RAM, OS, GPU) so you can understand why performance differs between machines.

The goal is to identify bottlenecks and optimize the most common operations: reading frames, applying phase correction, and writing to different formats.

```bash
mbo benchmark /path/to/raw                    # quick benchmark
mbo benchmark /path/to/raw --config full      # full suite
mbo benchmark /path/to/raw --zarr             # zarr chunking test
mbo benchmark /path/to/raw --plot             # generate visualization
```

**Presets:**

- `quick` - basic tests, ~1-2 minutes
- `full` - comprehensive suite, ~10-15 minutes
- `read-only` - skip write tests
- `write-only` - only test output formats
- `analysis` - throughput and scaling focus

**What gets measured:**

- Initialization time (opening files, parsing metadata)
- Frame indexing speed (single frames, batches, z-planes)
- Phase correction overhead (disabled vs correlation vs FFT)
- Write performance across formats (zarr, tiff, h5, bin)
- Throughput in MB/s to compare against disk limits
- Random vs sequential access to measure seek overhead
- Cold vs warm reads to understand caching effects
- Zarr chunk/shard configurations to find optimal settings

**Output:**

- Dark-mode plots matching MBO documentation style
- JSON files with full results, system info, and git commit for reproducibility

See commits: [6535625](../../commit/6535625), [260f289](../../commit/260f289), [905bb3a](../../commit/905bb3a), [ea811d7](../../commit/ea811d7)

## Improvements

### GUI

- Stats property on ZarrArray with zstats backwards compatibility ([b92ae08](../../commit/b92ae08))
- Improved slider dimension handling via `get_slider_dims()` ([e855efc](../../commit/e855efc), [44bff05](../../commit/44bff05))
- Metadata search and keybinds ([be92328](../../commit/be92328))

Added features to data-arrays to control how metadata are shown to the user:

- `REQUIRED_METADATA` for metadata that requires user input, e.g. `dz` (z-step in um/px) for LBM acquisition, prevent saving without this input
- `METADATA_CONTEXT` for metadata that requires user input, e.g. `dz` (z-step in um/px) for LBM acquisition, prevent saving without this input

```python

METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Raw TIFF page height in pixels.",
        "Lx": "Raw TIFF page width in pixels.",
        "num_zplanes": "Number of z-planes in the stack.",
        "dz": "Z-step size in micrometers.",
    }
```

This gives tooltips to the above metadata in the viewer.

Added a very cool function for `imgui-tables`. Since `imgui.columns()` is deprecated, we should instead work with tables. `gui/_widgets/draw_checkbox_grid` will use the available content width and the item (checkbox + text) width to auto-adjust the number of columns shown.

### Writers

- Fixed zarr shard divisibility for chunked writes ([911a835](../../commit/911a835))
- Zarr compression level benchmarks L1/L3/L5/L9 ([da2d9e](../../commit/da2d9e))

### Documentation

- Sphinx theme updates with semantic styling ([6b4285f](../../commit/6b4285f), [995eb1f](../../commit/995eb1f))
- Better admonitions and table styling
- Updated array types, user-guide for new ScanImage arrays and metadata

## Module Reorganization

### Renamed/Moved

| Old Path | New Path |
|----------|----------|
| `mbo_utilities/graphics/` | `mbo_utilities/gui/` |
| `mbo_utilities/metadata.py` | `mbo_utilities/metadata/` (package) |
| `mbo_utilities/install_checker.py` | `mbo_utilities/install.py` |
| `mbo_utilities/phasecorr.py` | `mbo_utilities/analysis/phasecorr.py` |
| `mbo_utilities/metrics.py` | `mbo_utilities/analysis/metrics.py` |

### Removed

Consolidated or unused modules ([6b47097](../../commit/6b47097), [099a62b](../../commit/099a62b), [155cca9](../../commit/155cca9)):

- `mbo_utilities/array_types.py` - use `arrays/` directly
- `mbo_utilities/lazy_array.py` - use `reader`/`writer` directly
- `mbo_utilities/widgets.py` - use `gui.simple_selector`
- `mbo_utilities/_installation.py` - merged into `install.py`
- `mbo_utilities/_benchmark.py` - replaced by `benchmarks.py`
- `mbo_utilities/plot_util.py` - unused
- `mbo_utilities/formats/` - consolidated into `arrays/`
- `mbo_utilities/__main__.py` - unused

## Breaking Changes

- Import paths changed for moved modules (backwards compat shims removed)
- `graphics` module renamed to `gui`
- Direct imports from `array_types` and `lazy_array` no longer work

## Migration Guide

```python
# Old imports
from mbo_utilities.graphics import run_gui
from mbo_utilities.array_types import MboRawArray
from mbo_utilities.lazy_array import imread
from mbo_utilities.install_checker import check_installation

# New imports
from mbo_utilities.gui import run_gui
from mbo_utilities.arrays import MboRawArray
from mbo_utilities.reader import imread  # or: from mbo_utilities import imread
from mbo_utilities.install import check_installation
```

## Full Commit History

49 commits since v2.4.1. See [compare view](../../compare/v2.4.1...HEAD) for complete diff.

## TO-DO

### Pipeline Registry

Pipeline registry needs to be more carefully implemented.

There need to be clearly defined steps. Pre-processing, registration, segmentation, extraction (of traces). These steps are specifically for functional imaging. We may want different steps in the future, e.g. for brain clearing.

Additionally, we should allow configuration of pre-processing steps.

- Assemble, or to be more programmatically specific, make contiguous
  - de-interleave data: `zT,Y,X` -> `Z,T,Y,X LBM-Only)
  - Blend cameras: `cam, Z, X, Y` -> `Z, X, Y`
- Scan-phase correction (All scanimage / raster scanning)
- Scan-phase correction (All scanimage / raster scanning)
- Registration (All data)
- Segmentation (Functional Imaging Only)
