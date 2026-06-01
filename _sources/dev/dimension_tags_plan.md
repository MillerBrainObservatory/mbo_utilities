# Dimension Tags System Refactor Plan

## Overview

Create an extensible dimension tag system for filename generation that supports:
- Volumetric (TZYX) vs planar (TYX) output modes
- ImageJ-compatible hyperstack TIFFs
- Descriptive filenames with dimension ranges
- Future extensibility for Camera, Beamlet, View, etc.

## Filename Convention

```
{tag1}_{tag2}_{...}_stack.tif
```

**Tag format:** `{label}{start}[-{stop}[-{step}]]`

**Examples:**
- `zplane01-14_tp00001-10000_stack.tif` → TZYX (14 planes, 10000 timepoints)
- `zplane01-14_tp00001_stack.tif` → ZYX (14 planes, single timepoint)
- `zplane01_tp00001-10000_stack.tif` → TYX (single plane, 10000 timepoints)
- `zplane01-14-2_tp00001-10000_stack.tif` → TZYX with step=2 (planes 1,3,5,7,9,11,13)

**Future examples:**
- `cam0_zplane01-14_tp00001-10000_stack.tif` → Camera 0, volumetric
- `beamlet01-04_zplane01_tp00001-10000_stack.tif` → Pollen calibration

## Architecture

### 1. DimensionTag Class

Location: `mbo_utilities/arrays/features/_dim_tags.py`

```python
@dataclass
class DimensionTag:
    """single dimension tag for filename generation."""
    label: str           # e.g., "zplane", "tp", "cam", "beamlet"
    dim_char: str        # single char: Z, T, C, V, B, etc.
    start: int           # 1-based start index
    stop: int | None     # 1-based stop index (None = single value)
    step: int = 1        # step size (default 1)
    zero_pad: int = 2    # zero-padding width for indices

    def to_string(self) -> str:
        """format as filename component."""
        ...

    def to_slice(self) -> slice:
        """convert to array slice (0-based)."""
        ...

    @classmethod
    def from_string(cls, s: str) -> "DimensionTag":
        """parse from filename component."""
        ...
```

### 2. DimensionTagRegistry

Location: `mbo_utilities/arrays/features/_dim_tags.py`

```python
class DimensionTagRegistry:
    """registry of known dimension tag types."""

    # built-in tags
    TAGS = {
        "T": TagDefinition(label="tp", description="timepoint", zero_pad=5),
        "Z": TagDefinition(label="zplane", description="z-plane", zero_pad=2),
        "C": TagDefinition(label="ch", description="channel", zero_pad=2),
        "V": TagDefinition(label="view", description="view", zero_pad=2),
        "R": TagDefinition(label="roi", description="region of interest", zero_pad=2),
        # future
        "B": TagDefinition(label="beamlet", description="beamlet", zero_pad=2),
        "A": TagDefinition(label="cam", description="camera", zero_pad=1),
    }

    def register(self, dim_char: str, definition: TagDefinition):
        """register custom tag type."""
        ...

    def get_label(self, dim_char: str) -> str:
        """get filename label for dimension."""
        ...
```

### 3. OutputFilename Builder

Location: `mbo_utilities/arrays/features/_dim_tags.py`

```python
class OutputFilename:
    """builds output filename from dimension tags."""

    def __init__(self, tags: list[DimensionTag], suffix: str = "stack"):
        self.tags = tags
        self.suffix = suffix

    def build(self, ext: str = ".tif") -> str:
        """build filename string."""
        parts = [tag.to_string() for tag in self.tags]
        return "_".join(parts + [self.suffix]) + ext

    @classmethod
    def from_array(
        cls,
        arr,
        planes: slice | list | None = None,
        frames: slice | list | None = None,
        suffix: str = "stack",
    ) -> "OutputFilename":
        """create from array and selection."""
        ...
```

### 4. Integration with imwrite

Location: `mbo_utilities/_writers.py`

Modify `_write_tiff()` to:
1. Accept `volumetric: bool = False` parameter
2. When `volumetric=True`:
   - Write all Z planes to single file as hyperstack
   - Use proper ImageJ TZCYX ordering
   - Generate filename with dimension tags
3. When `volumetric=False` (default, current behavior):
   - Write separate file per plane (backward compatible)

### 5. Array _imwrite() Method Updates

Each array class's `_imwrite()` method gains:
- `volumetric: bool = False` - write as single hyperstack
- `filename_template: str | None = None` - override auto-generated name

## Implementation Steps

### Phase 1: Core Tag System

1. Create `_dim_tags.py` with:
   - `TagDefinition` dataclass
   - `DimensionTag` dataclass
   - `DimensionTagRegistry` singleton
   - `OutputFilename` builder

2. Add unit tests for tag parsing and generation

### Phase 2: TIFF Hyperstack Support

1. Update `_write_tiff()` in `_writers.py`:
   - Add `volumetric` parameter
   - Implement TZCYX hyperstack writing
   - Proper ImageJ metadata for hyperstacks

2. Update `_build_imagej_metadata()`:
   - Handle multi-plane hyperstacks
   - Set correct frames/slices/channels

### Phase 3: Array Integration

1. Update `_imwrite_base()` in `arrays/_base.py`:
   - Accept `volumetric` parameter
   - Use `OutputFilename` for path generation
   - Route to appropriate writer mode

2. Update individual array `_imwrite()` methods:
   - `TiffArray`
   - `ScanImageArray` and subclasses
   - `ZarrArray`, `H5Array`, etc.

### Phase 4: API and Documentation

1. Update `imwrite()` in `writer.py`:
   - Expose `volumetric` parameter
   - Document new filename conventions

2. Update documentation:
   - New filename format examples
   - Hyperstack usage guide

## File Changes Summary

| File | Changes |
|------|---------|
| `arrays/features/_dim_tags.py` | **NEW** - Tag system implementation |
| `arrays/features/__init__.py` | Export new classes |
| `_writers.py` | Add volumetric tiff writing |
| `arrays/_base.py` | Update `_build_output_path()`, `_imwrite_base()` |
| `writer.py` | Add volumetric parameter to `imwrite()` |

## Backward Compatibility

- Default behavior unchanged (`volumetric=False`)
- Existing per-plane output still works
- New filenames only when explicitly requested
- Existing filename patterns still parsed correctly

## Example Usage

```python
import mbo_utilities as mbo

arr = mbo.imread("data.tif")  # shape: (10000, 14, 512, 512) TZYX

# current behavior (per-plane files)
mbo.imwrite(arr, "output/", ext=".tiff")
# creates: output/plane01.tiff, output/plane02.tiff, ...

# new volumetric mode
mbo.imwrite(arr, "output/", ext=".tiff", volumetric=True)
# creates: output/zplane01-14_tp00001-10000_stack.tif

# with plane selection
mbo.imwrite(arr, "output/", ext=".tiff", volumetric=True, planes=[1, 2, 4])
# creates: output/zplane01-02-04_tp00001-10000_stack.tif

# single timepoint
mbo.imwrite(arr[0:1], "output/", ext=".tiff", volumetric=True)
# creates: output/zplane01-14_tp00001_stack.tif
```

## Open Questions

1. Should `_stack` suffix be configurable or always present for volumetric?
2. How to handle ROI dimension in filename? `roi01-04_zplane01-14_stack.tif`?
3. Should non-tiff formats (zarr, h5) also use this naming convention?
