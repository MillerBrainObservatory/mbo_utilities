# array dimension specification

all array types in mbo_utilities use a fixed 5D layout: **TCZYX**

```
T - time (frames, volumes, timepoints)
C - channels (color channels, camera views)
Z - z-planes (depth, beamlets, z-slices)
Y - height (spatial)
X - width (spatial)
```

singleton dimensions are kept, never squeezed. a single-plane grayscale
timeseries has shape `(T, 1, 1, Y, X)`, not `(T, Y, X)`.

## why

the previous system had 3D, 4D, or 5D arrays depending on array type and
runtime state. writers branched on ndim with hardcoded index assumptions.
every branch was a bug surface. every squeeze erased dimension identity.

with always-5D:
- `shape[2]` is always Z. no guessing, no `_get_num_planes()` heuristics.
- writers have one code path. no ndim checks, no squeeze, no np.newaxis.
- `dims` is a constant, not computed per-instance.
- consistent with OME-NGFF 0.5, bioio/aicsimageio, and napari conventions.

## protocol

```python
DIMS = ("T", "C", "Z", "Y", "X")

class ArrayProtocol:
    shape: tuple[int, int, int, int, int]  # always length 5
    dims = DIMS                             # constant
    ndim = 5                                # constant
    dtype: np.dtype

    @property
    def nt(self) -> int: return self.shape[0]
    @property
    def nc(self) -> int: return self.shape[1]
    @property
    def nz(self) -> int: return self.shape[2]
    @property
    def ny(self) -> int: return self.shape[3]
    @property
    def nx(self) -> int: return self.shape[4]
```

`__getitem__` follows standard numpy conventions: integer indices squeeze
their axis, slices keep it. the array object is always 5D. numpy arrays
returned from indexing follow normal numpy rules.

## dimension mapping per array type

| array type | T | C | Z | Y | X |
|---|---|---|---|---|---|
| LBMArray | timepoints | color channels (1-2) | beamlets | height | width |
| PiezoArray | volumes | 1 | z-slices | height | width |
| SinglePlaneArray | timepoints | color channels | 1 | height | width |
| TiffArray (volume) | timepoints | color channels | planes | height | width |
| TiffArray (single) | timepoints | 1 | 1 | height | width |
| ZarrArray | timepoints | 1 | planes | height | width |
| Suite2pArray | timepoints | 1 | planes or 1 | height | width |
| BinArray | timepoints | 1 | 1 | height | width |
| IsoviewArray | timepoints | views (cameras) | depth | height | width |
| LBMPiezoArray | 1 | 1 | z-planes | height | width |
| H5Array | varies | varies | varies | height | width |
| NumpyArray | varies | varies | varies | height | width |

semantic differences (volumes vs timepoints, views vs channels, beamlets
vs z-planes) are display labels, not structural differences. structure is
always TCZYX.

## what this replaces

### removed patterns

- **ndim branching in writers**: `if ndim == 5 ... elif ndim == 4 ... elif ndim == 3`
  replaced by one code path that always handles 5D.

- **squeeze operations**: `chunk.squeeze(axis=1)` in `_write_plane`, `_write_bin`,
  `_write_npy`. these assumed axis 1 was Z without checking. removed entirely.

- **np.newaxis insertions**: `chunk[:, np.newaxis, :, :]` to promote 3D to 4D.
  unnecessary when input is always 5D.

- **DEFAULT_DIMS mapping**: `{2: ("Y","X"), 3: ("T","Y","X"), 4: ("T","Z","Y","X"), ...}`.
  replaced by the single constant `DIMS`.

- **DimensionSpecMixin / DimLabels complexity**: computed dims per-instance based on
  ndim and array type. replaced by a constant.

- **descriptive dim labels**: `("timepoints", "z-planes", "Y", "X")`. structure uses
  canonical `DIMS`. gui display labels are a separate concern (e.g. `dim_display_names`
  property for slider widgets).

- **`_ChannelView` wrapper**: wrapped 5D as 4D for suite2p. replaced by direct
  indexing: `arr[:, channel, :, :, :]`.

- **`_get_num_planes()` heuristics** (lbm_suite2p_python): checked `num_planes`,
  then `num_channels`, then `shape[1]`. replaced by `arr.nz` or `arr.shape[2]`.

- **`ndim` vs `len(shape)` inconsistency**: ScanImageArray had ndim returning
  metadata-derived value while shape returned actual shape. with always-5D,
  both are always 5.

### hardcoded index assumptions removed

these positions were assumed by ndim without validation:

| location | assumption | failure mode |
|---|---|---|
| `_write_plane:497` | 5D index as `[t, c, z, y, x]` | TZCYX data |
| `_write_plane:504` | 4D index as `[t, z, y, x]` | TCYX data |
| `_write_plane:528` | `shape[1]==1` is Z | could be C=1 |
| `_write_volumetric_tiff:1211` | 5D input is TCZYX | TZCYX input |
| `_imwrite_base:430` | 5D `shape[2]` is Z | TZCYX |
| `_imwrite_base:434` | 4D `shape[1]` is Z | TCYX |
| `read_chunk:579` | 5D index as `[t, c, z, y, x]` | non-TCZYX order |
| `_get_num_planes:48` | 4D `shape[1]` is Z | TCYX |

with always-5D TCZYX, none of these need conditional logic. Z is always
`shape[2]`, C is always `shape[1]`, T is always `shape[0]`.

## writer behavior

### _write_volumetric_tiff

input is always 5D TCZYX. transpose to TZCYX for ImageJ:

```python
chunk_data = chunk_data.transpose(0, 2, 1, 3, 4)  # always, no ndim check
```

ImageJ TIFF page ordering is TZCYX: `page = t*(Z*C) + z*C + c`.

### _write_plane (bin/h5/npy per-plane writer)

caller extracts the 3D slice before calling:

```python
plane_data = data[:, channel_index, plane_index, :, :]  # always valid
_write_plane(plane_data, ...)  # receives (T, Y, X)
```

no plane_index or channel_index params needed on `_write_plane` itself.

### _write_volumetric_zarr

input is always 5D. store as 5D zarr with OME-NGFF axes metadata.

### ArraySlicing

always has T, C, Z selections. no conditional dimension detection.
`read_chunk` always indexes as `arr[t_sel, c_sel, z_sel, :, :]`.

## lbm_suite2p_python changes

### pipeline()

```python
# before
if arr.ndim == 4 and not hasattr(arr, "num_planes"):
    is_volumetric = True
elif arr.ndim == 3:
    is_volumetric = False

# after
is_volumetric = arr.nz > 1
```

### run_volume()

```python
# before
num_planes = _get_num_planes(input_arr)  # heuristic chain

# after
num_planes = input_arr.nz
```

### run_plane()

```python
# before
write_planes = [plane] if file.ndim >= 4 else None
imwrite(file, ..., planes=write_planes)

# after
plane_data = arr[:, 0, plane_idx, :, :]  # 3D (T, Y, X) for suite2p binary
```

### _compute_projection()

```python
# before
if ndim == 4:
    data = arr[:, plane_idx, :, :]
elif ndim == 3:
    data = arr[:]
else:
    raise ValueError(...)

# after
data = arr[:, 0, plane_idx, :, :]  # always valid
```

## migration

### phase 1: add 5D accessors alongside existing shape

add `nt`, `nc`, `nz`, `ny`, `nx` properties to all array types. add a
`shape5d` property that returns the 5D shape. validate these match the
existing shape for every array type via tests. this breaks nothing.

### phase 2: switch shape to always 5D

change every array's `shape` property to return 5D with singletons.
update `ndim` to always return 5. update `dims` to always return
`("T", "C", "Z", "Y", "X")`. update writers to drop all ndim branching.
update lbm_suite2p_python to use `arr.nz` etc.

### phase 3: delete dead code

remove DimLabels complexity, squeeze operations, DEFAULT_DIMS mapping,
_ChannelView, ndim branching, _get_num_planes heuristics, descriptive
dim label overrides.

## design decisions

**why not xarray?** adds a heavy dependency and serialization complexity
for lazy TIFF readers. the benefit of named dims is fully captured by a
fixed 5D layout.

**why not configurable dimension order?** the whole point is one order.
making it configurable reintroduces the current problem.

**why not keep backwards-compatible 4D shape?** two code paths forever
is worse than a clean break. this is a research library, not a public API.

**why TCZYX and not TZCYX?** TCZYX matches OME-NGFF 0.5 and bioio.
ImageJ uses TZCYX page ordering, but that's a serialization detail
handled by the writer transpose. internal representation follows the
standard.

**what about `__getitem__` returning variable ndim?** this is standard
numpy behavior. `arr[0]` squeezes T, returning 4D `(C, Z, Y, X)`.
`arr[0, 0, 0]` returns 2D `(Y, X)`. the array *object* is always 5D.
numpy arrays from indexing follow normal rules. users expect this.
