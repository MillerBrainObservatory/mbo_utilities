---
orphan: true
---

# Dim/metadata contract consolidation — change ledger

Goal: one place (`LazyArray`) defines the dims/metadata/dimension_specs
contract so a new array class implements only `_shape5d`, `__getitem__`,
`dtype`, `can_open`, and populates `self._metadata`. Natural-rank classes
(BinArray 3D, MP4Array 3D, _ChannelView 4D) opt out by overriding
`ndim`/`shape`. Suite2p input reader (BinArray) stays 3D.

## Removed

Classes / files:
- `DimLabels` (class) — `arrays/features/_dim_labels.py` (was event-driven, unsubscribed)
- `DimLabelsMixin` (class) + whole file `arrays/features/_mixin.py`
- `DimensionSpecMixin` (class) — `arrays/features/_dim_spec.py` (folded into `LazyArray`)

Per-class overrides deleted (now inherited from `LazyArray`):
- TiffArray: `shape`, `ndim`, `dims`, `metadata` get/set
- ScanImageArray: `ndim`, `dims`
- ZarrArray: `shape`, `ndim`, `dims`, `_read_ome_dims`, `_init_dim_labels` calls
- Suite2pArray: `shape`, `ndim`, two `dims`, `metadata` get/set, `DimLabels` build
- H5Array: `shape`, `ndim`
- NumpyArray: `shape`, `ndim`
- BinArray: `metadata` get/set
- MP4Array: `metadata` get/set
- _ChannelView: `dims`
- IsoviewArray: `ndim`, `dims`
- tiff.py: 6x `self._dim_labels = None`

Other:
- `DimensionSpec.to_ome_axis` body (delegates to `_dim_tags.dim_to_ome_axis`)
- `get_dims` `_dim_labels` branch (dead)
- exports of `DimLabels`, `DimLabelsMixin`, `DimensionSpecMixin` from `features/__init__`

## Added

`LazyArray` (base) now provides the whole contract:
- `dims` get/set (declared-or-canonical-by-rank; setter invalidates specs)
- `metadata` get/set (dict-backed; setter routes a `"dims"` key)
- `dimension_specs`, `invalidate_dimension_specs`
- `spatial_dims`, `iteratable_dims`, `batch_dims`, `slider_dims`
- `dim_index`, `has_dim`
- module const `_DEFAULT_DIMS_BY_NDIM`

Tests: `tests/test_lazyarray_contract.py` (8 tests).

New-class contract: implement `can_open`, `_shape5d`, `__getitem__`, `dtype`,
set `self._metadata`. Natural-rank classes also override `ndim`/`shape`.

## Behavior held

- BinArray stays 3D `(T,Y,X)` (suite2p input contract).
- slider_dims ↔ shape/ndim lockstep across T/C/Z singleton cases.
- shape ↔ __getitem__ contract.
- wrapper __array__/astype stay explicit (_ChannelView).

## Regression found + fixed during pipeline verification

The base `metadata` setter initially routed a `"dims"` key onto `self.dims`,
which raised when a single-channel write left a 4-element `dims` in a 5D
array's metadata (`writer.py` re-assigns metadata on the `.bin` hop). Fixed:
the base setter now just stores the dict; only `NumpyArray` routes dims.
Verified end-to-end on real LBM data (raw SI tiff -> zarr+register_z -> axial
-> tiff -> bin/ops): dims stays TCZYX and dx/dy/fs/pixel_resolution/num_planes
carry through every hop; `plane_shifts` computed by register_z, applied by the
axial view, and correctly hidden in the baked tiff. Guard:
`tests/local/test_metadata_carrythrough.py`.
