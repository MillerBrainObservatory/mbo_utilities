# IsoView readers

A reference for `mbo_utilities/arrays/isoview.py`. This module reads the four
output trees produced by the Keller-lab IsoView lightsheet pipeline:

| Kind         | Layout                                                                                                  | View key             | Channel labels        |
| ------------ | ------------------------------------------------------------------------------------------------------- | -------------------- | --------------------- |
| `corrected`  | `<root>.corrected/SPM##/TM######/SPM##_TM######_CM##.<ext>`                                              | `cam` (int)          | `CM00`, `CM01`, …     |
| `fused`      | `<root>.corrected/Results/MultiFused_<method>/SPM##/TM######/SPM##_TM######_CM##_CM##_VW##(.fusedStack)?.<ext>` | `(cam0, cam1, vw)`   | `VW00_fused`, …       |
| `raw`        | `<root>/SPC##_TM#####_ANG###_CM#_CHN##_PH#.stack`                                                       | `(cam, chn)`         | `CM0_CHN00`, …        |
| `clusterpt`  | `<root>/TM######/SPM##_TM######_CM##_CHN##.klb`                                                         | `(cam, chn)`         | `CM00_CHN00`, …       |

All four kinds expose the same `(T, C, Z, Y, X)` shape, the same `__getitem__`
contract, and the same `_imwrite` path. The only per-kind variation lives in
a single `_KINDS` registry; there are no subclasses.

## Top-level entry points

| Symbol                                  | Purpose                                                                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| [`IsoviewArray(path, kind=None)`](#class-isoviewarray) | The reader. Auto-detects `kind` from the path unless one is supplied. |
| [`detect_isoview_kind(path)`](#detect_isoview_kind) | Walks a path (file or dir) and returns `"corrected"`, `"fused"`, `"raw"`, `"clusterpt"`, or `None`. |
| [`isoview_to_ome_zarr(src, out, kind=None, …)`](#isoview_to_ome_zarr) | Convenience wrapper: open an IsoView tree and write it to a single OME-Zarr v0.5 group. |

`imread(path)` routes any path inside an isoview tree through `IsoviewArray`
before any other file-extension dispatch, so opening a single `.zarr` inside
`<root>.corrected/SPM00/TM000000/` and opening the `SPM00/` folder produce
the identical `IsoviewArray(kind="corrected")`.

## Class: `IsoviewArray`

Lazy 5D reader. Built on top of `Shape5DMixin`.

### Construction

```python
IsoviewArray(path)                  # auto-detect
IsoviewArray(path, kind="fused")    # force a specific reader
```

Construction steps (in order):

1. Resolve `kind` via `detect_isoview_kind` if not supplied.
2. Resolve `scan_root` via `_KINDS[kind]["resolve"]` (one of the `_resolve_*`
   helpers below). For `corrected`, this walks ancestors until it finds the
   `.corrected/SPM##` directory.
3. If the kind needs `.stack` dimensions (raw only), call `_probe_raw_xml` to
   read the XML sidecar **before** trying to probe the binary's shape.
4. Run the per-kind scanner (`_scan_corrected` / `_scan_fused` /
   `_scan_raw` / `_scan_klb_tm`). Each returns `(tp_paths, view_keys, channel_names)`
   where `tp_paths[t][view_key] = Path`.
5. Probe the first volume via `LazyVolume` to capture `dtype`, `(nz, ny, nx)`,
   and any embedded OME-Zarr / ImageJ scale metadata.

### Properties

| Property              | Returns                                                                          |
| --------------------- | -------------------------------------------------------------------------------- |
| `shape`               | `(T, C, Z, Y, X)`.                                                              |
| `shape5d` (mixin)     | Same as `shape`.                                                                |
| `dims`                | `("T", "C", "Z", "Y", "X")`.                                                    |
| `ndim`                | `5`.                                                                            |
| `dtype`               | numpy dtype of the on-disk volumes.                                             |
| `size`                | total element count (`prod(shape)`).                                            |
| `kind`                | `"corrected"`, `"fused"`, `"raw"`, or `"clusterpt"`.                            |
| `stack_type`          | `"isoview-<kind>"`. Used by the GUI's projections widget.                       |
| `views`               | List of view keys (cam int for corrected, tuples for fused/raw/clusterpt).      |
| `channel_names`       | Human-readable labels per channel.                                              |
| `num_timepoints`      | `T` size.                                                                       |
| `num_planes`          | `Z` size.                                                                       |
| `num_views`           | `C` size (alias).                                                               |
| `num_color_channels`  | `C` size (alias).                                                               |
| `filenames`           | All on-disk volume paths backing the array.                                     |
| `metadata`            | Dict combining shape, dtype, scale (dx/dy/dz/fs), XML sidecar fields, etc.      |
| `source_path` (mixin) | Canonical path used by `imread` to reconstruct the array.                       |

### Methods

| Method                            | Behavior                                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__getitem__(key)`                | NumPy-style indexing into `(T, C, Z, Y, X)`. Single-`(t, c)` reads take the narrow-zarr fast path; multi-`(t, c)` reads use a cached full volume. |
| `__array__(dtype=None)`           | Full materialization. Used by `np.asarray(arr)`.                                                                                                  |
| `astype(dtype, …)`                | Materialize-then-cast.                                                                                                                            |
| `__len__()`                       | `T` size.                                                                                                                                         |
| `projections()`                   | Returns the projection TIFFs for *this* stack, or `None`. Format: `{"axes", "views", "files"}`. See [projections](#projection-helpers).           |
| `close()`                         | Clear the volume cache.                                                                                                                           |
| `_imwrite(outpath, **kwargs)`     | Write to disk via the shared `_imwrite_base` writer. Honors `planes`, `frames`, `channels`, `ext`, sharding, pyramid, etc.                        |
| `_probe_shape(sample_path)`       | Internal. Reads first volume to fill `(nz, ny, nx)` and dtype. For raw `.stack`, finalizes depth from file size.                                  |
| `_probe_raw_xml()`                | Internal, raw kind only. Pulls `(W, H)` and acquisition metadata from the XML sidecar so `LazyVolume` can map the binary.                         |
| `_read_volume(path, t, c)`        | Internal. Cached `arr[:]` read.                                                                                                                   |
| `_read_slab(path, t, c, z, y, x)` | Internal. Uncached narrow read — used when only one `(t, c)` is hit per call.                                                                     |

### The `_KINDS` registry

`_KINDS[kind]` is a dict with:

| Key               | Type           | Role                                                                                                |
| ----------------- | -------------- | --------------------------------------------------------------------------------------------------- |
| `stack_type`      | `str`          | Public label (`"isoview-corrected"` etc.) surfaced via `metadata["stack_type"]`.                    |
| `resolve`         | `Path -> Path` | Maps a user-supplied path to the directory the scanner should walk.                                 |
| `scan`            | `Path -> (tp_paths, view_keys, channel_names)` | The discovery function for this kind.                                     |
| `projections`     | `IsoviewArray -> dict \| None` | Locates the projection TIFFs for this stack, or returns `None`.                       |
| `needs_raw_dims`  | `bool`         | Set on `raw` only; tells `__init__` to read the XML sidecar before probing shape.                   |

To add a fifth kind you'd append one entry here plus its scanner / resolver — no
class changes required.

## Per-kind helpers

### `detect_isoview_kind(path) -> str | None`

Decision order (most specific first):

1. If any ancestor matches `Results/MultiFused_*/SPM##/…` → `"fused"`.
2. If any ancestor matches `.corrected/SPM##/…` (and isn't under `Results/`) → `"corrected"`.
3. If the directory contains raw `.stack` files → `"raw"`.
4. If `TM######/` subfolders contain `.klb` files → `"clusterpt"`.
5. Otherwise → `None`.

A file path is normalized to its parent directory before testing.

### Path resolvers

| Helper                                  | Maps                                                                                  | Returns                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `_resolve_corrected_spm_dir(p)`         | the `.corrected/` root, an SPM## descendant, a TM###### descendant, or any ancestor   | the `SPM##` directory under `.corrected/`                          |
| `_resolve_fused_method_dir(p)`          | the `.corrected/Results/MultiFused_<method>/` tree (or any ancestor)                  | the `MultiFused_<method>` directory                                |
| `_first_method_dir(results_dir)`        | `Results/` → first `MultiFused_*` dir (preferring `MultiFused_geometric`)             |                                                                    |

### Volume scanners (one per kind)

| Scanner                       | Returns                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `_scan_corrected(spm_dir)`    | `tp_paths[t][cam] = Path`, view_keys = sorted cam ints, channel_names = `["CM00", "CM01", …]`.        |
| `_scan_fused(method_dir)`     | `tp_paths[t][(cam0, cam1, vw)] = Path`, channel_names = `["VW00_fused", …]`.                          |
| `_scan_raw(base)`             | `tp_paths[t][(cam, chn)] = Path`, channel_names = `["CM0_CHN00", …]`.                                 |
| `_scan_klb_tm(base)`          | `tp_paths[t][(cam, chn)] = Path`, channel_names = `["CM00_CHN00", …]`.                                |

All four scanners return `({}, [], [])` when no volumes are found.

### Projection helpers

| Helper                            | Behavior                                                                                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `_scan_flat_projections(dir)`     | Walks a flat `<dataset>.{raw,corrected}.projections/` directory. Matches `SPM##_TM##_CM##.{xy,xz,yz}Projection.tif`. Returns `{"axes", "views", "files"}` or `None`.            |
| `_scan_fused_projections(method)` | Walks `MultiFused_*/SPM##/TM######/` for `*Projection.tif` files. Handles both `CM##_CM##_VW##` and bare `VW##` naming.                                                       |
| `_corrected_projections(arr)`     | Looks for `<root>.corrected.projections/` next to the array's `.corrected` directory.                                                                                          |
| `_fused_projections(arr)`         | Delegates to `_scan_fused_projections(arr.scan_root)`.                                                                                                                         |
| `_raw_projections(arr)`           | Looks for `<root>.raw.projections/` next to the raw acquisition directory.                                                                                                     |
| `_finalize_projections(axes, views, files)` | Normalizes a scan result: orders axes as `[xy, xz, yz]`, sorts views, drops empty groups (returns `None`).                                                          |

The projections dict format consumed by the GUI widget:

```python
{
    "axes": ["xy", "xz", "yz"],            # subset of these, in this order
    "views": ["CM00", "CM01", ...],        # or ["VW00", "VW90"] for fused
    "files": {(axis, view, t): Path, ...},  # t is the TM integer
}
```

## Metadata extraction

| Helper                         | Source                                                                                                                  | Fields produced                                                                              |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `_parse_isoview_xml(xml_path)` | Acquisition XML sidecar (`ch00_spec00.xml`, etc.). Used by raw `.stack` reader to learn `(W, H, D)`.                    | `dimensions`, `z_step`, `exposure_time`, `detection_objective`, `objective_mag`, `pixel_resolution_um`, `fps`, `vps`, `zplanes`, camera + illumination fields, `specimen_name`, `timestamp`, … |
| `_find_isoview_xml(dir)`       | Returns the first matching XML in a directory (`ch*_spec*.xml`, `TL*_ch*.xml`, fallback `*.xml`).                       | path or `None`.                                                                              |
| `_extract_zarr_scale(attrs)`   | OME-Zarr `multiscales[0].datasets[0].coordinateTransformations[type=scale]`.                                            | `dx`, `dy`, `dz`.                                                                            |
| `_extract_tiff_scale(tif)`     | TIFF tags + ImageJ metadata. Reads ImageJ `spacing` for `dz`, `finterval` or `fps` for `fs`, XResolution/YResolution for `dx`/`dy`. | `dz`, `dy`, `dx`, `fs`.                                                                |

The `IsoviewArray.metadata` property merges:

1. Fields from `_extract_zarr_scale` / `_extract_tiff_scale` (called during `_probe_shape`).
2. Fields from `_parse_isoview_xml` (only on the raw kind).
3. Shape-derived fields: `Ly`, `Lx`, `num_zplanes` / `nplanes` / `num_planes`,
   `num_timepoints` / `nframes` / `num_frames`, `num_color_channels`, `num_views`.
4. Stack-identifying fields: `stack_type`, `pipeline_stage`, `dtype`, `shape`, `views`, `channel_names`.
5. Convenience aliases: `pixel_resolution_um` → `dx`, `dy`; `z_step` → `dz`; `fps` → `fs`.

## Pure utilities

| Helper                            | Purpose                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `_has_tm_pattern(name)`           | `True` if a string contains `TM##### `or `TM######`.                                                          |
| `_extract_timepoint(folder_name)` | Returns the integer timepoint from a `TM######` folder name.                                                  |
| `_find_tm_folders(base_path)`     | Lists `TM######` child directories sorted by timepoint.                                                       |
| `_is_aux(path)`                   | `True` for masks / projections / intensity files that should be skipped during volume scans.                  |
| `_chunks_touched(shape, chunks, key)` | Number of zarr chunks a given index would decompress. Used in debug-level perf logging of narrow reads.   |
| `_to_indices(k, max_val)`         | Normalizes an int/slice/list/None index into a Python list of ints.                                           |
| `_axis_size(k, max_val)`          | Output size along one axis for a given index spec.                                                            |

## `isoview_to_ome_zarr`

Thin wrapper around `IsoviewArray._imwrite(ext=".zarr", ...)`. Use it when you
just want to convert one isoview tree into a single OME-Zarr group.

```python
isoview_to_ome_zarr(
    src,                       # path to any isoview tree (auto-detected)
    out,                       # output directory
    kind=None,                 # optional override: "corrected" | "fused" | "raw" | "clusterpt"
    timepoints=None,           # 1-based selection (forwarded as frames=)
    channels=None,
    planes=None,
    overwrite=False,
    target_chunk_mb=64,
    sharded=True,              # zarr v3 sharding
    compressor="zstd",         # "none" | "gzip" | "zstd" | "blosc-lz4" | "blosc-zstd"
    compression_level=3,
    shuffle=None,
    pyramid=False,             # OME-NGFF multiscale pyramid (Y/X downsample)
    pyramid_max_layers=4,
    pyramid_method="mean",
    output_suffix=None,        # defaults to the stack_type
    progress_callback=None,
    show_progress=True,
    debug=False,
)
```

## End-to-end load paths

All three of these produce the same `IsoviewArray(kind="corrected", shape=(61, 4, 38, 1848, 768))`:

```bash
# CLI
uv run mbo "D:/.../<dataset>.corrected/SPM00"
```

```python
# Open Folder dialog → load_new_data → imread(...)
imread("D:/.../<dataset>.corrected/SPM00")
```

```python
# Open File dialog on a zarr deep inside the tree
imread("D:/.../<dataset>.corrected/SPM00/TM000000/SPM00_TM000000_CM00.zarr")
```

`imread` runs `detect_isoview_kind` before any file-vs-directory branching, so
the choice between Open File, Open Folder, and the CLI never affects the
resulting reader.
