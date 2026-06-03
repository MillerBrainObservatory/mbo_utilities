# Zarr I/O & metadata audit

Map of every function involved in zarr reading, writing, chunk/shard layout,
compression codecs, and OME-NGFF metadata across `mbo_utilities`.
Generated 2026-06-03.

`Called by` legend: `path:line` = concrete caller; `public API` = exported /
called by external code or the package surface; `tests only` = exercised solely
by `tests/`; `CLI script entry` = run as a standalone script.

Note on `arrays/zarr.py`: despite the module name it holds **both** the reader
(`ZarrArray`) and a writer (`merge_zarr_zplanes`). There is no
`create_multiscale_pyramid` function anywhere — earlier notes that referenced it
were describing `merge_zarr_zplanes`.

## 1. Entry points & dispatch

| Path | Function | What it does | Called by |
|------|----------|--------------|-----------|
| mbo_utilities/writer.py | `imwrite(lazy_array, outpath, ext, **kwargs)` | Top-level write router; sends `.zarr` to the array's `_imwrite_base` or to `_try_generic_writers`, passing `sharded`/`ome`/`level`/`pyramid*` kwargs through. | public API; gui/_save_as.py:103, gui/tasks.py:284, cli.py:657 |
| mbo_utilities/reader.py | `imread(path, ...)` | Reader dispatch; routes `.zarr` paths to `ZarrArray` and isoview trees to `IsoviewArray`. | public API |
| mbo_utilities/cli.py | `convert(input_path, output_path, ext, ..., ome)` | CLI format-conversion command; `imread` then `imwrite`, adding `ome=` when `ext==.zarr`. | CLI script entry (`mbo convert ... -e .zarr --ome`) |
| mbo_utilities/arrays/_base.py | `_imwrite_base(self, outpath, ..., **kwargs)` | Array-base write method; delegates `.zarr` to `_write_volumetric_zarr`. | array `_imwrite()` methods via `imwrite` |
| mbo_utilities/arrays/h5.py | `H5Array._imwrite(outpath, ext, **kwargs)` | H5 array write hook; for `.zarr` defers to `_imwrite_base`. | `imwrite(h5_array, ...)` |
| mbo_utilities/arrays/suite2p.py | `Suite2pArray._imwrite(outpath, ext, **kwargs)` | Suite2p array write hook; defers to `_imwrite_base`. | `imwrite(suite2p_array, ...)` |
| mbo_utilities/arrays/zarr.py | `ZarrArray._imwrite(outpath, **kwargs)` (`:490`) | Reader's own write hook; re-encodes a loaded zarr via the base writer. | `imwrite(zarr_array, ...)` |

## 2. Writers

| Path | Function | What it does | Called by |
|------|----------|--------------|-----------|
| mbo_utilities/_writers.py | `_write_volumetric_zarr(data, path, ..., sharded, compressor, compression_level, pyramid, pyramid_max_layers, pyramid_method)` (`:1232`) | Writes TZYX/TCZYX as OME-NGFF v0.5 zarr v3; per-plane inner chunks `(1,1,1,ny,nx)`, T-major shards, builds multiscales, optional webknossos anisotropic pyramid (median/mode). | arrays/_base.py:402 (`_imwrite_base`); benchmarks.py |
| mbo_utilities/_writers.py | `_write_zarr(path, data, ..., **kwargs)` (`:1805`) | Streamed per-plane zarr writer; sets up ShardingCodec, optionally opens an OME group and writes metadata via `_build_ome_metadata`. | internal writer dispatch (`_writers.py:380`); benchmarks.py:1375 |
| mbo_utilities/_writers.py | `_try_generic_writers(data, outpath, ...)` (`:1953`) | Fallback writer for plain arrays; for `.zarr` creates a v3 array with per-plane inner chunk + sharding codec. | writer.py:375 (`imwrite` fallback) |
| mbo_utilities/_writers.py | `_build_inner_codecs(name, level, shuffle)` (nested in `_write_volumetric_zarr`) | Builds the inner codec chain (BytesCodec + gzip/zstd/blosc) used inside a shard. | `_write_volumetric_zarr` (level 0 + each pyramid level) |
| mbo_utilities/arrays/zarr.py | `merge_zarr_zplanes(zarr_paths, output_path, ..., suite2p_dirs, compression_level)` (`:528`) | Stacks single-plane `(T,Y,X)` zarrs into one OME-NGFF `(T,Z,Y,X)` volume (single resolution); writes OME metadata via `_build_ome_metadata`, optional Suite2p labels. | public API; tests/test_zarr_chunking.py:421 |
| mbo_utilities/arrays/suite2p.py | `_add_suite2p_labels(root_group, suite2p_dirs, T, Z, Y, X, dtype, compression_level)` (`:696`) | Writes an OME-Zarr `labels/` subgroup of Suite2p ROI masks with coordinate transforms. | arrays/zarr.py (`merge_zarr_zplanes`, ~`:759`) |
| mbo_utilities/arrays/features/_compression.py | `CompressionFeature.to_zarr_codecs()` (`:175`) | Maps compression settings to a zarr v3 codec list (`[BytesCodec()]` or `[BytesCodec(), GzipCodec(level)]`). | public API |

## 3. Readers

| Path | Function | What it does | Called by |
|------|----------|--------------|-----------|
| mbo_utilities/arrays/zarr.py | `ZarrArray.__init__(filenames, compressor, rois, dims)` (`:97`) | Opens zarr store(s) read-only, detects OME-Zarr group vs bare array, grabs the `"0"` array, reads attrs + OME dims. | reader.py:400/405/414; tests |
| mbo_utilities/arrays/zarr.py | `ZarrArray._read_ome_dims()` (`:165`) | Reads dimension labels from `ome.multiscales[0].axes`, pads to canonical 5D. | arrays/zarr.py:157 (`__init__`) |
| mbo_utilities/arrays/zarr.py | `ZarrArray.__getitem__(key)` (`:389`) | Indexes the level-0 zarr array (returns full-res data; pyramid levels not exposed). | indexing / readers / fpl |
| mbo_utilities/arrays/zarr.py | `ZarrArray.metadata` (property) | Returns the merged zarr/group attrs dict plus derived shape/dtype. | reader consumers |
| mbo_utilities/arrays/isoview/array.py | `LazyVolume.__init__` / `_init` (`:111`) | Opens one isoview volume (zarr `LocalStore` group `"0"`, or tiff/klb/stack) and probes shape/dtype lazily. | IsoviewArray scanners (array.py:1355/1527/1600/1855/1871) |
| mbo_utilities/arrays/isoview/array.py | `IsoviewArray.__init__` (`:1435`) | Lazy 5D `(T,C,Z,Y,X)` reader for corrected/fused/raw/clusterpt trees; per-`(t,c)` reads. | public API; reader.py:216 |
| mbo_utilities/arrays/isoview/array.py | `_ome_block(attrs)` | Extracts the OME block from zarr attrs (handles 0.4 flat vs 0.5 nested). | array.py:641/673 |
| mbo_utilities/arrays/isoview/array.py | `_extract_zarr_scale(attrs)` | Reads `(dz,dy,dx)` scale + translation from OME coordinateTransformations. | array.py:1604 |
| mbo_utilities/arrays/isoview/array.py | `_extract_isoview_attrs(attrs)` | Pulls isoview-specific OME fields (specimen/view/channel). | array.py:1605 |
| mbo_utilities/arrays/isoview/array.py | `isoview_zarr_chunks(shape)` | Resolves the `ISOVIEW_ZARR_CHUNK_ZYX` policy against a real shape. | arrays/isoview/__init__.py:36 (exported) |
| mbo_utilities/arrays/isoview/array.py | `ISOVIEW_ZARR_CHUNK_ZYX = (1,-1,-1)` (`:40`) | Module constant: one Z-plane per chunk, full Y, full X. | isoview/__init__.py:32/45; consolidate.py |
| mbo_utilities/file_io.py | `HAS_ZARR` (constant) | Flags whether `zarr` is importable. | arrays/zarr.py:20/642 |

## 4. isoview consolidation (`arrays/isoview/consolidate.py`)

All rows are `mbo_utilities/arrays/isoview/consolidate.py`.

| Function | What it does | Called by |
|----------|--------------|-----------|
| `consolidate_isoview(src, out, kind, overwrite, pyramid, pyramid_max_layers, compressor, compression_level, progress_callback)` | Public entry; detects kind and dispatches to corrected/fused consolidator. | public API; gui/tasks.py:794; arrays/__init__.py:90 |
| `to_bigstitcher(src, dest, compressor, compression_level, overwrite)` | Mirrors the consolidated v3 zarr into a transient zarr v2 (image pyramid only, OME 0.5→0.4, no sharding) for BigStitcher. | public API; arrays/__init__.py:125 |
| `_consolidate_corrected(...)` | Corrected-pipeline driver; writes image/seg/aux/projection pyramids + metadata. | consolidate.py:1582 |
| `_consolidate_fused(...)` | Fused-pipeline driver; writes image/mask/2D-mask/projection pyramids + metadata. | consolidate.py:1589 |
| `_setup_consolidation(iso, out, overwrite, pyramid, pyramid_max_layers)` | Opens the output group, computes pyramid mags, extracts physical scales. | consolidate.py:1308/1439 |
| `_open_output_group(out_path, overwrite)` | Opens/creates the output zarr v3 group (removes existing if overwrite). | consolidate.py:1275 |
| `_compute_anisotropic_mags(voxel_size, shape, max_layers, min_size=64)` | Per-level `(Z,Y,X)` factors via the webknossos algorithm (doubles smallest-physical axis). | consolidate.py:1266; **_writers.py:_write_volumetric_zarr** (lazy import) |
| `_create_sharded_array(group, name, shape, dtype, chunk_shape, shard_shape, compressor, compression_level)` | Creates a v3 array with per-plane inner chunks inside whole-volume shards. | consolidate.py:581/674/772/869/992/1086/1156 |
| `_make_compressors(name, level, itemsize)` | Builds the v3 codec list (gzip/zstd/blosc). | consolidate.py:441 |
| `_v2_compressor(name, level)` | Builds a single numcodecs compressor for the zarr v2 BigStitcher mirror. | consolidate.py:1682 |
| `_v2_attrs(attrs)` | Recursively rewrites OME `version` `0.5`→`0.4` for the v2 mirror. | consolidate.py:400/402/1680 |
| `_multiscales_block(name, axes, dataset_paths, scales, method="median")` | Assembles one OME-NGFF multiscales entry (datasets + downsample type). | consolidate.py:711/799/918/1002/1183/1394/1515 |
| `_omero_block(channel_names, cam_metadata, default_z)` | Builds the per-channel OMERO block (labels/colors/windows). | consolidate.py:1396/1517 |
| `_scale_5d(dt_s, dz, dy, dx, mag_zyx)` | Builds a `(T,C,Z,Y,X)` scale vector for one pyramid level. | consolidate.py:586/679/875/1002 |
| `_json_safe(v)` | Coerces numpy scalars/arrays to JSON for zarr attrs. | consolidate.py:524 |
| `_write_image_pyramid(group, iso_arr, mags, dx, dy, dz, dt_s, compressor, compression_level, progress_callback)` | Writes `/0../N` intensity pyramid; level 0 from IsoviewArray, deeper levels median-downsampled on disk. | consolidate.py:1319/1449 |
| `_write_segmentation_pyramid(labels_group, scanner, iso_arr, mags, ...)` | Writes `/labels/background_mask` pyramid; level 0 from companions, deeper via mode. | consolidate.py:1331/1461 |
| `_write_aux_2d_per_tc(root, name, scanner, iso_arr, ...)` | Writes a 2D auxiliary mask group `(T,C,1,d0,d1)` with multiscales. | consolidate.py:1338/1342/1473/1477/1484/1491 |
| `_write_disk_xy_projections(root, projections, iso_arr, tm_int_by_index, mags_yx, ...)` | Reads precomputed raw XY projection TIFFs into `/raw/projections/max_xy` pyramid. | consolidate.py:1377 |
| `_write_computed_projections(root, iso_arr, image_mags, ...)` | Computes max projections from the volume and writes `/projections/max_{xy,xz,yz}` pyramids. | consolidate.py:1348/1501 |
| `_write_min_intensity(root, scanner, iso_arr, ...)` | Writes `/min_intensity/0` `(T,C,2)` float32 from npz companions. | consolidate.py:1383 |
| `_write_metadata(root, iso_arr)` | Inlines parsed XML + isoview_config.json into `/metadata` attrs. | consolidate.py:1387/1507 |
| `_write_backgrounds(root, raw_root)` | Writes `/raw/background` `(1,C,1,Y,X)` from `Background_*.tif`. | consolidate.py:1388/1509 |
| `_write_raw_xml(root, raw_root)` | Stores each raw `*.xml` as a 1D uint8 array under `/raw/metadata`. | consolidate.py:1389/1510 |
| `_read_companion(path)` | Reads one companion (zarr/tif/klb) as numpy. | consolidate.py:687/754/782 |

## 5. Metadata & pyramid builders

| Path | Function | What it does | Called by |
|------|----------|--------------|-----------|
| mbo_utilities/metadata/io.py | `_build_ome_metadata(shape, dtype, metadata, dims)` (`:924`) | Builds the full OME-NGFF v0.5 dict (axes, multiscales, coordinateTransformations, omero). | _writers.py:1915 (`_write_zarr`); arrays/zarr.py:762 (`merge_zarr_zplanes`) |
| mbo_utilities/metadata/io.py | `_build_omero_metadata(shape, dtype, metadata)` (`:1157`) | Builds the OMERO channel/window/render block. | metadata/io.py:1019 (`_build_ome_metadata`) |
| mbo_utilities/metadata/output.py | `OutputMetadata.to_ome_ngff(dims)` (`:465`) | Builds OME-NGFF axes + coordinateTransformations from reactive voxel size. | _writers.py:1492 (`_write_volumetric_zarr`) |
| mbo_utilities/metadata/output.py | `OutputMetadata.to_napari_scale(dims)` (`:509`) | Builds a napari scale tuple (finterval + voxel). | gui/run_gui.py:1100 |
| mbo_utilities/metadata/output.py | `OutputMetadata.voxel_size` (`:326`) | Returns adjusted `(dx,dy,dz)` after selections; supplies pyramid voxel input. | _writers.py:916; output.py internal; gui/run_gui.py:1096 |
| mbo_utilities/arrays/features/_pyramid.py | `downsample_block(data, factors, method)` (`:137`) | Block-downsamples by per-axis factors using mean/median/mode/nearest/gaussian/local_mean. | _writers.py:1722; consolidate.py:603/699/906/1036 |
| mbo_utilities/arrays/features/_pyramid.py | `_downsample_mean/_median/_mode/_gaussian/_local_mean(data, factors)` | Per-method reducers (median/mode match webknossos defaults). | `downsample_block` dispatch (`:177/180/183/186/189`) |
| mbo_utilities/arrays/features/_pyramid.py | `compute_pyramid_shapes(base_shape, config)` (`:78`) | Computes fixed-factor level shapes/scales/paths (stops below min_size). | _pyramid.py:352 (`generate_pyramid`); (no longer used by `_write_volumetric_zarr`) |
| mbo_utilities/arrays/features/_pyramid.py | `generate_pyramid(data, config)` (`:327`) | Generator yielding `(level, downsampled)` from full-res data. | public API; no internal callers |
| mbo_utilities/arrays/features/_pyramid.py | `build_multiscales_metadata(levels, base_scale, axes, name, downsample_type)` (`:364`) | Builds OME-NGFF multiscales metadata for a set of levels. | public API; no internal callers |
| mbo_utilities/arrays/features/_pyramid.py | `build_napari_scale_attrs(levels, base_scale)` (`:421`) | Builds per-level napari scale attrs. | public API; no internal callers |

## 6. Scripts (`scripts/`)

| Path | Function | What it does | Called by |
|------|----------|--------------|-----------|
| scripts/backfill_bdv_pyramid.py | `<module main>` | Adds pyramid levels to a single-level BDV `dataset.ome.zarr` and rewrites multiscales. | CLI script entry |
| scripts/reencode_bdv_zarr.py | `<module main>` | Re-encodes BDV zarrs (big-endian/zstd → little-endian/gzip) for BigStitcher, swapping atomically. | CLI script entry |
| scripts/rechunk_isoview.py | `rechunk_file(src_path, dry_run, keep_backup)` | Rewrites isoview zarr arrays to the per-plane Y×X chunk layout with a sharding codec, round-trip verified. | scripts/rechunk_isoview.py:372 (main) |
| scripts/check_zarr_chunks.py | `<module main>` | Walks a zarr store and prints each array's shape/chunks/dtype. | CLI script entry |
| scripts/fix_bdv_multiscales_scale.py | `<module main>` | Fixes multiscales `scale` axis order in BDV `.zattrs`. | CLI script entry |

## Files checked with no zarr/OME functions

- `mbo_utilities/writer.py` — router only (delegates; no direct zarr calls).
- `mbo_utilities/file_io.py` — only the `HAS_ZARR` availability flag.
- `mbo_utilities/gui/_metadata.py` — metadata display UI, no I/O.
- `mbo_utilities/metadata/__init__.py`, `mbo_utilities/arrays/features/__init__.py` — re-exports only.
