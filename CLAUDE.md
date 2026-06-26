# mbo_utilities

- never git push or add yourself to github or any attributions in git related tasks.
- Before proposing changes review the relevant codebase sections first.
- No fluff comments that restate what the code already says.
- UI/CLI text: status + action only, minimum words. Don't explain *why* a feature exists.

## Structure

```
mbo_utilities/
  reader.py            imread()  — path/array -> LazyArray (dispatch)
  writer.py            imwrite() — LazyArray  -> disk (any format)
  lazy_array.py        LazyArray base + plugin registry + _dispatch()
  _writers.py          format write backends + processing-history stamping
  squeeze.py           SqueezedView (drop size-1 T/C/Z for display)

  arrays/              every concrete LazyArray subclass
    _base.py           Shape5DMixin / ReductionMixin / TiffReaderMixin, _index_5d_into_raw
    tiff.py            TiffArray + ScanImageArray family (LBM/Piezo/SinglePlane/...)
    bin.py h5.py zarr.py mp4.py numpy.py suite2p.py
    _channel_view.py   _ChannelView (one channel of a 5D array as 4D)
    _registration.py   axial z-registration + AxialShiftView
    _phasecorr_view.py PhaseCorrectedView (read-time scan-phase for any array)
    isoview/           IsoviewArray (4-kind light-sheet reader) + consolidate
    features/          composable dim/ROI/phase/pyramid/stats machinery

  metadata/            canonical metadata model + ScanImage parsing + OutputMetadata
  analysis/            scanphase.py, phasecorr.py (bidirectional offset detect/correct)
  gui/                 Miller Brain Studio (fastplotlib + imgui-bundle)
  hpc/                 submitit-based SLURM runner (config-driven)
  cli.py               `mbo` command (click)
  gpu.py log.py preferences.py file_io.py env_cache.py _sysmem.py
```

`mbo_utilities/__init__.py` uses `__getattr__` lazy imports so the CLI starts
fast — heavy deps (numpy/tifffile/zarr/torch) load only when first used. Keep
new public symbols on this lazy path; don't add eager top-level imports.

## 5D TCZYX

Every array `imread()` returns is a **`LazyArray`** (`lazy_array.py`) and reports
itself as **always-5D `(T, C, Z, Y, X)`** — time, channel, z-plane, height,
width — with unused axes as singletons. OME-NGFF 0.5 axis order.

- Subclasses implement `_shape5d() -> (T,C,Z,Y,X)`, `__getitem__`, `dtype`,
  `_metadata`, and classmethod `can_open(path)`. The base provides
  `shape`/`ndim`/`nt`/`nc`/`nz`/`ny`/`nx`, dims, metadata, and `dimension_specs`.
- **Reported rank exception:** a few classes override `ndim`/`shape` to natural
  rank because a downstream consumer needs it — **`BinArray` (3D)** and
  **`Suite2p` data.bin readers stay 3D for suite2p `run_plane`**,
  **`MP4Array` (3D)**, **`_ChannelView` (4D)**. Everything else
  (TiffArray, all ScanImage subclasses, ZarrArray, H5Array, NumpyArray,
  **Suite2pArray**, IsoviewArray, AxialShiftView, PhaseCorrectedView) reports 5D.
- **Hard invariant:** never override `_shape5d`/`shape` without making
  `__getitem__` return that exact layout. The reported shape must match what
  indexing yields. (See `_index_5d_into_raw` in `_base.py` for how a 5D key maps
  onto a natural-rank backing array — front-pads missing C/Z/T/Y as singletons.)
- Array wrappers must define `__array__`/`astype` explicitly. `__getattr__`
  delegation silently leaks the *underlying* shape through numpy's protocol.

### imread dispatch (`reader.py` + `lazy_array.py:_dispatch`)

1. ndarray -> `NumpyArray`; already-loaded LazyArray passes through; `SqueezedView`
   normalizes back to its 5D base.
2. **Plugin dispatch** (`_dispatch`): highest-`PRIORITY` registered class whose
   `can_open(path)` returns True wins. Built-ins self-register; third parties add
   formats via the `mbo_utilities.lazy_arrays` entry-point group (see *Extending*).
3. Legacy fall-through in `_imread_impl` handles inputs no class claims:
   multi-file lists, `.bin`/`.klb`, suite2p folders, isoview trees, mixed dirs.
   ScanImage TIFFs try `TIFF_ARRAY_CLASSES` most-specific-first
   (LBM → Piezo → LBMPiezo → SinglePlane → ScanImage → TiffArray).
4. Detection precedence inside a dir: **suite2p** (`ops.npy`/plane dirs) >
   **isoview** (`detect_isoview_kind` parent-walk) > zarr > tiff volume.
5. `imread(path, channel=N)` wraps the result in `_ChannelView` (4D TZYX of one
   channel) — picklable through `reader_kwargs` so subprocess workers re-create it.

### imwrite (`writer.py` -> `arr._imwrite` -> `_writers.py`)

`imwrite(arr, outpath, ext=".tiff", ...)`. Canonical selection params are
**1-based**: `planes`, `timepoints`, `channels` (int / list / range), plus
first-N counts `num_zplanes`, `num_timepoints`. `frames`/`num_frames` are
deprecated aliases (warn, still work — canonical is `timepoints`).

- Formats: `.tiff` (ImageJ hyperstack, BigTIFF >4GB), `.bin` (suite2p + ops.npy),
  `.zarr` (v3, OME-NGFF 0.5; `sharded`/`compressor`/`pyramid` kwargs),
  `.h5` (`dataset_name`, default `"mov"`), `.mp4`, `.npy`.
- ROI handling via `roi_mode` (`RoiMode.concat_y` stitch — default — vs
  `separate` per-ROI files) and `roi` (None=stitch / 0=split-all / int / list).
  **`roi=0`/list is a fan-out in `_imwrite_base`** producing `roiNN/` subdirs;
  it is *not* an imwrite `roi_mode`.
- `register_z=True` computes per-plane rigid shifts (phase correlation, GPU if
  cupy+CUDA) and stores `metadata["plane_shifts"]` — **not applied to pixels**;
  apply non-destructively at read time with `with_axial_shifts(arr)`.
- Every write appends a `processing_history` entry via
  `add_processing_step` (`_writers.py`): step, in/out files, duration, params.
- An object is writable if it has `_imwrite`; otherwise `_try_generic_writers`.
  Pass a raw ndarray with `dim_order="TZYX"` etc. to declare its axes.

## Features subsystem (`arrays/features/`)

Composable machinery the array base and subclasses delegate to. Feature support
is detected by **duck typing** (attribute presence), never `isinstance`.

- **`_dim_labels.py`** — canonical dim labels. `DEFAULT_DIMS` by rank
  (3D=TYX, 4D=TZYX, 5D=TCZYX), `parse_dims`, `get_slider_dims` (non-spatial,
  size>1 dims → fastplotlib sliders), `find_slider_name`.
- **`_dim_spec.py`** — `DimensionSpecs.from_array(arr)` builds a reactive spec
  from dims+shape+metadata (roles SPATIAL/ITERATABLE/BATCH, physical scales
  dx/dy/dz/fs). Cached on the array; `invalidate_dimension_specs()` on dims
  change. Backs `arr.num_timepoints`, `arr.dz`, `arr.fs`, etc.
- **`_slicing.py`** — `ArraySlicing`, `parse_selection` (1-based in, 0-based out),
  `read_chunk` (frame-by-frame; avoids `np.ix_` which breaks laziness),
  `bytes_per_frame`, `iter_chunks` for streaming writes.
- **`_roi.py`** — `RoiMode`, `RoiFeatureMixin` (`roi`, `num_rois`, `roi_mode`,
  `iter_rois`). Duck-typed via `hasattr(arr, "roi_mode")`.
- **`_phase_correction.py`** — `PhaseCorrectionFeature` + `PhaseCorrectionMixin`
  (`fix_phase`, `use_fft`, `phasecorr_method`, `border`, `max_offset`). GUI
  duck-types `hasattr(arr, "phase_correction")`.
- **`_pyramid.py`** — `downsample_block(data, factors, method)` with
  mean/nearest/gaussian/median/mode (median/mode are webknossos-parity for
  intensity/label volumes); `PyramidConfig`, `compute_pyramid_shapes`.
- **`_summary_stats.py`** — `SummaryStatsSpec` drives the GUI Signal-Quality tab:
  classifies dims as image vs scrollable and assigns SERIES/GROUP/REDUCE roles.
  Override `summary_stats_dim_role`/`summary_stats_metrics`/`summary_stats_spec`
  on an array to reclassify (e.g. tiled-T → GROUP instead of averaged).
- **`_segmentation.py`** — Cellpose dense masks ↔ Suite2p sparse `stat` conversion.

**Slider/squeeze invariant:** `get_slider_dims`, the `SqueezedView`/
`_SqueezeSingletonDims` wrapper, and fastplotlib's `ndim-2` slider count must
stay in lockstep. When touching one, account for all T/C/Z singleton
combinations, not just the one in front of you.

## Metadata (`metadata/`)

- **Canonical model** (`base.py`): `METADATA_PARAMS` registry of
  `MetadataParameter`s (canonical name, aliases, dtype, unit, transforms).
  `ALIAS_MAP`/`get_canonical_name` resolve any alias; `get_param(md, name)`
  (`params.py`) looks up canonical + aliases + reciprocal transforms
  (e.g. `fs`↔`finterval`, `dx`↔`XResolution`) + shape fallback.
- **Readers deposit raw keys in `self._metadata`; do not pre-normalize.** Writers
  call `normalize_metadata` / `VoxelSize.to_dict(include_aliases=True)` to emit
  format-specific keys. `arr.dx`/`arr.fs`/`arr.num_zplanes` resolve through the
  registry.
- **ScanImage parsing** (`io.get_metadata` → `scanimage.py` getters →
  `clean_scanimage_metadata`): raw SI metadata is flattened, nested under `si`,
  and stack type is detected. `detect_stack_type` → lbm / piezo / single_plane /
  pollen drives which array subclass and z-plane semantics apply. **For LBM
  stacks `dz` is user-supplied** (no reliable acquisition value).
- **`query_tiff_pages`** gets page count **O(1)** from the IFD stride + file size.
  **Never `len(tf.pages)` or walk the IFD chain** — large raw files (≈730k pages)
  take minutes.
- **`OutputMetadata`** (`output.py`) recomputes metadata for *subsetted* writes
  (adjusts Ly/Lx/counts, scales dz/fs/dx/dy by selection stride, carries a
  provenance stamp so multi-hop writes raw→zarr→tiff don't double-scale).
  `to_imagej`, `to_ome_ngff`, `to_napari_scale`.
- **`EXPORT_DENYLIST`/`strip_for_export`** drop suite2p-only fields (meanImg,
  xoff, regPC, ...) from non-suite2p output. `default_ops()` is the suite2p ops
  template; `_build_ome_metadata` builds the OME-NGFF dict.
- **`_filename_parser.py`** extracts indicator/model/region/line from filenames.

## Read-time views (non-destructive, reversible via `.enabled`)

- **`with_axial_shifts(arr)` / `AxialShiftView`** (`_registration.py`) — applies
  per-plane `plane_shifts` on a padded canvas at read time; 5D TCZYX, reversible.
- **`with_phasecorr(arr)` / `PhaseCorrectedView`** (`_phasecorr_view.py`) —
  read-time bidirectional scan-phase correction for **any** T-dim array
  (zarr/h5/npy/mp4/tiff), not just native ScanImage. Holds a real
  `PhaseCorrectionFeature` so GUI duck-typing works. `run_gui` auto-wraps
  non-native T>1 arrays (disabled by default; toggle flips `.enabled`).
- **`_ChannelView`** — fixes one channel of a 5D array, presenting it as 4D TZYX
  for pipelines (e.g. suite2p) that expect single-channel input.

## Isoview (light-sheet) — `arrays/isoview/`

`IsoviewArray` is one class, **four kinds** via `_KINDS` (raw / corrected /
fused / clusterpt), always 5D TCZYX, lazy per-(t, view) reads, PRIORITY 90 (wins
ZarrArray). `detect_isoview_kind` parent-walks to the stack root. This is a deep
subsystem with its own pipeline (correct → fuse → BigStitcher export) shared with
the upstream `isoview` package. Before editing it, read the relevant memory notes
under `~/.claude/projects/.../memory/` (bigstitcher export, crop, projections,
tile-grid orientation, fusion, VW naming, layout) — many hard-won conventions
live there, not in the code.

## GUI — Miller Brain Studio (`gui/`)

fastplotlib (wgpu) canvas + imgui-bundle side panel. `run_gui.run_gui(path)` →
`imread` → wrap (squeeze singletons, auto axial-shift / phasecorr) → fpl
`ImageWidget` + `PreviewDataWidget` (docked imgui panel).

- **Widgets** (`gui/widgets/*.py`): auto-discovered `Widget` subclasses gated by
  classmethod `is_supported(parent)` (duck-type the array), rendered per frame by
  `draw_all_widgets`. Gate by **marker fields in metadata, not isinstance + `_arr`
  unwrap loops**.
- **Pipelines** (`gui/widgets/pipelines/`): `PipelineWidget` subclasses with
  `applies_to(arr)` + `draw_config()`. `suite2p.py` and `isoview.py`. Skip/Run/
  Force radios must map to a parameter that actually gates a stage
  (`do_registration`/`do_detection`/`do_deconvolution`).
- **Tasks/workers** (`tasks.py`, `_worker.py`): processing runs in a spawned
  subprocess (`python -m mbo_utilities.gui._worker`), reporting progress to a
  sidecar JSON; `_sysmem.py` powers the optional memory monitor.
- **imgui gotchas (latent bugs):** `end_child()` must run even when
  `begin_child()` returns False (else the window stack corrupts on resize);
  `input_float` is 32-bit (strict `==` vs Python float64 defaults fails every
  frame); never enumerate wgpu adapters inside an imgui frame.
- **Log audience:** INFO = user-facing confirmation, DEBUG = internal. Toggle via
  `MBO_DEBUG=1` or File → Options → Debug logging.

## CLI (`cli.py`, entry point `mbo`)

A bare path (`mbo /data`) opens the viewer. Subcommands: `view`, `convert`
(full imread/imwrite surface), `info`, `formats`, `scanphase`, `processes`,
`gpu`, `init` (starter notebooks), `download-models` (prefetch cellpose, avoids a
parallel-worker download race), `shortcut`, and the `hpc` group.

## HPC / SLURM (`hpc/`)

Primary path is config-driven submitit: `mbo hpc init/run/status` or
`from mbo_utilities.hpc import HpcConfig, submit`. TOML config (`io` / `slurm` /
`pipeline` / `parameters` sections) — no `.sh` editing, no `MBO_*` env vars.
`pipeline.planes_per_gpu` is the pack factor F. Modes: single / array (per-plane
shards + dependent aggregate) / local.

## Other subsystems

- **`analysis/phasecorr.py`** — `bidir_phasecorr` / `_phase_corr_2d`: detect &
  apply the bidirectional scan offset between odd/even rows (rFFT phase
  correlation or integer roll). `analysis/scanphase.py` — `run_scanphase_analysis`
  diagnostic (offset vs averaging window, spatial map).
- **`gpu.py`** — `MBO_GPU` → `CUDA_VISIBLE_DEVICES` governs torch + cupy +
  cellpose uniformly (inherited by workers). `mbo gpu` reports render vs compute
  GPU.
- **`log.py`** — `log.get()` is the root `mbo` logger (propagates, user-facing).
  `log.get("sub")` children have `propagate=False` + no handler, so **INFO is
  silent in plain scripts** — use `log.get()` for terminal-facing lines.
- **`preferences.py`** — `get_mbo_dirs()` is the single source for `~/.mbo`
  (recent files, last dirs, gpu index, caches). `env_cache.py` is the one
  decoupled exception.

## Extending: add a new format

1. Subclass `LazyArray` (or compose `ReductionMixin`/`Shape5DMixin`): implement
   `_shape5d`, `__getitem__`, `dtype`, `_metadata`, classmethod `can_open`, set
   `PRIORITY`.
2. For write support add `_imwrite` (or route through `_imwrite_base`).
3. Register it: built-ins call `register_array_class` and are also listed in the
   `[project.entry-points."mbo_utilities.lazy_arrays"]` group in `pyproject.toml`.
   Third-party packages declare the same entry-point group to add formats without
   editing this repo (see `docs/forking.md`).

## Testing

```bash
uv run pytest tests/ -k "not test_99" -x --tb=short
```

~332 tests pass, ~38 skipped. The repo dev env keeps the full stack
(`tool.uv.default-groups`); base install is slim (viewer + IO + scanphase only,
heavy stacks are extras `processing`/`napari`/`notebooks`/`isoview`).

## Conventions / recurring gotchas

- 1-based selection at the public API (`planes`/`timepoints`/`channels`),
  0-based internally.
