# isoview metadata resolution (fuse / bigstitcher)

## Current behavior (verified 2026-05)

`multi_fuse` and `generate_bigstitcher_xml` resolve acquisition metadata
(pixel_spacing_z, objective_mag, camera_view_map) via
`read_all_xml_metadata(input_dir, specimen)` in `isoview/io.py`, which:

1. Globs **XML sidecars** (`ch*.xml`, `*_CHN*.xml`, `*_VW*.xml`).
2. Only **if no XML sidecars are found**, falls back to reading
   `ome.isoview.xml_metadata` embedded in OME-Zarr files.

So it is **XML-sidecar-first, zarr-attributes-second** — it does *not*
read metadata from the zarr data it is actually fusing first.

## Why "data-first" doesn't work for corrected trees today

- The corrected `.ome.zarr` volumes carry **no `ome.isoview` block**
  (only standard OME-NGFF `version`/`multiscales`/`omero`).
- The CORRECT step (`pipeline.py:_save_volume_file`) embeds an
  `isoview_meta` with per-volume identifiers (specimen/timepoint/camera/
  channel/stage_um) but **not** the `xml_metadata` blob the zarr-fallback
  path looks for. Only the **FUSE** step embeds full `xml_metadata`
  (which is why fused zarrs are self-describing, corrected ones aren't).
- Corrected trees also still ship XML sidecars next to every volume, so
  the XML branch always wins regardless.

## To make it data-first (isoview-side, two changes)

1. CORRECT step: embed full `xml_metadata` (pixel_spacing_z,
   objective_mag, camera_view_map, wavelengths) into corrected zarrs'
   `ome.isoview`, like the FUSE step does.
2. Read path: when the input volume is zarr, check its `ome.isoview`
   attributes **before** globbing XML sidecars; XML/raw only as legacy
   fallback.

Then fuse + bigstitcher read metadata straight from the corrected zarrs —
no XML/raw dependency. Note: only makes *future* corrected runs
self-describing; existing corrected trees still need their XML sidecars
(or a one-time backfill).

## Files

- `isoview/io.py` — `read_all_xml_metadata`, `read_isoview_zarr_attrs`
- `isoview/pipeline.py:_save_volume_file` — CORRECT-step write (~L450-505)
- `isoview/fusion.py` — FUSE-step write with `isoview_meta` (~L1140-1175)
- `isoview/config.py` — `fused_dir`/`stitcher_dir` derived from
  `strip_output_suffix(input_dir.name)` (string only, no disk dependency)
