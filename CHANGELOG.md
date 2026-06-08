# Changelog

The notable, user-facing changes for each release since v2.5.0. Bug fixes and
smaller internal tweaks are left out.

## v2.6.0 — 2026-02-14

- Full dual-channel LBM support — index, view, and save individual channels. The GUI is now channel-aware with a channel selector.
- Read ImageJ hyperstacks directly as a `TiffArray`.
- Dimension labels throughout, with time- and plane-based slicing and volumetric outputs.
- Isoview metadata support, and OME-NGFF bumped to v0.5.

## v2.7.0 — 2026-04-03

- Early 5D array support.
- Trimmed the GUI: the standalone Cellpose and Suite2p entries are gone.
- Fixed pollen-calibration X scaling.
- CUDA 12.x install fixes and a round of dead-code cleanup.

## v3.0.0 — 2026-04-24

- New `IsoviewArray`: a single 5D array covering raw, corrected, fused, and clusterpt data, read lazily one (t, c) at a time, and wired straight into the processing pipeline.
- Internal 3D registration with Suite2p and GPU support.
- Per-channel dimension and OME-NGFF compliance across the board.
- Suite2p is now a required dependency instead of an optional one.
- Isoview gains per-camera cropping at fuse time, projections written next to the tiles, streaming fusion to avoid out-of-memory crashes, and BigStitcher orientation control.

## v3.1.0 — 2026-06-05

- Non-destructive axial registration: apply or remove per-plane shifts at read time with `with_axial_shifts`, fully reversible.
- Multi-resolution pyramids on write, with median and mode downsampling.
- zscore support and a zarr I/O cleanup pass.
- Dropped the default `_stack` suffix from output filenames.

## v3.2.0 — 2026-06-08

- Arrays are now always 5D (TCZYX). If you want the old trimmed shape, use `arr.squeeze()` or `imread(path, squeeze=True)`.
- `.shape` is the canonical 5D shape everywhere — the old `.shape5d` property has been removed.
- Pluggable readers: third-party plugins can register their own formats and override the built-ins.
- `MP4Array` unifies reading and writing mp4s; `to_video` now streams frame by frame.
- GPU toggle via the `MBO_GPU` environment variable, the `mbo gpu` CLI command, or the Options panel. Warns if you select Cellpose without a GPU build of PyTorch.
- Python 3.13 is now supported.
