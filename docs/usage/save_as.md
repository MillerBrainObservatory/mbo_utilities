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
# `save_as`

{func}`mbo_utilities.save_as` is a convenience function for exporting common data formats processed from MBO datasets.

It can save **ScanImage tiffs** or **ScanMultiROIReordered arrays** into:

- `.tiff`
- `.zarr`
- `.h5`

This function is usable from:

- terminal (`mbo assemble` CLI)
- Python script
- Jupyter/IPython notebooks

`save_as` automatically handles large datasets, metadata, chunking, and scan phase correction.

It uses `tifffile`, `h5py`, and optionally `zarr` for efficient IO.

---

## Terminal usage (CLI)

```bash
# Quickly open file/folder, process, and save
assemble path/to/data/ --save path/to/output/ --zarr

# Optional arguments:
# --trimx, --trimy for trimming
# --target_chunk_mb for chunk size
# --summary to include per-plane statistics
```

When no `--save` path is given, only metadata is printed.
