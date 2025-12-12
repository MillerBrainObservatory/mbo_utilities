(cli_usage)=

# CLI Guide

The `mbo` command provides tools for viewing, converting, and analyzing imaging data.

## Overview

| Command | Description |
|---------|-------------|
| `mbo` | launch gui with file dialog |
| `mbo view` | view data in gui |
| `mbo convert` | convert between formats |
| `mbo info` | show array info |
| `mbo download` | download files from github |
| `mbo formats` | list supported formats |

## GUI Mode

```bash
mbo                            # file dialog
mbo /path/to/data              # open specific file
mbo /path/to/data --metadata   # show only metadata
mbo view /data --roi 0 --roi 2 # view specific ROIs
```

## Convert

Convert between formats with optional processing.

```bash
mbo convert input.tiff output/ -e .zarr           # tiff to zarr
mbo convert input.tiff output/ -e .bin            # tiff to suite2p binary
mbo convert input.zarr output/ -e .tiff           # zarr to tiff
mbo convert input.tiff output/ -e .zarr -p 1 -p 7 # specific planes
mbo convert input.tiff output/ --fix-phase        # with phase correction
mbo convert input.tiff output/ -n 1000            # first 1000 frames
```

**Options:**

| Option | Description |
|--------|-------------|
| `-e, --ext` | output format: .tiff, .zarr, .bin, .h5, .npy |
| `-p, --planes` | z-planes to export (1-based), repeatable |
| `-n, --num-frames` | limit number of frames |
| `--roi` | roi selection: None, 0, N, or "1,3" |
| `--fix-phase/--no-fix-phase` | bidirectional phase correction |
| `--phasecorr-method` | mean, median, or max |
| `--register-z` | z-plane registration via suite3d |
| `--ome/--no-ome` | ome-zarr metadata (zarr only) |
| `--overwrite` | replace existing files |
| `--chunk-mb` | streaming chunk size (default: 100) |
| `--debug` | verbose logging |

## Info

Display array information without loading data.

```bash
mbo info /data/raw.tiff
mbo info /data/volume.zarr
mbo info /data/suite2p/plane0
```

## Download

Download files from github (auto-converts blob to raw urls).

```bash
mbo download https://github.com/user/repo/blob/main/notebook.ipynb
mbo download https://github.com/user/repo/blob/main/data.npy -o ./data/
```

## Utilities

```bash
mbo --download-notebook     # download user guide notebook
mbo --download-file URL     # download any file
mbo --check-install         # verify installation and gpu config
```

## Formats

Show supported file formats:

```bash
mbo formats
```

**Input formats:** .tif, .tiff, .zarr, .bin, .h5, .hdf5, .npy, .json
**Output formats:** .tiff, .zarr, .bin, .h5, .npy
