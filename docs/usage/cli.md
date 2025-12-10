(cli_usage)=
# CLI Commands

The `mbo` command provides several utilities for working with imaging data.

## Launch GUI

```bash
mbo                     # open file dialog
mbo /path/to/data       # open specific file
mbo /path/to/data --metadata  # show only metadata
```

## Download Files

Download files from GitHub URLs:

```bash
mbo download https://github.com/user/repo/blob/main/notebook.ipynb
mbo download https://github.com/user/repo/blob/main/data.npy -o ./data/
```

## Scan-Phase Analysis

Analyze bidirectional scanning phase offset. See [scanphase](scanphase) for details.

```bash
mbo scanphase                          # open file dialog
mbo scanphase /path/to/data.tiff       # analyze specific file
mbo scanphase ./folder/ -n 5           # use first 5 tiffs from folder
mbo scanphase data.tiff --show         # show plots interactively
```

## File Conversion

Convert between supported formats:

```bash
mbo convert input.tiff output.zarr     # tiff to zarr
mbo convert input.tiff output.bin      # tiff to suite2p binary
mbo convert input.zarr output.tiff     # zarr to tiff
```

## File Info

Display information about a data file:

```bash
mbo info /path/to/data.tiff
```

## List Formats

Show supported input and output formats:

```bash
mbo formats
```

## Utilities

```bash
mbo --download-notebook    # download user guide notebook
mbo --check-install        # verify installation and GPU config
```
