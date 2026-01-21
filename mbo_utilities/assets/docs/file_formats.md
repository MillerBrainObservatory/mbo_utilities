# Supported File Formats

## Quick Reference

| Format | Read | Write | Description |
|--------|:----:|:-----:|-------------|
| `.tiff` | Yes | Yes | ScanImage, BigTIFF, OME-TIFF |
| `.zarr` | Yes | Yes | Zarr v3, OME-Zarr |
| `.bin` | Yes | Yes | Suite2p binary format |
| `.h5` | Yes | Yes | HDF5 files |
| `.npy` | Yes | Yes | NumPy arrays |

## ScanImage TIFF

Automatically detects acquisition mode:

- **LBM**: Z-planes as channels
- **Piezo**: Z-stacks with optional averaging
- **Single-plane**: Time series

## Zarr Output

Recommended for large datasets:

- Chunked storage for fast random access
- Optional sharding for cloud storage
- OME-NGFF v0.5 compliant metadata

## Suite2p Binary

Used by Suite2p processing pipeline:

- `data_raw.bin`: Unregistered frames
- `data.bin`: Registered frames
- `ops.npy`: Processing parameters

## Directory Detection

The reader auto-detects:

- `planeXX.tiff` - Multi-plane TIFF volume
- `planeXX/ops.npy` - Multi-plane Suite2p output
- `ops.npy` - Single-plane Suite2p output
