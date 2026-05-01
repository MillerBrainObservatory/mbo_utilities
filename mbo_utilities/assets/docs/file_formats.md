# Supported File Formats

## Input

- **ScanImage TIFF** — primary, originally what this tool was built for. LBM mode (Z-planes as channels), piezo (Z-stacks with optional `logAverageFactor` averaging), and single-plane time series are auto-detected.
- **Generic 2D-4D TIFF** — any tifffile-readable stack with reasonable axis order.
- **OME-TIFF / OME-Zarr** — read via OME-NGFF metadata.
- **Suite2p binary (`.bin`)** — `data.bin` (registered) or `data_raw.bin` (unregistered) when paired with an `ops.npy` sibling.

## Output

- **TIFF** — Fiji / ImageJ compatible.
- **Zarr** — chunked, OME-NGFF v0.5 metadata. Recommended for archival and large data.
- **Suite2p `.bin`** — produced by Run tab, paired with `ops.npy`.
- **HDF5 `.h5`** — for downstream NWB / MATLAB tooling.

Save As copies the displayed data — phase correction, frame range, plane subset all apply. The viewer keeps showing the original, so reopen the saved file to work with it. Run always processes the original raw data (not a Save-As output).

## Suite2p output layout

Each processed plane becomes its own subdir:

```
<save_path>/
  zplane01_tp00001-01574/
    ops.npy           # parameters + per-step results
    stat.npy          # ROI shapes
    F.npy / Fneu.npy  # fluorescence + neuropil
    spks.npy          # deconvolved spikes
    iscell.npy        # classifier output
    data.bin          # registered binary (kept by default)
    data_raw.bin      # unregistered (only if keep_raw=True)
  zplane02_tp00001-01574/
  ...
```

Open the *parent* folder to reload the whole volume. Open a `zplane0N_…/` folder or any `data.bin` inside it to reload a single plane.

When re-running on a loaded plane, set Registration -> **Skip**: the pipeline links to the existing `data.bin` rather than copying it, so detection sweeps stay fast and disk-light.
