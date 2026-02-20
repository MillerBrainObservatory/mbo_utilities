# GUI Quick Start

## Launch

```bash
mbo                    # opens file dialog
mbo /path/to/data      # opens specific file
mbo --metadata         # metadata only mode
```

## Main Features

- **Time/Z sliders**: Navigate through frames and z-planes
- **Window functions**: Mean, max, std, mean-subtracted projections
- **Scan-phase correction**: Preview bidirectional correction
- **Contrast controls**: Adjust vmin/vmax
- **Export**: Save to .tiff, .zarr, .bin, .h5

## Window Functions

| Function | Description |
|----------|-------------|
| mean | Average intensity over window |
| max | Maximum projection |
| std | Standard deviation |
| mean-sub | Mean-subtracted (highlights changes) |

## Scan-Phase Correction

Preview bidirectional phase correction:

1. View mean-subtracted projection (window 3-15)
2. Toggle Fix Phase on/off to compare
3. Adjust border-px and max-offset
4. Enable Sub-Pixel for refinement

## Opening Data

- **Open File** (`o`): loads exactly the file(s) you select
- **Open Folder** (`Shift+O`): loads all compatible files in a directory

Open File always loads only what you pick. Open Folder scans
the directory and loads files that match the detected format
(e.g. only raw ScanImage TIFFs, ignoring previously saved outputs).

## Saving Data

Access via **File > Save As** or press **s**

Output formats: `.zarr` (recommended), `.tiff`, `.bin`, `.h5`

Save As exports a copy of the current data to a new file.
The viewer continues to display the original dataset.
To work with the saved file, open it with File > Open File.

Running Suite2p after Save As still processes the original
dataset that is loaded in the viewer, not the saved file.
