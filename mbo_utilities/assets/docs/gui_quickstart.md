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

## Saving Data

Access via **File > Save As** or press **s**

Output formats: `.zarr` (recommended), `.tiff`, `.bin`, `.h5`
