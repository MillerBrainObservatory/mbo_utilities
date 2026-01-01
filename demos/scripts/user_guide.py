# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "mbo_utilities",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", app_title="MBO Utilities User Guide")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    return Path, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    # Sidebar with file browser and navigation
    file_browser = mo.ui.file_browser(
        initial_path="D:/",
        multiple=False,
        selection_mode="directory",
        filetypes=[".tif", ".tiff", ".zarr", ".h5", ".bin"],
        label="Select data directory",
    )

    save_path_input = mo.ui.text(
        value="D:/SERVER_DATA/raw_scanimage_tiffs/volume",
        label="Save path",
        full_width=True,
    )

    mo.sidebar(
        [
            mo.md("# MBO Utilities"),
            mo.md("**User Guide**"),
            mo.md("---"),
            file_browser,
            mo.md("---"),
            save_path_input,
            mo.md("---"),
            mo.md("""
[Documentation](https://millerbrainobservatory.github.io/mbo_utilities/)

[GitHub](https://github.com/millerbrainobservatory/mbo_utilities)

[MBO Hub](https://millerbrainobservatory.github.io/)
            """),
        ],
        footer=mo.md("*Select a data directory to begin*"),
    )
    return file_browser, save_path_input


@app.cell(hide_code=True)
def _(Path, file_browser, save_path_input):
    RAW_PATH = Path(file_browser.value[0].path) if file_browser.value else Path("D:/SERVER_DATA/raw_scanimage_tiffs")
    SAVE_PATH = Path(save_path_input.value) if save_path_input.value else Path("D:/SERVER_DATA/raw_scanimage_tiffs/volume")
    return RAW_PATH, SAVE_PATH


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
# mbo_utilities: User Guide

An image I/O library with an intuitive GUI for scientific imaging data.

!!! tip "Best viewed as an app"
    Press **Cmd/Ctrl + .** or click the **app view** button (bottom right) to hide all code cells.
    """)
    return


@app.cell(hide_code=True)
def _(mo, reading_tab, writing_tab, phase_tab, roi_tab, metadata_tab, video_tab, numpy_tab, gui_tab):
    mo.ui.tabs({
        "Reading Data": reading_tab,
        "Writing Data": writing_tab,
        "Phase Correction": phase_tab,
        "ROI Handling": roi_tab,
        "Metadata": metadata_tab,
        "Video Export": video_tab,
        "NumPy Arrays": numpy_tab,
        "GUI": gui_tab,
    })
    return


# =============================================================================
# Tab: Reading Data
# =============================================================================
@app.cell(hide_code=True)
def _(RAW_PATH, mo, np, plt):
    import mbo_utilities as mbo

    reading_content = mo.md("""
## Reading Data with `imread`

`imread` is a lazy-file reader that automatically detects the file format.

- Pass a **file**, **list of files**, or **directory**
- Wrap a **numpy array** to access lazy operations
- Volumetric data: pass the directory containing z-plane files
    """)

    reading_api = mo.accordion({
        "imread() Parameters": mo.md("""
| Parameter | Default | Description |
|-----------|---------|-------------|
| `inputs` | *required* | Path, list of paths, directory, or numpy array |
| `roi` | `None` | ROI selection: `None`=stitch, `0`=split, `N`=specific ROI |
| `fix_phase` | `True` | Enable bidirectional scan-phase correction |
| `phasecorr_method` | `"mean"` | Phase estimation: `"mean"`, `"median"`, or `"max"` |
| `use_fft` | `False` | Use FFT-based subpixel correction (slower, more precise) |
| `upsample` | `5` | Upsampling factor for subpixel phase estimation |
| `border` | `3` | Border pixels to exclude from phase calculation |
| `max_offset` | `4` | Maximum phase offset to search (pixels) |
        """),
    })

    try:
        arr = mbo.imread(RAW_PATH)

        # Data info table
        data_info = mo.md(f"""
### Loaded Data

| Property | Value |
|----------|-------|
| **Path** | `{RAW_PATH}` |
| **Type** | `{type(arr).__name__}` |
| **Shape** | `{arr.shape}` (T, Z, Y, X) |
| **Planes** | {arr.num_planes} |
| **Frames** | {arr.num_frames} |
| **ROIs** | {arr.num_rois} |
| **Dtype** | `{arr.dtype}` |
        """)

        # Create visualization - max projection of middle plane
        mid_plane = arr.num_planes // 2
        preview_img = arr[:, mid_plane, :, :].max(axis=0)

        preview_fig, preview_axes = plt.subplots(1, 2, figsize=(14, 5))

        # Max projection
        im = preview_axes[0].imshow(preview_img, cmap='gray')
        preview_axes[0].set_title(f'Max Projection (Plane {mid_plane})', fontweight='bold')
        preview_axes[0].axis('off')
        plt.colorbar(im, ax=preview_axes[0], fraction=0.046, pad=0.04)

        # Z-stack montage (sample of planes)
        num_sample = min(6, arr.num_planes)
        plane_indices = [int(i * (arr.num_planes - 1) / (num_sample - 1)) for i in range(num_sample)]

        montage_height = preview_img.shape[0]
        montage_width = preview_img.shape[1] * num_sample
        montage = np.zeros((montage_height, montage_width), dtype=preview_img.dtype)

        for i, p in enumerate(plane_indices):
            plane_img = arr[0, p, :, :]
            montage[:, i*preview_img.shape[1]:(i+1)*preview_img.shape[1]] = plane_img

        preview_axes[1].imshow(montage, cmap='gray')
        preview_axes[1].set_title(f'Z-Stack Sample (Planes {plane_indices})', fontweight='bold')
        preview_axes[1].axis('off')

        plt.tight_layout()

        visualization = mo.vstack([
            mo.md("### Data Preview"),
            mo.as_html(preview_fig),
        ])

        # Key metadata display
        key_metadata = {
            "dx (um)": arr.metadata.get("dx", "N/A"),
            "dy (um)": arr.metadata.get("dy", "N/A"),
            "dz (um)": arr.metadata.get("dz", "N/A"),
            "Frame Rate (Hz)": arr.metadata.get("fs", "N/A"),
            "Bidirectional": arr.metadata.get("bidirectional_scan", "N/A"),
        }
        key_metadata_items = [{"Key": k, "Value": str(v)} for k, v in key_metadata.items()]

        key_metadata_display = mo.vstack([
            mo.md("### Key Metadata"),
            mo.ui.table(key_metadata_items, selection=None, pagination=False),
        ])

        load_success = True

    except Exception as e:
        data_info = mo.callout(
            mo.md(f"**Could not load data:** {e}\n\nSelect a valid data directory in the sidebar."),
            kind="warn"
        )
        visualization = mo.md("")
        key_metadata_display = mo.md("")
        arr = None
        load_success = False

    reading_tab = mo.vstack([
        reading_content,
        reading_api,
        mo.md("---"),
        data_info,
        visualization,
        key_metadata_display,
    ])
    return arr, load_success, mbo, reading_tab


# =============================================================================
# Tab: Writing Data
# =============================================================================
@app.cell(hide_code=True)
def _(arr, load_success, mo):

    writing_content = mo.md("""
## Writing Data with `imwrite`

Convert between any supported formats: `.zarr`, `.tiff`, `.h5`, `.bin`, `.npy`

!!! warning "Frame Count Warning"
    When saving z-planes with inconsistent frame counts, only the minimum
    number of frames across all planes will be written.
    """)

    writing_api = mo.accordion({
        "imwrite() Parameters": mo.md("""
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lazy_array` | *required* | Source array from imread |
| `outpath` | *required* | Output directory or file path |
| `ext` | `".tiff"` | Output format: `.tiff`, `.zarr`, `.h5`, `.bin`, `.npy` |
| `planes` | `None` | Z-planes to export (1-based): `None`=all, int, or list |
| `num_frames` | `None` | Number of frames to write (`None`=all) |
| `roi_mode` | `"concat_y"` | ROI handling: `"concat_y"` or `"separate"` |
| `register_z` | `False` | Enable Suite3D axial registration |
| `overwrite` | `False` | Overwrite existing files |
| `output_suffix` | `None` | Custom suffix (default: `"_stitched"`) |
        """),
        "Zarr-Specific Options": mo.md("""
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sharded` | `True` | Use sharding (recommended for large data) |
| `ome` | `True` | Write OME-NGFF metadata |
| `level` | `1` | Compression level (0=none, 1-9=gzip) |
| `shard_frames` | `None` | Frames per shard (default: 100) |
| `chunk_shape` | `None` | Inner chunk shape (t, y, x) |
        """),
    })

    if load_success and arr is not None:
        plane_options = [str(i) for i in range(1, arr.num_planes + 1)]
        write_controls = mo.vstack([
            mo.md("### Export Settings"),
            mo.hstack([
                mo.ui.dropdown(
                    [".zarr", ".tiff", ".h5", ".bin"],
                    value=".zarr",
                    label="Format",
                ),
                mo.ui.slider(
                    start=10, stop=min(500, arr.num_frames), value=100, step=10,
                    label="Frames",
                ),
            ], justify="start"),
            mo.ui.multiselect(
                options=plane_options,
                value=["7", "8", "9"] if arr.num_planes >= 9 else plane_options[:3],
                label="Planes (1-based)",
            ),
            mo.ui.run_button(label="Export Data"),
        ])
    else:
        write_controls = mo.callout(
            mo.md("Load data first using the sidebar file browser."),
            kind="info"
        )

    writing_tab = mo.vstack([writing_content, writing_api, mo.md("---"), write_controls])
    return (writing_tab,)


# =============================================================================
# Tab: Phase Correction
# =============================================================================
@app.cell(hide_code=True)
def _(RAW_PATH, load_success, mbo, mo, plt):

    phase_content = mo.md("""
## Bidirectional Scan-Phase Correction

ScanImage TIFFs acquired with bidirectional scanning have phase offsets
between alternating scan lines. This creates a "zipper" artifact.

**Methods:**
- `fix_phase=True`: Standard integer-pixel correction (fast)
- `use_fft=True`: FFT-based subpixel correction (slower, more precise)
    """)

    # Interactive controls
    fix_phase_switch = mo.ui.switch(label="Enable correction", value=True)
    use_fft_switch = mo.ui.switch(label="Use FFT (subpixel)", value=False)
    method_select = mo.ui.dropdown(["mean", "median", "max"], value="mean", label="Estimation method")
    plane_slider = mo.ui.slider(start=0, stop=14, value=7, label="Z-Plane")

    phase_controls = mo.vstack([
        mo.md("### Interactive Demo"),
        mo.hstack([fix_phase_switch, use_fft_switch], justify="start"),
        mo.hstack([method_select, plane_slider], justify="start"),
    ])

    if load_success:
        try:
            scan = mbo.imread(RAW_PATH)
            scan.roi = None
            plane_idx = plane_slider.value

            scan.fix_phase = False
            img_no_corr = scan[:, plane_idx, :, :].max(axis=0)

            scan.fix_phase = fix_phase_switch.value
            scan.use_fft = use_fft_switch.value
            scan.phasecorr_method = method_select.value
            img_corr = scan[:, plane_idx, :, :].max(axis=0)

            phase_fig, phase_axes = plt.subplots(1, 2, figsize=(12, 5))
            cy, cx = img_no_corr.shape[0] // 2, img_no_corr.shape[1] // 2
            crop_size = 60
            crop = (slice(cy-crop_size, cy+crop_size), slice(cx-crop_size, cx+crop_size))

            phase_axes[0].imshow(img_no_corr[crop], cmap='gray')
            phase_axes[0].set_title('No Correction', fontweight='bold')
            phase_axes[0].axis('off')

            title = f"Corrected ({method_select.value}"
            if use_fft_switch.value:
                title += ", FFT"
            title += ")"
            phase_axes[1].imshow(img_corr[crop], cmap='gray')
            phase_axes[1].set_title(title, fontweight='bold')
            phase_axes[1].axis('off')
            plt.tight_layout()

            phase_result = mo.vstack([
                mo.md(f"**Plane {plane_idx}** - Max projection (cropped to center)"),
                mo.as_html(phase_fig),
            ])
        except Exception as e:
            phase_result = mo.callout(mo.md(f"Error: {e}"), kind="danger")
    else:
        phase_result = mo.callout(mo.md("Load ScanImage data to see phase correction demo."), kind="info")

    phase_tab = mo.vstack([phase_content, mo.md("---"), phase_controls, mo.md("---"), phase_result])
    return (phase_tab,)


# =============================================================================
# Tab: ROI Handling
# =============================================================================
@app.cell(hide_code=True)
def _(RAW_PATH, load_success, mbo, mo):

    roi_content = mo.md("""
## Multi-ROI Handling

ScanImage can acquire multiple regions of interest (ROIs) in a single scan.
Control how they are handled:

| `roi` value | Behavior |
|-------------|----------|
| `None` | Stitch ROIs horizontally (default) |
| `0` | Split into separate arrays |
| `N` | Select specific ROI (1-indexed) |
    """)

    if load_success:
        try:
            arr_roi = mbo.imread(RAW_PATH)
            num_rois = arr_roi.num_rois

            roi_mode = mo.ui.dropdown(
                ["Stitched (None)", "Split All (0)"] + [f"ROI {i}" for i in range(1, num_rois + 1)],
                value="Stitched (None)",
                label="ROI Mode",
            )

            roi_controls = mo.vstack([
                mo.md(f"### Data has **{num_rois} ROIs**"),
                roi_mode,
            ])

            if "Stitched" in roi_mode.value:
                arr_roi.roi = None
                shape_info = f"Stitched shape: `{arr_roi[0, 0].shape}`"
            elif "Split" in roi_mode.value:
                arr_roi.roi = 0
                result = arr_roi[0, 0]
                if isinstance(result, tuple):
                    shapes = [str(r.shape) for r in result]
                    shape_info = f"Split shapes: {', '.join(shapes)}"
                else:
                    shape_info = f"Shape: `{result.shape}`"
            else:
                roi_num = int(roi_mode.value.split()[-1])
                arr_roi.roi = roi_num
                shape_info = f"ROI {roi_num} shape: `{arr_roi[0, 0].shape}`"

            roi_result = mo.md(f"**Result:** {shape_info}")

        except Exception as e:
            roi_controls = mo.callout(mo.md(f"Error: {e}"), kind="danger")
            roi_result = None
    else:
        roi_controls = mo.callout(mo.md("Load ScanImage data to see ROI handling."), kind="info")
        roi_result = None

    roi_tab = mo.vstack([roi_content, mo.md("---"), roi_controls, roi_result if roi_result else mo.md("")])
    return (roi_tab,)


# =============================================================================
# Tab: Metadata
# =============================================================================
@app.cell(hide_code=True)
def _(arr, load_success, mo):

    metadata_content = mo.md("""
## Accessing Metadata

All lazy arrays expose metadata via the `.metadata` property.

**CLI Access:**
```bash
uv run mbo info /path/to/data
uv run mbo /path/to/data --metadata
```
    """)

    metadata_types = mo.accordion({
        "Imaging Metadata": mo.md("""
Physical data properties:
- `dx`, `dy`, `dz` - voxel size in um
- `fs` - frame rate in Hz
- `Lx`, `Ly` - image dimensions in pixels
- `num_zplanes`, `num_timepoints`, `dtype`
        """),
        "Acquisition Metadata": mo.md("""
ScanImage-specific collection parameters:
- `num_mrois` - number of scan regions
- `roi_groups`, `roi_heights` - ROI configuration
- `fix_phase`, `phasecorr_method` - scan-phase settings
- `zoom_factor`, `objective_resolution` - optical parameters
        """),
        "Stack Type Differences": mo.md("""
| Parameter | LBM Stack | Piezo Stack | Single Plane |
|-----------|:---------:|:-----------:|:------------:|
| `num_zplanes` | > 1 | > 1 | 1 |
| `dz` | user-supplied | from SI | N/A |
| `frame_averaging` | no | yes | yes |
| `bidirectional_scan` | yes | varies | varies |
        """),
    })

    if load_success and arr is not None:
        full_metadata_items = [
            {"Key": str(k), "Value": str(v)[:80] + ("..." if len(str(v)) > 80 else "")}
            for k, v in arr.metadata.items()
        ]
        metadata_table = mo.ui.table(
            full_metadata_items,
            selection=None,
            pagination=True,
            page_size=15,
        )
        metadata_display = mo.vstack([mo.md("### Current Data Metadata"), metadata_table])
    else:
        metadata_display = mo.callout(mo.md("Load data to view its metadata."), kind="info")

    metadata_tab = mo.vstack([metadata_content, metadata_types, mo.md("---"), metadata_display])
    return (metadata_tab,)


# =============================================================================
# Tab: Video Export
# =============================================================================
@app.cell(hide_code=True)
def _(mo):

    video_content = mo.md("""
## Video Export

Export calcium imaging data to video files for presentations with `to_video`.

```python
from mbo_utilities import imread, to_video

arr = imread("data.tif")

# Basic export
to_video(arr, "output.mp4")

# Quick preview (10x playback)
to_video(arr, "preview.mp4", speed_factor=10)

# High-quality for presentations
to_video(
    arr,
    "movie.mp4",
    fps=30,
    speed_factor=5,
    temporal_smooth=3,
    gamma=0.8,
    quality=10,
)
```
    """)

    video_params = mo.accordion({
        "to_video() Parameters": mo.md("""
| Parameter | Default | Description |
|-----------|---------|-------------|
| `fps` | 30 | Base frame rate |
| `speed_factor` | 1.0 | Playback speed multiplier |
| `plane` | None | Z-plane to export (for 4D data) |
| `temporal_smooth` | 0 | Rolling average window (frames) |
| `spatial_smooth` | 0 | Gaussian blur sigma (pixels) |
| `gamma` | 1.0 | Gamma correction (<1 = brighter) |
| `cmap` | None | Matplotlib colormap name |
| `quality` | 9 | Video quality (1-10) |
| `vmin_percentile` | 1.0 | Percentile for auto min |
| `vmax_percentile` | 99.5 | Percentile for auto max |
        """),
    })

    video_example = mo.md("""
### 4D Data

For 4D data `(T, Z, Y, X)`, select which z-plane to export:

```python
to_video(arr_4d, "plane5.mp4", plane=5)
```
    """)

    video_tab = mo.vstack([video_content, video_params, video_example])
    return (video_tab,)


# =============================================================================
# Tab: NumPy Arrays
# =============================================================================
@app.cell(hide_code=True)
def _(mbo, mo, np):

    numpy_content = mo.md("""
## Working with NumPy Arrays

Wrap any numpy array with `imread()` to get full `imwrite()` support.
This is useful for:
- Processing data in memory before saving
- Converting between array types
- Working with synthetic or generated data
    """)

    data = np.random.randn(100, 512, 512).astype(np.float32)
    arr_np = mbo.imread(data)

    numpy_example = mo.md(f"""
### Example Usage

```python
import numpy as np
import mbo_utilities as mbo

# 3D array (T, Y, X)
data = np.random.randn(100, 512, 512).astype(np.float32)
arr = mbo.imread(data)
print(arr)  # {arr_np}

# 4D array (T, Z, Y, X)
volume = np.random.randn(100, 15, 512, 512).astype(np.float32)
arr4d = mbo.imread(volume)
print(f"dims: {{arr4d.dims}}, num_planes: {{arr4d.num_planes}}")

# Write to any format
mbo.imwrite(arr, "output", ext=".zarr")
mbo.imwrite(arr4d, "output", ext=".zarr", planes=[1, 7, 14])
```
    """)

    numpy_api = mo.accordion({
        "Supported Array Shapes": mo.md("""
| Dimensions | Shape | Interpretation |
|------------|-------|----------------|
| 2D | `(Y, X)` | Single frame |
| 3D | `(T, Y, X)` | Time series (single plane) |
| 4D | `(T, Z, Y, X)` | Volumetric time series |
        """),
    })

    numpy_tab = mo.vstack([numpy_content, numpy_example, numpy_api])
    return (numpy_tab,)


# =============================================================================
# Tab: GUI
# =============================================================================
@app.cell(hide_code=True)
def _(mo):

    gui_content = mo.md("""
## MBO-GUI

The interactive viewer provides real-time visualization and analysis of imaging data.

### Launch Methods

**From Python** (Jupyter Lab/Notebook):
```python
import mbo_utilities as mbo

arr = mbo.imread("data.tif")
mbo.run_gui(arr)
```

**From command line** (works anywhere):
```bash
# Open GUI with data
uv run mbo /path/to/data

# Open GUI file browser
uv run mbo --gui
```
    """)

    gui_features = mo.accordion({
        "GUI Features": mo.md("""
- **Time series playback** with adjustable speed
- **Z-plane navigation** for volumetric data
- **ROI selection** and intensity extraction
- **Contrast adjustment** with auto-scaling
- **Export** selected regions or frames
        """),
        "Keyboard Shortcuts": mo.md("""
| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `Left/Right` | Previous/Next frame |
| `Up/Down` | Previous/Next z-plane |
| `R` | Reset view |
| `S` | Screenshot |
        """),
    })

    gui_link = mo.md("""
See the [GUI documentation](https://millerbrainobservatory.github.io/mbo_utilities/gui.html) for detailed usage.
    """)

    gui_tab = mo.vstack([gui_content, gui_features, gui_link])
    return (gui_tab,)


if __name__ == "__main__":
    app.run()
