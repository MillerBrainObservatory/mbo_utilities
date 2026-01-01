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
    return Path, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.sidebar(
        [
            mo.md("# MBO Utilities"),
            mo.md("**User Guide**"),
            mo.md("---"),
            mo.md("""
    [Documentation](https://millerbrainobservatory.github.io/mbo_utilities/)

    [GitHub](https://github.com/millerbrainobservatory/mbo_utilities)

    [MBO Hub](https://millerbrainobservatory.github.io/)
            """),
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("""
    # mbo_utilities: User Guide

    An image I/O library with an intuitive GUI for scientific imaging data.
        """),
        mo.callout(
            mo.md("**Best viewed as an app:** Press **Cmd/Ctrl + .** or click the **app view** button (bottom right) to hide all code cells."),
            kind="info"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    input_browser = mo.ui.file_browser(
        initial_path="D:/Server_Data/raw_scanimage_tiffs",
        multiple=False,
        selection_mode="directory",
        filetypes=[".tif", ".tiff", ".zarr", ".h5", ".bin"],
        label="Input directory",
    )
    return (input_browser,)


@app.cell(hide_code=True)
def _(mo):
    output_browser = mo.ui.file_browser(
        initial_path="D:/",
        multiple=False,
        selection_mode="directory",
        label="Output directory",
    )
    return (output_browser,)


@app.cell(hide_code=True)
def _(mo):
    output_name = mo.ui.text(
        value="converted",
        label="Output folder name",
        full_width=True,
    )
    return (output_name,)


@app.cell(hide_code=True)
def _(input_browser, mo, output_browser, output_name):
    mo.vstack([
        mo.hstack([input_browser, output_browser], justify="start", widths=[0.5, 0.5]),
        output_name,
    ])
    return


@app.cell(hide_code=True)
def _(Path, input_browser, output_browser, output_name):
    RAW_PATH = Path(input_browser.value[0].path) if input_browser.value else Path("D:/Server_Data/raw_scanimage_tiffs")
    _output_base = Path(output_browser.value[0].path) if output_browser.value else Path("D:/output")
    SAVE_PATH = _output_base / output_name.value if output_name.value else _output_base
    return RAW_PATH, SAVE_PATH


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(
    gui_tab,
    metadata_tab,
    mo,
    numpy_tab,
    phase_tab,
    reading_tab,
    roi_tab,
    video_tab,
    writing_tab,
):
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


@app.cell(hide_code=True)
def _(RAW_PATH, mo):
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

        # Key metadata
        key_metadata = {
            "dx (um)": arr.metadata.get("dx", "N/A"),
            "dy (um)": arr.metadata.get("dy", "N/A"),
            "dz (um)": arr.metadata.get("dz", "N/A"),
            "Frame Rate (Hz)": arr.metadata.get("fs", "N/A"),
        }
        key_metadata_items = [{"Key": k, "Value": str(v)} for k, v in key_metadata.items()]
        key_metadata_display = mo.ui.table(key_metadata_items, selection=None, pagination=False)

        load_success = True

    except Exception as e:
        data_info = mo.callout(
            mo.md(f"**Could not load data:** {e}\n\nSelect a valid data directory above."),
            kind="warn"
        )
        key_metadata_display = mo.md("")
        arr = None
        load_success = False

    return arr, data_info, key_metadata_display, load_success, mbo, reading_api, reading_content


@app.cell(hide_code=True)
def _(arr, load_success, mo):
    # Create sliders in separate cell - only return them, don't access .value
    if load_success and arr is not None:
        frame_slider = mo.ui.slider(
            start=0, stop=min(arr.num_frames - 1, 99), value=0, step=1,
            label=f"Frame (0-{min(arr.num_frames - 1, 99)})", show_value=True
        )
        plane_slider = mo.ui.slider(
            start=0, stop=arr.num_planes - 1, value=arr.num_planes // 2, step=1,
            label=f"Z-Plane (0-{arr.num_planes - 1})", show_value=True
        )
    else:
        frame_slider = None
        plane_slider = None
    return frame_slider, plane_slider


@app.cell(hide_code=True)
def _(arr, frame_slider, load_success, mo, np, plane_slider):
    # Access slider values and create viewer in separate cell
    if load_success and arr is not None and frame_slider is not None:
        frame_data = arr[frame_slider.value, plane_slider.value, :, :]

        # Normalize for display
        frame_min, frame_max = frame_data.min(), frame_data.max()
        if frame_max > frame_min:
            display_frame = ((frame_data - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
        else:
            display_frame = np.zeros_like(frame_data, dtype=np.uint8)

        viewer = mo.vstack([
            mo.md("### Lazy Array Viewer"),
            mo.md("*Drag sliders to load individual frames on-demand*"),
            mo.hstack([frame_slider, plane_slider], justify="start"),
            mo.image(display_frame),
            mo.md(f"Frame shape: `{frame_data.shape}` | Min: `{frame_min:.1f}` | Max: `{frame_max:.1f}`"),
        ])
    else:
        viewer = mo.md("")
    return (viewer,)


@app.cell(hide_code=True)
def _(data_info, key_metadata_display, mo, reading_api, reading_content, viewer):
    reading_tab = mo.vstack([
        reading_content,
        reading_api,
        mo.md("---"),
        data_info,
        viewer,
        key_metadata_display,
    ])
    return (reading_tab,)


@app.cell(hide_code=True)
def _(mo):
    writing_content = mo.md("""
    ## Writing Data with `imwrite`

    Convert between any supported formats: `.zarr`, `.tiff`, `.h5`, `.bin`, `.npy`
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
    | `overwrite` | `False` | Overwrite existing files |
        """),
        "Zarr-Specific Options": mo.md("""
    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `sharded` | `True` | Use sharding (recommended for large data) |
    | `ome` | `True` | Write OME-NGFF metadata |
    | `level` | `1` | Compression level (0=none, 1-9=gzip) |
        """),
    })
    return writing_api, writing_content


@app.cell(hide_code=True)
def _(arr, load_success, mo):
    # Create write UI controls in separate cell
    if load_success and arr is not None:
        plane_options = [str(i) for i in range(1, arr.num_planes + 1)]
        format_dropdown = mo.ui.dropdown(
            [".zarr", ".tiff", ".h5", ".bin"],
            value=".zarr",
            label="Format",
        )
        frames_slider = mo.ui.slider(
            start=10, stop=min(500, arr.num_frames), value=min(100, arr.num_frames), step=10,
            label="Frames", show_value=True
        )
        planes_select = mo.ui.multiselect(
            options=plane_options,
            value=plane_options[:3] if len(plane_options) >= 3 else plane_options,
            label="Planes (1-based)",
        )
        write_button = mo.ui.run_button(label="Write Data")
    else:
        format_dropdown = None
        frames_slider = None
        planes_select = None
        write_button = None
    return format_dropdown, frames_slider, planes_select, write_button


@app.cell(hide_code=True)
def _(Path, arr, format_dropdown, frames_slider, load_success, mbo, mo, output_browser, output_name, planes_select, write_button):
    # Access UI values and handle write action - compute SAVE_PATH here for reactivity
    if load_success and arr is not None and write_button is not None:
        _output_base = Path(output_browser.value[0].path) if output_browser.value else Path("D:/output")
        current_save_path = _output_base / output_name.value if output_name.value else _output_base

        write_controls = mo.vstack([
            mo.md("### Export Settings"),
            mo.md(f"**Output path:** `{current_save_path}`"),
            mo.hstack([format_dropdown, frames_slider], justify="start"),
            planes_select,
            write_button,
        ])

        if write_button.value:
            try:
                # Create directory if it doesn't exist
                current_save_path.parent.mkdir(parents=True, exist_ok=True)
                selected_planes = [int(p) for p in planes_select.value] if planes_select.value else None
                mbo.imwrite(
                    arr,
                    current_save_path,
                    ext=format_dropdown.value,
                    planes=selected_planes,
                    num_frames=frames_slider.value,
                )
                write_status = mo.callout(
                    mo.md(f"Successfully wrote data to `{current_save_path}`"),
                    kind="success"
                )
            except Exception as e:
                write_status = mo.callout(
                    mo.md(f"**Write failed:** {e}"),
                    kind="danger"
                )
        else:
            write_status = mo.md("")
    else:
        write_controls = mo.callout(
            mo.md("Load data first using the file browser above."),
            kind="info"
        )
        write_status = mo.md("")
    return write_controls, write_status


@app.cell(hide_code=True)
def _(mo, write_controls, write_status, writing_api, writing_content):
    writing_tab = mo.vstack([writing_content, writing_api, mo.md("---"), write_controls, write_status])
    return (writing_tab,)


@app.cell(hide_code=True)
def _(RAW_PATH, load_success, mbo, mo, np):

    phase_content = mo.md("""
    ## Bidirectional Scan-Phase Correction

    ScanImage TIFFs acquired with bidirectional scanning have phase offsets
    between alternating scan lines. This creates a "zipper" artifact.

    **Methods:**
    - `fix_phase=True`: Standard integer-pixel correction (fast)
    - `use_fft=True`: FFT-based subpixel correction (slower, more precise)
    """)

    if load_success:
        try:
            scan = mbo.imread(RAW_PATH)

            # Interactive controls
            fix_phase_switch = mo.ui.switch(label="Enable correction", value=True)
            use_fft_switch = mo.ui.switch(label="Use FFT (subpixel)", value=False)
            method_select = mo.ui.dropdown(["mean", "median", "max"], value="mean", label="Method")
            phase_plane_slider = mo.ui.slider(start=0, stop=scan.num_planes - 1, value=scan.num_planes // 2, label="Z-Plane", show_value=True)

            phase_controls = mo.vstack([
                mo.md("### Interactive Demo"),
                mo.hstack([fix_phase_switch, use_fft_switch, method_select], justify="start"),
                phase_plane_slider,
            ])

            scan.roi = None
            plane_idx = phase_plane_slider.value

            # Get uncorrected frame
            scan.fix_phase = False
            img_no_corr = scan[0, plane_idx, :, :]

            # Get corrected frame
            scan.fix_phase = fix_phase_switch.value
            scan.use_fft = use_fft_switch.value
            scan.phasecorr_method = method_select.value
            img_corr = scan[0, plane_idx, :, :]

            # Crop to center for detail
            cy, cx = img_no_corr.shape[0] // 2, img_no_corr.shape[1] // 2
            crop_size = 80
            crop = (slice(cy-crop_size, cy+crop_size), slice(cx-crop_size, cx+crop_size))

            img_no_corr_crop = img_no_corr[crop]
            img_corr_crop = img_corr[crop]

            # Normalize for display
            def normalize_img(img):
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                return np.zeros_like(img, dtype=np.uint8)

            corr_label = f"Corrected ({method_select.value}"
            if use_fft_switch.value:
                corr_label += ", FFT"
            corr_label += ")"

            phase_result = mo.vstack([
                mo.md(f"**Plane {plane_idx}** - Center crop comparison"),
                mo.hstack([
                    mo.vstack([mo.md("**No Correction**"), mo.image(normalize_img(img_no_corr_crop))]),
                    mo.vstack([mo.md(f"**{corr_label}**"), mo.image(normalize_img(img_corr_crop))]),
                ], justify="start"),
            ])
        except Exception as e:
            phase_controls = mo.md("")
            phase_result = mo.callout(mo.md(f"Error: {e}"), kind="danger")
    else:
        phase_controls = mo.md("")
        phase_result = mo.callout(mo.md("Load ScanImage data to see phase correction demo."), kind="info")

    phase_tab = mo.vstack([phase_content, mo.md("---"), phase_controls, phase_result])
    return (phase_tab,)


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
        """),
        "Acquisition Metadata": mo.md("""
    ScanImage-specific collection parameters:
    - `num_mrois` - number of scan regions
    - `fix_phase`, `phasecorr_method` - scan-phase settings
    - `zoom_factor`, `objective_resolution` - optical parameters
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
    | `gamma` | 1.0 | Gamma correction (<1 = brighter) |
    | `quality` | 9 | Video quality (1-10) |
        """),
    })

    video_tab = mo.vstack([video_content, video_params])
    return (video_tab,)


@app.cell(hide_code=True)
def _(mbo, mo, np):

    numpy_content = mo.md("""
    ## Working with NumPy Arrays

    Wrap any numpy array with `imread()` to get full `imwrite()` support.
    """)

    demo_data = np.random.randn(100, 512, 512).astype(np.float32)
    arr_np = mbo.imread(demo_data)

    numpy_example = mo.md(f"""
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

    # Write to any format
    mbo.imwrite(arr, "output", ext=".zarr")
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


@app.cell(hide_code=True)
def _(mo):

    gui_content = mo.md("""
    ## MBO-GUI

    The interactive viewer provides real-time visualization of imaging data.

    **From Python:**
    ```python
    import mbo_utilities as mbo
    arr = mbo.imread("data.tif")
    mbo.run_gui(arr)
    ```

    **From command line:**
    ```bash
    uv run mbo /path/to/data
    uv run mbo --gui  # file browser
    ```
    """)

    gui_features = mo.accordion({
        "Features & Shortcuts": mo.md("""
    | Feature | Shortcut |
    |---------|----------|
    | Play/Pause | `Space` |
    | Prev/Next frame | `Left/Right` |
    | Prev/Next z-plane | `Up/Down` |
    | Reset view | `R` |
    | Screenshot | `S` |
        """),
    })

    gui_link = mo.md("[Full GUI documentation](https://millerbrainobservatory.github.io/mbo_utilities/gui.html)")

    gui_tab = mo.vstack([gui_content, gui_features, gui_link])
    return (gui_tab,)


if __name__ == "__main__":
    app.run()
