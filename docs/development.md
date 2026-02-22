# Development

## Setup

```bash
git clone https://github.com/MillerBrainObservatory/mbo_utilities.git
cd mbo_utilities
uv sync --all-extras
```

## Testing

Tests use synthetic data by default.

```bash
# run all tests
uv run pytest tests/ -v

# run specific test file
uv run pytest tests/test_arrays.py -v

# for CI: run only synthetic data tests
uv run pytest tests/test_arrays.py tests/test_to_video.py -v -k "synthetic or Synthetic"

# keep output files
KEEP_TEST_OUTPUT=1 uv run pytest tests/
```

Test structure:
- `tests/conftest.py` - fixtures for synthetic 3D/4D data, temp files, comparison helpers
- `tests/test_arrays.py` - array class tests (indexing, protocols, volume detection)
- `tests/test_roundtrip.py` - format conversion tests
- `tests/test_metadata_module.py` - metadata system tests
- `tests/test_to_video.py` - video export tests

## Code Formatting

[ruff](https://github.com/astral-sh/ruff) for formatting and linting.

```bash
# format code
uv tool run ruff format .

# check linting
uv tool run ruff check .

# fix auto-fixable issues
uv tool run ruff check --fix .
```

## Building Docs

```bash
uv pip install "mbo_utilities[docs]"
cd docs
uv run make clean
uv run make html
```

---

## Internals

Reference for internal systems.

### Logging

`mbo_utilities.log`.

All modules use a hierarchical logger under the `mbo` namespace.

```python
from mbo_utilities import log

# get a logger for your module
logger = log.get("arrays.suite2p")  # creates mbo.arrays.suite2p

logger.debug("loading file")
logger.info("processing complete")
logger.warning("missing metadata")
```

The GUI uses this module to add/remove entries in the debug logger.

```python
import logging
from mbo_utilities import log

# set global level (affects all mbo.* loggers)
log.set_global_level(logging.DEBUG)

# enable/disable specific loggers
log.disable("metadata.io")  # silence noisy module
log.enable("metadata.io")   # re-enable

# attach custom handler (e.g., for GUI)
log.attach(my_handler)

# list active loggers
log.get_package_loggers()  # ["mbo.arrays", "mbo.metadata.io", ...]
```

The `MBO_DEBUG=1` environment variable enables debug logging globally.

#### Logging API

| Function | Description |
|----------|-------------|
| `log.get(subname)` | Get logger for `mbo.{subname}` |
| `log.set_global_level(level)` | Set level for all mbo loggers |
| `log.attach(handler)` | Add handler to all mbo loggers |
| `log.enable(*subs)` | Enable specific subloggers |
| `log.disable(*subs)` | Disable specific subloggers |
| `log.get_package_loggers()` | List active logger names |

---

### Preferences

User preferences stored in `~/mbo/settings/preferences.json`.

Handles recent files, dialog directories, GUI state, and pipeline defaults.

```python
from mbo_utilities.preferences import (
    add_recent_file,
    get_recent_files,
    get_last_dir,
    set_last_dir,
    get_gui_preference,
    set_gui_preference,
)

# recent files
add_recent_file("/path/to/data.tiff")
for entry in get_recent_files():
    print(entry["path"], entry["timestamp"])

# context-specific directories (dialogs remember their last location)
set_last_dir("open_file", "/data/imaging")
set_last_dir("save_as", "/output")
start_dir = get_last_dir("suite2p_output") or Path.home()

# GUI preferences
set_gui_preference("split_rois", True)
split = get_gui_preference("split_rois", default=False)
```

#### Directory Contexts

Each dialog type maintains its own last-used directory:

| Context | Description |
|---------|-------------|
| `open_file` | File > Open File |
| `open_folder` | File > Open Folder |
| `save_as` | Save As dialog |
| `suite2p_output` | Suite2p output directory |
| `suite2p_chan2` | Channel 2 file selection |
| `suite2p_stat` | stat.npy selection |
| `suite2p_ops` | ops.npy selection |
| `suite2p_diagnostics` | Diagnostics plane folder |
| `grid_search` | Grid search results |

#### Preferences API

| Function | Description |
|----------|-------------|
| `get_recent_files()` | List of recent file dicts |
| `add_recent_file(path, type)` | Add to recent (max 20) |
| `remove_recent_file(path)` | Remove from recent |
| `clear_recent_files()` | Clear all recent |
| `get_last_dir(context)` | Get last dir for context |
| `set_last_dir(context, path)` | Set last dir for context |
| `get_default_open_dir()` | Smart fallback for open dialogs |
| `get_gui_preference(key, default)` | Get GUI preference |
| `set_gui_preference(key, value)` | Set GUI preference |
| `get_pipeline_defaults()` | Get pipeline settings dict |
| `set_pipeline_default(key, value)` | Set pipeline setting |
| `reset_preferences()` | Clear all preferences |
| `export_preferences(path)` | Backup to file |
| `import_preferences(path, merge)` | Restore from file |

---

### Metadata

Standardized metadata handling with alias resolution across formats (ScanImage, Suite2p, OME, TIFF tags).

#### Parameter Registry

Parameters are defined in `METADATA_PARAMS` with canonical names and aliases:

```python
from mbo_utilities.metadata import get_param, get_canonical_name, METADATA_PARAMS

# get parameter checking all aliases
meta = {"PhysicalSizeX": 0.5, "frame_rate": 7.5}
dx = get_param(meta, "dx")  # 0.5 (via PhysicalSizeX alias)
fs = get_param(meta, "fs")  # 7.5 (via frame_rate alias)

# resolve alias to canonical name
get_canonical_name("pixel_size_x")  # "dx"
get_canonical_name("nplanes")       # "num_zplanes"
get_canonical_name("fps")           # "fs"

# check registered parameters
param = METADATA_PARAMS["dx"]
print(param.aliases)  # ("PhysicalSizeX", "pixel_size_x", ...)
print(param.unit)     # "µm"
print(param.default)  # 1.0
```

#### Voxel Size

```python
from mbo_utilities.metadata import get_voxel_size, VoxelSize

# extract from metadata (checks all alias patterns)
vs = get_voxel_size(metadata)
print(vs.dx, vs.dy, vs.dz)  # µm/px

# with overrides
vs = get_voxel_size(metadata, dz=20.0)

# normalize metadata (add all aliases)
from mbo_utilities.metadata import normalize_resolution
normalize_resolution(metadata)  # adds PhysicalSizeX, z_step, etc.
```

#### ScanImage Detection

```python
from mbo_utilities.metadata import (
    detect_stack_type,
    get_num_zplanes,
    get_frame_rate,
    get_roi_info,
)

# detect acquisition type
stack_type = detect_stack_type(metadata)  # "lbm", "piezo", or "single_plane"

# extract parameters
nz = get_num_zplanes(metadata)
fs = get_frame_rate(metadata)
roi_info = get_roi_info(metadata)  # {"num_mrois": 7, "roi": (68, 68), "fov": (476, 68)}
```

#### Registered Parameters

The canonical names for dx and Lx have been chosen to match suite2p, as that was the first pipeline integrated in `mbo_utilities`.

| Parameter | Aliases | Unit | Description |
|-----------|---------|------|-------------|
| `dx` | PhysicalSizeX, pixel_size_x | µm | X pixel size |
| `dy` | PhysicalSizeY, pixel_size_y | µm | Y pixel size |
| `dz` | PhysicalSizeZ, z_step | µm | Z step size |
| `fs` | frame_rate, fps, fr | Hz | Frame rate |
| `Lx` | width, nx, size_x | px | Image width |
| `Ly` | height, ny, size_y | px | Image height |
| `num_timepoints` | nframes, num_frames, T | - | Timepoint count (T dimension) |
| `num_zplanes` | num_planes, nplanes, Z | - | Z-plane count |
| `num_mrois` | num_rois, nrois | - | mROI count |

Note: `num_timepoints` is the canonical name for the T dimension. The alias `nframes` is maintained for Suite2p compatibility (Suite2p's `ops["nframes"]` maps to `num_timepoints`).

---

### Pipeline Registry

Central registry for array types and processing pipelines. Tracks file patterns, extensions, and marker files.

#### Registering a Pipeline

```{warning}
Pipeline registry is a WIP.
```

```python
from mbo_utilities.pipeline_registry import register_pipeline, PipelineInfo

# register via function
register_pipeline(PipelineInfo(
    name="my_pipeline",
    description="Custom processing pipeline",
    input_patterns=["**/*.tif"],
    output_patterns=["**/output.zarr"],
    input_extensions=["tif", "tiff"],
    output_extensions=["zarr"],
    marker_files=["pipeline_done.json"],
    category="processor",
))

# or via decorator
from mbo_utilities.pipeline_registry import pipeline

@pipeline(
    name="custom_array",
    description="Reads custom format",
    input_extensions=["custom"],
    marker_files=["meta.json"],
    category="reader",
)
class CustomArray:
    ...
```

#### Querying the Registry

```python
from mbo_utilities.pipeline_registry import (
    get_pipeline_info,
    get_all_pipelines,
    get_pipelines_by_category,
    get_readable_extensions,
)

# get specific pipeline
info = get_pipeline_info("suite2p")
print(info.marker_files)  # ["ops.npy"]

# get all readers
readers = get_pipelines_by_category("reader")

# get all readable extensions
exts = get_readable_extensions()  # {"tif", "tiff", "zarr", "h5", ...}
```

#### PipelineInfo Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Unique identifier |
| `description` | str | Human-readable description |
| `input_patterns` | list[str] | Glob patterns for input files |
| `output_patterns` | list[str] | Glob patterns for output files |
| `input_extensions` | list[str] | File extensions read (no dot) |
| `output_extensions` | list[str] | File extensions written |
| `marker_files` | list[str] | Files identifying output directories |
| `validator` | Callable | Optional validation function |
| `category` | str | Grouping: reader, writer, processor, segmentation |

---

### Installation Validation

Validates installation, GPU configuration, and optional dependencies.

```python
from mbo_utilities.install_checker import check_installation, Status

status = check_installation()

print(f"mbo_utilities v{status.mbo_version}")
print(f"Python {status.python_version}")

# check cuda
if status.cuda_info.device_name:
    print(f"GPU: {status.cuda_info.device_name}")
    print(f"CUDA Toolkit: {status.cuda_info.nvcc_version}")

# check features
for feature in status.features:
    if feature.status == Status.OK:
        print(f"✓ {feature.name} v{feature.version}")
    elif feature.status == Status.WARN:
        print(f"! {feature.name}: {feature.message}")
    elif feature.status == Status.MISSING:
        print(f"- {feature.name} not installed")

# overall status
if status.all_ok:
    print("Installation OK")
```

#### Status Enums

| Status | Meaning |
|--------|---------|
| `OK` | Installed and working |
| `WARN` | Installed but degraded (e.g., no GPU) |
| `ERROR` | Installed but broken |
| `MISSING` | Not installed |

#### Checked Features

- **PyTorch**: CUDA availability, version
- **CuPy**: CUDA runtime, NVRTC compiler
- **Suite2p**: Installation, GPU linkage
- **Suite3D**: Installation, CuPy dependency
- **Rastermap**: Installation

---

### Availability Flags

Quick boolean checks for optional dependencies (no import overhead):

```python
from mbo_utilities._installation import (
    HAS_SUITE2P,
    HAS_SUITE3D,
    HAS_CUPY,
    HAS_TORCH,
    HAS_RASTERMAP,
    HAS_IMGUI,
    HAS_FASTPLOTLIB,
    HAS_PYSIDE6,
)

if HAS_SUITE2P:
    from suite2p import run_s2p
```

These are evaluated at module load using `importlib.util.find_spec()` (no actual import).

---

### Directory Structure

Standard locations for user data:

```python
from mbo_utilities.file_io import get_mbo_dirs

dirs = get_mbo_dirs()
# {
#     "base": Path("~/mbo"),
#     "imgui": Path("~/mbo/imgui"),
#     "cache": Path("~/mbo/cache"),
#     "logs": Path("~/mbo/logs"),
#     "assets": Path("~/mbo/imgui/assets"),
#     "settings": Path("~/mbo/imgui/assets/app_settings"),
#     "data": Path("~/mbo/data"),
#     "tests": Path("~/mbo/tests"),
# }
```

| Directory | Purpose |
|-----------|---------|
| `~/mbo/` | Root directory |
| `~/mbo/settings/` | Preferences JSON |
| `~/mbo/cache/` | Temporary cached data |
| `~/mbo/logs/` | Application logs |
| `~/mbo/data/` | Sample/user data |

---

### GUI Internals

Reference for the imgui_bundle-based GUI system.

#### Launch Flow

```text
run_gui(path) or `mbo` CLI
  → _cli_entry() handles flags (--splash, --check-upgrade, etc.)
  → _run_gui_impl() does heavy imports
  → _launch_standard_viewer()
      ├─ imread(path) → array object
      ├─ _create_image_widget(array) → fastplotlib.ImageWidget
      │   sets slider_dim_names from array.dims
      │   applies window_funcs (mean/max/std projections)
      └─ PreviewDataWidget(iw, fpath) → EdgeWindow attached to figure
          ├─ selects viewer via get_viewer_class(array)
          ├─ initializes panels (debug, stats, metadata, process, pipeline)
          └─ spawns background thread for z-stats computation
```

#### Setup (`_setup.py`)

Runs on import. Configures:

- `RENDERCANVAS_BACKEND=qt` for PyQt6 integration
- wgpu backend (Vulkan/DX12/Metal)
- copies imgui_bundle fonts to `~/.mbo/imgui/assets/`
- `hello_imgui.set_assets_folder()` for font/icon loading
- `set_qt_icon()` sets window icons on QApplication

#### Render Loop

`PreviewDataWidget.update()` is called every frame by fastplotlib:

```text
update()
  ├─ handle_keyboard_shortcuts()        # o=open, s=save, m=metadata, etc.
  ├─ check_file_dialogs()               # poll pfd dialogs, trigger load_new_data()
  ├─ draw_menu_bar()                    # File/Docs/Settings + process status
  ├─ draw popups (metadata, process console, save-as, help)
  ├─ _viewer.draw()                     # delegates to viewer → panels
  ├─ draw_all_widgets()                 # auto-discovered widgets
  └─ draw_stats_section() if ready      # implot charts for z-stats
```

#### Viewer Hierarchy

```text
BaseViewer (abstract)
  draw(), on_data_loaded(), draw_menu_bar(), cleanup()
  ├─ TimeSeriesViewer      (default for TZYX volumetric data)
  │   panels: DebugPanel, ProcessPanel, MetadataPanel, StatsPanel, PipelinePanel
  │   properties: proj, window_size, gaussian_sigma, mean_subtraction
  │   phase correction: fix_phase, use_fft, border, max_offset, phase_upsample
  └─ PollenCalibrationViewer   (stack_type == "pollen")
```

#### Panel System

```text
BasePanel (abstract)
  draw(), cleanup(), show()/hide()/toggle()
  ├─ DebugPanel         # routes logging.Handler to scrollable table, filters by level/logger
  ├─ ProcessPanel       # shows running background tasks from ProcessManager
  ├─ MetadataPanel      # recursive tree display of array metadata, search/filter
  ├─ StatsPanel         # implot visualization of z-stats (snr/mean/std per slice)
  └─ PipelinePanel      # suite2p pipeline config and execution
```

All lazy-imported via `panels/__init__.py` `__getattr__`.

#### Widget System (Plugin Architecture)

```python
class Widget(ABC):
    name: str
    priority: int          # lower = rendered first
    is_supported(parent)   # checked per frame
    draw()                 # imgui rendering
```

Auto-discovery: `pkgutil.iter_modules()` scans `widgets/`, finds `Widget` subclasses, calls `is_supported()`, instantiates and sorts by priority.

| Widget | Purpose |
|--------|---------|
| `MenuBarWidget` | file/docs/settings menus |
| `RasterScanWidget` | phase correction controls (for `SupportsRasterScan` arrays) |
| `FrameAveragingWidget` | piezo frame grouping |
| `Suite2pEmbedded` | pipeline config |
| `IntegratedClassifierWindow` | ROI classification |

#### Z-Stats Computation (`_stats.py`)

```text
compute_zstats(parent)
  → threading.Thread per array in image_widget.data
    → compute_zstats_single_array(parent, idx, arr)
        ├─ check arr.stats/arr.zstats for cached values
        ├─ _get_slice_range() → z-plane or channel indices
        ├─ for each slice:
        │   stack = _get_zslice(arr, slice(None,None,10), z)  # every 10th frame
        │   mean_img = np.mean(stack, axis=0)
        │   SNR = (fg_mean - bg_mean) / bg_std   # AAPM, top20%/bottom50%
        └─ stores in parent._zstats[idx]
```

Visualization via implot:

- single z-plane: bar chart + stats table
- dual z-plane: grouped bars
- multi z-plane: line plot with error bands
- combined view: gray per-ROI lines + shaded mean +/- std

#### Background Tasks (`tasks.py`, `_worker.py`)

```text
ProcessManager.launch_task(task_type, args)
  → subprocess: python -m mbo_utilities.gui._worker <type> <json>
  → writes progress to ~/.mbo/logs/progress_{uuid}.json
  → survives GUI closure
```

Task types:

- `save_as`: `imread` -> optional `register_zplanes_s3d` -> `imwrite`
- `suite2p`: `imread` -> `_ChannelView` (if 5D) -> `lbm_suite2p_python.pipeline`

#### Data Loading on File Change

```
load_new_data(parent, path)
  ├─ imread(path) → new array
  ├─ parent.image_widget.data[0] = arr    # iw-array replacement API
  ├─ detect nz/nc from shape or dims
  ├─ reset window functions if ndim changed
  ├─ get_viewer_class(arr) → reinstantiate viewer
  └─ refresh_zstats() → new background thread
```

#### imgui_bundle Primitives

```python
# layout
imgui.begin_child("id")          # scrollable container
imgui.begin_table("id", cols)    # multi-column
imgui.collapsing_header("label") # collapsible section
imgui.tree_node("label")         # nested tree

# inputs
imgui.checkbox("label", val)
imgui.input_int("##id", val)
imgui.input_float("##id", val)
imgui.input_text("##id", val)
imgui.slider_int("##id", val, min, max)

# popups
imgui.open_popup("title")
imgui.begin_popup_modal("title")

# plotting (implot)
implot.begin_plot("title", size)
implot.setup_axes("X", "Y")
implot.plot_line("series", x, y)
implot.plot_bars("series", x, heights)

# file dialogs (portable_file_dialogs)
pfd.open_file("title", default_path, filters)
pfd.select_folder("title", default_path)

# icons
icons_fontawesome.ICON_FA_FOLDER_OPEN
icons_fontawesome.ICON_FA_SAVE

# runner
params = hello_imgui.RunnerParams()
params.callbacks.show_gui = render_func   # per-frame callback
immapp.run(runner_params=params, add_ons_params=addons)
```

#### Protocols (`_protocols.py`)

Runtime-checkable protocols gate widget visibility:

- `SupportsRasterScan` : `fix_phase`, `use_fft`, `border`, `max_offset`, `phase_upsample`
- `SupportsMetadata` : metadata dict access
- `SupportsROI` : ROI info access

#### Signal/Slot (PyQt6)

`SharedDataModel(QObject)` provides reactive state for suite2p integration:

- signals: `roi_selected`, `iscell_changed`, `iscell_batch_changed`, `data_loaded`, `data_saved`
- properties: `stat`, `iscell`, `F`, etc. with change detection

#### Key State on PreviewDataWidget

| Field | Type | Description |
|-------|------|-------------|
| `image_widget` | `ImageWidget` | fastplotlib widget |
| `fpath` | `str \| list[str]` | current file path(s) |
| `shape` | `tuple` | data shape |
| `nz`, `nc` | `int` | z-planes, channels |
| `_viewer` | `BaseViewer` | active viewer |
| `_zstats` | `list[dict]` | `[{mean:[], std:[], snr:[]}]` per graphic |
| `_zstats_done` | `list[bool]` | completion flags |
| `_zstats_progress` | `list[float]` | 0.0-1.0 |
| `_file_dialog` | pfd dialog | current open file dialog |
| `_folder_dialog` | pfd dialog | current folder dialog |

---

### Metadata Internals

Detailed reference for the metadata extraction and normalization system.

#### Module Map

| File | Purpose |
|------|---------|
| `base.py` | core types: `MetadataParameter`, `VoxelSize`, `RoiMode`, parameter registry |
| `params.py` | alias resolution (`get_param`), voxel size extraction, normalization |
| `scanimage.py` | stack type detection, z-plane/channel counting, ROI extraction |
| `io.py` | TIFF metadata I/O, raw ScanImage detection, OME builder |
| `output.py` | `OutputMetadata` : reactive metadata for subsetted writes |
| `_filename_parser.py` | regex extraction of experiment info from filenames |

#### Stack Type Detection

`detect_stack_type(metadata)` classifies acquisitions:

```
channelSave > 2 AND hStackManager.enable → "pollen"  (LBM + piezo calibration)
channelSave > 2                          → "lbm"     (beamlets as z-planes)
hStackManager.enable                     → "piezo"   (z-stack scanning)
else                                     → "single_plane"
```

Convenience: `is_lbm_stack()`, `is_piezo_stack()`, `is_pollen_stack()`

#### Z-Plane Counting

`get_num_zplanes(metadata)`:

- LBM/pollen: `len(channelSave) // num_color_channels` (beamlets = z-planes)
- piezo: `hStackManager.numSlices`
- single_plane: 1

#### Color Channel Detection

`get_num_color_channels(metadata)`:

- counts unique AI sources from `hScan2D.virtualChannelSettings__N.source`
- AI0 only = 1 channel, AI0 + AI1 = 2 channels
- fallback: channelSave length

#### Timepoint Computation

`compute_num_timepoints(total_frames, metadata)`:

- LBM/single_plane: `total_frames` (1 frame = 1 timepoint)
- piezo/pollen: `total_frames // frames_per_volume`
  - `frames_per_volume = numSlices * framesPerSlice` (or just `numSlices` if log-averaged)

#### Voxel Size Resolution Priority

`get_voxel_size(metadata, dx, dy, dz)` checks in order:

1. user-provided parameters (dx, dy, dz)
2. canonical keys (dx, dy, dz)
3. `pixel_resolution` tuple
4. legacy keys (`umPerPixX/Y/Z`)
5. OME keys (`PhysicalSizeX/Y/Z`)
6. ScanImage SI keys (except dz for LBM, which must be user-supplied)
7. defaults: 1.0 for dx/dy, None for dz

#### Parameter Alias System

Every parameter has one canonical name and multiple aliases. `get_param(meta, name)` resolves any alias to its value:

```python
meta = {"PhysicalSizeX": 0.5, "frame_rate": 7.5}
get_param(meta, "dx")  # 0.5 (via PhysicalSizeX alias)
get_param(meta, "fs")  # 7.5 (via frame_rate alias)
```

`normalize_metadata(metadata)` adds all aliases to the dict for cross-tool compatibility.

#### Metadata I/O (`io.py`)

**`is_raw_scanimage(file)`**: True if TIFF has `scanimage_metadata` AND no tag 50839 AND no `shaped_metadata`.

**`get_metadata(file, dx, dy, dz)`**: entry point. accepts path, directory, list of paths, or array with `.metadata`.

**`get_metadata_single(file)`**: extraction from a single file:

- non-raw: tries `shaped_metadata[0]`, tag 50839 JSON, first-page description JSON
- raw ScanImage: parses `scanimage_metadata`, extracts ROI groups, computes `pixel_resolution`
- fallback: looks for `ops.npy` in directory tree (suite2p)

**`clean_scanimage_metadata(meta)`**: transforms flat `SI.*` keys to nested dict under `si`, strips numpy types, appends derived fields (`stack_type`, `num_zplanes`, `num_color_channels`, `fs`, `dz`, ROI info), calls `normalize_metadata()`.

**`query_tiff_pages(file_path)`**: fast page count from TIFF header (classic v42 and BigTIFF v43) without loading data.

**`_build_ome_metadata(shape, dtype, metadata, dims)`**: OME-NGFF v0.5 with multiscales, coordinate transforms, OMERO rendering, custom sections (scanimage, roi_groups, acquisition, processing, source_files).

#### Other Extractors (`scanimage.py`)

| Function | Returns |
|----------|---------|
| `get_frame_rate()` | Hz, from `hRoiManager.scanFrameRate` or `1/scanFramePeriod` |
| `get_z_step_size()` | microns, from `hStackManager.actualStackZStepSize` (None for LBM) |
| `get_frames_per_slice()` | from `hStackManager.framesPerSlice` (NOT `logFramesPerSlice`) |
| `get_log_average_factor()` | `hScan2D.logAverageFactor` |
| `get_roi_info()` | `{num_mrois, roi: (w,h), fov: (total_w, h)}` |
| `get_stack_info()` | aggregates all of the above |
| `extract_roi_slices()` | pixel boundaries per ROI accounting for fly-to lines |

#### Filename Parser (`_filename_parser.py`)

`parse_filename_metadata(filename)` extracts experiment info via regex:

- `calcium_indicator`: GCaMP6f, jGCaMP8m, Cal-520, OGB-1, Fluo-4, RCaMP, etc.
- `animal_model`: Mouse, Rat, Zebrafish
- `brain_region`: V1, S1, M1, A1, OB, PBN, NTS, ACC, PFC, HPC
- `transgenic_line`: Cux2, Emx1, CaMKII, Thy1, PV, SST, VIP, etc.
- `induction_method`: AAV, Transgenic, Acute injection

#### End-to-End Metadata Flow

Reading raw ScanImage TIFF:

```
get_metadata(path, dz=5.0)
  → is_raw_scanimage() = True
  → get_metadata_single()
      → parse tifffile.scanimage_metadata
      → extract ROI groups, pixel_resolution
  → clean_scanimage_metadata()
      → nest SI.* keys under 'si'
      → detect_stack_type() → "lbm"
      → get_num_zplanes() → 28
      → get_num_color_channels() → 1
      → get_frame_rate() → 9.6 Hz
      → get_roi_info() → {num_mrois: 2, roi: (256,512), fov: (512,512)}
  → normalize_resolution(dz=5.0)
      → adds dx/dy/dz + all aliases
  → return dict
```

Writing subsetted data:

```
OutputMetadata(source_meta, (100,28,512,512), ("T","Z","Y","X"),
               selections={"Z": [0,2,4,6]})
  → dz = 5.0 * 2 = 10.0  (every other z-plane)
  → num_zplanes = 4
  → fs = 9.6 (unchanged, T not subsetted)
  → to_imagej((100,4,512,512))
      → {"images":204800, "frames":100, "slices":4, "spacing":10.0, ...}
```
