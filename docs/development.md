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
| `nframes` | num_frames, T | - | Frame count |
| `num_zplanes` | num_planes, nplanes, Z | - | Z-plane count |
| `num_mrois` | num_rois, nrois | - | mROI count |

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
