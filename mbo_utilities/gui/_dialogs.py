"""
File dialog handlers and data loading.

This module contains file/folder dialog handling and data loading logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from mbo_utilities.reader import imread
from mbo_utilities.arrays import ScanImageArray
from mbo_utilities.preferences import add_recent_file, set_last_dir


def _try_hydrate_s2p_from_binary(parent: Any, path: str | Path) -> bool:
    """If `path` looks like a suite2p binary (data.bin / data_raw.bin),
    hydrate parent.s2p / parent.s2p_db from the plane folder's settings
    files AND point parent._s2p_outdir at the suite2p output ROOT (the
    plane folder's parent) so re-running the pipeline recreates the same
    layout in the same place.

    Source-of-truth preference:
      1. settings.npy + db.npy (canonical upstream-shape pair, exactly
         what was passed into lsp.pipeline)
      2. ops.npy (flat fused dict — may have been mutated by reactive
         fs/dz rescaling, includes detection outputs)

    No-op when the path isn't a suite2p binary or none of the sibling
    settings files exist. Returns True iff settings were applied.
    """
    p = Path(path)
    if p.name not in ("data.bin", "data_raw.bin"):
        return False
    plane_dir = p.parent
    settings_file = plane_dir / "settings.npy"
    db_file = plane_dir / "db.npy"
    ops_file = plane_dir / "ops.npy"

    import numpy as np  # local — keep _dialogs.py import surface tight

    def _load_npy_dict(fp: Path) -> dict | None:
        try:
            arr = np.load(str(fp), allow_pickle=True)
            d = arr.item() if hasattr(arr, "item") and arr.ndim == 0 else arr
            return d if isinstance(d, dict) else None
        except Exception as e:
            parent.logger.warning(
                f"suite2p hydrate: failed to read {fp}: {e}"
            )
            return None

    loaded: dict = {}
    sources: list[str] = []
    try:
        from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
            from_structured as _from_structured,
            from_flat as _from_flat,
            warm_up_suite2p_schema as _warm_up_schema,
        )
    except Exception as e:
        parent.logger.warning(f"suite2p hydrate: schema import failed: {e}")
        return False
    # Force-load suite2p.parameters.SETTINGS now (synchronous, idempotent).
    # `is_default` short-circuits to True while the schema is unloaded, so
    # without this the hydrated values would not be flagged as modified
    # in the Run tab summary or tinted in the Pipeline Settings popup
    # until something else triggered the import. The user is opening
    # suite2p data here, so paying the import cost during file-load is
    # appropriate.
    try:
        _warm_up_schema()
    except Exception as e:
        parent.logger.warning(f"suite2p hydrate: schema warm-up failed: {e}")

    if settings_file.is_file():
        d = _load_npy_dict(settings_file)
        if d:
            loaded.update(_from_structured(d))
            sources.append("settings.npy")
    if db_file.is_file():
        d = _load_npy_dict(db_file)
        if d:
            # db.npy is flat (paths/nplanes/keep_movie_raw at top level),
            # so route via from_flat — it covers keep_movie_raw and any
            # other db fields with entries in _FLAT_TO_MBO.
            loaded.update(_from_flat(d))
            sources.append("db.npy")
    # ALSO read ops.npy: it's the only file that carries lsp's mbo-only
    # post-processing knobs (dff_window_size / dff_percentile /
    # dff_smooth_window — written at the dff_calculation step in
    # run_lsp.py). We `setdefault` instead of `update` so existing
    # entries from settings.npy / db.npy aren't overwritten by ops.npy
    # (settings.npy is the canonical source for suite2p settings; ops.npy
    # may have stale or run-time-mutated values for those same keys).
    if ops_file.is_file():
        d = _load_npy_dict(ops_file)
        if d:
            ops_view = _from_flat(d)
            new_keys = []
            for k, v in ops_view.items():
                if k not in loaded:
                    loaded[k] = v
                    new_keys.append(k)
            if new_keys:
                if "ops.npy" not in sources:
                    sources.append("ops.npy")
                parent.logger.debug(
                    f"suite2p hydrate: ops.npy contributed {len(new_keys)} fields "
                    f"not in settings.npy/db.npy: {sorted(new_keys)}"
                )

    if not loaded:
        parent.logger.info(
            f"suite2p hydrate: no settings/db/ops sibling files at {plane_dir}; "
            "leaving pipeline settings untouched."
        )
        return False

    # access via lazy properties so the dataclasses get instantiated
    # if they haven't been touched yet.
    s2p = getattr(parent, "s2p", None)
    s2p_db = getattr(parent, "s2p_db", None)
    s2p_extras = getattr(parent, "s2p_extras", None)
    if s2p is None or s2p_db is None:
        parent.logger.info(
            "suite2p hydrate: Suite2pSettings/Suite2pDB unavailable "
            "(suite2p not installed?), skipping."
        )
        return False

    n_settings = 0
    n_db = 0
    n_extras = 0
    skipped: list[str] = []
    for field, value in loaded.items():
        target = None
        if hasattr(s2p, field):
            target = s2p
        elif hasattr(s2p_db, field):
            target = s2p_db
        elif s2p_extras is not None and hasattr(s2p_extras, field):
            target = s2p_extras
        else:
            skipped.append(field)
            continue
        # coerce to current field's type — the loaded value may be a
        # numpy scalar (int64, float32, bool_) which imgui inputs don't
        # handle uniformly. _s2p_schema._to_py normalizes numpy types
        # to Python equivalents at the loader boundary, but a leaked
        # ndarray here would still crash bool()/int()/float() with
        # `truth value ambiguous`. The whole block is best-effort: any
        # failure leaves the existing (well-typed) default in place
        # rather than tanking the load.
        cur = getattr(target, field)
        if value is not None and cur is not None:
            # defensive: skip multi-element arrays — they shouldn't
            # reach a scalar field, but if they do, leaving the
            # default is safer than raising.
            if type(value).__name__ == "ndarray" and getattr(value, "size", 1) > 1:
                parent.logger.debug(
                    f"suite2p hydrate: skipping {field}: got ndarray "
                    f"size={value.size}, expected scalar"
                )
                continue
            try:
                if isinstance(cur, bool):
                    value = bool(value)
                elif isinstance(cur, int) and not isinstance(value, bool):
                    value = int(value)
                elif isinstance(cur, float):
                    value = float(value)
                elif isinstance(cur, str):
                    value = str(value)
            except Exception as e:
                parent.logger.debug(
                    f"suite2p hydrate: type-coerce skipped for {field}: {e}"
                )
                continue
        try:
            setattr(target, field, value)
            if target is s2p:
                n_settings += 1
            elif target is s2p_db:
                n_db += 1
            else:
                n_extras += 1
        except Exception as e:
            parent.logger.debug(
                f"suite2p hydrate: could not set {field}={value!r}: {e}"
            )

    # plane_dir is e.g. .../res/zplane01_tp00001-01574 — the ROOT is its
    # parent (.../res). re-running with output=ROOT recreates the same
    # plane subdir in place.
    parent._s2p_outdir = str(plane_dir.parent)

    parent.logger.info(
        f"suite2p hydrate: loaded {n_settings} settings + {n_db} db + "
        f"{n_extras} mbo-extras fields from {' + '.join(sources)}; "
        f"output dir set to {parent._s2p_outdir}"
        + (f" (skipped {len(skipped)} unmapped keys)" if skipped else "")
    )
    return True


def check_file_dialogs(parent: Any):
    """Check if file/folder dialogs have results and load data if so."""
    # Check file dialog
    if parent._file_dialog is not None and parent._file_dialog.ready():
        result = parent._file_dialog.result()
        parent.logger.info(f"File dialog result: {result}")
        if result and len(result) > 0:
            # Save to recent files and context-specific preferences
            add_recent_file(result[0], file_type="file")
            set_last_dir("open_file", result[0])
            load_new_data(parent, result[0])
        else:
            parent.logger.info("File dialog cancelled or empty result")
        parent._file_dialog = None

    # Check folder dialog
    if parent._folder_dialog is not None and parent._folder_dialog.ready():
        result = parent._folder_dialog.result()
        if result:
            # Save to recent files and context-specific preferences
            add_recent_file(result, file_type="folder")
            set_last_dir("open_folder", result)
            load_new_data(parent, result)
        parent._folder_dialog = None


def _reset_per_data_state(parent: Any) -> None:
    """Clear per-dataset display/widget state that must not carry across loads.

    Widget controls in the side panel bind to attributes on the parent
    (e.g. `parent.gaussian_sigma`), not on the widget instance. The
    widget objects themselves are reconstructed by `_refresh_widgets()`
    on every load, but the bound parent attributes persist unless
    explicitly cleared here. Without this, opening a new file leaves
    the previous file's gaussian sigma, projection mode, window size,
    auto-contrast toggle, and mean-subtraction checkbox in place.

    Mirror the defaults in `PreviewDataWidget._init_state` so a reload
    starts in the same state as initial launch.

    Lives as a standalone helper so the reset contract can be
    unit-tested without spinning up the full GUI module-load chain.
    """
    # spatial functions
    parent._mean_subtraction = False
    parent._gaussian_sigma = 0.0
    # window functions
    parent._proj = "mean"
    parent._window_size = 1
    # contrast / z-tracking
    parent._auto_contrast_on_z = False
    parent._last_z_idx = 0
    # save-as dialog selections
    parent._saveas_selected_roi = set()
    parent._saveas_rois = False


def load_new_data(parent: Any, path: str):
    """
    Load new data from the specified path using iw-array API.

    Uses iw.set_data() to swap data arrays, which handles shape changes.
    """
    from mbo_utilities.arrays import TiffArray
    from mbo_utilities.gui.run_gui import _SqueezeSingletonDims

    path_obj = Path(path)
    if not path_obj.exists():
        parent.logger.error(f"Path does not exist: {path}")
        parent._load_status_msg = "Error: Path does not exist"
        parent._load_status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
        return

    try:
        parent.logger.info(f"Loading data from: {path}")
        parent._load_status_msg = "Loading..."
        parent._load_status_color = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)

        parent.logger.debug(f"Calling imread on: {path}")
        raw_data = imread(path)
        parent.logger.debug(f"imread returned: type={type(raw_data).__name__}, shape={getattr(raw_data, 'shape', 'N/A')}")

        # Apply the same singleton-dim squeeze that _create_image_widget
        # uses on initial launch, so reload via file dialog produces the
        # same natural-rank view. Without this, a 5D Suite2pArray (T,1,Z,Y,X)
        # reload would expose Z at index 2 here, but at index 1 on initial
        # launch — different code paths giving different shapes for the
        # same file. Squeeze keeps them in lockstep.
        if hasattr(raw_data, "shape") and len(raw_data.shape) == 5 and any(
            raw_data.shape[i] == 1 for i in range(3)
        ):
            new_data = _SqueezeSingletonDims(raw_data)
        else:
            new_data = raw_data

        # Reset stale per-data display state via the standalone helper
        # so the reset contract can be tested in isolation.
        _reset_per_data_state(parent)

        # Drop stale closures *before* swapping data. The processor's
        # spatial_func is a closure that captured the previous dataset's
        # mean_img, so the next render after data[0] = new_data would crash
        # if the new shape differs. _reset_per_data_state already cleared
        # _mean_subtraction and _gaussian_sigma, so _rebuild_spatial_func
        # will install the identity passthrough. Window funcs/sizes get
        # cleared too since they're bound to the old t-rank.
        if hasattr(parent, "_rebuild_spatial_func"):
            parent._rebuild_spatial_func()
        if hasattr(parent, "image_widget") and parent.image_widget is not None:
            for proc in parent.image_widget._image_processors:
                proc.window_funcs = None
                proc.window_sizes = None
                proc.window_order = None

        # iw-array API: use data indexer for replacing data
        # data[0] = new_array triggers _reset_dimensions() automatically
        parent.image_widget.data[0] = new_data

        # reset indices to start of data using public API
        if parent.image_widget.n_sliders > 0:
            parent.image_widget.indices = [0] * parent.image_widget.n_sliders

        # Update internal state
        parent.fpath = path
        parent.shape = new_data.shape

        # Check if this is MBO data (ScanImage or volumetric TIFF).
        # Peel _SqueezeSingletonDims so the wrapper doesn't hide the
        # underlying class from isinstance.
        underlying = getattr(new_data, "_arr", new_data)
        parent.is_mbo_scan = (
            isinstance(underlying, ScanImageArray) or
            isinstance(underlying, TiffArray)
        )

        # If the loaded file is a suite2p binary, hydrate Suite2pSettings
        # / Suite2pDB from the sibling ops.npy and point _s2p_outdir at
        # the output ROOT so the user can re-run the exact same config
        # (with whatever tweaks) and land the new run alongside the old
        # one. Otherwise fall back to the generic "suggest output dir".
        first_path = path[0] if isinstance(path, (list, tuple)) else path
        if not _try_hydrate_s2p_from_binary(parent, first_path):
            if not parent._s2p_outdir:
                parent._s2p_outdir = str(Path(first_path).parent / "suite2p")

        # Update nz/nc for z-plane and channel counts
        if len(parent.shape) == 5:
            # TCZYX: shape[1]=C, shape[2]=Z
            parent.nc = parent.shape[1]
            parent.nz = parent.shape[2]
        elif len(parent.shape) == 4:
            parent.nz = parent.shape[1]
            parent.nc = 1
        else:
            parent.nz = 1
            parent.nc = 1

        parent._load_status_msg = f"Loaded: {path_obj.name}"
        parent._load_status_color = imgui.ImVec4(0.3, 1.0, 0.3, 1.0)
        parent.logger.info(f"Loaded successfully, shape: {new_data.shape}")
        parent.set_context_info()

        # update image widget subplot title with new filename
        try:
            base_name = path_obj.stem
            # for suite2p arrays (data.bin), use parent folder name
            if base_name in ("data", "data_raw"):
                base_name = path_obj.parent.name
            if len(base_name) > 24:
                base_name = base_name[:21] + "..."
            parent.image_widget.figure[0, 0].title = base_name
        except Exception:
            pass  # ignore if subplot title update fails

        # refresh widgets based on new data capabilities
        parent._refresh_widgets()

        # clear stale metadata overrides from previous file
        parent._custom_metadata = {}

        # Reinitialize viewer based on new data type (new architecture)
        from mbo_utilities.gui.viewers import get_viewer_class, TimeSeriesViewer
        if hasattr(parent, "_viewer") and parent._viewer:
            parent._viewer.cleanup()
        viewer_cls = get_viewer_class(new_data)
        parent._viewer = viewer_cls(parent.image_widget, parent.fpath, parent=parent)
        parent._viewer.on_data_loaded()
        parent.logger.info(f"Viewer switched to: {parent._viewer.name}")

        # Automatically recompute z-stats for new data (only for time series)
        if isinstance(parent._viewer, TimeSeriesViewer):
            parent.refresh_zstats()

    except Exception as e:
        parent.logger.exception(f"Error loading data: {e}")
        parent._load_status_msg = f"Error: {e!s}"
        parent._load_status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
