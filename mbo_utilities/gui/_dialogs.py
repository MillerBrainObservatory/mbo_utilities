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


def load_new_data(parent: Any, path: str):
    """
    Load new data from the specified path using iw-array API.

    Uses iw.set_data() to swap data arrays, which handles shape changes.
    """
    from mbo_utilities.arrays import TiffArray

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
        new_data = imread(path)
        parent.logger.debug(f"imread returned: type={type(new_data).__name__}, shape={getattr(new_data, 'shape', 'N/A')}")

        # Check if dimensionality is changing - if so, reset window functions
        # to avoid IndexError in fastplotlib's _apply_window_function
        old_ndim = 0
        if hasattr(parent, "shape") and parent.shape and isinstance(parent.shape, tuple):
            old_ndim = len(parent.shape)
        new_ndim = new_data.ndim

        # Reset window functions if dimensionality changes
        if old_ndim != new_ndim:
            parent.image_widget.window_funcs = None

        # iw-array API: use data indexer for replacing data
        # data[0] = new_array triggers _reset_dimensions() automatically
        parent.image_widget.data[0] = new_data

        # reset indices to start of data
        if parent.image_widget.slider_dims:
            parent.image_widget.current_index = {d: 0 for d in parent.image_widget.slider_dims}

        # Update internal state
        parent.fpath = path
        parent.shape = new_data.shape

        # Check if this is MBO data (ScanImage or volumetric TIFF)
        parent.is_mbo_scan = (
            isinstance(new_data, ScanImageArray) or
            isinstance(new_data, TiffArray)
        )

        # Suggest s2p output directory if not set
        if not parent._s2p_outdir:
            path_obj = Path(path[0] if isinstance(path, (list, tuple)) else path)
            parent._s2p_outdir = str(path_obj.parent / "suite2p")

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

        # Reset save dialog state for new data
        parent._saveas_selected_roi = set()
        parent._saveas_rois = False

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

        # Reinitialize viewer based on new data type (new architecture)
        from mbo_utilities.gui.viewers import get_viewer_class, TimeSeriesViewer
        if hasattr(parent, "_viewer") and parent._viewer:
            parent._viewer.cleanup()
        viewer_cls = get_viewer_class(new_data)
        parent._viewer = viewer_cls(parent.image_widget, parent.fpath, parent=parent)
        parent._viewer.on_data_loaded()
        parent.logger.info(f"Viewer switched to: {parent._viewer.name}")

        # Keep _main_widget reference for backwards compatibility
        parent._main_widget = parent._viewer._main_widget

        # Automatically recompute z-stats for new data (only for time series)
        if isinstance(parent._viewer, TimeSeriesViewer):
            parent.refresh_zstats()

        # Automatically reset vmin/vmax for initial view of new data
        if parent.image_widget:
            parent.image_widget.reset_vmin_vmax_frame()

    except Exception as e:
        parent.logger.exception(f"Error loading data: {e}")
        parent._load_status_msg = f"Error: {e!s}"
        parent._load_status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
