import logging
import webbrowser
from pathlib import Path
from typing import Literal
import threading
import os
import importlib.util

# Force rendercanvas to use Qt backend if PySide6 is available
# This must happen BEFORE importing fastplotlib to avoid glfw selection
# Note: rendercanvas.qt requires PySide6 to be IMPORTED, not just available
if importlib.util.find_spec("PySide6") is not None:
    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
    import PySide6  # noqa: F401 - Must be imported before rendercanvas.qt can load

    # Fix suite2p PySide6 compatibility - must happen before any suite2p GUI imports
    # suite2p's RangeSlider uses self.NoTicks which doesn't exist in PySide6
    from PySide6.QtWidgets import QSlider
    if not hasattr(QSlider, "NoTicks"):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks

import imgui_bundle
import numpy as np
from numpy import ndarray
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter

from imgui_bundle import (
    imgui,
    hello_imgui,
    imgui_ctx,
    implot,
    portable_file_dialogs as pfd,
)

from mbo_utilities.file_io import (
    get_mbo_dirs,
)
from mbo_utilities.reader import MBO_SUPPORTED_FTYPES
from mbo_utilities.preferences import (
    get_last_dir,
    set_last_dir,
    add_recent_file,
)
from mbo_utilities.array_types import MboRawArray
from mbo_utilities.graphics._imgui import (
    begin_popup_size,
    ndim_to_frame,
    style_seaborn_dark,
)
from mbo_utilities.graphics._widgets import (
    set_tooltip,
    checkbox_with_tooltip,
    draw_scope,
)
from mbo_utilities.graphics.progress_bar import (
    draw_status_indicator,
    reset_progress_state,
    start_output_capture,
)
from mbo_utilities.graphics.widgets import get_supported_widgets, draw_all_widgets
from mbo_utilities.graphics._availability import HAS_SUITE2P, HAS_SUITE3D
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities.arrays import _sanitize_suffix
from mbo_utilities.graphics.gui_logger import GuiLogger, GuiLogHandler
from mbo_utilities import log

try:
    import cupy as cp  # noqa
    from cusignal import (
        register_translation,
    )  # GPU version of phase_cross_correlation # noqa

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    register_translation = phase_cross_correlation  # noqa

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

REGION_TYPES = ["Full FOV", "Sub-FOV"]


def _save_as_worker(path, **imwrite_kwargs):
    # Don't pass roi to imread - let it load all ROIs
    # Then imwrite will handle splitting/filtering based on roi parameter
    data = imread(path)

    # Apply scan-phase correction settings to the array before writing
    # These must be set on the array object for MboRawArray phase correction
    fix_phase = imwrite_kwargs.pop("fix_phase", False)
    use_fft = imwrite_kwargs.pop("use_fft", False)
    phase_upsample = imwrite_kwargs.pop("phase_upsample", 10)
    border = imwrite_kwargs.pop("border", 10)
    mean_subtraction = imwrite_kwargs.pop("mean_subtraction", False)

    if hasattr(data, "fix_phase"):
        data.fix_phase = fix_phase
    if hasattr(data, "use_fft"):
        data.use_fft = use_fft
    if hasattr(data, "phase_upsample"):
        data.phase_upsample = phase_upsample
    if hasattr(data, "border"):
        data.border = border
    if hasattr(data, "mean_subtraction"):
        data.mean_subtraction = mean_subtraction

    imwrite(data, **imwrite_kwargs)


def draw_menu(parent):
    # (accessible from the "Tools" menu)
    if parent.show_scope_window:
        size = begin_popup_size()
        imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)  # type: ignore # noqa
        _, parent.show_scope_window = imgui.begin(
            "Scope Inspector",
            parent.show_scope_window,
        )
        draw_scope()
        imgui.end()
    if parent.show_debug_panel:
        size = begin_popup_size()
        imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)  # type: ignore # noqa
        opened, _ = imgui.begin(
            "MBO Debug Panel",
            parent.show_debug_panel,
        )
        if opened:
            parent.debug_panel.draw()
        imgui.end()
    if parent.show_metadata_viewer:
        size = begin_popup_size()
        imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)
        _, parent.show_metadata_viewer = imgui.begin(
            "Metadata Viewer",
            parent.show_metadata_viewer,
        )
        if parent.image_widget and parent.image_widget.data:
            metadata = parent.image_widget.data[0].metadata
            from mbo_utilities.graphics._widgets import draw_metadata_inspector
            draw_metadata_inspector(metadata)
        else:
            imgui.text("No data loaded")
        imgui.end()
    with imgui_ctx.begin_child(
        "menu",
        window_flags=imgui.WindowFlags_.menu_bar,  # noqa,
        child_flags=imgui.ChildFlags_.auto_resize_y
        | imgui.ChildFlags_.always_auto_resize,
    ):
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File", True):
                # Open File - iw-array API
                if imgui.menu_item("Open File", "Ctrl+O", p_selected=False, enabled=True)[0]:
                    # Handle fpath being a list or a string
                    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
                    if fpath and Path(fpath).exists():
                        start_dir = str(Path(fpath).parent)
                    else:
                        # Use open_file context-specific preference
                        start_dir = str(get_last_dir("open_file") or Path.home())
                    parent._file_dialog = pfd.open_file(
                        "Select Data File",
                        start_dir
                    )
                # Open Folder - iw-array API
                if imgui.menu_item("Open Folder", "", p_selected=False, enabled=True)[0]:
                    # Handle fpath being a list or a string
                    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
                    if fpath and Path(fpath).exists():
                        start_dir = str(Path(fpath).parent)
                    else:
                        # Use open_folder context-specific preference
                        start_dir = str(get_last_dir("open_folder") or Path.home())
                    parent._folder_dialog = pfd.select_folder("Select Data Folder", start_dir)
                imgui.separator()
                # Check if current data supports imwrite
                can_save = parent.is_mbo_scan
                if parent.image_widget and parent.image_widget.data:
                    arr = parent.image_widget.data[0]
                    can_save = hasattr(arr, "_imwrite")
                if imgui.menu_item(
                    "Save as", "Ctrl+S", p_selected=False, enabled=can_save
                )[0]:
                    parent._saveas_popup_open = True
                if not can_save and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    imgui.begin_tooltip()
                    arr_type = type(parent.image_widget.data[0]).__name__ if parent.image_widget and parent.image_widget.data else "Unknown"
                    imgui.text(f"{arr_type} does not support saving.")
                    imgui.end_tooltip()
                imgui.end_menu()
            if imgui.begin_menu("Docs", True):
                if imgui.menu_item(
                    "Open Docs", "Ctrl+I", p_selected=False, enabled=True
                )[0]:
                    webbrowser.open(
                        "https://millerbrainobservatory.github.io/mbo_utilities/"
                    )
                if imgui.menu_item(
                    "Download User Guide Notebook", "", p_selected=False, enabled=True
                )[0]:
                    webbrowser.open(
                        "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/demos/user_guide.ipynb"
                    )
                imgui.end_menu()
            if imgui.begin_menu("Settings", True):
                imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Tools")
                imgui.separator()
                imgui.spacing()
                _, parent.show_debug_panel = imgui.menu_item(
                    "Debug Panel",
                    "",
                    p_selected=parent.show_debug_panel,
                    enabled=True,
                )
                _, parent.show_scope_window = imgui.menu_item(
                    "Scope Inspector", "", parent.show_scope_window, True
                )
                imgui.spacing()
                imgui.separator()
                imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Display")
                imgui.separator()
                imgui.spacing()
                _, parent._show_progress_overlay = imgui.menu_item(
                    "Status Indicator", "", parent._show_progress_overlay, True
                )
                imgui.end_menu()
        imgui.end_menu_bar()

        # Draw status indicator below menu bar (in same child window)
        if parent._show_progress_overlay:
            draw_status_indicator(parent)


def draw_tabs(parent):
    # Don't create an outer child window - let each tab manage its own scrolling
    # For single z-plane data, show all tabs
    # For multi-zplane data, show all tabs (user wants all tabs visible)
    if imgui.begin_tab_bar("MainPreviewTabs"):
        if imgui.begin_tab_item("Preview")[0]:
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

            # Add metadata button at top of Preview tab
            imgui.spacing()
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.0, 0.0, 0.0, 1.0))  # Black button
            imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White border
            imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.0)
            if imgui.button("Show Metadata"):
                parent.show_metadata_viewer = not parent.show_metadata_viewer
            imgui.pop_style_var()
            imgui.pop_style_color(2)
            imgui.spacing()

            parent.draw_preview_section()
            imgui.pop_style_var()
            imgui.pop_style_var()
            imgui.end_tab_item()
        imgui.begin_disabled(not all(parent._zstats_done))
        if imgui.begin_tab_item("Summary Stats")[0]:
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
            # Create scrollable child for stats content
            with imgui_ctx.begin_child("##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
                parent.draw_stats_section()
            imgui.pop_style_var()
            imgui.pop_style_var()
            imgui.end_tab_item()
        imgui.end_disabled()
        # Run tab for processing pipelines
        if imgui.begin_tab_item("Run")[0]:
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
            from mbo_utilities.graphics.widgets.pipelines import draw_run_tab
            draw_run_tab(parent)
            imgui.pop_style_var()
            imgui.pop_style_var()
            imgui.end_tab_item()
        imgui.end_tab_bar()


def draw_saveas_popup(parent):
    if getattr(parent, "_saveas_popup_open"):
        imgui.open_popup("Save As")
        parent._saveas_popup_open = False

    if imgui.begin_popup_modal("Save As")[0]:
        imgui.dummy(imgui.ImVec2(0, 5))

        imgui.set_next_item_width(hello_imgui.em_size(25))

        # Directory + Ext
        current_dir_str = (
            str(Path(parent._saveas_outdir).expanduser().resolve())
            if parent._saveas_outdir
            else ""
        )

        # Track last known value to detect external changes (e.g., from Browse dialog)
        if not hasattr(parent, '_saveas_input_last_value'):
            parent._saveas_input_last_value = current_dir_str

        # Check if value changed externally (e.g., Browse dialog selected a new folder)
        # If so, we need to force imgui to update its internal buffer
        value_changed_externally = (parent._saveas_input_last_value != current_dir_str)
        if value_changed_externally:
            parent._saveas_input_last_value = current_dir_str

        # Use unique ID that changes when value updates externally to reset imgui's buffer
        input_id = f"Save Dir##{hash(current_dir_str) if value_changed_externally else 'stable'}"
        changed, new_str = imgui.input_text(input_id, current_dir_str)
        if changed:
            parent._saveas_outdir = new_str
            parent._saveas_input_last_value = new_str

        imgui.same_line()
        if imgui.button("Browse"):
            # Use save_as context-specific directory, fall back to home
            default_dir = parent._saveas_outdir or str(get_last_dir("save_as") or Path.home())
            parent._saveas_folder_dialog = pfd.select_folder("Select output folder", default_dir)

        # Check if async folder dialog has a result
        if parent._saveas_folder_dialog is not None and parent._saveas_folder_dialog.ready():
            result = parent._saveas_folder_dialog.result()
            if result:
                parent._saveas_outdir = str(result)
                set_last_dir("save_as", result)
            parent._saveas_folder_dialog = None

        imgui.set_next_item_width(hello_imgui.em_size(25))
        _, parent._ext_idx = imgui.combo("Ext", parent._ext_idx, MBO_SUPPORTED_FTYPES)
        parent._ext = MBO_SUPPORTED_FTYPES[parent._ext_idx]

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Options Section - Multi-ROI only for raw ScanImage data with multiple ROIs
        try:
            num_rois = parent.image_widget.data[0].num_rois
        except (AttributeError, Exception):
            num_rois = 1

        # Only show multi-ROI option if data actually has multiple ROIs
        if num_rois > 1:
            parent._saveas_rois = checkbox_with_tooltip(
                "Save ScanImage multi-ROI Separately",
                parent._saveas_rois,
                "Enable to save each mROI individually."
                " mROI's are saved to subfolders: plane1_roi1, plane1_roi2, etc."
                " These subfolders can be merged later using mbo_utilities.merge_rois()."
                " This can be helpful as often mROI's are non-contiguous and can drift in orthogonal directions over time.",
            )
            if parent._saveas_rois:
                imgui.spacing()
                imgui.separator()
                imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Choose mROI(s):")
                imgui.dummy(imgui.ImVec2(0, 5))

                if imgui.button("All##roi"):
                    parent._saveas_selected_roi = set(range(num_rois))
                imgui.same_line()
                if imgui.button("None##roi"):
                    parent._saveas_selected_roi = set()

                imgui.columns(2, borders=False)
                for i in range(num_rois):
                    imgui.push_id(f"roi_{i}")
                    selected = i in parent._saveas_selected_roi
                    _, selected = imgui.checkbox(f"mROI {i + 1}", selected)
                    if selected:
                        parent._saveas_selected_roi.add(i)
                    else:
                        parent._saveas_selected_roi.discard(i)
                    imgui.pop_id()
                    imgui.next_column()
                imgui.columns(1)
        else:
            # Reset multi-ROI state when not applicable
            parent._saveas_rois = False

        imgui.spacing()
        imgui.separator()

        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Options")
        set_tooltip(
            "Note: Current values for upsample and max-offset are applied during scan-phase correction.",
            True,
        )

        imgui.dummy(imgui.ImVec2(0, 5))

        parent._overwrite = checkbox_with_tooltip(
            "Overwrite", parent._overwrite, "Replace any existing output files."
        )
        # suite3d z-plane registration - show disabled with reason if unavailable
        can_register_z = HAS_SUITE3D and parent.nz > 1
        if not can_register_z:
            imgui.begin_disabled()
        _changed, _reg_value = imgui.checkbox(
            "Register Z-Planes Axially", parent._register_z if can_register_z else False
        )
        if can_register_z and _changed:
            parent._register_z = _reg_value
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            if not HAS_SUITE3D:
                imgui.text_unformatted("suite3d is not installed. Install with: pip install suite3d")
            elif parent.nz <= 1:
                imgui.text_unformatted("Requires multi-plane (4D) data with more than one z-plane.")
            else:
                imgui.text_unformatted("Register adjacent z-planes to each other using Suite3D.")
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        if not can_register_z:
            imgui.end_disabled()
        fix_phase_changed, fix_phase_value = imgui.checkbox(
            "Fix Scan Phase", parent.fix_phase
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted("Correct for bi-directional scan phase offsets.")
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        if fix_phase_changed:
            parent.fix_phase = fix_phase_value

        use_fft, use_fft_value = imgui.checkbox(
            "Subpixel Phase Correction", parent.use_fft
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted(
                "Use FFT-based subpixel registration (slower, more precise)."
            )
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        if use_fft:
            parent.use_fft = use_fft_value

        parent._debug = checkbox_with_tooltip(
            "Debug",
            parent._debug,
            "Print additional information to the terminal during process.",
        )

        imgui.spacing()
        imgui.text("Chunk Size (MB)")
        set_tooltip(
            "The size of the chunk, in MB, to read and write at a time. Larger chunks may be faster but use more memory.",
        )

        imgui.set_next_item_width(hello_imgui.em_size(20))
        _, parent._saveas_chunk_mb = imgui.drag_int(
            "##chunk_size_mb_mb",
            parent._saveas_chunk_mb,
            v_speed=1,
            v_min=1,
            v_max=1024,
        )

        # Output suffix section (only show for multi-ROI data when stitching)
        if num_rois > 1 and not parent._saveas_rois:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Filename Suffix")
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted(
                    "Custom suffix appended to output filenames.\n"
                    "Default: '_stitched' for stitched multi-ROI data.\n"
                    "Examples: '_stitched', '_processed', '_session1'\n\n"
                    "Illegal characters (<>:\"/\\|?*) are removed.\n"
                    "Underscore prefix is added if missing."
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()

            imgui.set_next_item_width(hello_imgui.em_size(15))
            changed, new_suffix = imgui.input_text(
                "##output_suffix",
                parent._saveas_output_suffix,
            )
            if changed:
                parent._saveas_output_suffix = new_suffix

            # Live filename preview
            sanitized = _sanitize_suffix(parent._saveas_output_suffix)
            preview_ext = parent._ext.lstrip(".")
            preview_name = f"plane01{sanitized}.{preview_ext}"
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), f"Preview: {preview_name}")

        # Format-specific options
        if parent._ext in (".zarr",):
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Zarr Options")
            imgui.dummy(imgui.ImVec2(0, 5))

            _, parent._zarr_sharded = imgui.checkbox("Sharded", parent._zarr_sharded)
            set_tooltip(
                "Use sharding to group multiple chunks into single files (100 frames/shard). "
                "Improves read/write performance for large datasets by reducing filesystem overhead.",
            )

            _, parent._zarr_ome = imgui.checkbox("OME-Zarr", parent._zarr_ome)
            set_tooltip(
                "Write OME-NGFF v0.5 metadata for compatibility with OME-Zarr viewers "
                "(napari, vizarr, etc). Includes multiscales, axes, and coordinate transforms.",
            )

            imgui.text("Compression Level")
            set_tooltip(
                "GZip compression level (0-9). Higher = smaller files, slower write. "
                "Level 1 is fast with decent compression. Level 0 disables compression.",
            )
            imgui.set_next_item_width(hello_imgui.em_size(10))
            _, parent._zarr_compression_level = imgui.slider_int(
                "##zarr_level", parent._zarr_compression_level, 0, 9
            )

        imgui.spacing()
        imgui.separator()

        # Metadata Section (collapsible)
        imgui.spacing()
        if imgui.collapsing_header("Metadata Overrides"):
            imgui.dummy(imgui.ImVec2(0, 5))

            # Get current metadata from data for showing auto-detected values
            try:
                current_meta = parent.image_widget.data[0].metadata or {}
            except (IndexError, AttributeError):
                current_meta = {}

            # Frame Rate
            auto_fs = current_meta.get("frame_rate") or current_meta.get("fs") or ""
            imgui.text("Frame Rate (Hz)")
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted(
                    f"Acquisition frame rate in Hz.\n"
                    f"Auto-detected: {auto_fs if auto_fs else 'Not found'}\n"
                    f"Aliases: fs, frame_rate"
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
            imgui.set_next_item_width(hello_imgui.em_size(12))
            _, parent._saveas_fs = imgui.input_text(
                "##fs_input",
                parent._saveas_fs,
                flags=imgui.InputTextFlags_.chars_decimal,
            )
            imgui.same_line()
            imgui.text_disabled(f"(auto: {auto_fs})" if auto_fs else "(not detected)")

            # Pixel Resolution X
            auto_dx = ""
            pixel_res = current_meta.get("pixel_resolution")
            if pixel_res and isinstance(pixel_res, (list, tuple)) and len(pixel_res) >= 1:
                auto_dx = pixel_res[0]
            elif current_meta.get("dx"):
                auto_dx = current_meta.get("dx")
            imgui.text("Pixel Size X (µm)")
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted(
                    f"X pixel size in micrometers.\n"
                    f"Auto-detected: {auto_dx if auto_dx else 'Not found'}\n"
                    f"Aliases: dx, umPerPixX, PhysicalSizeX, pixel_resolution[0]"
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
            imgui.set_next_item_width(hello_imgui.em_size(12))
            _, parent._saveas_dx = imgui.input_text(
                "##dx_input",
                parent._saveas_dx,
                flags=imgui.InputTextFlags_.chars_decimal,
            )
            imgui.same_line()
            imgui.text_disabled(f"(auto: {auto_dx})" if auto_dx else "(not detected)")

            # Pixel Resolution Y
            auto_dy = ""
            if pixel_res and isinstance(pixel_res, (list, tuple)) and len(pixel_res) >= 2:
                auto_dy = pixel_res[1]
            elif current_meta.get("dy"):
                auto_dy = current_meta.get("dy")
            imgui.text("Pixel Size Y (µm)")
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted(
                    f"Y pixel size in micrometers.\n"
                    f"Auto-detected: {auto_dy if auto_dy else 'Not found'}\n"
                    f"Aliases: dy, umPerPixY, PhysicalSizeY, pixel_resolution[1]"
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
            imgui.set_next_item_width(hello_imgui.em_size(12))
            _, parent._saveas_dy = imgui.input_text(
                "##dy_input",
                parent._saveas_dy,
                flags=imgui.InputTextFlags_.chars_decimal,
            )
            imgui.same_line()
            imgui.text_disabled(f"(auto: {auto_dy})" if auto_dy else "(not detected)")

            # Z Step
            auto_dz = current_meta.get("dz") or current_meta.get("z_step") or ""
            imgui.text("Z Step (µm)")
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted(
                    f"Distance between z-planes in micrometers.\n"
                    f"Auto-detected: {auto_dz if auto_dz else 'Not found'}\n"
                    f"Aliases: dz, z_step, umPerPixZ, PhysicalSizeZ, spacing"
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
            imgui.set_next_item_width(hello_imgui.em_size(12))
            _, parent._saveas_dz = imgui.input_text(
                "##dz_input",
                parent._saveas_dz,
                flags=imgui.InputTextFlags_.chars_decimal,
            )
            imgui.same_line()
            imgui.text_disabled(f"(auto: {auto_dz})" if auto_dz else "(not detected)")

            imgui.spacing()
            imgui.separator()

            # Custom metadata key-value pairs
            imgui.text("Custom Metadata")
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted(
                    "Add custom key-value pairs to the output metadata.\n"
                    "Values will be stored as strings unless they parse as numbers."
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()

            imgui.dummy(imgui.ImVec2(0, 3))

            # Show existing custom metadata entries
            to_remove = None
            for key, value in list(parent._saveas_custom_metadata.items()):
                imgui.push_id(f"custom_{key}")
                imgui.text(f"  {key}:")
                imgui.same_line()
                imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), str(value))
                imgui.same_line()
                if imgui.small_button("X"):
                    to_remove = key
                imgui.pop_id()
            if to_remove:
                del parent._saveas_custom_metadata[to_remove]

            # Add new key-value pair
            imgui.set_next_item_width(hello_imgui.em_size(8))
            _, parent._saveas_custom_key = imgui.input_text(
                "##custom_key", parent._saveas_custom_key
            )
            imgui.same_line()
            imgui.text("=")
            imgui.same_line()
            imgui.set_next_item_width(hello_imgui.em_size(10))
            _, parent._saveas_custom_value = imgui.input_text(
                "##custom_value", parent._saveas_custom_value
            )
            imgui.same_line()
            if imgui.button("Add"):
                if parent._saveas_custom_key.strip():
                    # Try to parse value as number
                    val = parent._saveas_custom_value
                    try:
                        if "." in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass  # Keep as string
                    parent._saveas_custom_metadata[parent._saveas_custom_key.strip()] = val
                    parent._saveas_custom_key = ""
                    parent._saveas_custom_value = ""

            imgui.spacing()

        imgui.spacing()
        imgui.separator()

        # Num frames slider
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Frames")
        imgui.dummy(imgui.ImVec2(0, 5))

        # Get max frames from data
        try:
            first_array = parent.image_widget.data[0]
            max_frames = first_array.shape[0]
        except (IndexError, AttributeError):
            max_frames = 1000

        # Initialize num_frames if not set or if max changed
        if not hasattr(parent, '_saveas_num_frames') or parent._saveas_num_frames is None:
            parent._saveas_num_frames = max_frames
        if not hasattr(parent, '_saveas_last_max_frames'):
            parent._saveas_last_max_frames = max_frames
        elif parent._saveas_last_max_frames != max_frames:
            parent._saveas_last_max_frames = max_frames
            parent._saveas_num_frames = max_frames

        imgui.set_next_item_width(hello_imgui.em_size(8))
        changed, new_value = imgui.input_int("##frames_input", parent._saveas_num_frames, step=1, step_fast=100)
        if changed:
            parent._saveas_num_frames = max(1, min(new_value, max_frames))
        imgui.same_line()
        imgui.text(f"/ {max_frames}")
        set_tooltip(
            f"Number of frames to save (1-{max_frames}). "
            "Useful for testing on subsets before full conversion."
        )

        imgui.set_next_item_width(hello_imgui.em_size(20))
        slider_changed, slider_value = imgui.slider_int(
            "##frames_slider", parent._saveas_num_frames, 1, max_frames
        )
        if slider_changed:
            parent._saveas_num_frames = slider_value

        imgui.spacing()
        imgui.separator()

        # Z-plane selection
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Choose z-planes:")
        imgui.dummy(imgui.ImVec2(0, 5))

        try:
            data = parent.image_widget.data[0]
            # Try various attributes for plane count
            if hasattr(data, "num_planes"):
                num_planes = data.num_planes
            elif hasattr(data, "num_channels"):
                num_planes = data.num_channels
            elif len(data.shape) == 4:
                # 4D array: shape is (T, Z, Y, X)
                num_planes = data.shape[1]
            else:
                num_planes = 1
        except Exception as e:
            num_planes = 1
            hello_imgui.log(
                hello_imgui.LogLevel.error,
                f"Could not read number of planes: {e}",
            )

        # Auto-select current z-plane if none selected
        if not parent._selected_planes:
            # Get current z index from image widget
            names = parent.image_widget._slider_dim_names or ()
            try:
                current_z = parent.image_widget.indices["z"] if "z" in names else 0
            except (IndexError, KeyError):
                current_z = 0
            parent._selected_planes = {current_z}

        if imgui.button("All"):
            parent._selected_planes = set(range(num_planes))
        imgui.same_line()
        if imgui.button("None"):
            parent._selected_planes = set()
        imgui.same_line()
        if imgui.button("Current"):
            names = parent.image_widget._slider_dim_names or ()
            try:
                current_z = parent.image_widget.indices["z"] if "z" in names else 0
            except (IndexError, KeyError):
                current_z = 0
            parent._selected_planes = {current_z}

        imgui.columns(2, borders=False)
        for i in range(num_planes):
            imgui.push_id(i)
            selected = i in parent._selected_planes
            _, selected = imgui.checkbox(f"Plane {i + 1}", selected)
            if selected:
                parent._selected_planes.add(i)
            else:
                parent._selected_planes.discard(i)
            imgui.pop_id()
            imgui.next_column()
        imgui.columns(1)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.button("Save", imgui.ImVec2(100, 0)):
            if not parent._saveas_outdir:
                last_dir = get_last_dir("save_as") or Path().home()
                parent._saveas_outdir = str(last_dir)
            try:
                save_planes = [p + 1 for p in parent._selected_planes]

                # Validate that at least one plane is selected
                if not save_planes:
                    parent.logger.error("No z-planes selected! Please select at least one plane.")
                else:
                    parent._saveas_total = len(save_planes)
                    if parent._saveas_rois:
                        if (
                            not parent._saveas_selected_roi
                            or len(parent._saveas_selected_roi) == set()
                        ):
                            # Get mROI count from data array (ScanImage-specific)
                            try:
                                mroi_count = parent.image_widget.data[0].num_rois
                            except Exception:
                                mroi_count = 1
                            parent._saveas_selected_roi = set(range(mroi_count))
                        # Convert 0-indexed UI values to 1-indexed ROI values for MboRawArray
                        rois = sorted([r + 1 for r in parent._saveas_selected_roi])
                    else:
                        rois = None

                    outdir = Path(parent._saveas_outdir).expanduser()
                    if not outdir.exists():
                        outdir.mkdir(parents=True, exist_ok=True)

                    # Get num_frames (None means all frames)
                    num_frames = getattr(parent, '_saveas_num_frames', None)
                    try:
                        max_frames = parent.image_widget.data[0].shape[0]
                        if num_frames is not None and num_frames >= max_frames:
                            num_frames = None  # All frames, don't limit
                    except (IndexError, AttributeError):
                        pass

                    # Build metadata overrides dict
                    metadata_overrides = dict(parent._saveas_custom_metadata)
                    # Add resolution overrides if provided (non-empty strings)
                    if parent._saveas_fs.strip():
                        try:
                            fs_val = float(parent._saveas_fs)
                            metadata_overrides["fs"] = fs_val
                            metadata_overrides["frame_rate"] = fs_val
                        except ValueError:
                            pass
                    if parent._saveas_dx.strip():
                        try:
                            metadata_overrides["dx"] = float(parent._saveas_dx)
                        except ValueError:
                            pass
                    if parent._saveas_dy.strip():
                        try:
                            metadata_overrides["dy"] = float(parent._saveas_dy)
                        except ValueError:
                            pass
                    if parent._saveas_dz.strip():
                        try:
                            metadata_overrides["dz"] = float(parent._saveas_dz)
                        except ValueError:
                            pass

                    # Determine output_suffix: only use custom suffix for multi-ROI stitched data
                    output_suffix = None
                    if rois is None:
                        # Stitching all ROIs - use custom suffix (or default "_stitched")
                        output_suffix = parent._saveas_output_suffix

                    save_kwargs = {
                        "path": parent.fpath,
                        "outpath": parent._saveas_outdir,
                        "planes": save_planes,
                        "roi": rois,
                        "overwrite": parent._overwrite,
                        "debug": parent._debug,
                        "ext": parent._ext,
                        "target_chunk_mb": parent._saveas_chunk_mb,
                        "num_frames": num_frames,
                        # scan-phase correction settings
                        "fix_phase": parent.fix_phase,
                        "use_fft": parent.use_fft,
                        "phase_upsample": parent.phase_upsample,
                        "border": parent.border,
                        "register_z": parent._register_z,
                        "mean_subtraction": parent.mean_subtraction,
                        "progress_callback": lambda frac,
                        current_plane: parent.gui_progress_callback(frac, current_plane),
                        # metadata overrides
                        "metadata": metadata_overrides if metadata_overrides else None,
                        # filename suffix
                        "output_suffix": output_suffix,
                    }
                    # Add zarr-specific options if saving to zarr
                    if parent._ext == ".zarr":
                        save_kwargs["sharded"] = parent._zarr_sharded
                        save_kwargs["ome"] = parent._zarr_ome
                        save_kwargs["level"] = parent._zarr_compression_level
                    frames_msg = f"{num_frames} frames" if num_frames else "all frames"
                    parent.logger.info(f"Saving planes {save_planes} ({frames_msg}) with ROIs {rois if rois else 'stitched'}")
                    parent.logger.info(
                        f"Saving to {parent._saveas_outdir} as {parent._ext}"
                    )
                    # Reset progress state to allow new progress display
                    reset_progress_state("saveas")
                    parent._saveas_progress = 0.0
                    parent._saveas_done = False
                    # Also reset register_z progress if enabled
                    if parent._register_z:
                        reset_progress_state("register_z")
                        parent._register_z_progress = 0.0
                        parent._register_z_done = False
                        parent._register_z_current_msg = ""
                    threading.Thread(
                        target=_save_as_worker, kwargs=save_kwargs, daemon=True
                    ).start()
                imgui.close_current_popup()
            except Exception as e:
                parent.logger.info(f"Error saving data: {e}")
                imgui.close_current_popup()

        imgui.same_line()
        if imgui.button("Cancel"):
            imgui.close_current_popup()

        imgui.end_popup()


class PreviewDataWidget(EdgeWindow):
    def __init__(
        self,
        iw: fpl.ImageWidget,
        fpath: str | None | list = None,
        threading_enabled: bool = True,
        size: int = None,
        location: Literal["bottom", "right"] = "right",
        title: str = "Data Preview",
        show_title: bool = False,
        movable: bool = False,
        resizable: bool = False,
        scrollable: bool = False,
        auto_resize: bool = True,
        window_flags: int | None = None,
        **kwargs,
    ):
        flags = (
            (imgui.WindowFlags_.no_title_bar if not show_title else 0)
            | (imgui.WindowFlags_.no_move if not movable else 0)
            | (imgui.WindowFlags_.no_resize if not resizable else 0)
            | (imgui.WindowFlags_.no_scrollbar if not scrollable else 0)
            | (imgui.WindowFlags_.always_auto_resize if auto_resize else 0)
            | (window_flags or 0)
        )
        super().__init__(
            figure=iw.figure,
            size=250 if size is None else size,
            location=location,
            title=title,
            window_flags=flags,
        )

        # logger / debugger
        self.debug_panel = GuiLogger()
        gui_handler = GuiLogHandler(self.debug_panel)
        gui_handler.setFormatter(logging.Formatter("%(message)s"))
        gui_handler.setLevel(logging.DEBUG)
        log.attach(gui_handler)

        # Also add console handler so logs appear in terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))

        # Only show DEBUG logs if MBO_DEBUG is set
        import os
        if bool(int(os.getenv("MBO_DEBUG", "0"))):
            console_handler.setLevel(logging.DEBUG)
            log.set_global_level(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
            log.set_global_level(logging.INFO)

        log.attach(console_handler)
        self.logger = log.get("gui")

        self.logger.info("Logger initialized.")

        # Start capturing stdout/stderr for the console output popup
        start_output_capture()

        # Only initialize Suite2p settings if suite2p is installed
        if HAS_SUITE2P:
            from mbo_utilities.graphics.pipeline_widgets import Suite2pSettings
            self.s2p = Suite2pSettings()
        else:
            self.s2p = None
        self._s2p_dir = ""
        self._s2p_savepath_flash_start = None  # Track when flash animation starts
        self._s2p_savepath_flash_count = 0  # Number of flashes
        self._s2p_show_savepath_popup = False  # Show popup when save path is missing
        self._s2p_folder_dialog = None  # Async folder dialog for Run tab Browse button
        self.kwargs = kwargs

        if implot.get_current_context() is None:
            implot.create_context()

        io = imgui.get_io()
        font_config = imgui.ImFontConfig()
        font_config.merge_mode = True

        fd_settings_dir = (
            Path(get_mbo_dirs()["imgui"])
            .joinpath("assets", "app_settings", "preview_settings.ini")
            .expanduser()
            .resolve()
        )
        io.set_ini_filename(str(fd_settings_dir))

        sans_serif_font = str(
            Path(imgui_bundle.__file__).parent.joinpath(
                "assets", "fonts", "Roboto", "Roboto-Regular.ttf"
            )
        )

        self._default_imgui_font = io.fonts.add_font_from_file_ttf(
            sans_serif_font, 14, imgui.ImFontConfig()
        )

        imgui.push_font(self._default_imgui_font, self._default_imgui_font.legacy_size)

        self.fpath = fpath if fpath else getattr(iw, "fpath", None)

        # image widget setup
        self.image_widget = iw

        # Unified naming: num_graphics matches len(iw.graphics)
        self.num_graphics = len(self.image_widget.graphics)
        self.shape = self.image_widget.data[0].shape
        self.is_mbo_scan = (
            True if isinstance(self.image_widget.data[0], MboRawArray) else False
        )
        self.logger.info(f"Data type: {type(self.image_widget.data[0]).__name__}, is_mbo_scan: {self.is_mbo_scan}")

        # Only set if not already configured - the ImageWidget/processor handles defaults
        # We just need to track the window size for our UI
        self._window_size = 1
        self._gaussian_sigma = 0.0  # Track gaussian sigma locally, applied via spatial_func

        if len(self.shape) == 4:
            self.nz = self.shape[1]
        elif len(self.shape) == 3:
            self.nz = 1
        else:
            self.nz = 1

        for subplot in self.image_widget.figure:
            subplot.toolbar = False
        self.image_widget._sliders_ui._loop = True  # noqa

        self._zstats = [
            {"mean": [], "std": [], "snr": []} for _ in range(self.num_graphics)
        ]
        self._zstats_means = [None] * self.num_graphics
        self._zstats_mean_scalar = [0.0] * self.num_graphics
        self._zstats_done = [False] * self.num_graphics
        self._zstats_progress = [0.0] * self.num_graphics
        self._zstats_current_z = [0] * self.num_graphics

        # Settings menu flags
        self.show_debug_panel = False
        self.show_scope_window = False
        self.show_metadata_viewer = False
        self.show_diagnostics_window = False
        self._diagnostics_widget = None  # Lazy-loaded diagnostics widget
        self._show_progress_overlay = True  # Global progress overlay (bottom-right)

        # Processing properties are now on the processor, not the widget
        # We just track UI state here
        self._auto_update = False
        self._proj = "mean"
        self._mean_subtraction = False
        self._last_z_idx = 0  # track z-index for mean subtraction updates

        self._register_z = False
        self._register_z_progress = 0.0
        self._register_z_done = False
        self._register_z_current_msg = ""

        self._selected_pipelines = None
        self._selected_array = 0
        self._selected_planes = set()
        self._planes_str = str(getattr(self, "_planes_str", ""))

        # properties for saving to another filetype
        self._ext = str(getattr(self, "_ext", ".tiff"))
        self._ext_idx = MBO_SUPPORTED_FTYPES.index(".tiff")

        self._overwrite = True
        self._debug = False

        self._saveas_chunk_mb = 100

        # zarr-specific options
        self._zarr_sharded = True
        self._zarr_ome = True
        self._zarr_compression_level = 1

        self._saveas_popup_open = False
        self._saveas_done = False
        self._saveas_progress = 0.0
        self._saveas_current_index = 0
        # pre-fill with context-specific saved directory if available
        save_as_dir = get_last_dir("save_as")
        self._saveas_outdir = (
            str(save_as_dir) if save_as_dir else str(getattr(self, "_save_dir", ""))
        )
        # suite2p output path (separate from save-as)
        s2p_output_dir = get_last_dir("suite2p_output")
        self._s2p_outdir = str(s2p_output_dir) if s2p_output_dir else ""
        self._saveas_folder_dialog = None  # async folder dialog for browse button
        self._saveas_total = 0

        self._saveas_selected_roi = set()  # -1 means all ROIs
        self._saveas_rois = False
        self._saveas_selected_roi_mode = "All"

        # Metadata override state for save dialog
        self._saveas_metadata_expanded = False
        self._saveas_custom_metadata = {}  # user-added key-value pairs
        self._saveas_custom_key = ""  # temp input for new key
        self._saveas_custom_value = ""  # temp input for new value
        # Resolution overrides (empty string = use auto-detected value)
        self._saveas_fs = ""  # frame rate override
        self._saveas_dx = ""  # x pixel resolution override
        self._saveas_dy = ""  # y pixel resolution override
        self._saveas_dz = ""  # z step override

        # Output suffix for filename customization (default: "_stitched" for multi-ROI)
        self._saveas_output_suffix = "_stitched"

        # File/folder dialog state for loading new data (iw-array API)
        self._file_dialog = None
        self._folder_dialog = None
        self._load_status_msg = ""
        self._load_status_color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)

        # initialize widgets based on data capabilities
        self._widgets = get_supported_widgets(self)

        self.set_context_info()

        if threading_enabled:
            self.logger.info("Starting zstats computation in a separate thread.")
            threading.Thread(target=self.compute_zstats, daemon=True).start()

    def set_context_info(self):
        if self.fpath is None:
            title = "Test Data"
        elif isinstance(self.fpath, list):
            title = f"{[Path(f).stem for f in self.fpath]}"
        else:
            title = f"Filepath: {Path(self.fpath).stem}"
        self.image_widget.figure.canvas.set_title(str(title))

    def _refresh_image_widget(self):
        """
        Trigger a frame refresh on the ImageWidget.

        Forces the widget to re-render the current frame by re-setting indices,
        which causes the processor's get() method to be called again.
        """
        # Force refresh by re-assigning current indices
        # This triggers the indices setter which calls processor.get() for each graphic
        current_indices = list(self.image_widget.indices)
        self.image_widget.indices = current_indices

    def _set_processor_attr(self, attr: str, value):
        """
        Set processor attribute without expensive histogram recomputation.

        Uses the proper fastplotlib ImageWidget API but temporarily disables
        histogram computation to avoid expensive full-array reads on lazy arrays.

        Parameters
        ----------
        attr : str
            Attribute name to set (window_funcs, window_sizes, spatial_func)
        value
            Value to set. Applied to all processors via ImageWidget API.
        """
        if not self.processors:
            self.logger.warning(f"No processors available to set {attr}")
            return

        # Save original compute_histogram states and disable on all processors
        original_states = []
        for proc in self.processors:
            original_states.append(proc._compute_histogram)
            proc._compute_histogram = False

        try:
            # Use proper ImageWidget API - this handles validation and index refresh
            # The histogram recomputation is skipped because _compute_histogram is False
            if attr == "window_funcs":
                # ImageWidget.window_funcs expects a list with one tuple per processor
                # Always wrap in a list, even for single processor
                if isinstance(value, tuple):
                    # Single tuple like (mean_wrapper, None) - wrap for all processors
                    value = [value] * len(self.processors)
                self.logger.debug(f"Setting window_funcs: {value}, n_processors={len(self.processors)}")
                self.logger.debug(f"Processor n_slider_dims: {[p.n_slider_dims for p in self.processors]}")
                self.image_widget.window_funcs = value
            elif attr == "window_sizes":
                # ImageWidget expects a list with one entry per processor
                if isinstance(value, (tuple, list)) and not isinstance(value[0], (tuple, list, type(None))):
                    # Single tuple like (5, None) - wrap for all processors
                    value = [value] * len(self.processors)
                self.image_widget.window_sizes = value
            elif attr == "spatial_func":
                self.image_widget.spatial_func = value
            else:
                # Fallback for other attributes
                for proc in self.processors:
                    setattr(proc, attr, value)
                self._refresh_image_widget()
        except Exception as e:
            self.logger.error(f"Error setting {attr}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Restore original compute_histogram states
            for proc, orig in zip(self.processors, original_states):
                proc._compute_histogram = orig

        try:
            self.image_widget.reset_vmin_vmax_frame()
        except Exception as e:
            self.logger.warning(f"Could not reset vmin/vmax: {e}")

    def _refresh_widgets(self):
        """
        refresh widgets based on current data capabilities.

        call this after loading new data to update which widgets are shown.
        """
        self._widgets = get_supported_widgets(self)
        self.logger.debug(
            f"refreshed widgets: {[w.name for w in self._widgets]}"
        )

    def gui_progress_callback(self, frac, meta=None):
        """
        Handles both saving progress (z-plane) and Suite3D registration progress.
        The `meta` parameter may be a plane index (int) or message (str).
        """
        if isinstance(meta, (int, np.integer)):
            # This is standard save progress
            self._saveas_progress = frac
            self._saveas_current_index = meta
            self._saveas_done = frac >= 1.0

        elif isinstance(meta, str):
            # Suite3D progress message
            self._register_z_progress = frac
            self._register_z_current_msg = meta
            self._register_z_done = frac >= 1.0

    @property
    def s2p_dir(self):
        return self._s2p_dir

    @s2p_dir.setter
    def s2p_dir(self, value):
        self.logger.info(f"Setting Suite2p directory to {value}")
        self._s2p_dir = value

    @property
    def register_z(self):
        return self._register_z

    @register_z.setter
    def register_z(self, value):
        self._register_z = value

    @property
    def processors(self) -> list:
        """Access to underlying NDImageProcessor instances."""
        return self.image_widget._image_processors

    def _get_data_arrays(self) -> list:
        """Get underlying data arrays from image processors."""
        return [proc.data for proc in self.processors]

    @property
    def current_offset(self) -> list[float]:
        """Get current phase offset from each data array (MboRawArray)."""
        offsets = []
        for arr in self._get_data_arrays():
            if hasattr(arr, 'offset'):
                arr_offset = arr.offset
                if isinstance(arr_offset, np.ndarray):
                    offsets.append(float(arr_offset.mean()) if arr_offset.size > 0 else 0.0)
                else:
                    offsets.append(float(arr_offset) if arr_offset else 0.0)
            else:
                offsets.append(0.0)
        return offsets

    @property
    def has_raster_scan_support(self) -> bool:
        """Check if any data array supports raster scan phase correction."""
        arrays = self._get_data_arrays()
        self.logger.debug(f"Checking raster scan support for {len(arrays)} arrays")
        for arr in arrays:
            self.logger.debug(f"  Array type: {type(arr).__name__}, has fix_phase: {hasattr(arr, 'fix_phase')}, has use_fft: {hasattr(arr, 'use_fft')}")
            if hasattr(arr, 'fix_phase') and hasattr(arr, 'use_fft'):
                return True
        return False

    @property
    def fix_phase(self) -> bool:
        """Whether bidirectional phase correction is enabled."""
        arrays = self._get_data_arrays()
        return getattr(arrays[0], 'fix_phase', False) if arrays else False

    @fix_phase.setter
    def fix_phase(self, value: bool):
        self.logger.info(f"Setting fix_phase to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, 'fix_phase'):
                arr.fix_phase = value
        self._refresh_image_widget()

    @property
    def use_fft(self) -> bool:
        """Whether FFT-based phase correlation is used."""
        arrays = self._get_data_arrays()
        return getattr(arrays[0], 'use_fft', False) if arrays else False

    @use_fft.setter
    def use_fft(self, value: bool):
        self.logger.info(f"Setting use_fft to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, 'use_fft'):
                arr.use_fft = value
        self._refresh_image_widget()

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase correlation."""
        arrays = self._get_data_arrays()
        return getattr(arrays[0], 'border', 3) if arrays else 3

    @border.setter
    def border(self, value: int):
        self.logger.info(f"Setting border to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, 'border'):
                arr.border = value
        self._refresh_image_widget()

    @property
    def max_offset(self) -> int:
        """Maximum pixel offset for phase correction."""
        arrays = self._get_data_arrays()
        return getattr(arrays[0], 'max_offset', 3) if arrays else 3

    @max_offset.setter
    def max_offset(self, value: int):
        self.logger.info(f"Setting max_offset to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, 'max_offset'):
                arr.max_offset = value
        self._refresh_image_widget()

    @property
    def selected_array(self) -> int:
        return self._selected_array

    @selected_array.setter
    def selected_array(self, value: int):
        if value < 0 or value >= self.num_graphics:
            raise ValueError(
                f"Invalid array index: {value}. "
                f"Must be between 0 and {self.num_graphics - 1}."
            )
        self._selected_array = value
        self.logger.info(f"Selected array index set to {value}.")

    @property
    def gaussian_sigma(self) -> float:
        """Sigma for Gaussian blur (0 = disabled). Uses fastplotlib spatial_func API."""
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value: float):
        """Set gaussian blur using fastplotlib's spatial_func API."""
        self._gaussian_sigma = max(0.0, value)
        # Rebuild spatial_func which combines mean subtraction and gaussian blur
        self._rebuild_spatial_func()
        self._refresh_image_widget()

    @property
    def proj(self) -> str:
        """Current projection mode (mean, max, std)."""
        return self._proj

    @proj.setter
    def proj(self, value: str):
        if value != self._proj:
            self._proj = value
            self._update_window_funcs()

    @property
    def mean_subtraction(self) -> bool:
        """Whether mean subtraction is enabled (spatial function)."""
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool):
        if value != self._mean_subtraction:
            self._mean_subtraction = value
            self._update_mean_subtraction()

    def _update_mean_subtraction(self):
        """Update spatial_func to apply mean subtraction."""
        self._rebuild_spatial_func()
        self._refresh_image_widget()

    def _rebuild_spatial_func(self):
        """Rebuild and apply the combined spatial function (mean subtraction + gaussian blur)."""
        names = self.image_widget._slider_dim_names or ()
        try:
            z_idx = self.image_widget.indices["z"] if "z" in names else 0
        except (IndexError, KeyError):
            z_idx = 0

        # Get gaussian sigma (shared across all processors)
        sigma = self.gaussian_sigma if self.gaussian_sigma > 0 else None

        # Check if any processing is needed
        any_mean_sub = self._mean_subtraction and any(
            self._zstats_done[i] and self._zstats_means[i] is not None
            for i in range(self.num_graphics)
        )

        if not any_mean_sub and sigma is None:
            # No processing needed - clear spatial_func by setting identity
            # fastplotlib doesn't accept None, so we use a passthrough function
            def identity(frame):
                return frame
            self.image_widget.spatial_func = identity
            self.logger.info("Spatial functions cleared (no mean subtraction or gaussian)")
            return

        # Build spatial_func list for each processor
        spatial_funcs = []
        for i in range(self.num_graphics):
            # Get mean image for this processor/z-plane if mean subtraction enabled
            mean_img = None
            if self._mean_subtraction and self._zstats_done[i] and self._zstats_means[i] is not None:
                mean_img = self._zstats_means[i][z_idx].astype(np.float32)
                self.logger.info(
                    f"Mean subtraction enabled for graphic {i}, z={z_idx}, "
                    f"mean_img shape={mean_img.shape}, mean value={mean_img.mean():.1f}"
                )

            # Build combined spatial function (always create one, fastplotlib doesn't accept None in list)
            spatial_funcs.append(self._make_spatial_func(mean_img, sigma))

        # Apply to image widget
        self.image_widget.spatial_func = spatial_funcs

    def _make_spatial_func(self, mean_img: np.ndarray | None, sigma: float | None):
        """Create a spatial function that applies mean subtraction and/or gaussian blur."""
        def spatial_func(frame):
            result = frame
            # Apply mean subtraction first
            if mean_img is not None:
                result = result.astype(np.float32) - mean_img
            # Apply gaussian blur second
            if sigma is not None and sigma > 0:
                from scipy.ndimage import gaussian_filter
                result = gaussian_filter(result, sigma=sigma)
            return result
        return spatial_func

    def _update_window_funcs(self):
        """Update window_funcs on image widget based on current projection mode."""
        if not self.processors:
            self.logger.warning("No processors available to update window funcs")
            return

        # Wrapper functions that match fastplotlib's expected signature
        # WindowFuncCallable must accept (array, axis, keepdims) parameters
        def mean_wrapper(data, axis, keepdims):
            return np.mean(data, axis=axis, keepdims=keepdims)

        def max_wrapper(data, axis, keepdims):
            return np.max(data, axis=axis, keepdims=keepdims)

        def std_wrapper(data, axis, keepdims):
            return np.std(data, axis=axis, keepdims=keepdims)

        proj_funcs = {
            "mean": mean_wrapper,
            "max": max_wrapper,
            "std": std_wrapper,
        }
        proj_func = proj_funcs.get(self._proj, mean_wrapper)

        # build window_funcs tuple based on data dimensionality
        n_slider_dims = self.processors[0].n_slider_dims if self.processors else 1

        if n_slider_dims == 1:
            window_funcs = (proj_func,)
        elif n_slider_dims == 2:
            window_funcs = (proj_func, None)
        else:
            window_funcs = (proj_func,) + (None,) * (n_slider_dims - 1)

        self.logger.debug(f"Updating window_funcs to {self._proj} with {n_slider_dims} slider dims")
        self._set_processor_attr("window_funcs", window_funcs)

    @property
    def window_size(self) -> int:
        """
        Window size for temporal projection.

        This sets the window size for the first slider dimension (typically 't').
        Uses fastplotlib's window_sizes API which expects a tuple per slider dim.
        """
        return self._window_size

    @window_size.setter
    def window_size(self, value: int):
        self._window_size = value
        self.logger.info(f"Window size set to {value}.")

        if not self.processors:
            self.logger.warning("No processors available to set window size")
            return

        # Use fastplotlib's window_sizes API
        # ImageWidget.window_sizes expects a list with one entry per processor
        # Each entry is a tuple with one value per slider dim
        # For 4D data (t, z, y, x) we have 2 slider dims: (t_window, z_window)
        # For 3D data (t, y, x) we have 1 slider dim: (t_window,)
        n_slider_dims = self.processors[0].n_slider_dims if self.processors else 1

        if n_slider_dims == 1:
            # Only temporal dimension
            per_processor_sizes = (value,)
        elif n_slider_dims == 2:
            # Temporal and z dimensions - only apply window to temporal (first dim)
            per_processor_sizes = (value, None)
        else:
            # More dimensions - apply to first, None for rest
            per_processor_sizes = (value,) + (None,) * (n_slider_dims - 1)

        # Set directly on processors with histogram computation disabled
        # to avoid expensive full-array histogram recomputation
        self._set_processor_attr("window_sizes", per_processor_sizes)

    @property
    def phase_upsample(self) -> int:
        """Upsampling factor for subpixel phase correlation."""
        if not self.has_raster_scan_support:
            return 5
        arrays = self._get_data_arrays()
        return getattr(arrays[0], 'upsample', 5) if arrays else 5

    @phase_upsample.setter
    def phase_upsample(self, value: int):
        if not self.has_raster_scan_support:
            return
        self.logger.info(f"Setting phase_upsample to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, 'upsample'):
                arr.upsample = value
        self._refresh_image_widget()

    def update(self):
        # Check for file/folder dialog results (iw-array API)
        self._check_file_dialogs()
        draw_saveas_popup(self)
        draw_menu(self)
        draw_tabs(self)

        # Update mean subtraction when z-plane changes
        if self._mean_subtraction:
            names = self.image_widget._slider_dim_names or ()
            try:
                z_idx = self.image_widget.indices["z"] if "z" in names else 0
            except (IndexError, KeyError):
                z_idx = 0
            if z_idx != self._last_z_idx:
                self._last_z_idx = z_idx
                self._update_mean_subtraction()

    def _check_file_dialogs(self):
        """Check if file/folder dialogs have results and load data if so."""
        # Check file dialog
        if self._file_dialog is not None and self._file_dialog.ready():
            result = self._file_dialog.result()
            if result and len(result) > 0:
                # Save to recent files and context-specific preferences
                add_recent_file(result[0], file_type="file")
                set_last_dir("open_file", result[0])
                self._load_new_data(result[0])
            self._file_dialog = None

        # Check folder dialog
        if self._folder_dialog is not None and self._folder_dialog.ready():
            result = self._folder_dialog.result()
            if result:
                # Save to recent files and context-specific preferences
                add_recent_file(result, file_type="folder")
                set_last_dir("open_folder", result)
                self._load_new_data(result)
            self._folder_dialog = None

    def _load_new_data(self, path: str):
        """
        Load new data from the specified path using iw-array API.

        Uses iw.set_data() to swap data arrays, which handles shape changes.
        """
        from mbo_utilities.lazy_array import imread

        path_obj = Path(path)
        if not path_obj.exists():
            self.logger.error(f"Path does not exist: {path}")
            self._load_status_msg = f"Error: Path does not exist"
            self._load_status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
            return

        try:
            self.logger.info(f"Loading data from: {path}")
            self._load_status_msg = "Loading..."
            self._load_status_color = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)

            new_data = imread(path)

            # Check if dimensionality is changing - if so, reset window functions
            # to avoid IndexError in fastplotlib's _apply_window_function
            old_ndim = len(self.shape) if hasattr(self, 'shape') and self.shape else 0
            new_ndim = new_data.ndim

            # Reset window functions on processors if dimensionality changes
            # This prevents tuple index out of range errors when going 3D->4D or vice versa
            if old_ndim != new_ndim:
                for proc in self.image_widget._image_processors:
                    proc.window_funcs = None
                    proc.window_sizes = None
                    proc.window_order = None

            # iw-array API: use data indexer for replacing data
            # iw.data[0] = new_array handles shape changes automatically
            self.image_widget.data[0] = new_data

            # Reset indices
            try:
                names = self.image_widget._slider_dim_names or ()
                if "t" in names:
                    self.image_widget.indices["t"] = 0
                if "z" in names and new_data.ndim >= 4:
                    self.image_widget.indices["z"] = 0
            except (IndexError, KeyError):
                pass  # Indices not available yet

            # Update internal state
            self.fpath = path
            self.shape = new_data.shape
            self.is_mbo_scan = isinstance(new_data, MboRawArray)

            # Update nz for z-plane count
            if len(self.shape) == 4:
                self.nz = self.shape[1]
            elif len(self.shape) == 3:
                self.nz = 1
            else:
                self.nz = 1

            # Reset save dialog state for new data
            self._saveas_selected_roi = set()
            self._saveas_rois = False

            self._load_status_msg = f"Loaded: {path_obj.name}"
            self._load_status_color = imgui.ImVec4(0.3, 1.0, 0.3, 1.0)
            self.logger.info(f"Loaded successfully, shape: {new_data.shape}")
            self.set_context_info()

            # refresh widgets based on new data capabilities
            self._refresh_widgets()

            # Automatically recompute z-stats for new data
            self.refresh_zstats()

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self._load_status_msg = f"Error: {str(e)}"
            self._load_status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)

    def draw_stats_section(self):
        if not any(self._zstats_done):
            return

        stats_list = self._zstats
        is_single_zplane = self.nz == 1  # Single bar for 1 plane
        is_dual_zplane = self.nz == 2    # Grouped bars for 2 planes
        is_multi_zplane = self.nz > 2    # Line graph for 3+ planes

        # Different title for single vs multi z-plane
        if is_single_zplane or is_dual_zplane:
            imgui.text_colored(
                imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Signal Quality Summary"
            )
        else:
            imgui.text_colored(
                imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Z-Plane Summary Stats"
            )

        # ROI selector
        array_labels = [
            f"{"graphic"} {i + 1}"
            for i in range(len(stats_list))
            if stats_list[i] and "mean" in stats_list[i]
        ]
        # Only show "Combined" if there are multiple arrays
        if len(array_labels) > 1:
            array_labels.append("Combined")

        # Ensure selected array is within bounds
        if self._selected_array >= len(array_labels):
            self._selected_array = 0

        avail = imgui.get_content_region_avail().x
        xpos = 0

        for i, label in enumerate(array_labels):
            if imgui.radio_button(label, self._selected_array == i):
                self._selected_array = i
            button_width = (
                imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 4
            )
            xpos += button_width + imgui.get_style().item_spacing.x

            if xpos >= avail:
                xpos = button_width
                imgui.new_line()
            else:
                imgui.same_line()

        imgui.separator()

        # Check if "Combined" view is selected (only valid if there are multiple arrays)
        has_combined = len(array_labels) > 1 and array_labels[-1] == "Combined"
        is_combined_selected = has_combined and self._selected_array == len(array_labels) - 1

        if is_combined_selected:  # Combined
            imgui.text(f"Stats for Combined {"graphic"}s")
            mean_vals = np.mean(
                [np.array(s["mean"]) for s in stats_list if s and "mean" in s], axis=0
            )

            if len(mean_vals) == 0:
                return

            std_vals = np.mean(
                [np.array(s["std"]) for s in stats_list if s and "std" in s], axis=0
            )
            snr_vals = np.mean(
                [np.array(s["snr"]) for s in stats_list if s and "snr" in s], axis=0
            )

            z_vals = np.ascontiguousarray(
                np.arange(1, len(mean_vals) + 1, dtype=np.float64)
            )
            mean_vals = np.ascontiguousarray(mean_vals, dtype=np.float64)
            std_vals = np.ascontiguousarray(std_vals, dtype=np.float64)

            # For single/dual z-plane, show simplified combined view
            if is_single_zplane or is_dual_zplane:
                # Show stats table
                n_cols = 4 if is_dual_zplane else 3
                if imgui.begin_table(
                    f"Stats (averaged over {"graphic"}s)",
                    n_cols,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
                ):
                    if is_dual_zplane:
                        for col in ["Metric", "Z1", "Z2", "Unit"]:
                            imgui.table_setup_column(
                                col, imgui.TableColumnFlags_.width_stretch
                            )
                    else:
                        for col in ["Metric", "Value", "Unit"]:
                            imgui.table_setup_column(
                                col, imgui.TableColumnFlags_.width_stretch
                            )
                    imgui.table_headers_row()

                    if is_dual_zplane:
                        metrics = [
                            ("Mean Fluorescence", mean_vals[0], mean_vals[1], "a.u."),
                            ("Std. Deviation", std_vals[0], std_vals[1], "a.u."),
                            ("Signal-to-Noise", snr_vals[0], snr_vals[1], "ratio"),
                        ]
                        for metric_name, val1, val2, unit in metrics:
                            imgui.table_next_row()
                            imgui.table_next_column()
                            imgui.text(metric_name)
                            imgui.table_next_column()
                            imgui.text(f"{val1:.2f}")
                            imgui.table_next_column()
                            imgui.text(f"{val2:.2f}")
                            imgui.table_next_column()
                            imgui.text(unit)
                    else:
                        metrics = [
                            ("Mean Fluorescence", mean_vals[0], "a.u."),
                            ("Std. Deviation", std_vals[0], "a.u."),
                            ("Signal-to-Noise", snr_vals[0], "ratio"),
                        ]
                        for metric_name, value, unit in metrics:
                            imgui.table_next_row()
                            imgui.table_next_column()
                            imgui.text(metric_name)
                            imgui.table_next_column()
                            imgui.text(f"{value:.2f}")
                            imgui.table_next_column()
                            imgui.text(unit)
                    imgui.end_table()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                imgui.text("Signal Quality Comparison")
                set_tooltip(
                    f"Comparison of mean fluorescence across all {"graphic"}s"
                    + (" and z-planes" if is_dual_zplane else ""),
                    True,
                )

                plot_width = imgui.get_content_region_avail().x

                if is_dual_zplane:
                    # Grouped bar chart for 2 z-planes
                    # Get per-graphic, per-z-plane mean values
                    graphic_means_z1 = [
                        np.asarray(self._zstats[r]["mean"][0], float)
                        for r in range(self.num_graphics)
                        if self._zstats[r] and "mean" in self._zstats[r] and len(self._zstats[r]["mean"]) >= 1
                    ]
                    graphic_means_z2 = [
                        np.asarray(self._zstats[r]["mean"][1], float)
                        for r in range(self.num_graphics)
                        if self._zstats[r] and "mean" in self._zstats[r] and len(self._zstats[r]["mean"]) >= 2
                    ]

                    if graphic_means_z1 and graphic_means_z2 and implot.begin_plot(
                        "Signal Comparison", imgui.ImVec2(plot_width, 350)
                    ):
                        try:
                            style_seaborn_dark()
                            implot.setup_axes(
                                "Graphic",
                                "Mean Fluorescence (a.u.)",
                                implot.AxisFlags_.none.value,
                                implot.AxisFlags_.auto_fit.value,
                            )

                            n_graphics = len(graphic_means_z1)
                            bar_width = 0.35
                            x_pos = np.arange(n_graphics, dtype=np.float64)

                            labels = [f"{i + 1}" for i in range(n_graphics)]
                            implot.setup_axis_limits(
                                implot.ImAxis_.x1.value, -0.5, n_graphics - 0.5
                            )
                            implot.setup_axis_ticks(
                                implot.ImAxis_.x1.value, x_pos.tolist(), labels, False
                            )

                            # Z-plane 1 bars (offset left)
                            x_z1 = x_pos - bar_width / 2
                            heights_z1 = np.array(graphic_means_z1, dtype=np.float64)
                            implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                            implot.push_style_color(
                                implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                            )
                            implot.plot_bars("Z-Plane 1", x_z1, heights_z1, bar_width)
                            implot.pop_style_color()
                            implot.pop_style_var()

                            # Z-plane 2 bars (offset right)
                            x_z2 = x_pos + bar_width / 2
                            heights_z2 = np.array(graphic_means_z2, dtype=np.float64)
                            implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                            implot.push_style_color(
                                implot.Col_.fill.value, (0.9, 0.4, 0.2, 0.8)
                            )
                            implot.plot_bars("Z-Plane 2", x_z2, heights_z2, bar_width)
                            implot.pop_style_color()
                            implot.pop_style_var()

                        finally:
                            implot.end_plot()
                else:
                    # Single z-plane: simple bar chart
                    graphic_means = [
                        np.asarray(self._zstats[r]["mean"][0], float)
                        for r in range(self.num_graphics)
                        if self._zstats[r] and "mean" in self._zstats[r]
                    ]

                    if graphic_means and implot.begin_plot(
                        "Signal Comparison", imgui.ImVec2(plot_width, 350)
                    ):
                        try:
                            style_seaborn_dark()
                            implot.setup_axes(
                                "Graphic",
                                "Mean Fluorescence (a.u.)",
                                implot.AxisFlags_.none.value,
                                implot.AxisFlags_.auto_fit.value,
                            )

                            x_pos = np.arange(len(graphic_means), dtype=np.float64)
                            heights = np.array(graphic_means, dtype=np.float64)

                            labels = [f"{i + 1}" for i in range(len(graphic_means))]
                            implot.setup_axis_limits(
                                implot.ImAxis_.x1.value, -0.5, len(graphic_means) - 0.5
                            )
                            implot.setup_axis_ticks(
                                implot.ImAxis_.x1.value, x_pos.tolist(), labels, False
                            )

                            implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                            implot.push_style_color(
                                implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                            )
                            implot.plot_bars(
                                "Graphic Signal",
                                x_pos,
                                heights,
                                0.6,
                            )
                            implot.pop_style_color()
                            implot.pop_style_var()

                            # Add mean line
                            mean_line = np.full_like(heights, mean_vals[0])
                            implot.push_style_var(implot.StyleVar_.line_weight.value, 2)
                            implot.push_style_color(
                                implot.Col_.line.value, (1.0, 0.4, 0.2, 0.8)
                            )
                            implot.plot_line("Average", x_pos, mean_line)
                            implot.pop_style_color()
                            implot.pop_style_var()
                        finally:
                            implot.end_plot()

            else:
                # Multi-z-plane: show original table and combined plot
                # Table
                if imgui.begin_table(
                    f"Stats, averaged over {"graphic"}s",
                    4,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,  # type: ignore # noqa
                ):  # type: ignore # noqa
                    for col in ["Z", "Mean", "Std", "SNR"]:
                        imgui.table_setup_column(
                            col, imgui.TableColumnFlags_.width_stretch
                        )  # type: ignore # noqa
                    imgui.table_headers_row()
                    for i in range(len(z_vals)):
                        imgui.table_next_row()
                        for val in (
                            z_vals[i],
                            mean_vals[i],
                            std_vals[i],
                            snr_vals[i],
                        ):
                            imgui.table_next_column()
                            imgui.text(f"{val:.2f}")
                    imgui.end_table()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                imgui.text("Z-plane Signal: Combined")
                set_tooltip(
                    f"Gray = per-ROI z-profiles (mean over frames)."
                    f" Blue shade = across-ROI mean ± std; blue line = mean."
                    f" Hover gray lines for values.",
                    True,
                )

                # build per-graphic series
                graphic_series = [
                    np.asarray(self._zstats[r]["mean"], float)
                    for r in range(self.num_graphics)
                ]

                L = min(len(s) for s in graphic_series)
                z = np.asarray(z_vals[:L], float)
                graphic_series = [s[:L] for s in graphic_series]
                stack = np.vstack(graphic_series)
                mean_vals = stack.mean(axis=0)
                std_vals = stack.std(axis=0)
                lower = mean_vals - std_vals
                upper = mean_vals + std_vals

                # Use available width to prevent cutoff
                plot_width = imgui.get_content_region_avail().x
                if implot.begin_plot(
                    "Z-Plane Plot (Combined)", imgui.ImVec2(plot_width, 300)
                ):
                    try:
                        style_seaborn_dark()
                        implot.setup_axes(
                            "Z-Plane",
                            "Mean Fluorescence",
                            implot.AxisFlags_.none.value,
                            implot.AxisFlags_.auto_fit.value,
                        )

                        implot.setup_axis_limits(
                            implot.ImAxis_.x1.value, float(z[0]), float(z[-1])
                        )
                        implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")

                        for i, ys in enumerate(graphic_series):
                            label = f"ROI {i + 1}##roi{i}"
                            implot.push_style_var(implot.StyleVar_.line_weight.value, 1)
                            implot.push_style_color(
                                implot.Col_.line.value, (0.6, 0.6, 0.6, 0.35)
                            )
                            implot.plot_line(label, z, ys)
                            implot.pop_style_color()
                            implot.pop_style_var()

                        implot.push_style_color(
                            implot.Col_.fill.value, (0.2, 0.4, 0.8, 0.25)
                        )
                        implot.plot_shaded("Mean ± Std##band", z, lower, upper)
                        implot.pop_style_color()

                        implot.push_style_var(implot.StyleVar_.line_weight.value, 2)
                        implot.plot_line("Mean##line", z, mean_vals)
                        implot.pop_style_var()
                    finally:
                        implot.end_plot()

        else:
            array_idx = self._selected_array
            stats = stats_list[array_idx]
            if not stats or "mean" not in stats:
                return

            mean_vals = np.array(stats["mean"])
            std_vals = np.array(stats["std"])
            snr_vals = np.array(stats["snr"])
            n = min(len(mean_vals), len(std_vals), len(snr_vals))

            mean_vals, std_vals, snr_vals = mean_vals[:n], std_vals[:n], snr_vals[:n]

            z_vals = np.ascontiguousarray(np.arange(1, n + 1, dtype=np.float64))
            mean_vals = np.ascontiguousarray(mean_vals, dtype=np.float64)
            std_vals = np.ascontiguousarray(std_vals, dtype=np.float64)

            imgui.text(f"Stats for {"graphic"} {array_idx + 1}")

            # For single/dual z-plane, show simplified table and visualization
            if is_single_zplane or is_dual_zplane:
                # Show stats table with appropriate columns
                n_cols = 4 if is_dual_zplane else 3
                if imgui.begin_table(
                    f"stats{array_idx}",
                    n_cols,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
                ):
                    if is_dual_zplane:
                        for col in ["Metric", "Z1", "Z2", "Unit"]:
                            imgui.table_setup_column(
                                col, imgui.TableColumnFlags_.width_stretch
                            )
                    else:
                        for col in ["Metric", "Value", "Unit"]:
                            imgui.table_setup_column(
                                col, imgui.TableColumnFlags_.width_stretch
                            )
                    imgui.table_headers_row()

                    if is_dual_zplane:
                        metrics = [
                            ("Mean Fluorescence", mean_vals[0], mean_vals[1], "a.u."),
                            ("Std. Deviation", std_vals[0], std_vals[1], "a.u."),
                            ("Signal-to-Noise", snr_vals[0], snr_vals[1], "ratio"),
                        ]
                        for metric_name, val1, val2, unit in metrics:
                            imgui.table_next_row()
                            imgui.table_next_column()
                            imgui.text(metric_name)
                            imgui.table_next_column()
                            imgui.text(f"{val1:.2f}")
                            imgui.table_next_column()
                            imgui.text(f"{val2:.2f}")
                            imgui.table_next_column()
                            imgui.text(unit)
                    else:
                        metrics = [
                            ("Mean Fluorescence", mean_vals[0], "a.u."),
                            ("Std. Deviation", std_vals[0], "a.u."),
                            ("Signal-to-Noise", snr_vals[0], "ratio"),
                        ]
                        for metric_name, value, unit in metrics:
                            imgui.table_next_row()
                            imgui.table_next_column()
                            imgui.text(metric_name)
                            imgui.table_next_column()
                            imgui.text(f"{value:.2f}")
                            imgui.table_next_column()
                            imgui.text(unit)
                    imgui.end_table()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                style_seaborn_dark()
                imgui.text("Signal Quality Metrics")
                set_tooltip(
                    "Bar chart showing mean fluorescence, standard deviation, and SNR"
                    + (" for each z-plane" if is_dual_zplane else ""),
                    True,
                )

                plot_width = imgui.get_content_region_avail().x
                if implot.begin_plot(
                    f"Signal Metrics {array_idx}", imgui.ImVec2(plot_width, 350)
                ):
                    try:
                        implot.setup_axes(
                            "Metric",
                            "Value",
                            implot.AxisFlags_.none.value,
                            implot.AxisFlags_.auto_fit.value,
                        )

                        x_pos = np.array([0.0, 1.0, 2.0], dtype=np.float64)
                        implot.setup_axis_limits(implot.ImAxis_.x1.value, -0.5, 2.5)
                        implot.setup_axis_ticks(
                            implot.ImAxis_.x1.value, x_pos.tolist(), ["Mean", "Std Dev", "SNR"], False
                        )

                        if is_dual_zplane:
                            # Grouped bars for Z1 and Z2
                            bar_width = 0.35
                            x_z1 = x_pos - bar_width / 2
                            x_z2 = x_pos + bar_width / 2

                            heights_z1 = np.array([mean_vals[0], std_vals[0], snr_vals[0]], dtype=np.float64)
                            heights_z2 = np.array([mean_vals[1], std_vals[1], snr_vals[1]], dtype=np.float64)

                            implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                            implot.push_style_color(
                                implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                            )
                            implot.plot_bars("Z-Plane 1", x_z1, heights_z1, bar_width)
                            implot.pop_style_color()
                            implot.pop_style_var()

                            implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                            implot.push_style_color(
                                implot.Col_.fill.value, (0.9, 0.4, 0.2, 0.8)
                            )
                            implot.plot_bars("Z-Plane 2", x_z2, heights_z2, bar_width)
                            implot.pop_style_color()
                            implot.pop_style_var()
                        else:
                            # Single bars for single z-plane
                            heights = np.array([mean_vals[0], std_vals[0], snr_vals[0]], dtype=np.float64)

                            implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                            implot.push_style_color(
                                implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                            )
                            implot.plot_bars("Signal Metrics", x_pos, heights, 0.6)
                            implot.pop_style_color()
                            implot.pop_style_var()
                    finally:
                        implot.end_plot()

            else:
                # Multi-z-plane: show original table and line plot
                if imgui.begin_table(
                    f"zstats{array_idx}",
                    4,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
                ):
                    for col in ["Z", "Mean", "Std", "SNR"]:
                        imgui.table_setup_column(
                            col, imgui.TableColumnFlags_.width_stretch
                        )
                    imgui.table_headers_row()
                    for j in range(n):
                        imgui.table_next_row()
                        for val in (
                            int(z_vals[j]),
                            mean_vals[j],
                            std_vals[j],
                            snr_vals[j],
                        ):
                            imgui.table_next_column()
                            imgui.text(f"{val:.2f}")
                    imgui.end_table()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                style_seaborn_dark()
                imgui.text("Z-plane Signal: Mean ± Std")
                plot_width = imgui.get_content_region_avail().x
                if implot.begin_plot(
                    f"Z-Plane Signal {array_idx}", imgui.ImVec2(plot_width, 300)
                ):
                    try:
                        implot.setup_axes(
                            "Z-Plane",
                            "Mean Fluorescence",
                            implot.AxisFlags_.auto_fit.value,
                            implot.AxisFlags_.auto_fit.value,
                        )
                        implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")
                        implot.plot_error_bars(
                            f"Mean ± Std {array_idx}", z_vals, mean_vals, std_vals
                        )
                        implot.plot_line(f"Mean {array_idx}", z_vals, mean_vals)
                    finally:
                        implot.end_plot()

    def draw_preview_section(self):
        """Draw preview section using modular UI sections based on data capabilities."""
        imgui.dummy(imgui.ImVec2(0, 5))
        cflags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize
        with imgui_ctx.begin_child("##PreviewChild", imgui.ImVec2(0, 0), cflags):
            # draw all supported widgets
            draw_all_widgets(self, self._widgets)

    def get_raw_frame(self) -> tuple[ndarray, ...]:
        # iw-array API: use indices property for named dimension access
        idx = self.image_widget.indices
        names = self.image_widget._slider_dim_names or ()
        t = idx["t"] if "t" in names else 0
        z = idx["z"] if "z" in names else 0
        return tuple(ndim_to_frame(arr, t, z) for arr in self.image_widget.data)

    # NOTE: _compute_phase_offsets, update_frame_apply, and _combined_frame_apply
    # have been removed. Processing logic is now on MboImageProcessor.get()

    def _compute_zstats_single_roi(self, roi, fpath):
        arr = imread(fpath)
        if hasattr(arr, "fix_phase"):
            arr.fix_phase = False
        if hasattr(arr, "roi"):
            arr.roi = roi

        stats, means = {"mean": [], "std": [], "snr": []}, []
        self._tiff_lock = threading.Lock()
        for z in range(self.nz):
            with self._tiff_lock:
                stack = arr[::10, z].astype(np.float32)  # Z, Y, X
                mean_img = np.mean(stack, axis=0)
                std_img = np.std(stack, axis=0)
                snr_img = np.divide(mean_img, std_img + 1e-5, where=(std_img > 1e-5))
                stats["mean"].append(float(np.mean(mean_img)))
                stats["std"].append(float(np.mean(std_img)))
                stats["snr"].append(float(np.mean(snr_img)))
                means.append(mean_img)
                self._zstats_progress[roi - 1] = (z + 1) / self.nz
                self._zstats_current_z[roi - 1] = z

        self._zstats[roi - 1] = stats
        means_stack = np.stack(means)

        self._zstats_means[roi - 1] = means_stack
        self._zstats_mean_scalar[roi - 1] = means_stack.mean(axis=(1, 2))
        self._zstats_done[roi - 1] = True

    def _compute_zstats_single_array(self, idx, arr):
        # Check for pre-computed z-stats in zarr metadata (instant loading)
        if hasattr(arr, "zstats") and arr.zstats is not None:
            stats = arr.zstats
            self._zstats[idx - 1] = stats
            # Still need to compute mean images for visualization
            means = []
            self._tiff_lock = threading.Lock()
            for z in [0] if arr.ndim == 3 else range(self.nz):
                with self._tiff_lock:
                    stack = (
                        arr[::10].astype(np.float32)
                        if arr.ndim == 3
                        else arr[::10, z].astype(np.float32)
                    )
                    mean_img = np.mean(stack, axis=0)
                    means.append(mean_img)
                    self._zstats_progress[idx - 1] = (z + 1) / self.nz
                    self._zstats_current_z[idx - 1] = z
            means_stack = np.stack(means)
            self._zstats_means[idx - 1] = means_stack
            self._zstats_mean_scalar[idx - 1] = means_stack.mean(axis=(1, 2))
            self._zstats_done[idx - 1] = True
            self.logger.info(f"Loaded pre-computed z-stats from zarr metadata for array {idx}")
            return

        stats, means = {"mean": [], "std": [], "snr": []}, []
        self._tiff_lock = threading.Lock()

        for z in [0] if arr.ndim == 3 else range(self.nz):
            with self._tiff_lock:
                stack = (
                    arr[::10].astype(np.float32)
                    if arr.ndim == 3
                    else arr[::10, z].astype(np.float32)
                )

                mean_img = np.mean(stack, axis=0)
                std_img = np.std(stack, axis=0)
                snr_img = np.divide(mean_img, std_img + 1e-5, where=(std_img > 1e-5))

                stats["mean"].append(float(np.mean(mean_img)))
                stats["std"].append(float(np.mean(std_img)))
                stats["snr"].append(float(np.mean(snr_img)))

                means.append(mean_img)
                self._zstats_progress[idx - 1] = (z + 1) / self.nz
                self._zstats_current_z[idx - 1] = z

        self._zstats[idx - 1] = stats
        means_stack = np.stack(means)
        self._zstats_means[idx - 1] = means_stack
        self._zstats_mean_scalar[idx - 1] = means_stack.mean(axis=(1, 2))
        self._zstats_done[idx - 1] = True

        # Save z-stats to array metadata for persistence (zarr files)
        if hasattr(arr, "zstats"):
            try:
                arr.zstats = stats
                self.logger.info(f"Saved z-stats to array {idx} metadata")
            except Exception as e:
                self.logger.debug(f"Could not save z-stats to array metadata: {e}")

    def compute_zstats(self):
        if not self.image_widget or not self.image_widget.data:
            return

        # Compute z-stats for each graphic (array)
        for idx, arr in enumerate(self.image_widget.data, start=1):
            threading.Thread(
                target=self._compute_zstats_single_array,
                args=(idx, arr),
                daemon=True,
            ).start()

    def refresh_zstats(self):
        """
        Reset and recompute z-stats for all arrays.

        This is useful after loading new data or when z-stats need to be
        recalculated (e.g., after changing the number of z-planes).
        """
        if not self.image_widget:
            return

        # Use num_graphics which matches len(iw.graphics)
        n = self.num_graphics

        # Reset z-stats state
        self._zstats = [{"mean": [], "std": [], "snr": []} for _ in range(n)]
        self._zstats_means = [None] * n
        self._zstats_mean_scalar = [0.0] * n
        self._zstats_done = [False] * n
        self._zstats_progress = [0.0] * n
        self._zstats_current_z = [0] * n

        # Reset progress state for each graphic to allow new progress display
        for i in range(n):
            reset_progress_state(f"zstats_{i}")

        # Update nz based on current data shape
        if len(self.shape) >= 4:
            self.nz = self.shape[1]
        elif len(self.shape) == 3:
            self.nz = 1
        else:
            self.nz = 1

        self.logger.info(f"Refreshing z-stats for {n} arrays, nz={self.nz}")

        # Recompute z-stats
        self.compute_zstats()

    def cleanup(self):
        """Clean up resources when the GUI is closing.

        Should be called before the application exits to properly release
        resources like open windows, file handles, and pending operations.
        """
        # Clean up pipeline instances (suite2p window, etc)
        from mbo_utilities.graphics.widgets.pipelines import cleanup_pipelines
        cleanup_pipelines(self)

        # Clean up all widgets
        from mbo_utilities.graphics.widgets import cleanup_all_widgets
        cleanup_all_widgets(self._widgets)

        # Clear file dialogs
        self._file_dialog = None
        self._folder_dialog = None
        if hasattr(self, '_s2p_folder_dialog'):
            self._s2p_folder_dialog = None

        self.logger.info("GUI cleanup complete")
