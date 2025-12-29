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
    ImVec2,
)
from mbo_utilities.metadata import get_param
from mbo_utilities.file_io import (
    get_mbo_dirs,
)
from mbo_utilities.reader import MBO_SUPPORTED_FTYPES
from mbo_utilities.arrays.features import PhaseCorrectionFeature
from mbo_utilities.preferences import (
    get_last_dir,
    set_last_dir,
    add_recent_file,
)
from mbo_utilities.arrays import ScanImageArray
from mbo_utilities.gui._imgui import (
    begin_popup_size,
    ndim_to_frame,
    style_seaborn_dark,
)
from mbo_utilities.gui._widgets import (
    set_tooltip,
    checkbox_with_tooltip,
    draw_scope,
    draw_metadata_inspector,
    draw_checkbox_grid,
)
from mbo_utilities.gui.process_manager import get_process_manager
from mbo_utilities.gui.progress_bar import (
    draw_status_indicator,
    reset_progress_state,
    start_output_capture,
)
from mbo_utilities.gui.widgets import get_supported_widgets, draw_all_widgets
from mbo_utilities.gui._availability import HAS_SUITE2P, HAS_SUITE3D
from mbo_utilities.reader import imread
from mbo_utilities.writer import imwrite
from mbo_utilities.arrays import _sanitize_suffix
from mbo_utilities.gui.gui_logger import GuiLogger, GuiLogHandler
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
    # These must be set on the array object for ScanImageArray phase correction
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


def draw_tools_popups(parent):
    """Draw independent popup windows (Scope, Debug, Metadata)."""
    if parent.show_scope_window:
        size = begin_popup_size()
        imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)  # type: ignore # noqa
        _, parent.show_scope_window = imgui.begin(
            "Scope Inspector",
            parent.show_scope_window,
        )
        draw_scope()
        imgui.end()
    if parent.show_metadata_viewer:
        # use absolute screen positioning so window is visible even when widget collapsed
        io = imgui.get_io()
        screen_w, screen_h = io.display_size.x, io.display_size.y
        win_w, win_h = min(600, screen_w * 0.5), min(500, screen_h * 0.6)
        # center on screen
        imgui.set_next_window_pos(
            ImVec2((screen_w - win_w) / 2, (screen_h - win_h) / 2),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(ImVec2(win_w, win_h), imgui.Cond_.first_use_ever)
        _, parent.show_metadata_viewer = imgui.begin(
            "Metadata Viewer",
            parent.show_metadata_viewer,
        )
        if parent.image_widget and parent.image_widget.data:
            data_arr = parent.image_widget.data[0]
            metadata = data_arr.metadata
            draw_metadata_inspector(metadata, data_array=data_arr)
        else:
            imgui.text("No data loaded")
        imgui.end()


def draw_menu_bar(parent):
    """Draw the menu bar within the current window/child scope."""
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
                        "Select Data File(s)",
                        start_dir,
                        ["Image Files", "*.tif *.tiff *.zarr *.npy *.bin", "All Files", "*"],
                        pfd.opt.multiselect
                    )
                # Open Folder - iw-array API
                if imgui.menu_item("Open Folder", "Ctrl+Shift+O", p_selected=False, enabled=True)[0]:
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
                    "Save as", "s", p_selected=False, enabled=can_save
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
                imgui.end_menu()
            if imgui.begin_menu("Settings", True):
                imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Tools")
                imgui.separator()
                imgui.spacing()
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

        # Draw process status indicator (consolidated)
        draw_process_status_indicator(parent)
        if parent._show_progress_overlay:
            parent._clear_stale_progress()
            # Overlay is deprecated/merged into status indicator, but keeping logic if needed internally
            # but NOT drawing the second text-based indicator
            pass


def draw_tabs(parent):
    # Don't create an outer child window - let each tab manage its own scrolling
    # For single z-plane data, show all tabs
    # For multi-zplane data, show all tabs (user wants all tabs visible)
    if imgui.begin_tab_bar("MainPreviewTabs"):
        if imgui.begin_tab_item("Preview")[0]:
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

            parent.draw_preview_section()
            imgui.pop_style_var()
            imgui.pop_style_var()
            imgui.end_tab_item()

        imgui.begin_disabled(not all(parent._zstats_done))
        if imgui.begin_tab_item("Signal Quality")[0]:
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
            from mbo_utilities.gui.widgets.pipelines import draw_run_tab
            draw_run_tab(parent)
            imgui.pop_style_var()
            imgui.pop_style_var()
            imgui.end_tab_item()
        imgui.end_tab_bar()


def draw_process_status_indicator(parent):
    """Draw compact process status indicator in top-left with color coding."""
    # Import icons
    try:
        from imgui_bundle import icons_fontawesome as fa
        ICON_IDLE = fa.ICON_FA_CIRCLE
        ICON_RUNNING = fa.ICON_FA_SPINNER
        ICON_ERROR = fa.ICON_FA_EXCLAMATION_TRIANGLE
    except (ImportError, AttributeError):
        # Fallback to unicode
        ICON_IDLE = "\uf111"  # circle
        ICON_RUNNING = "\uf110"  # spinner
        ICON_ERROR = "\uf071"  # exclamation-triangle

    pm = get_process_manager()
    pm.cleanup_finished()
    running_procs = pm.get_running()

    # Get in-app progress items
    from mbo_utilities.gui.progress_bar import _get_active_progress_items
    progress_items = _get_active_progress_items(parent)

    total_active = len(running_procs) + len(progress_items)

    # Determine status color and icon
    if total_active == 0:
        status_color = imgui.ImVec4(0.15, 0.55, 0.15, 1.0)  # Dark Green - idle
        status_text = f"{ICON_IDLE} Idle"
    else:
        # Check if any have errors (only relevant for procs currently)
        has_error = any(p.status == "error" for p in running_procs)

        if has_error:
            status_color = imgui.ImVec4(0.8, 0.2, 0.2, 1.0)  # Dark Red - error
            status_text = f"{ICON_ERROR} {total_active} Error"
        else:
            status_color = imgui.ImVec4(0.85, 0.45, 0.0, 1.0)  # Dark Orange - running

            # Add percentage if we have progress items
            if progress_items:
                avg_progress = sum(item["progress"] for item in progress_items) / len(progress_items)
                status_text = f"{ICON_RUNNING} Running ({total_active}) {int(avg_progress * 100)}%"
            else:
                status_text = f"{ICON_RUNNING} Running ({total_active})"

    # Draw rounded buttons
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)

    # 1. Status Button
    # Use distinct background color based on status
    imgui.push_style_color(imgui.Col_.button, status_color)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1, 1, 1, 1)) # Always white text

    # Slightly lighter hover color
    hover_col = imgui.ImVec4(
        min(status_color.x + 0.1, 1.0),
        min(status_color.y + 0.1, 1.0),
        min(status_color.z + 0.1, 1.0),
        status_color.w
    )
    imgui.push_style_color(imgui.Col_.button_hovered, hover_col)
    imgui.push_style_color(imgui.Col_.button_active, status_color)

    if imgui.button(status_text + "##process_status"):
        parent._show_process_console = True

    imgui.pop_style_color(4) # button, text, hovered, active

    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
        imgui.set_tooltip("Click to view process console")

    # 2. Metadata Button
    imgui.same_line()
    # Dark grey background
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.2, 0.2, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.3, 0.3, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.15, 0.15, 0.15, 1.0))
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.9, 0.9, 0.9, 1.0))

    if imgui.button("Metadata (m)"):
        parent.show_metadata_viewer = not parent.show_metadata_viewer

    imgui.pop_style_color(4)
    imgui.pop_style_var() # frame_rounding


def draw_process_console_popup(parent):
    """Draw popup showing process outputs and debug logs."""
    if not hasattr(parent, '_show_process_console'):
        parent._show_process_console = False

    if parent._show_process_console:
        imgui.open_popup("Process Console")
        parent._show_process_console = False

    center = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size(imgui.ImVec2(900, 600), imgui.Cond_.first_use_ever)

    if imgui.begin_popup_modal("Process Console")[0]:
        pm = get_process_manager()
        pm.cleanup_finished()
        running = pm.get_running()

        # Use tabs instead of collapsible headers
        if imgui.begin_tab_bar("ProcessConsoleTabs"):
            # Tab 1: Processes
            if imgui.begin_tab_item("Processes")[0]:
                with imgui_ctx.begin_child("##BGTasksContent", imgui.ImVec2(0, -30), imgui.ChildFlags_.none):
                    from mbo_utilities.gui.progress_bar import _get_active_progress_items
                    progress_items = _get_active_progress_items(parent)

                    if progress_items:
                        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), f"Active Tasks ({len(progress_items)})")
                        imgui.separator()
                        for item in progress_items:
                            pct = int(item["progress"] * 100)
                            if item.get("done", False):
                                imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"{item['text']} [Done]")
                            else:
                                imgui.text(f"{item['text']} [{pct}%]")

                            imgui.progress_bar(item["progress"], imgui.ImVec2(-1, 0), "")
                            imgui.spacing()
                        imgui.separator()
                        imgui.spacing()

                    if not running and not progress_items:
                        imgui.text_disabled("No active tasks or background processes.")
                    elif running:
                        if progress_items:
                            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), f"Background Processes ({len(running)})")
                            imgui.separator()

                        for proc in running:
                            imgui.push_id(f"proc_{proc.pid}")
                            imgui.bullet()

                            # Color code status
                            if proc.status == "error":
                                imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), f"[ERROR] {proc.description}")
                            elif proc.status == "completed":
                                imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"[DONE] {proc.description}")
                            else:
                                imgui.text(proc.description)

                            imgui.indent()
                            imgui.text_disabled(f"PID: {proc.pid} | Started: {proc.elapsed_str()}")

                            # Move Kill button here (next to PID) if active
                            if proc.is_alive():
                                imgui.same_line()
                                if imgui.small_button(f"Kill##{proc.pid}"):
                                    pm.kill(proc.pid)

                            # Show process output (tail of log)
                            if proc.output_path and Path(proc.output_path).is_file():
                                if imgui.tree_node(f"Output##proc_{proc.pid}"):
                                    lines = proc.tail_log(20)
                                    if imgui.begin_child(f"##proc_output_{proc.pid}", imgui.ImVec2(0, 150), imgui.ChildFlags_.borders):
                                        for line in lines:
                                            line_stripped = line.strip()
                                            if "error" in line_stripped.lower():
                                                imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), line_stripped)
                                            elif "warning" in line_stripped.lower():
                                                imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.2, 1.0), line_stripped)
                                            else:
                                                imgui.text(line_stripped)
                                        # Removed auto-scroll to allow reading history
                                        imgui.end_child()
                                    imgui.tree_pop()

                            # Control buttons (Dismiss / Copy)
                            if not proc.is_alive():
                                if imgui.small_button(f"Dismiss##{proc.pid}"):
                                    if proc.pid in pm._processes:
                                        del pm._processes[proc.pid]
                                        pm._save()
                                imgui.same_line()

                            # Copy Log Button (always available if log exists)
                            if proc.output_path and Path(proc.output_path).is_file():
                                if imgui.small_button(f"Copy Log##{proc.pid}"):
                                    try:
                                        with open(proc.output_path, "r", encoding="utf-8") as f:
                                            full_log = f.read()
                                        imgui.set_clipboard_text(full_log)
                                    except Exception:
                                        pass

                            imgui.unindent()
                            imgui.spacing()
                            imgui.pop_id()
                imgui.end_tab_item()

            # Tab 2: System Logs
            if imgui.begin_tab_item("System Logs")[0]:
                with imgui_ctx.begin_child("##SysLogsContent", imgui.ImVec2(0, -30), imgui.ChildFlags_.none):
                    parent.debug_panel.draw()
                imgui.end_tab_item()

            imgui.end_tab_bar()

        # Close button
        imgui.separator()
        if imgui.button("Close", imgui.ImVec2(100, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


def draw_saveas_popup(parent):
    just_opened = False
    if getattr(parent, "_saveas_popup_open"):
        imgui.open_popup("Save As")
        parent._saveas_popup_open = False
        # reset modal open state when reopening popup
        parent._saveas_modal_open = True
        just_opened = True

    # track if popup should remain open
    if not hasattr(parent, "_saveas_modal_open"):
        parent._saveas_modal_open = True

    # set initial size (resizable by user)
    imgui.set_next_window_size(imgui.ImVec2(500, 650), imgui.Cond_.first_use_ever)

    # modal_open is a bool, so we handle the 'X' button manually
    # by checking the second return value of begin_popup_modal.
    opened, visible = imgui.begin_popup_modal(
        "Save As",
        p_open=parent._saveas_modal_open,
        flags=imgui.WindowFlags_.no_saved_settings
    )

    if opened:
        if not visible:
            # user closed via X button or Escape
            parent._saveas_modal_open = False
            imgui.close_current_popup()
            imgui.end_popup()
            return
    else:
        # If not opened, and we didn't just try to open it, ensure state is synced
        if not just_opened:
            parent._saveas_modal_open = False
        return

    # If we are here, popup is open and visible
    if opened:
        # We need to ensure we set modal_open=True while it's open,
        # but we rely on ImGui keeping it open via p_open ref (handled via visible)
        parent._saveas_modal_open = True
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

        # Metadata section
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Metadata")
        imgui.dummy(imgui.ImVec2(0, 5))

        # get array data and metadata
        try:
            current_data = parent.image_widget.data[0]
            current_meta = current_data.metadata or {}
        except (IndexError, AttributeError):
            current_data = None
            current_meta = {}

        # check for required metadata fields from the array
        required_fields = []
        if current_data and hasattr(current_data, 'get_required_metadata'):
            required_fields = current_data.get_required_metadata()

        # track which required fields are missing (not in source metadata or custom)
        missing_required = []
        for field in required_fields:
            canonical = field["canonical"]
            # check if value exists in custom metadata or source
            custom_val = parent._saveas_custom_metadata.get(canonical)
            source_val = field.get("value")  # from get_required_metadata
            if custom_val is None and source_val is None:
                missing_required.append(field)

        # show required metadata fields (always visible, red/green status)
        if required_fields:
            imgui.text("Required:")
            imgui.dummy(imgui.ImVec2(0, 2))

            for field in required_fields:
                canonical = field["canonical"]
                label = field["label"]
                unit = field["unit"]
                dtype = field["dtype"]
                desc = field["description"]

                # check current value (custom overrides source)
                custom_val = parent._saveas_custom_metadata.get(canonical)
                source_val = field.get("value")
                value = custom_val if custom_val is not None else source_val

                # row: label | value/input | set button
                is_set = value is not None
                label_color = imgui.ImVec4(0.4, 0.9, 0.4, 1.0) if is_set else imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
                imgui.text_colored(label_color, f"{label}")
                imgui.same_line(hello_imgui.em_size(8))

                if is_set:
                    imgui.text_colored(imgui.ImVec4(0.4, 0.9, 0.4, 1.0), f"{value} {unit}")
                else:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), "required")

                # tooltip with description
                imgui.same_line()
                imgui.text_disabled("(?)")
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.push_text_wrap_pos(imgui.get_font_size() * 25.0)
                    imgui.text_unformatted(desc)
                    imgui.pop_text_wrap_pos()
                    imgui.end_tooltip()

                # input field
                imgui.same_line(hello_imgui.em_size(18))
                input_key = f"_meta_input_{canonical}"
                if not hasattr(parent, input_key):
                    setattr(parent, input_key, "")

                imgui.set_next_item_width(hello_imgui.em_size(6))
                flags = imgui.InputTextFlags_.chars_decimal if dtype in (float, int) else 0
                _, new_val = imgui.input_text(f"##{canonical}_input", getattr(parent, input_key), flags=flags)
                setattr(parent, input_key, new_val)

                # set button
                imgui.same_line()
                if imgui.small_button(f"Set##{canonical}"):
                    input_val = getattr(parent, input_key).strip()
                    if input_val:
                        try:
                            parsed = dtype(input_val)
                            parent._saveas_custom_metadata[canonical] = parsed
                            if current_data and hasattr(current_data, 'metadata'):
                                if isinstance(current_data.metadata, dict):
                                    current_data.metadata[canonical] = parsed
                            setattr(parent, input_key, "")
                        except (ValueError, TypeError):
                            pass

            imgui.spacing()

        # custom metadata section (always visible, no dropdown)
        imgui.text("Custom:")
        imgui.dummy(imgui.ImVec2(0, 2))

        # show existing custom metadata entries
        to_remove = None
        for key, value in list(parent._saveas_custom_metadata.items()):
            # skip required fields (shown above)
            if any(f["canonical"] == key for f in required_fields):
                continue
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

        # add new key-value pair
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
                val = parent._saveas_custom_value
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                parent._saveas_custom_metadata[parent._saveas_custom_key.strip()] = val
                parent._saveas_custom_key = ""
                parent._saveas_custom_value = ""

        imgui.spacing()

        # store missing required state for save button validation
        parent._saveas_missing_required = missing_required

        imgui.spacing()
        imgui.separator()

        # Timepoints section
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Timepoints")
        imgui.dummy(imgui.ImVec2(0, 5))

        # Get max frames from data
        try:
            first_array = parent.image_widget.data[0]
            max_frames = first_array.shape[0]
        except (IndexError, AttributeError):
            max_frames = 1000

        # initialize num_timepoints if not set or if max changed
        if not hasattr(parent, '_saveas_num_timepoints') or parent._saveas_num_timepoints is None:
            parent._saveas_num_timepoints = max_frames
        if not hasattr(parent, '_saveas_last_max_timepoints'):
            parent._saveas_last_max_timepoints = max_frames
        elif parent._saveas_last_max_timepoints != max_frames:
            parent._saveas_last_max_timepoints = max_frames
            parent._saveas_num_timepoints = max_frames

        imgui.set_next_item_width(hello_imgui.em_size(8))
        changed, new_value = imgui.input_int("##timepoints_input", parent._saveas_num_timepoints, step=1, step_fast=100)
        if changed:
            parent._saveas_num_timepoints = max(1, min(new_value, max_frames))
        imgui.same_line()
        imgui.text(f"/ {max_frames}")
        set_tooltip(
            f"Number of timepoints to save (1-{max_frames}). "
            "Useful for testing on subsets before full conversion."
        )

        imgui.set_next_item_width(hello_imgui.em_size(20))
        slider_changed, slider_value = imgui.slider_int(
            "##timepoints_slider", parent._saveas_num_timepoints, 1, max_frames
        )
        if slider_changed:
            parent._saveas_num_timepoints = slider_value

        imgui.spacing()
        imgui.separator()

        # Z-Plane Selection section
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Z-Plane Selection")
        imgui.dummy(imgui.ImVec2(0, 5))

        try:
            data = parent.image_widget.data[0]
            if hasattr(data, "num_planes"):
                num_planes = data.num_planes
            elif hasattr(data, "num_channels"):
                num_planes = data.num_channels
            elif len(data.shape) == 4:
                num_planes = data.shape[1]
            else:
                num_planes = 1
        except Exception as e:
            num_planes = 1
            hello_imgui.log(
                hello_imgui.LogLevel.error,
                f"Could not read number of planes: {e}",
            )

        # default to all planes selected (only on first open, not when user selects "None")
        if parent._selected_planes is None:
            parent._selected_planes = set(range(num_planes))

        # show summary and button to open selector popup
        n_selected = len(parent._selected_planes)
        if n_selected == num_planes:
            summary = "All planes"
        elif n_selected == 0:
            summary = "None"
        elif n_selected <= 3:
            summary = ", ".join(str(p + 1) for p in sorted(parent._selected_planes))
        else:
            summary = f"{n_selected} of {num_planes}"

        imgui.text(f"Z-planes: {summary}")
        imgui.same_line()
        if imgui.button("Select...##zplanes"):
            imgui.open_popup("Select Z-Planes##saveas")

        # plane selection popup
        popup_height = min(400, 130 + num_planes * 24)
        imgui.set_next_window_size(imgui.ImVec2(300, popup_height), imgui.Cond_.first_use_ever)
        if imgui.begin_popup_modal("Select Z-Planes##saveas", flags=imgui.WindowFlags_.no_saved_settings)[0]:
            # get current z for highlighting
            names = parent.image_widget._slider_dim_names or ()
            try:
                current_z = parent.image_widget.indices["z"] if "z" in names else 0
            except (IndexError, KeyError):
                current_z = 0

            imgui.text("Select planes to save:")
            imgui.spacing()

            if imgui.button("All"):
                parent._selected_planes = set(range(num_planes))
            imgui.same_line()
            if imgui.button("None"):
                parent._selected_planes = set()
            imgui.same_line()
            if imgui.button("Current"):
                parent._selected_planes = {current_z}

            imgui.separator()
            imgui.spacing()

            # adaptive grid of checkboxes
            items = [(f"Plane {i + 1}", i in parent._selected_planes) for i in range(num_planes)]

            def on_plane_change(idx, checked):
                if checked:
                    parent._selected_planes.add(idx)
                else:
                    parent._selected_planes.discard(idx)

            draw_checkbox_grid(items, "saveas_plane", on_plane_change)

            imgui.spacing()
            imgui.separator()
            if imgui.button("Done", imgui.ImVec2(-1, 0)):
                imgui.close_current_popup()

            imgui.end_popup()

        imgui.spacing()
        imgui.separator()

        # run in background option (default to True)
        if not hasattr(parent, '_saveas_background'):
            parent._saveas_background = True
        _, parent._saveas_background = imgui.checkbox(
            "Run in background", parent._saveas_background
        )
        set_tooltip(
            "Run save operation as a separate process that continues after closing the GUI. "
            "Progress will be logged to a file in the output directory."
        )

        imgui.spacing()

        # disable save button if required metadata is missing or no z-planes selected
        missing_fields = getattr(parent, '_saveas_missing_required', None)
        no_planes = parent._selected_planes is not None and len(parent._selected_planes) == 0

        if missing_fields or no_planes:
            imgui.begin_disabled()
            imgui.button("Save", imgui.ImVec2(100, 0))
            imgui.end_disabled()
            imgui.same_line()
            # show error message
            if no_planes:
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.4, 0.4, 1.0),
                    "Select at least one z-plane"
                )
            elif missing_fields:
                missing_names = ", ".join(f["label"] for f in missing_fields)
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.4, 0.4, 1.0),
                    f"Please set required metadata: {missing_names}"
                )
        elif imgui.button("Save", imgui.ImVec2(100, 0)):
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
                        # Convert 0-indexed UI values to 1-indexed ROI values for ScanImageArray
                        rois = sorted([r + 1 for r in parent._saveas_selected_roi])
                    else:
                        rois = None

                    outdir = Path(parent._saveas_outdir).expanduser()
                    if not outdir.exists():
                        outdir.mkdir(parents=True, exist_ok=True)

                    # Get num_timepoints (None means all timepoints)
                    num_timepoints = getattr(parent, '_saveas_num_timepoints', None)
                    try:
                        max_timepoints = parent.image_widget.data[0].shape[0]
                        if num_timepoints is not None and num_timepoints >= max_timepoints:
                            num_timepoints = None  # All timepoints, don't limit
                    except (IndexError, AttributeError):
                        pass

                    # Build metadata overrides dict from custom metadata
                    # (all standard fields are added via the Set buttons in the Metadata section)
                    metadata_overrides = dict(parent._saveas_custom_metadata)

                    # Determine output_suffix: only use custom suffix for multi-ROI stitched data
                    output_suffix = None
                    if rois is None:
                        # Stitching all ROIs - use custom suffix (or default "_stitched")
                        output_suffix = parent._saveas_output_suffix

                    # determine roi_mode based on whether splitting ROIs
                    from mbo_utilities.metadata import RoiMode
                    roi_mode = RoiMode.separate if rois else RoiMode.concat_y

                    save_kwargs = {
                        "path": parent.fpath,
                        "outpath": parent._saveas_outdir,
                        "planes": save_planes,
                        "roi": rois,
                        "roi_mode": roi_mode,
                        "overwrite": parent._overwrite,
                        "debug": parent._debug,
                        "ext": parent._ext,
                        "target_chunk_mb": parent._saveas_chunk_mb,
                        "num_timepoints": num_timepoints,
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

                    frames_msg = f"{num_timepoints} timepoints" if num_timepoints else "all timepoints"
                    roi_msg = f"ROIs {rois}" if rois else roi_mode.description
                    parent.logger.info(f"Saving planes {save_planes} ({frames_msg}), {roi_msg}")
                    parent.logger.info(
                        f"Saving to {parent._saveas_outdir} as {parent._ext}"
                    )

                    # check if running as background process
                    if parent._saveas_background:
                        # spawn as detached subprocess via process manager
                        pm = get_process_manager()
                        # handle fpath being a list (from directory) or single path
                        if isinstance(parent.fpath, (list, tuple)):
                            input_path = str(parent.fpath[0]) if parent.fpath else ""
                            # use parent directory for display name
                            fname = Path(parent.fpath[0]).parent.name if parent.fpath else "data"
                        else:
                            input_path = str(parent.fpath) if parent.fpath else ""
                            fname = Path(parent.fpath).name if parent.fpath else "data"
                            name = Path(parent.fpath).name if parent.fpath else "data"
                        worker_args = {
                            "input_path": input_path,
                            "output_path": str(parent._saveas_outdir),
                            "ext": parent._ext,
                            "planes": save_planes,
                            "num_timepoints": num_timepoints,
                            "rois": rois,
                            "fix_phase": parent.fix_phase,
                            "use_fft": parent.use_fft,
                            "register_z": parent._register_z,
                            "metadata": metadata_overrides if metadata_overrides else {},
                            "kwargs": {
                                "sharded": parent._zarr_sharded if parent._ext == ".zarr" else False,
                                "ome": parent._zarr_ome if parent._ext == ".zarr" else False,
                                "output_suffix": output_suffix
                            }
                        }
                        pid = pm.spawn(
                            task_type="save_as",
                            args=worker_args,
                            description=f"Saving {fname} to {parent._ext}",
                            output_path=str(parent._saveas_outdir),
                        )
                        if pid:
                            parent.logger.info(f"Started background save process (PID {pid})")
                            parent.logger.info("You can close the GUI - the save will continue.")
                        else:
                            parent.logger.error("Failed to start background process")
                    else:
                        # run in foreground thread (existing behavior)
                        reset_progress_state("saveas")
                        parent._saveas_progress = 0.0
                        parent._saveas_done = False
                        parent._saveas_running = True
                        parent.logger.info("Starting save operation...")
                        # Also reset register_z progress if enabled
                        if parent._register_z:
                            reset_progress_state("register_z")
                            parent._register_z_progress = 0.0
                            parent._register_z_done = False
                            parent._register_z_running = True
                            parent._register_z_current_msg = "Starting..."
                        threading.Thread(
                            target=_save_as_worker, kwargs=save_kwargs, daemon=True
                        ).start()
                parent._saveas_modal_open = False
                imgui.close_current_popup()
            except Exception as e:
                parent.logger.info(f"Error saving data: {e}")
                parent._saveas_modal_open = False
                imgui.close_current_popup()

        imgui.same_line()
        if imgui.button("Cancel"):
            parent._saveas_modal_open = False
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
        # ensure assets (fonts + icons) are available for this imgui context
        from mbo_utilities.gui import _setup  # triggers setup on import

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
            from mbo_utilities.gui.pipeline_widgets import Suite2pSettings
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

        # num_graphics matches len(iw.graphics)
        self.num_graphics = len(self.image_widget.graphics)
        self.shape = self.image_widget.data[0].shape

        from mbo_utilities.metadata import has_mbo_metadata
        # Check if raw ScanImage OR has MBO metadata tagged tiff/zarr
        self.is_mbo_scan = (
            isinstance(self.image_widget.data[0], ScanImageArray) or
            has_mbo_metadata(self.fpath) if self.fpath else False
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
        self._zstats_running = [False] * self.num_graphics
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
        self._register_z_running = False
        self._register_z_current_msg = ""

        self._selected_pipelines = None
        self._selected_array = 0
        self._selected_planes = None  # None = uninitialized, empty set = user selected "None"
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
        self._saveas_running = False
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

        # Metadata state for save dialog
        self._saveas_custom_metadata = {}  # user-added key-value pairs
        self._saveas_custom_key = ""  # temp input for new key
        self._saveas_custom_value = ""  # temp input for new value

        # Process monitoring state
        self._viewing_process_pid = None  # selected process for log viewing

        # Output suffix for filename customization (default: "_stitched" for multi-ROI)
        self._saveas_output_suffix = "_stitched"

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
            self.logger.info("Starting zstats computation...")
            # mark all graphics as running immediately
            for i in range(self.num_graphics):
                self._zstats_running[i] = True
            threading.Thread(target=self.compute_zstats, daemon=True).start()

    def set_context_info(self):
        # Update app title with dataset name for easier identification in task manager/OS
        try:
            name = Path(self.fpath[0]).parent.name if isinstance(self.fpath, list) else Path(self.fpath).name
            hello_imgui.get_runner_params().app_shallow_settings.window_title = f"MBO Utilities - {name}"
        except RuntimeError:
            # HelloImGui runner not yet initialized (called during __init__)
            pass

    def draw_background_processes_section(self):
        """Draw listing of background processes and their logs."""
        pm = get_process_manager()
        pm.cleanup_finished()  # clean up dead processes
        running = pm.get_running()

        if not running:
            imgui.text_disabled("No background processes running.")
            return

        imgui.text_colored(
            imgui.ImVec4(0.9, 0.8, 0.3, 1.0),
            f"{len(running)} active process(es):"
        )
        imgui.separator()
        imgui.spacing()

        for proc in running:
            imgui.push_id(f"proc_{proc.pid}")

            # process description
            imgui.bullet()

            # Color code status
            if proc.status == "error":
                 imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), f"[ERROR] {proc.description}")
            elif proc.status == "completed":
                 imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"[DONE] {proc.description}")
            else:
                 imgui.text(proc.description)

            # details
            imgui.indent()
            imgui.text_disabled(f"PID: {proc.pid} | Started: {proc.elapsed_str()}")

            # last log line
            last_line = proc.get_last_log_line()
            if last_line:
                if len(last_line) > 100:
                    last_line = last_line[:97] + "..."
                imgui.text_colored(imgui.ImVec4(0.5, 0.7, 1.0, 0.8), f"> {last_line}")

            # Buttons (Kill/Dismiss and Show console)
            # Use small buttons to fit in the list
            if imgui.small_button("Show Console"):
                self._viewing_process_pid = proc.pid

            imgui.same_line()

            if proc.is_alive():
                if imgui.small_button("Kill"):
                    if pm.kill(proc.pid):
                        self.logger.info(f"Killed process {proc.pid}")
                        if self._viewing_process_pid == proc.pid:
                            self._viewing_process_pid = None
                    else:
                        self.logger.warning(f"Failed to kill process {proc.pid}")
            else:
                if imgui.small_button("Dismiss"):
                     if proc.pid in pm._processes:
                         if self._viewing_process_pid == proc.pid:
                             self._viewing_process_pid = None
                         del pm._processes[proc.pid]
                         pm._save()

            imgui.unindent()
            imgui.spacing()
            imgui.pop_id()

        # console output area for selected process
        if self._viewing_process_pid is not None:
            # find the process
            v_proc = next((p for p in running if p.pid == self._viewing_process_pid), None)
            if v_proc:
                imgui.dummy(imgui.ImVec2(0, 10))
                imgui.text_colored(imgui.ImVec4(0.3, 0.6, 1.0, 1.0), f"Console: {v_proc.description} (PID {v_proc.pid})")
                imgui.same_line(imgui.get_content_region_avail().x - 20)
                if imgui.small_button("x##close_console"):
                    self._viewing_process_pid = None

                # tail log
                lines = v_proc.tail_log(30)
                # Ensure we have a reasonable height for the console
                if imgui.begin_child("##proc_console", imgui.ImVec2(0, 250), imgui.ChildFlags_.borders):
                    for line in lines:
                        line_stripped = line.strip()
                        if "error" in line_stripped.lower():
                            imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), line_stripped)
                        elif "warning" in line_stripped.lower():
                            imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.2, 1.0), line_stripped)
                        elif "success" in line_stripped.lower() or "complete" in line_stripped.lower():
                            imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), line_stripped)
                        else:
                            imgui.text(line_stripped)
                    # auto-scroll
                    imgui.set_scroll_here_y(1.0)
                    imgui.end_child()
            else:
                self._viewing_process_pid = None

        imgui.spacing()
        if running:
            any_alive = any(p.is_alive() for p in running)
            if any_alive:
                if imgui.button("Kill All Processes", imgui.ImVec2(-1, 0)):
                    killed = pm.kill_all()
                    self.logger.info(f"Killed {killed} processes")
            else:
                if imgui.button("Clear Finished Processes", imgui.ImVec2(-1, 0)):
                    to_remove = [p.pid for p in running if not p.is_alive()]
                    for pid in to_remove:
                        del pm._processes[pid]
                    pm._save()

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
        import time

        if isinstance(meta, (int, np.integer)):
            # This is standard save progress
            self._saveas_progress = frac
            self._saveas_current_index = meta
            self._saveas_done = frac >= 1.0
            if frac >= 1.0:
                self._saveas_running = False
                self._saveas_complete_time = time.time()
                self.logger.info("Save complete")

        elif isinstance(meta, str):
            # Suite3D progress message
            self._register_z_progress = frac
            self._register_z_current_msg = meta
            self._register_z_done = frac >= 1.0
            if frac >= 1.0:
                self._register_z_running = False
                self._register_z_complete_time = time.time()

    def _clear_stale_progress(self):
        """Clear completed progress indicators after a delay."""
        import time
        now = time.time()
        clear_delay = 5.0  # seconds to show completion message

        if getattr(self, '_saveas_done', False):
            complete_time = getattr(self, '_saveas_complete_time', 0)
            if now - complete_time > clear_delay:
                self._saveas_done = False
                self._saveas_progress = 0.0

        if getattr(self, '_register_z_done', False):
            complete_time = getattr(self, '_register_z_complete_time', 0)
            if now - complete_time > clear_delay:
                self._register_z_done = False
                self._register_z_progress = 0.0
                self._register_z_current_msg = None

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
        """Get current phase offset from each data array (ScanImageArray)."""
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
            self.logger.debug(f"  Array type: {type(arr).__name__}, has feature: {hasattr(arr, 'phase_correction')}")
            # Check for feature first
            if hasattr(arr, 'phase_correction') and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                return True
            # Legacy fallback
            if hasattr(arr, 'fix_phase') and hasattr(arr, 'use_fft'):
                return True
        return False

    @property
    def has_frame_averaging_support(self) -> bool:
        """Check if any data array supports frame averaging (PiezoArray)."""
        arrays = self._get_data_arrays()
        for arr in arrays:
            if hasattr(arr, 'frames_per_slice') and hasattr(arr, 'can_average'):
                return True
        return False

    @property
    def fix_phase(self) -> bool:
        """Whether bidirectional phase correction is enabled."""
        arrays = self._get_data_arrays()
        if not arrays:
            return False
        arr = arrays[0]
        if hasattr(arr, 'phase_correction') and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.enabled
        return getattr(arr, 'fix_phase', False)

    @fix_phase.setter
    def fix_phase(self, value: bool):
        self.logger.info(f"Setting fix_phase to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, 'phase_correction') and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.enabled = value
            elif hasattr(arr, 'fix_phase'):
                arr.fix_phase = value
        self._refresh_image_widget()

    @property
    def use_fft(self) -> bool:
        """Whether FFT-based phase correlation is used."""
        arrays = self._get_data_arrays()
        if not arrays:
            return False
        arr = arrays[0]
        if hasattr(arr, 'phase_correction') and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.use_fft
        return getattr(arr, 'use_fft', False)

    @use_fft.setter
    def use_fft(self, value: bool):
        self.logger.info(f"Setting use_fft to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, 'phase_correction') and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.use_fft = value
            elif hasattr(arr, 'use_fft'):
                arr.use_fft = value
        self._refresh_image_widget()

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase correlation."""
        arrays = self._get_data_arrays()
        if not arrays:
            return 3
        arr = arrays[0]
        if hasattr(arr, 'phase_correction') and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.border
        return getattr(arr, 'border', 3)

    @border.setter
    def border(self, value: int):
        self.logger.info(f"Setting border to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, 'phase_correction') and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.border = value
            elif hasattr(arr, 'border'):
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
        # Automatically reset vmin/vmax to handle signal changes from blur
        if self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

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
        # Automatically reset vmin/vmax to handle scaled signal
        if self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

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
        # Automatically reset vmin/vmax for new projection scale
        if self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

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

        n_slider_dims = self.processors[0].n_slider_dims if self.processors else 1

        # Per-processor window sizes tuple
        per_processor_sizes = (self._window_size,) + (None,) * (n_slider_dims - 1)
        # Set directly on processors with histogram computation disabled
        # to avoid expensive full-array histogram recomputation
        self._set_processor_attr("window_sizes", per_processor_sizes)
        # Automatically reset vmin/vmax for new window integration
        if self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

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

    def _handle_keyboard_shortcuts(self):
        """Handle global keyboard shortcuts."""
        io = imgui.get_io()

        # skip if any widget has focus (typing in text field)
        # single-key shortcuts MUST be blocked when typing
        if io.want_text_input:
            return

        # o: open file (no modifiers)
        if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.o, False):
            if self._file_dialog is None and self._folder_dialog is None:
                self.logger.info("Shortcut: 'o' (Open File) triggered.")
                fpath = self.fpath[0] if isinstance(self.fpath, list) else self.fpath
                if fpath and Path(fpath).exists():
                    start_dir = str(Path(fpath).parent)
                else:
                    from mbo_utilities.preferences import get_last_dir
                    start_dir = str(get_last_dir("open_file") or Path.home())
                self._file_dialog = pfd.open_file(
                    "Select Data File(s)",
                    start_dir,
                    ["Image Files", "*.tif *.tiff *.zarr *.npy *.bin", "All Files", "*"],
                    pfd.opt.multiselect
                )

        # O (Shift + O): open folder (no ctrl)
        if not io.key_ctrl and io.key_shift and imgui.is_key_pressed(imgui.Key.o, False):
            if self._folder_dialog is None and self._file_dialog is None:
                self.logger.info("Shortcut: 'Shift+O' (Open Folder) triggered.")
                fpath = self.fpath[0] if isinstance(self.fpath, list) else self.fpath
                if fpath and Path(fpath).exists():
                    start_dir = str(Path(fpath).parent)
                else:
                    from mbo_utilities.preferences import get_last_dir
                    start_dir = str(get_last_dir("open_folder") or Path.home())
                self._folder_dialog = pfd.select_folder("Select Data Folder", start_dir)

        # s: toggle save as popup (no modifiers)
        if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.s, False):
            self.logger.info("Shortcut: 's' (Save As) triggered.")
            if getattr(self, "_saveas_modal_open", False):
                self._saveas_modal_open = False
            else:
                self._saveas_popup_open = True

        # m: toggle metadata viewer (no modifiers)
        if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.m, False):
            self.show_metadata_viewer = not self.show_metadata_viewer

        # [: toggle side panel collapse (no modifiers)
        if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.left_bracket, False):
            self.collapsed = not self.collapsed

        # v: reset vmin/vmax (no modifiers)
        if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.v, False):
            if self.image_widget:
                try:
                    self.image_widget.reset_vmin_vmax_frame()
                except Exception:
                    pass

        # enter: reset vmin/vmax for current frame
        if imgui.is_key_pressed(imgui.Key.enter, False) or imgui.is_key_pressed(imgui.Key.keypad_enter, False):
            if self.image_widget:
                try:
                    self.image_widget.reset_vmin_vmax_frame()
                except Exception:
                    pass

        # arrow keys for slider dimensions (only when data is loaded)
        try:
            self._handle_arrow_keys()
        except Exception:
            pass  # ignore errors during data transitions

    def _handle_arrow_keys(self):
        """Handle arrow key navigation for T and Z dimensions."""
        if not self.image_widget or not self.image_widget.data:
            return

        n_sliders = self.image_widget.n_sliders
        if n_sliders == 0:
            return

        # get shape from actual data
        shape = self.image_widget.data[0].shape
        if not isinstance(shape, tuple) or len(shape) < 3:
            return

        current_indices = list(self.image_widget.indices)

        # left/right: T dimension (index 0)
        t_max = shape[0] - 1
        current_t = current_indices[0]

        if imgui.is_key_pressed(imgui.Key.left_arrow):
            new_t = max(0, current_t - 1)
            if new_t != current_t:
                current_indices[0] = new_t
                self.image_widget.indices = current_indices
                return

        if imgui.is_key_pressed(imgui.Key.right_arrow):
            new_t = min(t_max, current_t + 1)
            if new_t != current_t:
                current_indices[0] = new_t
                self.image_widget.indices = current_indices
                return

        # up/down: Z dimension (index 1, only for 4D data)
        if n_sliders >= 2 and len(shape) >= 4:
            z_max = shape[1] - 1
            current_z = current_indices[1]

            if imgui.is_key_pressed(imgui.Key.down_arrow):
                new_z = max(0, current_z - 1)
                if new_z != current_z:
                    current_indices[1] = new_z
                    self.image_widget.indices = current_indices
                    return

            if imgui.is_key_pressed(imgui.Key.up_arrow):
                new_z = min(z_max, current_z + 1)
                if new_z != current_z:
                    current_indices[1] = new_z
                    self.image_widget.indices = current_indices

    def draw_window(self):
        """Override parent to handle keyboard shortcuts and global popups even when collapsed."""
        # always handle keyboard shortcuts regardless of collapsed state
        self._handle_keyboard_shortcuts()
        self._check_file_dialogs()

        # Draw independent floating windows (must be outside super().draw_window() path)
        draw_tools_popups(self)
        draw_saveas_popup(self)
        draw_process_console_popup(self)

        # call parent implementation (handles collapsed state and calls update())
        super().draw_window()

    def update(self):
        # keyboard shortcuts, file dialogs, and tools popups are handled in draw_window()
        # to ensure they work even when the widget is collapsed.
        # Inside update(), we draw things that belong inside the panel.
        draw_menu_bar(self)
        draw_tabs(self)

        # Update mean subtraction or reset vmin/vmax when z-plane changes
        names = self.image_widget._slider_dim_names or ()
        try:
            z_idx = self.image_widget.indices["z"] if "z" in names else 0
        except (IndexError, KeyError):
            z_idx = 0

        if z_idx != self._last_z_idx:
            self._last_z_idx = z_idx
            if self._mean_subtraction:
                # This already triggers a reset inside _update_mean_subtraction
                self._update_mean_subtraction()
            else:
                # Just reset for the new plane if not doing mean subtraction
                if self.image_widget:
                    self.image_widget.reset_vmin_vmax_frame()

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
        from mbo_utilities.reader import imread

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
            old_ndim = 0
            if hasattr(self, 'shape') and self.shape and isinstance(self.shape, tuple):
                old_ndim = len(self.shape)
            new_ndim = new_data.ndim

            # Reset window functions on processors if dimensionality changes
            # This prevents tuple index out of range errors when going 3D->4D or vice versa
            if old_ndim != new_ndim:
                for proc in self.image_widget._image_processors:
                    proc.window_funcs = None
                    proc.window_sizes = None
                    proc.window_order = None

            # iw-array API: use data indexer for replacing data
            # data[0] = new_array triggers _reset_dimensions() automatically
            self.image_widget.data[0] = new_data

            # reset indices to start of data using public API
            if self.image_widget.n_sliders > 0:
                self.image_widget.indices = [0] * self.image_widget.n_sliders

            # Update internal state
            self.fpath = path
            self.shape = new_data.shape

            from mbo_utilities.metadata import has_mbo_metadata
            self.is_mbo_scan = (
                isinstance(new_data, ScanImageArray) or
                has_mbo_metadata(path)
            )

            # Suggest s2p output directory if not set
            if not self._s2p_outdir:
                path_obj = Path(path[0] if isinstance(path, (list, tuple)) else path)
                self._s2p_outdir = str(path_obj.parent / "suite2p")

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

            # Automatically reset vmin/vmax for initial view of new data
            if self.image_widget:
                self.image_widget.reset_vmin_vmax_frame()

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
        self._zstats_running[roi - 1] = False

    def _compute_zstats_single_array(self, idx, arr):
        # Check for pre-computed stats in zarr metadata (instant loading)
        # supports both 'stats' (new) and 'zstats' (legacy) properties
        pre_stats = None
        if hasattr(arr, "stats") and arr.stats is not None:
            pre_stats = arr.stats
        elif hasattr(arr, "zstats") and arr.zstats is not None:
            pre_stats = arr.zstats
        if pre_stats is not None:
            stats = pre_stats
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
            self._zstats_running[idx - 1] = False
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
        self._zstats_running[idx - 1] = False

        # Save stats to array metadata for persistence (zarr files)
        # prefer 'stats' property but fall back to 'zstats' for backwards compat
        if hasattr(arr, "stats"):
            try:
                arr.stats = stats
                self.logger.info(f"Saved stats to array {idx} metadata")
            except Exception as e:
                self.logger.debug(f"Could not save stats to array metadata: {e}")
        elif hasattr(arr, "zstats"):
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
        self._zstats_running = [False] * n
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

        # Mark all as running before starting
        for i in range(n):
            self._zstats_running[i] = True

        # Recompute z-stats
        self.compute_zstats()

    def cleanup(self):
        """Clean up resources when the GUI is closing.

        Should be called before the application exits to properly release
        resources like open windows, file handles, and pending operations.
        """
        # Clean up pipeline instances (suite2p window, etc)
        from mbo_utilities.gui.widgets.pipelines import cleanup_pipelines
        cleanup_pipelines(self)

        # Clean up all widgets
        from mbo_utilities.gui.widgets import cleanup_all_widgets
        cleanup_all_widgets(self._widgets)

        # Clear file dialogs
        self._file_dialog = None
        self._folder_dialog = None
        if hasattr(self, '_s2p_folder_dialog'):
            self._s2p_folder_dialog = None

        self.logger.info("GUI cleanup complete")
