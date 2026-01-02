"""
Menu bar and process status indicator.

This module contains the main menu bar and process status indicator
for the PreviewDataWidget.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs as pfd

from mbo_utilities.preferences import get_last_dir
from mbo_utilities.gui.widgets.process_manager import get_process_manager

if TYPE_CHECKING:
    from .imgui import PreviewDataWidget


def draw_menu_bar(parent: PreviewDataWidget):
    """Draw the menu bar within the current window/child scope."""
    with imgui_ctx.begin_child(
        "menu",
        window_flags=imgui.WindowFlags_.menu_bar,
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


def draw_process_status_indicator(parent: PreviewDataWidget):
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
    from mbo_utilities.gui.widgets.progress_bar import _get_active_progress_items
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
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1, 1, 1, 1))  # Always white text

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

    imgui.pop_style_color(4)  # button, text, hovered, active

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
    imgui.pop_style_var()  # frame_rounding
