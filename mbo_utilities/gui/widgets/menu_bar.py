"""
Menu bar and process status indicator.

This module contains the main menu bar and process status indicator.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any

from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs as pfd

from mbo_utilities.preferences import get_last_dir
from mbo_utilities.gui.widgets.process_manager import get_process_manager


def draw_menu_bar(parent: Any):
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
                if imgui.menu_item("Open File", "o", p_selected=False, enabled=True)[0]:
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
                if imgui.menu_item("Open Folder", "Shift+O", p_selected=False, enabled=True)[0]:
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
                    "Help", "F1", p_selected=False, enabled=True
                )[0]:
                    parent._show_help_popup = True
                if imgui.menu_item(
                    "Keybinds", "/", p_selected=False, enabled=True
                )[0]:
                    parent._show_keybinds_popup = True
                imgui.separator()
                if imgui.menu_item(
                    "Online Docs", "", p_selected=False, enabled=True
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


def draw_process_status_indicator(parent: Any):
    """Draw compact process status indicator in top-left with color coding."""
    # Import icons
    try:
        from imgui_bundle import icons_fontawesome as fa
        ICON_IDLE = fa.ICON_FA_CIRCLE
        ICON_RUNNING = fa.ICON_FA_SPINNER
        ICON_ERROR = fa.ICON_FA_EXCLAMATION_TRIANGLE
        ICON_CHECK = fa.ICON_FA_CHECK_CIRCLE
    except (ImportError, AttributeError):
        # Fallback to unicode
        ICON_IDLE = "\uf111"  # circle
        ICON_RUNNING = "\uf110"  # spinner
        ICON_ERROR = "\uf071"  # exclamation-triangle
        ICON_CHECK = "\uf058"  # check-circle

    pm = get_process_manager()
    pm.cleanup_finished()
    all_procs = pm.get_running()

    # Get in-app progress items
    from mbo_utilities.gui.widgets.progress_bar import _get_active_progress_items
    progress_items = _get_active_progress_items(parent)

    # categorize processes
    running_procs = [p for p in all_procs if p.is_alive()]
    completed_procs = [p for p in all_procs if not p.is_alive() and p.status == "completed"]
    error_procs = [p for p in all_procs if not p.is_alive() and p.status == "error"]

    n_running = len(running_procs) + len(progress_items)
    n_completed = len(completed_procs)
    n_errors = len(error_procs)

    # Determine status color and icon
    if n_errors > 0:
        # errors take priority
        status_color = imgui.ImVec4(0.8, 0.2, 0.2, 1.0)  # Dark Red
        status_text = f"{ICON_ERROR} Error ({n_errors})"
    elif n_running > 0:
        # actively running tasks
        status_color = imgui.ImVec4(0.85, 0.45, 0.0, 1.0)  # Dark Orange

        # Add percentage if we have progress items
        if progress_items:
            avg_progress = sum(item["progress"] for item in progress_items) / len(progress_items)
            status_text = f"{ICON_RUNNING} Running ({n_running}) {int(avg_progress * 100)}%"
        else:
            status_text = f"{ICON_RUNNING} Running ({n_running})"
    elif n_completed > 0:
        # completed tasks waiting to be acknowledged
        status_color = imgui.ImVec4(0.15, 0.55, 0.15, 1.0)  # Dark Green
        task_word = "task" if n_completed == 1 else "tasks"
        status_text = f"{ICON_CHECK} Completed {n_completed} {task_word}"
    else:
        # idle
        status_color = imgui.ImVec4(0.15, 0.55, 0.15, 1.0)  # Dark Green
        status_text = f"{ICON_IDLE} Idle"

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


def draw_keybinds_popup(parent: Any):
    """Draw the keybinds cheatsheet popup."""
    if not hasattr(parent, "_show_keybinds_popup"):
        parent._show_keybinds_popup = False

    if parent._show_keybinds_popup:
        imgui.open_popup("Keybinds")
        parent._show_keybinds_popup = False

    imgui.set_next_window_size(imgui.ImVec2(320, 380), imgui.Cond_.first_use_ever)
    if imgui.begin_popup_modal("Keybinds", flags=imgui.WindowFlags_.no_saved_settings)[0]:
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Keyboard Shortcuts")
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 5))

        # keybinds data: (key, description)
        keybinds = [
            ("Navigation", None),
            ("\u2190 / \u2192", "Previous / Next frame (T)"),
            ("\u2191 / \u2193", "Previous / Next z-plane (Z)"),
            ("Shift + \u2190/\u2192", "Jump 10 frames (T)"),
            ("Shift + \u2191/\u2193", "Jump 10 z-planes (Z)"),
            ("", ""),
            ("File", None),
            ("o", "Open file"),
            ("Shift + O", "Open folder"),
            ("s", "Save as"),
            ("", ""),
            ("View", None),
            ("m", "Toggle metadata viewer"),
            ("[", "Toggle side panel"),
            ("v / Enter", "Reset vmin/vmax"),
            ("", ""),
            ("Help", None),
            ("F1 / ?", "Open help"),
            ("/", "Toggle keybinds popup"),
        ]

        table_flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
        if imgui.begin_table("keybinds_table", 2, table_flags):
            imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, 80)
            imgui.table_setup_column("desc", imgui.TableColumnFlags_.width_stretch)

            for key, desc in keybinds:
                if desc is None:
                    # section header
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.dummy(imgui.ImVec2(0, 3))
                    imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), key)
                    imgui.table_next_column()
                elif key == "":
                    # spacer
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.table_next_column()
                else:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text_colored(imgui.ImVec4(0.9, 0.9, 0.5, 1.0), key)
                    imgui.table_next_column()
                    imgui.text(desc)

            imgui.end_table()

        imgui.dummy(imgui.ImVec2(0, 10))
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()
