"""
Keyboard shortcut handlers.

This module contains keyboard shortcut handling for the PreviewDataWidget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui, portable_file_dialogs as pfd

from mbo_utilities.preferences import get_last_dir
import contextlib


def handle_keyboard_shortcuts(parent: Any):
    """Handle global keyboard shortcuts."""
    io = imgui.get_io()

    # skip if any widget has focus (typing in text field)
    # single-key shortcuts MUST be blocked when typing
    if io.want_text_input:
        return

    # o: open file (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.o, False):
        if parent._file_dialog is None and parent._folder_dialog is None:
            parent.logger.info("Shortcut: 'o' (Open File)")
            fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
            if fpath and Path(fpath).exists():
                start_dir = str(Path(fpath).parent)
            else:
                start_dir = str(get_last_dir("open_file") or Path.home())
            parent._file_dialog = pfd.open_file(
                "Select Data File(s)",
                start_dir,
                ["Image Files", "*.tif *.tiff *.zarr *.npy *.bin", "All Files", "*"],
                pfd.opt.multiselect
            )

    # O (Shift + O): open folder (no ctrl)
    if not io.key_ctrl and io.key_shift and imgui.is_key_pressed(imgui.Key.o, False):
        if parent._folder_dialog is None and parent._file_dialog is None:
            parent.logger.info("Shortcut: 'Shift+O' (Open Folder)")
            fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
            if fpath and Path(fpath).exists():
                start_dir = str(Path(fpath).parent)
            else:
                start_dir = str(get_last_dir("open_folder") or Path.home())
            parent._folder_dialog = pfd.select_folder("Select Data Folder", start_dir)

    # s: open save as popup (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.s, False):
        parent.logger.info("Shortcut: 's' (Save As)")
        parent._saveas_popup_open = True

    # m: toggle metadata viewer (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.m, False):
        parent.logger.info("Shortcut: 'm' (Metadata Viewer)")
        parent.show_metadata_viewer = not parent.show_metadata_viewer

    # p: toggle side panel collapse (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.p, False):
        parent.logger.info("Shortcut: 'p' (Toggle Side Panel)")
        parent.collapsed = not parent.collapsed

    # v: reset vmin/vmax (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.v, False):
        if parent.image_widget:
            parent.logger.info("Shortcut: 'v' (Reset vmin/vmax)")
            with contextlib.suppress(Exception):
                parent.image_widget.reset_vmin_vmax_frame()

    # c: toggle auto-contrast on z-change (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.c, False):
        parent.auto_contrast_on_z = not parent.auto_contrast_on_z
        state = "ON" if parent.auto_contrast_on_z else "OFF"
        parent.logger.info(f"Shortcut: 'c' (Auto-contrast on Z: {state})")

    # k: toggle keybinds popup (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.k, False):
        parent.logger.info("Shortcut: 'k' (Keybinds)")
        parent._show_keybinds_popup = not getattr(parent, "_show_keybinds_popup", False)

    # h: show help popup (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.h, False):
        parent.logger.info("Shortcut: 'h' (Help)")
        parent._show_help_popup = True

    # arrow keys for slider dimensions (only when data is loaded)
    try:
        handle_arrow_keys(parent)
    except Exception:
        pass  # ignore errors during data transitions


def handle_arrow_keys(parent: Any):
    """Handle arrow key navigation for T and Z dimensions."""
    io = imgui.get_io()

    # skip arrow keys when typing in text fields
    if io.want_text_input:
        return

    if not parent.image_widget or not parent.image_widget.data:
        return

    slider_dims = parent.image_widget.slider_dims
    if not slider_dims:
        return

    # get shape from actual data
    shape = parent.image_widget.data[0].shape
    if not isinstance(shape, tuple) or len(shape) < 3:
        return

    current_index = dict(parent.image_widget.current_index)

    # jump step: 10 when shift held, 1 otherwise
    step = 10 if io.key_shift else 1

    # left/right: T dimension
    if "t" in current_index:
        t_max = shape[0] - 1
        current_t = current_index["t"]

        if imgui.is_key_pressed(imgui.Key.left_arrow):
            new_t = max(0, current_t - step)
            if new_t != current_t:
                parent.image_widget.current_index = {"t": new_t}
                return

        if imgui.is_key_pressed(imgui.Key.right_arrow):
            new_t = min(t_max, current_t + step)
            if new_t != current_t:
                parent.image_widget.current_index = {"t": new_t}
                return

    # up/down: Z dimension (only for 4D+ data)
    if "z" in current_index and len(shape) >= 4:
        z_max = shape[1] - 1
        current_z = current_index["z"]

        if imgui.is_key_pressed(imgui.Key.down_arrow):
            new_z = max(0, current_z - step)
            if new_z != current_z:
                parent.image_widget.current_index = {"z": new_z}
                return

        if imgui.is_key_pressed(imgui.Key.up_arrow):
            new_z = min(z_max, current_z + step)
            if new_z != current_z:
                parent.image_widget.current_index = {"z": new_z}
