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

    # enter or p: toggle side panel collapse (no modifiers)
    if not io.key_ctrl and not io.key_shift and (
        imgui.is_key_pressed(imgui.Key.enter, False)
        or imgui.is_key_pressed(imgui.Key.p, False)
    ):
        toggle_side_panel(parent)

    # space: toggle play/pause on the T dim of fpl's built-in sliders widget
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.space, False):
        toggle_playback(parent)

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

    # k: toggle keybinds popup open/close (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.k, False):
        parent._show_keybinds_popup = not getattr(parent, "_show_keybinds_popup", False)
        parent.logger.info(
            f"Shortcut: 'k' (Keybinds {'OPEN' if parent._show_keybinds_popup else 'CLOSE'})"
        )

    # h: show help popup (no modifiers)
    if not io.key_ctrl and not io.key_shift and imgui.is_key_pressed(imgui.Key.h, False):
        parent.logger.info("Shortcut: 'h' (Help)")
        parent._show_help_popup = True

    # arrow keys for slider dimensions (only when data is loaded)
    try:
        handle_arrow_keys(parent)
    except Exception:
        pass  # ignore errors during data transitions


def _get_sliders_ui(parent: Any):
    """Return fpl's ImageWidgetSliders instance, or None."""
    iw = getattr(parent, "image_widget", None)
    if iw is None:
        return None
    sliders = getattr(iw, "_sliders_ui", None)
    if sliders is not None:
        return sliders
    figure = getattr(iw, "figure", None)
    guis = getattr(figure, "guis", None) or {}
    for gui in (guis.values() if hasattr(guis, "values") else guis):
        if gui is not None and gui.__class__.__name__ == "ImageWidgetSliders":
            return gui
    return None


def toggle_playback(parent: Any, dim_index: int = 0) -> None:
    """Toggle play/pause on the given slider dim (default T=0) via fpl's sliders widget."""
    sliders = _get_sliders_ui(parent)
    if sliders is None or not hasattr(sliders, "_playing"):
        return
    playing = sliders._playing
    if dim_index >= len(playing):
        return
    playing[dim_index] = not playing[dim_index]
    if hasattr(sliders, "_last_frame_time") and dim_index < len(sliders._last_frame_time):
        sliders._last_frame_time[dim_index] = 0
    state = "PLAY" if playing[dim_index] else "PAUSE"
    parent.logger.info(f"Shortcut: 'Space' ({state})")


def toggle_side_panel(parent: Any) -> None:
    """Toggle collapse state of the side panel."""
    with contextlib.suppress(Exception):
        parent.collapsed = not parent.collapsed
        parent.logger.info("Shortcut: 'Enter' (Toggle Side Panel)")


def handle_arrow_keys(parent: Any):
    """Handle arrow key navigation for T and Z dimensions."""
    io = imgui.get_io()

    # skip arrow keys when typing in text fields
    if io.want_text_input:
        return

    if not parent.image_widget or not parent.image_widget.data:
        return

    n_sliders = parent.image_widget.n_sliders
    if n_sliders == 0:
        return

    # get shape from actual data
    shape = parent.image_widget.data[0].shape
    if not isinstance(shape, tuple) or len(shape) < 3:
        return

    current_indices = list(parent.image_widget.indices)

    # jump step: 10 when shift held, 1 otherwise
    step = 10 if io.key_shift else 1

    # left/right: T dimension (index 0)
    t_max = shape[0] - 1
    current_t = current_indices[0]

    if imgui.is_key_pressed(imgui.Key.left_arrow):
        new_t = max(0, current_t - step)
        if new_t != current_t:
            current_indices[0] = new_t
            parent.image_widget.indices = current_indices
            return

    if imgui.is_key_pressed(imgui.Key.right_arrow):
        new_t = min(t_max, current_t + step)
        if new_t != current_t:
            current_indices[0] = new_t
            parent.image_widget.indices = current_indices
            return

    # up/down: Z dimension (index 1, only for 4D data)
    if n_sliders >= 2 and len(shape) >= 4:
        z_max = shape[1] - 1
        current_z = current_indices[1]

        if imgui.is_key_pressed(imgui.Key.down_arrow):
            new_z = max(0, current_z - step)
            if new_z != current_z:
                current_indices[1] = new_z
                parent.image_widget.indices = current_indices
                return

        if imgui.is_key_pressed(imgui.Key.up_arrow):
            new_z = min(z_max, current_z + step)
            if new_z != current_z:
                current_indices[1] = new_z
                parent.image_widget.indices = current_indices
