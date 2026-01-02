"""
Save As dialog and worker functions.

This module contains the Save As popup dialog for exporting data
to different file formats with various options.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from imgui_bundle import imgui, hello_imgui, portable_file_dialogs as pfd

from mbo_utilities.reader import MBO_SUPPORTED_FTYPES, imread
from mbo_utilities.writer import imwrite
from mbo_utilities.arrays import _sanitize_suffix
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.gui._imgui_helpers import set_tooltip, checkbox_with_tooltip, draw_checkbox_grid
from mbo_utilities.gui._availability import HAS_SUITE3D
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.gui.widgets.progress_bar import reset_progress_state
import contextlib

if TYPE_CHECKING:
    from .imgui import PreviewDataWidget


def _save_as_worker(path, **imwrite_kwargs):
    """Background worker for saving data to disk."""
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


def draw_saveas_popup(parent: PreviewDataWidget):
    """Draw the Save As popup dialog."""
    just_opened = False
    if parent._saveas_popup_open:
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
        if not hasattr(parent, "_saveas_input_last_value"):
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
                    'Illegal characters (<>:"/\\|?*) are removed.\n'
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
        _draw_metadata_section(parent)

        imgui.spacing()
        imgui.separator()

        # Timepoints section
        _draw_timepoints_section(parent)

        imgui.spacing()
        imgui.separator()

        # Z-Plane Selection section
        _draw_zplane_section(parent)

        imgui.spacing()
        imgui.separator()

        # run in background option (default to True)
        if not hasattr(parent, "_saveas_background"):
            parent._saveas_background = True
        _, parent._saveas_background = imgui.checkbox(
            "Run in background", parent._saveas_background
        )
        set_tooltip(
            "Run save operation as a separate process that continues after closing the GUI. "
            "Progress will be logged to a file in the output directory."
        )

        imgui.spacing()

        # Save button
        _draw_save_button(parent)

        imgui.end_popup()


def _draw_metadata_section(parent: PreviewDataWidget):
    """Draw the metadata section of the save dialog."""
    imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Metadata")
    imgui.dummy(imgui.ImVec2(0, 5))

    # get array data and metadata
    try:
        current_data = parent.image_widget.data[0]
    except (IndexError, AttributeError):
        current_data = None

    # check for required metadata fields from the array
    required_fields = []
    if current_data and hasattr(current_data, "get_required_metadata"):
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
                        if current_data and hasattr(current_data, "metadata"):
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
    if imgui.button("Add") and parent._saveas_custom_key.strip():
        val = parent._saveas_custom_value
        with contextlib.suppress(ValueError):
            val = float(val) if "." in val else int(val)
        parent._saveas_custom_metadata[parent._saveas_custom_key.strip()] = val
        parent._saveas_custom_key = ""
        parent._saveas_custom_value = ""

    imgui.spacing()

    # store missing required state for save button validation
    parent._saveas_missing_required = missing_required


def _draw_timepoints_section(parent: PreviewDataWidget):
    """Draw the timepoints section of the save dialog."""
    imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Timepoints")
    imgui.dummy(imgui.ImVec2(0, 5))

    # Get max frames from data
    try:
        first_array = parent.image_widget.data[0]
        max_frames = first_array.shape[0]
    except (IndexError, AttributeError):
        max_frames = 1000

    # initialize num_timepoints if not set or if max changed
    if not hasattr(parent, "_saveas_num_timepoints") or parent._saveas_num_timepoints is None:
        parent._saveas_num_timepoints = max_frames
    if not hasattr(parent, "_saveas_last_max_timepoints"):
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


def _draw_zplane_section(parent: PreviewDataWidget):
    """Draw the z-plane selection section of the save dialog."""
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


def _draw_save_button(parent: PreviewDataWidget):
    """Draw the save/cancel buttons and handle save logic."""
    # disable save button if required metadata is missing or no z-planes selected
    missing_fields = getattr(parent, "_saveas_missing_required", None)
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
                num_timepoints = getattr(parent, "_saveas_num_timepoints", None)
                try:
                    max_timepoints = parent.image_widget.data[0].shape[0]
                    if num_timepoints is not None and num_timepoints >= max_timepoints:
                        num_timepoints = None  # All timepoints, don't limit
                except (IndexError, AttributeError):
                    pass

                # Build metadata overrides dict from custom metadata
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
