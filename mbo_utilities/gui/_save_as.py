"""
Save As dialog and worker functions.

This module contains the Save As popup dialog for exporting data
to different file formats with various options.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from imgui_bundle import imgui, hello_imgui, portable_file_dialogs as pfd

from mbo_utilities.reader import MBO_SUPPORTED_FTYPES, imread
from mbo_utilities.writer import imwrite
from mbo_utilities.arrays import _sanitize_suffix
from mbo_utilities.arrays.features import DimensionTag, TAG_REGISTRY, parse_timepoint_selection, TimeSelection
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.gui._imgui_helpers import set_tooltip, checkbox_with_tooltip, draw_checkbox_grid
from mbo_utilities.gui._availability import HAS_SUITE3D
from mbo_utilities.gui._selection_ui import draw_selection_table
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.gui.widgets.progress_bar import reset_progress_state
import contextlib

def _get_array_features(widget: Any) -> dict[str, bool]:
    """
    Check which features are available on the current data array.

    Returns a dict mapping feature name to availability.
    Feature detection uses duck typing based on attribute presence.

    Parameters
    ----------
    widget : Any
        Widget with image_widget.data attribute (PreviewDataWidget, BaseViewer, etc.)

    Features
    --------
    phase_correction : bool
        Array supports bidirectional scan phase correction (ScanImageArray).
    z_registration : bool
        Z-plane registration available (suite3d installed + multi-plane data).
    multi_roi : bool
        Array has multiple ROIs that can be saved separately.
    frame_averaging : bool
        Array supports frame averaging (PiezoArray with frames_per_slice > 1).
    """
    try:
        data = widget.image_widget.data[0]
    except (IndexError, AttributeError):
        return {}

    # Get nz from widget (supports both PreviewDataWidget.nz and BaseViewer patterns)
    nz = getattr(widget, "nz", 1)
    if nz == 1 and hasattr(data, "shape") and len(data.shape) == 4:
        nz = data.shape[1]

    return {
        # Phase correction: presence of phase_correction attribute
        "phase_correction": hasattr(data, "phase_correction"),
        # Z-registration: requires suite3d and multi-plane data
        "z_registration": HAS_SUITE3D and nz > 1,
        # Multi-ROI: data has multiple ROIs
        "multi_roi": getattr(data, "num_rois", 1) > 1,
        # Frame averaging: piezo arrays with multiple frames per slice
        "frame_averaging": hasattr(data, "can_average") and getattr(data, "can_average", False),
    }


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


def draw_saveas_popup(parent: Any):
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

    # track options popup state
    if not hasattr(parent, "_saveas_options_open"):
        parent._saveas_options_open = False

    # initialize timepoint selection state (used by save button)
    if not hasattr(parent, "_saveas_tp_error"):
        parent._saveas_tp_error = ""
    if not hasattr(parent, "_saveas_tp_parsed"):
        parent._saveas_tp_parsed = None

    # ensure phase correction defaults (fix_phase=True for save, use_fft=True)
    if not hasattr(parent, "_saveas_fix_phase"):
        parent._saveas_fix_phase = True
    if not hasattr(parent, "_saveas_use_fft"):
        parent._saveas_use_fft = True

    # set initial size (resizable by user)
    imgui.set_next_window_size(imgui.ImVec2(500, 550), imgui.Cond_.first_use_ever)

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

        # === PATH SECTION ===
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Output")
        imgui.dummy(imgui.ImVec2(0, 5))

        imgui.set_next_item_width(hello_imgui.em_size(25))

        # Directory
        current_dir_str = parent._saveas_outdir or ""
        changed, new_str = imgui.input_text("Save Dir", current_dir_str)
        if changed:
            parent._saveas_outdir = new_str

        imgui.same_line()
        if imgui.button("Browse"):
            default_dir = parent._saveas_outdir or str(get_last_dir("save_as") or Path.home())
            parent._saveas_folder_dialog = pfd.select_folder("Select output folder", default_dir)

        # Check if async folder dialog has a result
        if parent._saveas_folder_dialog is not None and parent._saveas_folder_dialog.ready():
            result = parent._saveas_folder_dialog.result()
            if result:
                parent._saveas_outdir = str(result)
                set_last_dir("save_as", result)
            parent._saveas_folder_dialog = None

        # Extension
        imgui.set_next_item_width(hello_imgui.em_size(25))
        _, parent._ext_idx = imgui.combo("Ext", parent._ext_idx, MBO_SUPPORTED_FTYPES)
        parent._ext = MBO_SUPPORTED_FTYPES[parent._ext_idx]

        imgui.spacing()
        imgui.separator()

        # === SELECTION SECTION ===
        _draw_selection_section(parent)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # === SAVE/OPTIONS BUTTONS ===
        _draw_save_button(parent)

        # Options popup (opened by button in _draw_save_button)
        _draw_options_popup(parent)

        # Metadata popup (opened by button in _draw_save_button)
        _draw_metadata_popup(parent)

        imgui.end_popup()

        # mROI section - commented out for later use
        # try:
        #     num_rois = parent.image_widget.data[0].num_rois
        # except (AttributeError, Exception):
        #     num_rois = 1
        #
        # # Only show multi-ROI option if data actually has multiple ROIs
        # if num_rois > 1:
        #     parent._saveas_rois = checkbox_with_tooltip(
        #         "Save ScanImage multi-ROI Separately",
        #         parent._saveas_rois,
        #         "Enable to save each mROI individually."
        #         " mROI's are saved to subfolders: plane1_roi1, plane1_roi2, etc."
        #         " These subfolders can be merged later using mbo_utilities.merge_rois()."
        #         " This can be helpful as often mROI's are non-contiguous and can drift in orthogonal directions over time.",
        #     )
        #     if parent._saveas_rois:
        #         imgui.spacing()
        #         imgui.separator()
        #         imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Choose mROI(s):")
        #         imgui.dummy(imgui.ImVec2(0, 5))
        #
        #         if imgui.button("All##roi"):
        #             parent._saveas_selected_roi = set(range(num_rois))
        #         imgui.same_line()
        #         if imgui.button("None##roi"):
        #             parent._saveas_selected_roi = set()
        #
        #         imgui.columns(2, borders=False)
        #         for i in range(num_rois):
        #             imgui.push_id(f"roi_{i}")
        #             selected = i in parent._saveas_selected_roi
        #             _, selected = imgui.checkbox(f"mROI {i + 1}", selected)
        #             if selected:
        #                 parent._saveas_selected_roi.add(i)
        #             else:
        #                 parent._saveas_selected_roi.discard(i)
        #             imgui.pop_id()
        #             imgui.next_column()
        #         imgui.columns(1)
        # else:
        #     # Reset multi-ROI state when not applicable
        #     parent._saveas_rois = False


def _draw_options_popup(parent: Any):
    """Draw the options popup for advanced save settings."""
    # track when options popup is about to open (for resetting defaults)
    if parent._saveas_options_open:
        imgui.open_popup("Save Options")
        parent._saveas_options_open = False
        # mark that we need to reset defaults on next frame when popup actually opens
        parent._saveas_options_needs_reset = True

    imgui.set_next_window_size(imgui.ImVec2(350, 400), imgui.Cond_.first_use_ever)
    if imgui.begin_popup("Save Options"):
        # reset defaults on first frame popup is actually visible
        if getattr(parent, "_saveas_options_needs_reset", False):
            parent._saveas_fix_phase = True
            parent._saveas_options_needs_reset = False
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Options")
        imgui.dummy(imgui.ImVec2(0, 5))

        # Get available features for current data
        features = _get_array_features(parent)

        # run in background option
        if not hasattr(parent, "_saveas_background"):
            parent._saveas_background = True
        _, parent._saveas_background = imgui.checkbox(
            "Run in background", parent._saveas_background
        )
        set_tooltip(
            "Run save operation as a separate process that continues after closing the GUI. "
            "Progress will be logged to a file in the output directory."
        )

        parent._overwrite = checkbox_with_tooltip(
            "Overwrite", parent._overwrite, "Replace any existing output files."
        )

        # Z-registration: show disabled with reason if unavailable
        if not features.get("z_registration", False):
            imgui.begin_disabled()
        _changed, _reg_value = imgui.checkbox(
            "Register Z-Planes Axially", parent._register_z if features.get("z_registration") else False
        )
        if features.get("z_registration") and _changed:
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
        if not features.get("z_registration", False):
            imgui.end_disabled()

        # Phase correction: only show if data supports it
        # uses separate _saveas_* settings (default True) instead of display settings
        if features.get("phase_correction", False):
            fix_phase_changed, fix_phase_value = imgui.checkbox(
                "Fix Scan Phase", parent._saveas_fix_phase
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
                parent._saveas_fix_phase = fix_phase_value

            use_fft_changed, use_fft_value = imgui.checkbox(
                "Subpixel Phase Correction", parent._saveas_use_fft
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
            if use_fft_changed:
                parent._saveas_use_fft = use_fft_value

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

        imgui.set_next_item_width(hello_imgui.em_size(15))
        _, parent._saveas_chunk_mb = imgui.drag_int(
            "##chunk_size_mb_mb",
            parent._saveas_chunk_mb,
            v_speed=1,
            v_min=1,
            v_max=1024,
        )

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
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), "Pyramid (Multi-resolution)")
            imgui.dummy(imgui.ImVec2(0, 3))

            _, parent._zarr_pyramid = imgui.checkbox("Generate Pyramid", parent._zarr_pyramid)
            set_tooltip(
                "Generate multi-resolution pyramid for faster navigation in napari. "
                "Creates additional downsampled copies (2x per level) of Y and X dimensions. "
                "Increases file size by ~33% but enables smooth zooming in large datasets.",
            )

            if parent._zarr_pyramid:
                imgui.text("Max Levels")
                set_tooltip(
                    "Maximum pyramid levels (0 = full resolution only). "
                    "Each level is 2x smaller in Y and X. "
                    "Default 4 = up to 5 levels (1x, 2x, 4x, 8x, 16x).",
                )
                imgui.set_next_item_width(hello_imgui.em_size(8))
                _, parent._zarr_pyramid_max_layers = imgui.slider_int(
                    "##pyramid_levels", parent._zarr_pyramid_max_layers, 1, 6
                )

                imgui.text("Method")
                set_tooltip(
                    "Downsampling method. 'mean' averages pixels (best for intensity data). "
                    "'nearest' uses nearest neighbor (best for labels/masks).",
                )
                methods = ["mean", "nearest", "gaussian"]
                current_idx = methods.index(parent._zarr_pyramid_method) if parent._zarr_pyramid_method in methods else 0
                imgui.set_next_item_width(hello_imgui.em_size(10))
                if imgui.begin_combo("##pyramid_method", parent._zarr_pyramid_method):
                    for method in methods:
                        selected = method == parent._zarr_pyramid_method
                        if imgui.selectable(method, selected)[0]:
                            parent._zarr_pyramid_method = method
                        if selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()

        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


def _get_suggested_metadata(parent: Any) -> list:
    """Get suggested metadata fields from array."""
    try:
        current_data = parent.image_widget.data[0]
    except (IndexError, AttributeError):
        return []

    fields = []

    # get array-specific suggested fields (e.g., from LBMArray)
    if current_data and hasattr(current_data, "get_suggested_metadata"):
        fields.extend(current_data.get_suggested_metadata())
    # fallback to old name for backwards compat
    elif current_data and hasattr(current_data, "get_required_metadata"):
        fields.extend(current_data.get_required_metadata())

    return fields


def _check_missing_metadata(parent: Any) -> list:
    """Check for missing suggested metadata fields."""
    fields = _get_suggested_metadata(parent)

    missing = []
    for field in fields:
        canonical = field["canonical"]
        custom_val = parent._saveas_custom_metadata.get(canonical)
        source_val = field.get("value")
        if custom_val is None and source_val is None:
            missing.append(field)

    return missing


def _build_suggested_fields(parent: Any) -> list[dict]:
    """Build the complete list of suggested metadata fields."""
    from mbo_utilities.metadata import parse_filename_metadata, get_filename_suggestions

    # get array data
    try:
        current_data = parent.image_widget.data[0]
    except (IndexError, AttributeError):
        current_data = None

    # get array-specific suggested fields
    suggested_fields = _get_suggested_metadata(parent)
    existing_canonicals = {f["canonical"] for f in suggested_fields}

    # add z-step if not already provided
    z_step_canonicals = ("dz", "z_step_um", "axial_step_um")
    if not any(c in existing_canonicals for c in z_step_canonicals):
        z_step_field = {
            "canonical": "dz",
            "label": "Z Step",
            "unit": "\u03bcm",
            "dtype": float,
            "description": "Distance between Z-planes in micrometers.",
        }
        if current_data and hasattr(current_data, "metadata"):
            meta = current_data.metadata
            if isinstance(meta, dict):
                val = meta.get("dz") or meta.get("z_step_um") or meta.get("axial_step_um")
                if val:
                    z_step_field["value"] = val
        suggested_fields.append(z_step_field)
        existing_canonicals.add("dz")

    # parse filename for auto-detected metadata
    filename_meta = None
    if hasattr(parent, "fpath") and parent.fpath:
        fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
        if fpath:
            filename_meta = parse_filename_metadata(str(fpath))

    # add user-provided metadata fields from standard suggestions
    user_fields = get_filename_suggestions()
    for canonical, field_def in user_fields.items():
        if canonical in existing_canonicals:
            continue

        field = dict(field_def)
        # check if value detected from filename
        if filename_meta:
            detected_val = getattr(filename_meta, canonical, None)
            if detected_val:
                field["value"] = detected_val
                field["detected"] = True  # mark as auto-detected

        # check if value in array metadata
        if current_data and hasattr(current_data, "metadata"):
            meta = current_data.metadata
            if isinstance(meta, dict) and canonical in meta:
                field["value"] = meta[canonical]

        suggested_fields.append(field)
        existing_canonicals.add(canonical)

    return suggested_fields


def _draw_metadata_popup(parent: Any):
    """Draw the metadata popup for editing metadata fields."""
    if not hasattr(parent, "_saveas_metadata_open"):
        parent._saveas_metadata_open = False

    if parent._saveas_metadata_open:
        imgui.open_popup("Metadata")
        parent._saveas_metadata_open = False

    imgui.set_next_window_size(imgui.ImVec2(420, 400), imgui.Cond_.first_use_ever)
    if imgui.begin_popup("Metadata"):
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Metadata")
        imgui.dummy(imgui.ImVec2(0, 5))

        # get array data
        try:
            current_data = parent.image_widget.data[0]
        except (IndexError, AttributeError):
            current_data = None

        # build suggested fields (includes filename detection)
        suggested_fields = _build_suggested_fields(parent)

        # draw suggested fields in a table
        if suggested_fields:
            table_flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
            if imgui.begin_table("suggested_meta", 4, table_flags):
                imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(7))
                imgui.table_setup_column("value", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(10))
                imgui.table_setup_column("input", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(8))
                imgui.table_setup_column("btn", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(3))

                for field in suggested_fields:
                    canonical = field["canonical"]
                    label = field["label"]
                    unit = field.get("unit", "")
                    dtype = field.get("dtype", str)
                    desc = field.get("description", "")
                    examples = field.get("examples", [])
                    detected = field.get("detected", False)

                    # get current value (custom overrides source)
                    custom_val = parent._saveas_custom_metadata.get(canonical)
                    source_val = field.get("value")
                    value = custom_val if custom_val is not None else source_val
                    is_set = value is not None

                    imgui.table_next_row()

                    # label column
                    imgui.table_next_column()
                    if is_set:
                        color = imgui.ImVec4(0.5, 0.8, 0.5, 1.0)
                    else:
                        color = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
                    imgui.text_colored(color, label)
                    if imgui.is_item_hovered():
                        tooltip = desc
                        if examples:
                            tooltip += f"\n\nExamples: {', '.join(examples[:5])}"
                        imgui.set_tooltip(tooltip)

                    # value column
                    imgui.table_next_column()
                    if is_set:
                        val_str = f"{value} {unit}".strip()
                        if detected and custom_val is None:
                            # show detected values in cyan
                            imgui.text_colored(imgui.ImVec4(0.4, 0.8, 0.9, 1.0), val_str)
                            if imgui.is_item_hovered():
                                imgui.set_tooltip("Detected from filename")
                        else:
                            imgui.text_colored(imgui.ImVec4(0.5, 0.8, 0.5, 1.0), val_str)
                    else:
                        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "-")

                    # input column
                    imgui.table_next_column()
                    input_key = f"_meta_input_{canonical}"
                    if not hasattr(parent, input_key):
                        setattr(parent, input_key, "")

                    imgui.set_next_item_width(hello_imgui.em_size(7.5))
                    flags = imgui.InputTextFlags_.chars_decimal if dtype in (float, int) else 0
                    _, new_val = imgui.input_text(f"##{canonical}", getattr(parent, input_key), flags=flags)
                    setattr(parent, input_key, new_val)
                    if imgui.is_item_hovered():
                        tip = "Type a value and click Set to save"
                        if dtype == str:
                            tip += " (text)"
                        elif dtype == float:
                            tip += " (number)"
                        imgui.set_tooltip(tip)

                    # set button column
                    imgui.table_next_column()
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

                imgui.end_table()

            imgui.spacing()

        # show existing custom entries as removable tags
        suggested_keys = {f["canonical"] for f in suggested_fields}
        custom_entries = [(k, v) for k, v in parent._saveas_custom_metadata.items() if k not in suggested_keys]

        if custom_entries:
            imgui.dummy(imgui.ImVec2(0, 2))
            to_remove = None
            for key, value in custom_entries:
                imgui.push_id(f"custom_{key}")
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.25, 0.3, 1.0))
                imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.35, 0.4, 1.0))
                tag_text = f"{key}={value}"
                if imgui.small_button(f"{tag_text}  \u00d7"):
                    to_remove = key
                imgui.pop_style_color(2)
                imgui.same_line()
                imgui.pop_id()
            if to_remove:
                del parent._saveas_custom_metadata[to_remove]
            imgui.new_line()

        # add new custom entry row
        imgui.dummy(imgui.ImVec2(0, 2))
        imgui.set_next_item_width(hello_imgui.em_size(8))
        _, parent._saveas_custom_key = imgui.input_text("##key", parent._saveas_custom_key)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Custom key name")
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "=")
        imgui.same_line()
        imgui.set_next_item_width(hello_imgui.em_size(8))
        _, parent._saveas_custom_value = imgui.input_text("##val", parent._saveas_custom_value)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Custom value (auto-detects number vs text)")
        imgui.same_line()
        if imgui.button("+", imgui.ImVec2(hello_imgui.em_size(2), 0)) and parent._saveas_custom_key.strip():
            val = parent._saveas_custom_value
            with contextlib.suppress(ValueError):
                val = float(val) if "." in val else int(val)
            parent._saveas_custom_metadata[parent._saveas_custom_key.strip()] = val
            parent._saveas_custom_key = ""
            parent._saveas_custom_value = ""

        imgui.spacing()
        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


def _draw_selection_section(parent: Any):
    """Draw the selection section with text input for dimension slicing and output path."""
    imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Selection")
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(
            "Select which data to save using start:stop:step notation.\n\n"
            "Format: start:stop or start:stop:step\n"
            "  start = first index (1-based)\n"
            "  stop = last index (inclusive)\n"
            "  step = interval (default 1)\n\n"
            "To exclude frames, add comma + exclude range:\n"
            "  1:100,50:60 = frames 1-100 excluding 50-60\n\n"
            "Examples:\n"
            "  1:100 = frames 1-100\n"
            "  1:100:2 = every other frame (1,3,5...99)\n"
            "  1:1000,200:300 = 1-1000 excluding 200-300"
        )
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
    imgui.dummy(imgui.ImVec2(0, 5))

    # get data dimensions - store reference once
    data = None
    max_frames = 1000
    num_planes = 1
    try:
        data = parent.image_widget.data[0]
        max_frames = data.shape[0]
        if hasattr(data, "num_planes"):
            num_planes = data.num_planes
        elif hasattr(data, "num_channels"):
            num_planes = data.num_channels
        elif len(data.shape) == 4:
            num_planes = data.shape[1]
        else:
            num_planes = 1
    except Exception as e:
        hello_imgui.log(hello_imgui.LogLevel.error, f"Could not read data dimensions: {e}")

    # track file path to reset state when file changes
    current_fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
    current_fpath = str(current_fpath) if current_fpath else ""
    file_changed = False
    if not hasattr(parent, "_saveas_last_fpath"):
        parent._saveas_last_fpath = current_fpath
    elif parent._saveas_last_fpath != current_fpath:
        file_changed = True
        parent._saveas_last_fpath = current_fpath

    # reset selection state when file changes
    if file_changed:
        parent._saveas_tp_selection = f"1:{max_frames}"
        parent._saveas_tp_error = ""
        parent._saveas_tp_parsed = None
        parent._saveas_last_max_tp = max_frames
        parent._saveas_z_start = 1
        parent._saveas_z_stop = num_planes
        parent._saveas_z_step = 1
        parent._saveas_last_num_planes = num_planes

    # initialize timepoint selection state (new single text input)
    if not hasattr(parent, "_saveas_tp_selection"):
        parent._saveas_tp_selection = f"1:{max_frames}"
    if not hasattr(parent, "_saveas_tp_error"):
        parent._saveas_tp_error = ""
    if not hasattr(parent, "_saveas_tp_parsed"):
        parent._saveas_tp_parsed = None
    if not hasattr(parent, "_saveas_last_max_tp"):
        parent._saveas_last_max_tp = max_frames
    elif parent._saveas_last_max_tp != max_frames:
        # update default selection when data changes
        parent._saveas_last_max_tp = max_frames
        parent._saveas_tp_selection = f"1:{max_frames}"
        parent._saveas_tp_parsed = None
        parent._saveas_tp_error = ""

    # initialize z-plane selection state
    if not hasattr(parent, "_saveas_z_start"):
        parent._saveas_z_start = 1
    if not hasattr(parent, "_saveas_z_stop"):
        parent._saveas_z_stop = num_planes
    if not hasattr(parent, "_saveas_z_step"):
        parent._saveas_z_step = 1
    if not hasattr(parent, "_saveas_last_num_planes"):
        parent._saveas_last_num_planes = num_planes
    elif parent._saveas_last_num_planes != num_planes:
        parent._saveas_last_num_planes = num_planes
        parent._saveas_z_start = 1
        parent._saveas_z_stop = num_planes
        parent._saveas_z_step = 1

    # draw selection table using shared component (includes suffix row)
    tp_parsed, z_start, z_stop, z_step = draw_selection_table(
        parent,
        max_frames,
        num_planes,
        tp_attr="_saveas_tp",
        z_attr="_saveas_z",
        id_suffix="_saveas",
        suffix_attr="_saveas_output_suffix",
    )

    # update legacy _selected_planes for compatibility
    if num_planes > 1:
        selected_planes = list(range(z_start, z_stop + 1, z_step))
        parent._selected_planes = set(p - 1 for p in selected_planes)
    else:
        parent._selected_planes = {0}

    # parse selection if not already done (initial load)
    if parent._saveas_tp_parsed is None and not parent._saveas_tp_error:
        try:
            parent._saveas_tp_parsed = parse_timepoint_selection(parent._saveas_tp_selection, max_frames)
        except ValueError as e:
            parent._saveas_tp_error = str(e)

    imgui.spacing()

    # build filename preview
    ext = getattr(parent, "_ext", ".tiff").lstrip(".")
    tags = []

    # timepoint tag from parsed selection
    tp_parsed = parent._saveas_tp_parsed
    if tp_parsed:
        final_indices = tp_parsed.final_indices
        if final_indices:
            # convert 0-based back to 1-based for display
            tp_start_1 = final_indices[0] + 1
            tp_stop_1 = final_indices[-1] + 1
            # detect step from indices
            if len(final_indices) > 1:
                tp_step = final_indices[1] - final_indices[0]
            else:
                tp_step = 1

            if tp_start_1 == tp_stop_1:
                t_tag = DimensionTag(TAG_REGISTRY["T"], start=tp_start_1, stop=None, step=1)
            else:
                t_tag = DimensionTag(
                    TAG_REGISTRY["T"],
                    start=tp_start_1,
                    stop=tp_stop_1,
                    step=tp_step if tp_step != 1 else 1,
                )
            tags.append(t_tag)
        n_frames = tp_parsed.count
    else:
        # fallback if no valid selection
        n_frames = 0

    # z-plane tag - always include if multi-plane
    z_start = getattr(parent, "_saveas_z_start", 1)
    z_stop = getattr(parent, "_saveas_z_stop", 1)
    z_step = getattr(parent, "_saveas_z_step", 1)
    if num_planes > 1:
        if z_start == z_stop:
            z_tag = DimensionTag(TAG_REGISTRY["Z"], start=z_start, stop=None, step=1)
        else:
            z_tag = DimensionTag(
                TAG_REGISTRY["Z"],
                start=z_start,
                stop=z_stop,
                step=z_step if z_step != 1 else 1,
            )
        tags.append(z_tag)

    # build filename
    suffix = getattr(parent, "_saveas_output_suffix", "")
    sanitized_suffix = _sanitize_suffix(suffix).lstrip("_") if suffix else ""
    if tags:
        dim_parts = "_".join(tag.to_string() for tag in tags)
        if sanitized_suffix:
            filename = f"{dim_parts}_{sanitized_suffix}.{ext}"
        else:
            filename = f"{dim_parts}.{ext}"
    else:
        filename = f"{sanitized_suffix}.{ext}" if sanitized_suffix else f"output.{ext}"

    # calculate output info
    n_planes_out = len(range(z_start, z_stop + 1, z_step)) if num_planes > 1 else 1

    # get image dimensions and dtype for size estimate
    Ly, Lx = 512, 512
    dtype_size = 2  # default assume uint16
    if data is not None:
        try:
            # try shape directly (Y, X are last two dims)
            if hasattr(data, "shape") and len(data.shape) >= 2:
                Ly, Lx = data.shape[-2], data.shape[-1]
            # fallback to metadata if shape is lazy/not available
            elif hasattr(data, "metadata") and isinstance(data.metadata, dict):
                meta = data.metadata
                Ly = meta.get("Ly") or meta.get("height") or meta.get("frame_height") or 512
                Lx = meta.get("Lx") or meta.get("width") or meta.get("frame_width") or 512
            if hasattr(data, "dtype"):
                dtype_size = data.dtype.itemsize
        except Exception:
            pass  # keep defaults

    # estimate file size (raw data size, compression varies)
    raw_bytes = n_frames * n_planes_out * Ly * Lx * dtype_size
    if raw_bytes >= 1e9:
        size_str = f"~{raw_bytes / 1e9:.1f} GB"
    elif raw_bytes >= 1e6:
        size_str = f"~{raw_bytes / 1e6:.0f} MB"
    else:
        size_str = f"~{raw_bytes / 1e3:.0f} KB"

    # output shape string
    if num_planes > 1:
        shape_str = f"({n_frames}, {n_planes_out}, {Ly}, {Lx})"
        dims_str = "TZYX"
    else:
        shape_str = f"({n_frames}, {Ly}, {Lx})"
        dims_str = "TYX"

    imgui.spacing()

    # output preview table
    table_flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    if imgui.begin_table("output_preview", 2, table_flags):
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(6))
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)

        # filename row
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Filename")
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.6, 0.9, 0.6, 1.0), filename)
        outdir = getattr(parent, "_saveas_outdir", "")
        if imgui.is_item_hovered() and outdir:
            imgui.begin_tooltip()
            display_path = str(Path(outdir)).replace("\\", "/")
            imgui.text_unformatted(f"{display_path}/{filename}")
            imgui.end_tooltip()

        # size row
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Size")
        imgui.table_next_column()
        imgui.text(size_str)

        # shape row
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Shape")
        imgui.table_next_column()
        imgui.text(f"{shape_str} {dims_str}")

        imgui.end_table()


def _draw_save_button(parent: Any):
    """Draw the save/cancel buttons and handle save logic."""
    # check for missing metadata (for warning indicator, not blocking)
    missing_fields = _check_missing_metadata(parent)
    no_planes = parent._selected_planes is not None and len(parent._selected_planes) == 0
    no_valid_tp = parent._saveas_tp_error or parent._saveas_tp_parsed is None

    if no_planes:
        imgui.begin_disabled()
        imgui.button("Save", imgui.ImVec2(100, 0))
        imgui.end_disabled()
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.4, 0.4, 1.0),
            "Select at least one z-plane"
        )
    elif no_valid_tp:
        imgui.begin_disabled()
        imgui.button("Save", imgui.ImVec2(100, 0))
        imgui.end_disabled()
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.4, 0.4, 1.0),
            "Invalid timepoint selection"
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
                        or len(parent._saveas_selected_roi) == 0
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

                # build frames list from parsed timepoint selection
                tp_parsed = parent._saveas_tp_parsed
                try:
                    max_timepoints = parent.image_widget.data[0].shape[0]
                except (IndexError, AttributeError):
                    max_timepoints = 1000

                # get final frame indices (0-based)
                final_indices_0 = tp_parsed.final_indices if tp_parsed else list(range(max_timepoints))

                # check if selecting all frames (None means all)
                if len(final_indices_0) == max_timepoints and final_indices_0 == list(range(max_timepoints)):
                    frames = None
                else:
                    # convert to 1-based for imwrite
                    frames = [i + 1 for i in final_indices_0]

                # Build metadata overrides dict from custom metadata
                metadata_overrides = dict(parent._saveas_custom_metadata)

                # add timepoint selection metadata (include/exclude info)
                if tp_parsed:
                    tp_meta = tp_parsed.to_metadata()
                    metadata_overrides["timepoint_selection"] = tp_meta

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
                    "frames": frames,
                    "roi": rois,
                    "roi_mode": roi_mode,
                    "overwrite": parent._overwrite,
                    "debug": parent._debug,
                    "ext": parent._ext,
                    "target_chunk_mb": parent._saveas_chunk_mb,
                    # scan-phase correction settings (separate from display)
                    "fix_phase": parent._saveas_fix_phase,
                    "use_fft": parent._saveas_use_fft,
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
                    save_kwargs["pyramid"] = parent._zarr_pyramid
                    if parent._zarr_pyramid:
                        save_kwargs["pyramid_max_layers"] = parent._zarr_pyramid_max_layers
                        save_kwargs["pyramid_method"] = parent._zarr_pyramid_method

                n_frames = len(frames) if frames else max_timepoints
                # build frames message from parsed selection
                if tp_parsed and tp_parsed.exclude_str:
                    frames_msg = f"{n_frames} frames ({tp_parsed.include_str} excl. {tp_parsed.exclude_str})"
                elif tp_parsed:
                    frames_msg = f"{n_frames} frames ({tp_parsed.include_str})"
                else:
                    frames_msg = "all frames"
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
                        "frames": frames,
                        "rois": rois,
                        "fix_phase": parent._saveas_fix_phase,
                        "use_fft": parent._saveas_use_fft,
                        "register_z": parent._register_z,
                        "metadata": metadata_overrides if metadata_overrides else {},
                        "kwargs": {
                            "sharded": parent._zarr_sharded if parent._ext == ".zarr" else False,
                            "ome": parent._zarr_ome if parent._ext == ".zarr" else False,
                            "output_suffix": output_suffix,
                            "pyramid": parent._zarr_pyramid if parent._ext == ".zarr" else False,
                            "pyramid_max_layers": parent._zarr_pyramid_max_layers if parent._ext == ".zarr" and parent._zarr_pyramid else 4,
                            "pyramid_method": parent._zarr_pyramid_method if parent._ext == ".zarr" and parent._zarr_pyramid else "mean",
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

    # metadata button - yellow if missing required fields
    if missing_fields:
        # yellow/warning style for button
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.7, 0.6, 0.1, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.8, 0.7, 0.2, 1.0))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.6, 0.5, 0.1, 1.0))
        if imgui.button("Metadata", imgui.ImVec2(80, 0)):
            parent._saveas_metadata_open = True
        imgui.pop_style_color(3)
        if imgui.is_item_hovered():
            missing_names = ", ".join(f["label"] for f in missing_fields)
            imgui.set_tooltip(f"Missing: {missing_names}")
    else:
        if imgui.button("Metadata", imgui.ImVec2(80, 0)):
            parent._saveas_metadata_open = True

    imgui.same_line()
    if imgui.button("Options", imgui.ImVec2(80, 0)):
        parent._saveas_options_open = True

    imgui.same_line()
    if imgui.button("Cancel"):
        parent._saveas_modal_open = False
        imgui.close_current_popup()
