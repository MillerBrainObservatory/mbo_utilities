import inspect
from collections.abc import Mapping, Sequence
from imgui_bundle import (
    imgui,
    imgui_ctx,
)
from imgui_bundle import imgui_md

# font awesome icons (with fallback)
try:
    from imgui_bundle import icons_fontawesome as fa
    ICON_SEARCH = fa.ICON_FA_SEARCH
except (ImportError, AttributeError):
    ICON_SEARCH = "\uf002"  # font awesome search icon codepoint

_NAME_COLORS = (
    imgui.ImVec4(0.95, 0.80, 0.30, 1.0),  # gold/yellow for names
    imgui.ImVec4(0.60, 0.95, 0.40, 1.0),  # green for indices
)
_VALUE_COLOR = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
_TREE_NODE_COLOR = imgui.ImVec4(0.40, 0.80, 0.95, 1.0)  # cyan for tree nodes
_ALIAS_COLOR = imgui.ImVec4(0.5, 0.5, 0.5, 0.8)  # muted gray for aliases

# category-specific colors for metadata
_IMAGING_COLOR = imgui.ImVec4(0.4, 0.9, 0.6, 1.0)  # bright green for imaging params
_ACQUISITION_COLOR = imgui.ImVec4(0.9, 0.6, 0.4, 1.0)  # orange for acquisition params
_OTHER_COLOR = imgui.ImVec4(0.75, 0.85, 0.95, 1.0)  # light blue for uncategorized
_DISABLED_COLOR = imgui.ImVec4(0.9, 0.4, 0.4, 1.0)  # red for disabled modules

# custom tooltips for specific metadata keys
METADATA_TOOLTIPS: dict[str, str] = {
    "roi_groups": "ROI scan regions defined in ScanImage. Each group contains position, size, and scan parameters for a multi-ROI acquisition.",
    "roi_mode": "How multi-ROI data is handled: concat_y stitches ROIs horizontally, separate keeps them as individual files.",
}


def _colored_tree_node(label: str) -> bool:
    """Create a tree node with colored text."""
    imgui.push_style_color(imgui.Col_.text, _TREE_NODE_COLOR)
    result = imgui.tree_node(label)
    imgui.pop_style_color()
    return result


def checkbox_with_tooltip(_label, _value, _tooltip):
    _changed, _value = imgui.checkbox(_label, _value)
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(_tooltip)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
    return _value


def set_tooltip(_tooltip, _show_mark=True):
    """set a tooltip with or without a (?)"""
    if _show_mark:
        imgui.same_line()
        imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(_tooltip)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def draw_checkbox_grid(
    items: list[tuple[str, bool]],
    id_prefix: str,
    on_change: callable,
    item_width: float | None = None,
    min_columns: int = 1,
    max_columns: int = 6,
) -> None:
    """
    draw a grid of checkboxes that adapts column count to available width.

    dynamically calculates how many columns fit based on:
    1. available content width from imgui
    2. item width (checkbox + label + padding)

    columns are added until items no longer fit, maximizing space usage
    while keeping the layout readable.

    parameters
    ----------
    items : list[tuple[str, bool]]
        list of (label, checked) tuples for each checkbox
    id_prefix : str
        unique prefix for imgui IDs (e.g., "saveas" -> "##saveas0")
    on_change : callable
        callback(index, new_value) called when a checkbox changes
    item_width : float, optional
        width per item in pixels. if None, calculated from longest label
    min_columns : int
        minimum columns to show (default 1)
    max_columns : int
        maximum columns to allow (default 6)

    example
    -------
    >>> items = [(f"Plane {i+1}", i in selected) for i in range(num_planes)]
    >>> def on_change(idx, checked):
    ...     if checked:
    ...         selected.add(idx)
    ...     else:
    ...         selected.discard(idx)
    >>> draw_checkbox_grid(items, "planes", on_change)
    """
    if not items:
        return

    # calculate item width if not provided
    if item_width is None:
        # checkbox width (~20px) + spacing + text width + padding
        checkbox_width = 20
        padding = 16
        longest_label = max(len(label) for label, _ in items)
        char_width = imgui.get_font_size() * 0.5  # approximate char width
        item_width = checkbox_width + (longest_label * char_width) + padding

    # get available width and calculate column count
    available_width = imgui.get_content_region_avail().x
    num_columns = max(min_columns, min(max_columns, int(available_width / item_width)))

    # draw table with calculated columns
    if imgui.begin_table(f"##{id_prefix}_grid", num_columns, imgui.TableFlags_.none):
        for i, (label, checked) in enumerate(items):
            col = i % num_columns
            if col == 0:
                imgui.table_next_row()
            imgui.table_next_column()

            changed, new_checked = imgui.checkbox(f"{label}##{id_prefix}{i}", checked)
            if changed:
                on_change(i, new_checked)

        imgui.end_table()


def compact_header(label: str, default_open: bool = False) -> bool:
    """
    draw a compact collapsing header with reduced padding.

    returns True if the header is open, False if collapsed.
    use as: if compact_header("Section"): draw_content()
    """
    # reduce vertical spacing
    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 2))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 2))

    flags = imgui.TreeNodeFlags_.default_open if default_open else 0
    is_open = imgui.collapsing_header(label, flags)
    if isinstance(is_open, tuple):
        is_open = is_open[0]

    imgui.pop_style_var(2)
    return is_open


# Track popup states globally by popup_id
_popup_states: dict[str, bool] = {}


def settings_row_with_popup(
    popup_id: str,
    label: str,
    enabled: bool,
    draw_settings_content: callable,
    tooltip: str = "",
    checkbox_tooltip: str = "",
    popup_width: float = 400,
    popup_height: float = 0,
) -> tuple[bool, bool]:
    """
    Draw a compact settings row: [checkbox] Label [Settings button] -> popup.

    Parameters
    ----------
    popup_id : str
        Unique identifier for the popup (used for imgui ID and state tracking).
    label : str
        Label shown next to checkbox and as popup title.
    enabled : bool
        Current enabled state for the checkbox.
    draw_settings_content : callable
        Function to call to draw the popup content. Called with no arguments.
    tooltip : str, optional
        Tooltip shown when hovering the Settings button.
    checkbox_tooltip : str, optional
        Tooltip shown when hovering the checkbox.
    popup_width : float, optional
        Width of the popup window. Default 400.
    popup_height : float, optional
        Height of the popup window. 0 = auto-size. Default 0.

    Returns
    -------
    tuple[bool, bool]
        (enabled_changed, new_enabled_value)

    Example
    -------
    >>> changed, self.do_registration = settings_row_with_popup(
    ...     "reg_settings",
    ...     "Registration",
    ...     self.do_registration,
    ...     lambda: draw_registration_settings(self),
    ...     tooltip="Configure registration parameters"
    ... )
    """
    global _popup_states

    # Initialize popup state if needed
    if popup_id not in _popup_states:
        _popup_states[popup_id] = False

    # Draw checkbox
    changed, new_enabled = imgui.checkbox(f"##{popup_id}_checkbox", enabled)

    # Add checkbox tooltip
    if checkbox_tooltip:
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted(checkbox_tooltip)
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()

    # Draw label
    imgui.same_line()
    imgui.text(label)

    # Draw Settings button
    imgui.same_line()
    if imgui.button(f"Settings##{popup_id}"):
        _popup_states[popup_id] = True
        imgui.open_popup(f"{label} Settings##{popup_id}")

    # Add button tooltip
    if tooltip:
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted(tooltip)
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()

    # Draw popup
    if popup_height > 0:
        imgui.set_next_window_size(
            imgui.ImVec2(popup_width, popup_height), imgui.Cond_.first_use_ever
        )
    else:
        imgui.set_next_window_size(
            imgui.ImVec2(popup_width, 0), imgui.Cond_.first_use_ever
        )

    opened, visible = imgui.begin_popup_modal(
        f"{label} Settings##{popup_id}",
        p_open=True if _popup_states[popup_id] else None,
        flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
    )

    if opened:
        if not visible:
            # User closed via X button
            _popup_states[popup_id] = False
            imgui.close_current_popup()
        else:
            # Draw the settings content
            draw_settings_content()

            # Close button at bottom
            imgui.spacing()
            imgui.separator()
            if imgui.button("Close", imgui.ImVec2(80, 0)):
                _popup_states[popup_id] = False
                imgui.close_current_popup()

        imgui.end_popup()

    return changed, new_enabled


def _fmt_multivalue(value, max_items=8):
    """format a value that may be a list of per-camera values."""
    if isinstance(value, (list, tuple)):
        if len(value) <= max_items:
            # round floats for cleaner display
            formatted = []
            for v in value:
                if isinstance(v, float):
                    if v == int(v):
                        formatted.append(str(int(v)))
                    else:
                        formatted.append(f"{v:.4g}")
                else:
                    formatted.append(str(v))
            return "[" + ", ".join(formatted) + "]"
        else:
            # truncate long lists
            formatted = []
            for v in value[:max_items]:
                if isinstance(v, float):
                    formatted.append(f"{v:.4g}")
                else:
                    formatted.append(str(v))
            return "[" + ", ".join(formatted) + f", +{len(value)-max_items}...]"
    return _fmt(value)


# module-level state for metadata search
_metadata_search_filter = ""
_metadata_search_active = False
_metadata_search_focus_requested = False
_clipboard_toast_timer = 0.0  # for showing "copied" feedback


def _matches_filter_shallow(key: str, value, filter_text: str) -> bool:
    """check if key or stringified value matches the search filter (case-insensitive)."""
    if not filter_text:
        return True
    filter_lower = filter_text.lower()
    if filter_lower in str(key).lower():
        return True
    # only check stringified value for non-containers
    if not isinstance(value, (Mapping, list, tuple)):
        if filter_lower in str(value).lower():
            return True
    return False


def _matches_filter_recursive(key: str, value, filter_text: str) -> bool:
    """recursively check if key, value, or any nested children match the filter."""
    if not filter_text:
        return True

    # check current key/value
    if _matches_filter_shallow(key, value, filter_text):
        return True

    # recurse into nested structures
    if isinstance(value, Mapping):
        for k, v in value.items():
            if _matches_filter_recursive(k, v, filter_text):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for i, v in enumerate(value):
            if _matches_filter_recursive(f"[{i}]", v, filter_text):
                return True

    return False


def _copy_to_clipboard(text: str):
    """copy text to clipboard using imgui."""
    imgui.set_clipboard_text(str(text))


def _clickable_value(key: str, value, value_col: int, color=None, unit: str = ""):
    """render a value at the specified column."""
    if color is None:
        color = _VALUE_COLOR

    val_str = _fmt_multivalue(value)
    if unit:
        val_str += f" ({unit})"

    imgui.same_line(value_col)
    imgui.text_colored(color, val_str)


def _get_key_color(key: str, imaging_keys: set, acquisition_keys: set, alias_map: dict):
    """get color for a metadata key based on its category."""
    key_lower = key.lower()
    # check if it's an imaging param or alias
    canonical = alias_map.get(key_lower)
    if canonical is not None:
        if canonical in imaging_keys:
            return _IMAGING_COLOR
        if canonical in acquisition_keys:
            return _ACQUISITION_COLOR
    # also check if the key itself (case-insensitive) is in the sets
    if key in imaging_keys or key_lower in {k.lower() for k in imaging_keys}:
        return _IMAGING_COLOR
    if key in acquisition_keys or key_lower in {k.lower() for k in acquisition_keys}:
        return _ACQUISITION_COLOR
    return _OTHER_COLOR


def _is_disabled_si_module(value) -> bool:
    """check if a scanimage module dict has enable=false."""
    if not isinstance(value, Mapping):
        return False
    enable_val = value.get("enable")
    if enable_val is False:
        return True
    if isinstance(enable_val, str) and enable_val.lower() in ("false", "0"):
        return True
    return False


def draw_metadata_inspector(metadata: dict, data_array=None):
    """draw metadata with canonical params first, then other fields.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to display.
    data_array : object, optional
        Data array object that may provide contextual descriptions via
        get_param_description(). If provided, tooltips will show array-specific
        context for metadata parameters.
    """
    global _metadata_search_filter, _metadata_search_active, _metadata_search_focus_requested
    from mbo_utilities.metadata import METADATA_PARAMS, IMAGING_METADATA_KEYS, ALIAS_MAP

    with imgui_ctx.begin_child("Metadata Viewer"):
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4))

        try:
            shown_keys = set()
            value_col = 180

            # centered header with help tooltip
            avail_width = imgui.get_content_region_avail().x
            header_text = "Metadata (?)"
            text_width = imgui.calc_text_size(header_text).x
            imgui.set_cursor_pos_x((avail_width - text_width) / 2)
            imgui.text(header_text)
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
                imgui.text_unformatted(
                    "Hover parameter names to see aliases.\n\n"
                    "Imaging parameters (green) have standardized aliases for "
                    "compatibility with ImageJ, OME-TIFF, OME-Zarr, and other systems.\n\n"
                    "Acquisition parameters (orange) describe the recording mode.\n\n"
                    "imread/imwrite handles all alias conversions automatically."
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()

            # keyboard shortcuts
            io = imgui.get_io()
            # ctrl+L to toggle search and focus input
            if io.key_ctrl and imgui.is_key_pressed(imgui.Key.l):
                _metadata_search_active = not _metadata_search_active
                if _metadata_search_active:
                    _metadata_search_focus_requested = True
                else:
                    _metadata_search_filter = ""
            # escape to close search
            if _metadata_search_active and imgui.is_key_pressed(imgui.Key.escape):
                _metadata_search_active = False
                _metadata_search_filter = ""

            # search button on the right side
            imgui.same_line(avail_width - 24)
            search_color = _TREE_NODE_COLOR if _metadata_search_active else imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
            imgui.push_style_color(imgui.Col_.text, search_color)
            if imgui.small_button(ICON_SEARCH + "##search"):
                _metadata_search_active = not _metadata_search_active
                if _metadata_search_active:
                    _metadata_search_focus_requested = True
                else:
                    _metadata_search_filter = ""
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Search metadata (Ctrl+L)")

            # search input field
            if _metadata_search_active:
                if _metadata_search_focus_requested:
                    imgui.set_keyboard_focus_here()
                    _metadata_search_focus_requested = False
                imgui.set_next_item_width(-1)
                changed, _metadata_search_filter = imgui.input_text_with_hint(
                    "##metadata_search",
                    "filter by key or value...",
                    _metadata_search_filter,
                )

            imgui.spacing()

            # file info section (if source path available)
            source_path = metadata.get("source_path") or metadata.get("path") or metadata.get("file_path")
            if source_path:
                shown_keys.update({"source_path", "path", "file_path"})
                file_matches = not _metadata_search_filter or _matches_filter_recursive("source", source_path, _metadata_search_filter)
                if file_matches:
                    imgui.text_colored(_TREE_NODE_COLOR, "File")
                    imgui.separator()
                    # truncate long paths for display
                    path_str = str(source_path)
                    display_path = path_str if len(path_str) <= 50 else "..." + path_str[-47:]
                    imgui.text_colored(_NAME_COLORS[0], "path")
                    imgui.same_line(value_col)
                    imgui.text_colored(_VALUE_COLOR, display_path)
                    if imgui.is_item_hovered() and len(path_str) > 50:
                        imgui.set_tooltip(path_str)
                    imgui.spacing()

            # acquisition fields to show in separate section (not in Imaging)
            acq_field_keys = {"stack_type", "lbm_stack", "piezo_stack", "num_mrois", "num_color_channels"}

            # detect acquisition mode
            is_lbm = metadata.get("lbm_stack", False) or metadata.get("stack_type") == "lbm"

            # imaging section header
            imgui.text_colored(_TREE_NODE_COLOR, "Imaging")
            imgui.separator()

            # show only core imaging params (from IMAGING_METADATA_KEYS), always show even if None
            for key in IMAGING_METADATA_KEYS:
                param = METADATA_PARAMS.get(key)
                if not param:
                    continue

                value = metadata.get(param.canonical)
                if value is None:
                    for alias in param.aliases:
                        if alias in metadata:
                            value = metadata[alias]
                            break

                # mark as shown
                shown_keys.add(param.canonical)
                shown_keys.update(param.aliases)

                # skip if doesn't match filter
                if _metadata_search_filter and not _matches_filter_recursive(param.canonical, value if value is not None else "", _metadata_search_filter):
                    continue

                # parameter name with tooltip on hover
                imgui.text_colored(_IMAGING_COLOR, param.canonical)
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    # show contextual description if data_array provides it
                    if data_array and hasattr(data_array, 'get_param_description'):
                        desc = data_array.get_param_description(param.canonical)
                        if desc:
                            imgui.text_wrapped(desc)
                            imgui.separator()
                    elif param.description:
                        imgui.text_wrapped(param.description)
                        imgui.separator()
                    # show aliases
                    if param.aliases:
                        imgui.text("Aliases:")
                        for alias in param.aliases:
                            imgui.bullet()
                            imgui.same_line()
                            imgui.text_colored(_IMAGING_COLOR, alias)
                    imgui.end_tooltip()
                # value with optional unit, or placeholder for missing
                if value is not None:
                    _clickable_value(param.canonical, value, value_col, unit=param.unit or "")
                elif param.canonical == "dz" and is_lbm:
                    # LBM stacks require user-supplied dz
                    imgui.same_line(value_col)
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "Required")
                    if imgui.is_item_hovered():
                        imgui.begin_tooltip()
                        imgui.text("LBM stacks require user-supplied Z step size.")
                        imgui.text("Set via File > Save As > Metadata section.")
                        imgui.end_tooltip()
                else:
                    imgui.same_line(value_col)
                    imgui.text_disabled("—")

            # show roi_mode in imaging section only for LBM with multiple ROIs
            num_mrois = metadata.get("num_mrois", 1) or 1
            if is_lbm and num_mrois > 1:
                roi_mode_val = metadata.get("roi_mode")
                if roi_mode_val is not None:
                    shown_keys.add("roi_mode")
                    if not _metadata_search_filter or _matches_filter_recursive("roi_mode", roi_mode_val, _metadata_search_filter):
                        imgui.text_colored(_IMAGING_COLOR, "roi_mode")
                        if imgui.is_item_hovered():
                            imgui.set_tooltip(METADATA_TOOLTIPS.get("roi_mode", ""))
                        _clickable_value("roi_mode", roi_mode_val, value_col)

            # section: acquisition mode
            acq_fields = [
                ("stack_type", "Acquisition stack type: lbm, piezo, or single_plane"),
                ("lbm_stack", "Light Beads Microscopy acquisition mode"),
                ("piezo_stack", "Piezo-driven z-stack acquisition mode"),
                ("num_color_channels", "Number of color channels (1 or 2)"),
            ]
            # only show num_mrois if lbm_stack is True
            if is_lbm:
                acq_fields.append(("num_mrois", "Number of multi-ROI scan regions (LBM only)"))

            # extract scanimage version and imaging system (check multiple key formats)
            # metadata["si"] is the raw FrameData dict with keys like "SI.VERSION_MAJOR"
            si_data = metadata.get("si", {}) if isinstance(metadata.get("si"), dict) else {}
            si_version_major = (
                si_data.get("SI.VERSION_MAJOR") or si_data.get("VERSION_MAJOR") or
                metadata.get("SI.VERSION_MAJOR") or metadata.get("si.version_major")
            )
            si_version_minor = (
                si_data.get("SI.VERSION_MINOR") or si_data.get("VERSION_MINOR") or
                metadata.get("SI.VERSION_MINOR") or metadata.get("si.version_minor")
            )
            si_imaging_system = (
                si_data.get("SI.imagingSystem") or si_data.get("imagingSystem") or
                metadata.get("SI.imagingSystem") or metadata.get("si.imagingsystem") or
                metadata.get("imaging_system")
            )

            # always mark acquisition fields as shown to avoid duplicates
            acq_aliases = {
                "stack_type": ("stackType",),
                "lbm_stack": ("is_lbm", "lbmStack"),
                "piezo_stack": ("is_piezo", "piezoStack"),
                "num_mrois": ("num_rois", "scanimage_multirois", "numROIs", "nrois", "n_rois"),
                "num_color_channels": ("color_channels", "ncolors", "num_colors"),
                # nchannels aliases (not shown in acquisition but should be hidden from Other)
                "nchannels": ("num_channels", "n_channels", "channels", "C", "nc", "numChannels"),
            }
            for key, _ in acq_fields:
                shown_keys.add(key)
                for alias in acq_aliases.get(key, ()):
                    shown_keys.add(alias)
            # also hide nchannels aliases
            shown_keys.add("nchannels")
            for alias in acq_aliases.get("nchannels", ()):
                shown_keys.add(alias)
            # hide SI version keys from Other section (but not the whole SI dict - that stays visible)
            shown_keys.update({"SI.VERSION_MAJOR", "SI.VERSION_MINOR", "SI.imagingSystem",
                               "si.version_major", "si.version_minor", "si.imagingsystem",
                               "imaging_system"})

            # check if any acquisition fields have values (including SI version/imaging system)
            acq_values = {k: metadata.get(k) for k, _ in acq_fields}
            has_si_version = si_version_major is not None and si_version_minor is not None
            has_acq_data = any(v is not None for v in acq_values.values()) or has_si_version or si_imaging_system

            # filter check: show if no filter, or if any acq field matches, or if SI version/system matches
            acq_matches_filter = True
            if _metadata_search_filter:
                acq_field_matches = any(
                    _matches_filter_recursive(k, v, _metadata_search_filter)
                    for k, v in acq_values.items() if v is not None
                )
                si_version_matches = has_si_version and _matches_filter_shallow("scanimage_version", f"{si_version_major}.{si_version_minor}", _metadata_search_filter)
                si_system_matches = si_imaging_system and _matches_filter_shallow("imaging_system", si_imaging_system, _metadata_search_filter)
                acq_matches_filter = acq_field_matches or si_version_matches or si_system_matches

            if has_acq_data and acq_matches_filter:
                imgui.spacing()
                imgui.text_colored(_TREE_NODE_COLOR, "Acquisition")
                imgui.separator()

                # show imaging system if available
                if si_imaging_system:
                    if not _metadata_search_filter or _matches_filter_shallow("imaging_system", si_imaging_system, _metadata_search_filter):
                        imgui.text_colored(_ACQUISITION_COLOR, "imaging_system")
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("ScanImage imaging system identifier")
                        _clickable_value("imaging_system", si_imaging_system, value_col)

                # show scanimage version if available
                if has_si_version:
                    si_version_str = f"{si_version_major}.{si_version_minor}"
                    if not _metadata_search_filter or _matches_filter_shallow("scanimage_version", si_version_str, _metadata_search_filter):
                        imgui.text_colored(_ACQUISITION_COLOR, "scanimage_version")
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("ScanImage software version")
                        _clickable_value("scanimage_version", si_version_str, value_col)

                for key, tooltip in acq_fields:
                    value = metadata.get(key)
                    if value is None:
                        continue
                    if _metadata_search_filter and not _matches_filter_recursive(key, value, _metadata_search_filter):
                        continue
                    imgui.text_colored(_ACQUISITION_COLOR, key)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(tooltip)
                    # format booleans nicely, then make clickable
                    display_value = "Yes" if value is True else ("No" if value is False else value)
                    _clickable_value(key, display_value, value_col)

            # section: cameras (if present)
            cameras = metadata.get("cameras")
            if cameras and isinstance(cameras, dict):
                # check if any camera data matches filter (recursive)
                cameras_match = not _metadata_search_filter
                if _metadata_search_filter:
                    cameras_match = _matches_filter_recursive("cameras", cameras, _metadata_search_filter)

                if cameras_match:
                    imgui.spacing()
                    imgui.text_colored(_TREE_NODE_COLOR, "Cameras")
                    imgui.separator()
                    for cam_idx, cam_meta in sorted(cameras.items()):
                        # skip cameras that don't match filter
                        if _metadata_search_filter and not _matches_filter_recursive(f"camera_{cam_idx}", cam_meta, _metadata_search_filter):
                            continue
                        if _colored_tree_node(f"camera_{cam_idx}"):
                            for k, v in sorted(cam_meta.items()):
                                if k == "multiscales":
                                    continue  # skip verbose ome-ngff metadata
                                if _metadata_search_filter and not _matches_filter_recursive(k, v, _metadata_search_filter):
                                    continue
                                imgui.text_colored(_NAME_COLORS[0], k)
                                imgui.same_line(value_col)
                                imgui.text_colored(_VALUE_COLOR, _fmt_multivalue(v))
                            imgui.tree_pop()
                shown_keys.add("cameras")

            # section: other metadata
            remaining = {k: v for k, v in metadata.items() if k not in shown_keys}
            # filter remaining items (recursive)
            if _metadata_search_filter:
                remaining = {k: v for k, v in remaining.items() if _matches_filter_recursive(k, v, _metadata_search_filter)}
            if remaining:
                imgui.spacing()
                imgui.text_colored(_TREE_NODE_COLOR, "Other")
                imgui.separator()
                # build category sets for coloring aliases
                imaging_keys = set(IMAGING_METADATA_KEYS)
                acquisition_keys = {"stack_type", "lbm_stack", "piezo_stack", "num_mrois", "num_color_channels", "nchannels"}
                for k, v in sorted(remaining.items()):
                    key_color = _get_key_color(k, imaging_keys, acquisition_keys, ALIAS_MAP)
                    _render_item(k, v, filter_text=_metadata_search_filter, name_color=key_color)
        finally:
            imgui.pop_style_var()


def draw_scope():
    with imgui_ctx.begin_child("Scope Inspector"):
        frame = inspect.currentframe().f_back
        vars_all = {**frame.f_locals}
        imgui.push_style_var(  # type: ignore # noqa
            imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4)
        )
        try:
            for name, val in sorted(vars_all.items()):
                if (
                    inspect.ismodule(val)
                    or (name.startswith("_") or name.endswith("_"))
                    or callable(val)
                ):
                    continue
                _render_item(name, val)
        finally:
            imgui.pop_style_var()


def _render_item(name, val, prefix="", depth=0, filter_text="", name_color=None, is_disabled=False):
    full_name = f"{prefix}{name}"

    # check if this is a disabled SI module (hSomething with enable=false)
    is_disabled_module = name.startswith("h") and _is_disabled_si_module(val)

    # use red for disabled modules, provided color, or fallback to default
    if is_disabled_module:
        color = _DISABLED_COLOR
    else:
        color = name_color if name_color is not None else _NAME_COLORS[0]

    # skip items that don't match filter (recursively checks children too)
    if filter_text and not _matches_filter_recursive(name, val, filter_text):
        return

    if isinstance(val, Mapping):
        # filter out all-underscore keys and callables
        children = [
            (k, v)
            for k, v in val.items()
            if not (k.startswith("__") and k.endswith("__")) and not callable(v)
        ]
        # filter children if search is active
        if filter_text:
            children = [(k, v) for k, v in children if _matches_filter_recursive(k, v, filter_text)]
        if children:
            # use red for disabled SI modules, otherwise cyan
            if is_disabled_module:
                imgui.push_style_color(imgui.Col_.text, _DISABLED_COLOR)
            else:
                imgui.push_style_color(imgui.Col_.text, _TREE_NODE_COLOR)
            node_open = imgui.tree_node(full_name)
            imgui.pop_style_color()
            # tooltip for disabled modules or custom tooltips
            if imgui.is_item_hovered():
                if is_disabled_module:
                    imgui.set_tooltip("Disabled - when saving, these values will be discarded")
                elif name in METADATA_TOOLTIPS:
                    imgui.set_tooltip(METADATA_TOOLTIPS[name])
            if node_open:
                # grey out children if parent is disabled
                child_color = _OTHER_COLOR if (is_disabled_module or is_disabled) else None
                for k, v in children:
                    _render_item(str(k), v, prefix=full_name + ".", depth=depth + 1, filter_text=filter_text, name_color=child_color, is_disabled=(is_disabled_module or is_disabled))
                imgui.tree_pop()
        else:
            imgui.text_colored(color, full_name)
            if imgui.is_item_hovered():
                if is_disabled_module:
                    imgui.set_tooltip("Disabled - when saving, these values will be discarded")
                elif name in METADATA_TOOLTIPS:
                    imgui.set_tooltip(METADATA_TOOLTIPS[name])
            imgui.same_line(spacing=16)
            val_color = _OTHER_COLOR if (is_disabled or is_disabled_module) else _VALUE_COLOR
            imgui.text_colored(val_color, _fmt(val))
    elif isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray)):
        # grey out values if in disabled context
        val_color = _OTHER_COLOR if is_disabled else _VALUE_COLOR
        # Check if this is a list of file paths (strings that look like paths)
        is_path_list = (
            len(val) > 0
            and all(isinstance(v, str) for v in val)
            and any("\\" in v or "/" in v for v in val[:min(3, len(val))])
        )
        if is_path_list:
            # filter paths if search is active
            filtered_paths = list(enumerate(val))
            if filter_text:
                filtered_paths = [(i, p) for i, p in filtered_paths if filter_text.lower() in p.lower()]
            if filtered_paths:
                label = f"{full_name} ({len(filtered_paths)}/{len(val)} paths)" if filter_text else f"{full_name} ({len(val)} paths)"
                if _colored_tree_node(label):
                    for i, path in filtered_paths:
                        imgui.text_colored(_OTHER_COLOR if is_disabled else _NAME_COLORS[1], f"[{i}]")
                        imgui.same_line(spacing=8)
                        display_path = path if len(path) <= 60 else "..." + path[-57:]
                        imgui.text_colored(val_color, display_path)
                        if imgui.is_item_hovered() and len(path) > 60:
                            imgui.set_tooltip(path)
                    imgui.tree_pop()
        elif len(val) <= 8 and all(isinstance(v, (int, float, str, bool)) for v in val):
            imgui.text_colored(color, full_name)
            imgui.same_line(spacing=16)
            imgui.text_colored(val_color, repr(val))
        else:
            children = [(i, v) for i, v in enumerate(val) if not callable(v)]
            # filter children if search is active
            if filter_text:
                children = [(i, v) for i, v in children if _matches_filter_recursive(f"[{i}]", v, filter_text)]
            if children:
                if _colored_tree_node(f"{full_name} ({len(val)} items)"):
                    for i, v in children:
                        _render_item(f"[{i}]", v, prefix=full_name, depth=depth + 1, filter_text=filter_text, is_disabled=is_disabled)
                    imgui.tree_pop()
            else:
                imgui.text_colored(color, full_name)
                if name in METADATA_TOOLTIPS and imgui.is_item_hovered():
                    imgui.set_tooltip(METADATA_TOOLTIPS[name])
                imgui.same_line(spacing=16)
                imgui.text_colored(val_color, _fmt(val))

    else:
        # grey out values if in disabled context
        val_color = _OTHER_COLOR if is_disabled else _VALUE_COLOR
        cls = type(val)
        prop_names = [
            name_ for name_, attr in cls.__dict__.items() if isinstance(attr, property)
        ]
        fields = {}
        if hasattr(val, "__dict__"):
            fields = {
                n: v
                for n, v in vars(val).items()
                if not n.startswith("_") and not callable(v)
            }
        # if there are any fields or properties, show a tree node
        if fields or prop_names:
            if _colored_tree_node(f"{full_name} ({cls.__name__})"):
                # render instance attributes (filtered)
                for k, v in fields.items():
                    if filter_text and not _matches_filter_recursive(k, v, filter_text):
                        continue
                    _render_item(k, v, prefix=full_name + ".", depth=depth + 1, filter_text=filter_text, is_disabled=is_disabled)
                # render properties by retrieving their current value (filtered)
                for prop in prop_names:
                    try:
                        prop_val = getattr(val, prop)
                    except Exception:
                        continue
                    if filter_text and not _matches_filter_recursive(prop, prop_val, filter_text):
                        continue
                    _render_item(prop, prop_val, prefix=full_name + ".", depth=depth + 1, filter_text=filter_text, is_disabled=is_disabled)
                imgui.tree_pop()
        else:
            # leaf node: display name and formatted value
            imgui.text_colored(color, full_name)
            if name in METADATA_TOOLTIPS and imgui.is_item_hovered():
                imgui.set_tooltip(METADATA_TOOLTIPS[name])
            imgui.same_line(spacing=16)
            imgui.text_colored(val_color, _fmt(val))


def _fmt(x):
    if x is None:
        return "—"  # em dash for missing values
    if isinstance(x, (str, bool, int, float)):
        return repr(x)
    if isinstance(x, (bytes, bytearray)):
        return f"<{len(x)} bytes>"
    if isinstance(x, (tuple, list)):
        if len(x) <= 8:
            return repr(x)
        return f"[len={len(x)}]"
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        try:
            # convert small arrays to list
            if x.size <= 8:
                return repr(x.tolist())
            return f"<shape={tuple(x.shape)}, dtype={x.dtype}>"
        except Exception:
            return f"<array dtype={x.dtype}>"
    return f"<{type(x).__name__}>"
