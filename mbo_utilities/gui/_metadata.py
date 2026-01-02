"""
Metadata inspector for displaying array metadata in ImGui.

This is the comprehensive metadata viewer used in the sidebar.
"""

from collections.abc import Mapping, Sequence

from imgui_bundle import imgui, imgui_ctx

from ._imgui_helpers import fmt_value, fmt_multivalue

# Font awesome icons (with fallback)
try:
    from imgui_bundle import icons_fontawesome as fa
    ICON_SEARCH = fa.ICON_FA_SEARCH
except (ImportError, AttributeError):
    ICON_SEARCH = "\uf002"

# Colors
_NAME_COLORS = (
    imgui.ImVec4(0.95, 0.80, 0.30, 1.0),  # gold/yellow for names
    imgui.ImVec4(0.60, 0.95, 0.40, 1.0),  # green for indices
)
_VALUE_COLOR = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
_TREE_NODE_COLOR = imgui.ImVec4(0.40, 0.80, 0.95, 1.0)  # cyan for tree nodes
_ALIAS_COLOR = imgui.ImVec4(0.5, 0.5, 0.5, 0.8)  # muted gray for aliases

# Category-specific colors for metadata
_IMAGING_COLOR = imgui.ImVec4(0.4, 0.9, 0.6, 1.0)  # bright green for imaging params
_ACQUISITION_COLOR = imgui.ImVec4(0.9, 0.6, 0.4, 1.0)  # orange for acquisition params
_OTHER_COLOR = imgui.ImVec4(0.75, 0.85, 0.95, 1.0)  # light blue for uncategorized
_DISABLED_COLOR = imgui.ImVec4(0.9, 0.4, 0.4, 1.0)  # red for disabled modules

# Custom tooltips for specific metadata keys
METADATA_TOOLTIPS: dict[str, str] = {
    "roi_groups": "ROI scan regions defined in ScanImage. Each group contains position, size, and scan parameters for a multi-ROI acquisition.",
    "roi_mode": "How multi-ROI data is handled: concat_y stitches ROIs horizontally, separate keeps them as individual files.",
}

# Module-level state for metadata search
_metadata_search_filter = ""
_metadata_search_active = False
_metadata_search_focus_requested = False


def _colored_tree_node(label: str) -> bool:
    """Create a tree node with colored text."""
    imgui.push_style_color(imgui.Col_.text, _TREE_NODE_COLOR)
    result = imgui.tree_node(label)
    imgui.pop_style_color()
    return result


def _matches_filter_shallow(key: str, value, filter_text: str) -> bool:
    """Check if key or stringified value matches the search filter (case-insensitive)."""
    if not filter_text:
        return True
    filter_lower = filter_text.lower()
    if filter_lower in str(key).lower():
        return True
    if not isinstance(value, (Mapping, list, tuple)):
        if filter_lower in str(value).lower():
            return True
    return False


def _matches_filter_recursive(key: str, value, filter_text: str) -> bool:
    """Recursively check if key, value, or any nested children match the filter."""
    if not filter_text:
        return True

    if _matches_filter_shallow(key, value, filter_text):
        return True

    if isinstance(value, Mapping):
        for k, v in value.items():
            if _matches_filter_recursive(k, v, filter_text):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for i, v in enumerate(value):
            if _matches_filter_recursive(f"[{i}]", v, filter_text):
                return True

    return False


def _clickable_value(key: str, value, value_col: int, color=None, unit: str = ""):
    """Render a value at the specified column."""
    if color is None:
        color = _VALUE_COLOR

    val_str = fmt_multivalue(value)
    if unit:
        val_str += f" ({unit})"

    imgui.same_line(value_col)
    imgui.text_colored(color, val_str)


def _get_key_color(key: str, imaging_keys: set, acquisition_keys: set, alias_map: dict):
    """Get color for a metadata key based on its category."""
    key_lower = key.lower()
    canonical = alias_map.get(key_lower)
    if canonical is not None:
        if canonical in imaging_keys:
            return _IMAGING_COLOR
        if canonical in acquisition_keys:
            return _ACQUISITION_COLOR
    if key in imaging_keys or key_lower in {k.lower() for k in imaging_keys}:
        return _IMAGING_COLOR
    if key in acquisition_keys or key_lower in {k.lower() for k in acquisition_keys}:
        return _ACQUISITION_COLOR
    return _OTHER_COLOR


def _is_disabled_si_module(value) -> bool:
    """Check if a scanimage module dict has enable=false."""
    if not isinstance(value, Mapping):
        return False
    enable_val = value.get("enable")
    if enable_val is False:
        return True
    return bool(isinstance(enable_val, str) and enable_val.lower() in ("false", "0"))


def _render_item(name, val, prefix="", depth=0, filter_text="", name_color=None, is_disabled=False):
    full_name = f"{prefix}{name}"

    is_disabled_module = name.startswith("h") and _is_disabled_si_module(val)

    if is_disabled_module:
        color = _DISABLED_COLOR
    else:
        color = name_color if name_color is not None else _NAME_COLORS[0]

    if filter_text and not _matches_filter_recursive(name, val, filter_text):
        return

    if isinstance(val, Mapping):
        children = [
            (k, v)
            for k, v in val.items()
            if not (k.startswith("__") and k.endswith("__")) and not callable(v)
        ]
        if filter_text:
            children = [(k, v) for k, v in children if _matches_filter_recursive(k, v, filter_text)]
        if children:
            if is_disabled_module:
                imgui.push_style_color(imgui.Col_.text, _DISABLED_COLOR)
            else:
                imgui.push_style_color(imgui.Col_.text, _TREE_NODE_COLOR)
            node_open = imgui.tree_node(full_name)
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                if is_disabled_module:
                    imgui.set_tooltip("Disabled - when saving, these values will be discarded")
                elif name in METADATA_TOOLTIPS:
                    imgui.set_tooltip(METADATA_TOOLTIPS[name])
            if node_open:
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
            imgui.text_colored(val_color, fmt_value(val))
    elif isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray)):
        val_color = _OTHER_COLOR if is_disabled else _VALUE_COLOR
        is_path_list = (
            len(val) > 0
            and all(isinstance(v, str) for v in val)
            and any("\\" in v or "/" in v for v in val[:min(3, len(val))])
        )
        if is_path_list:
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
                imgui.text_colored(val_color, fmt_value(val))

    else:
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
        if fields or prop_names:
            if _colored_tree_node(f"{full_name} ({cls.__name__})"):
                for k, v in fields.items():
                    if filter_text and not _matches_filter_recursive(k, v, filter_text):
                        continue
                    _render_item(k, v, prefix=full_name + ".", depth=depth + 1, filter_text=filter_text, is_disabled=is_disabled)
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
            imgui.text_colored(color, full_name)
            if name in METADATA_TOOLTIPS and imgui.is_item_hovered():
                imgui.set_tooltip(METADATA_TOOLTIPS[name])
            imgui.same_line(spacing=16)
            imgui.text_colored(val_color, fmt_value(val))


def draw_metadata_inspector(metadata: dict, data_array=None):
    """Draw metadata with canonical params first, then other fields.

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

            # Centered header with help tooltip
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

            # Keyboard shortcuts
            io = imgui.get_io()
            if io.key_ctrl and imgui.is_key_pressed(imgui.Key.l):
                _metadata_search_active = not _metadata_search_active
                if _metadata_search_active:
                    _metadata_search_focus_requested = True
                else:
                    _metadata_search_filter = ""
            if _metadata_search_active and imgui.is_key_pressed(imgui.Key.escape):
                _metadata_search_active = False
                _metadata_search_filter = ""

            # Search button
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

            # Search input field
            if _metadata_search_active:
                if _metadata_search_focus_requested:
                    imgui.set_keyboard_focus_here()
                    _metadata_search_focus_requested = False
                imgui.set_next_item_width(-1)
                _changed, _metadata_search_filter = imgui.input_text_with_hint(
                    "##metadata_search",
                    "filter by key or value...",
                    _metadata_search_filter,
                )

            imgui.spacing()

            # File info section
            source_path = metadata.get("source_path") or metadata.get("path") or metadata.get("file_path")
            if source_path:
                shown_keys.update({"source_path", "path", "file_path"})
                file_matches = not _metadata_search_filter or _matches_filter_recursive("source", source_path, _metadata_search_filter)
                if file_matches:
                    imgui.text_colored(_TREE_NODE_COLOR, "File")
                    imgui.separator()
                    path_str = str(source_path)
                    display_path = path_str if len(path_str) <= 50 else "..." + path_str[-47:]
                    imgui.text_colored(_NAME_COLORS[0], "path")
                    imgui.same_line(value_col)
                    imgui.text_colored(_VALUE_COLOR, display_path)
                    if imgui.is_item_hovered() and len(path_str) > 50:
                        imgui.set_tooltip(path_str)
                    imgui.spacing()

            is_lbm = metadata.get("lbm_stack", False) or metadata.get("stack_type") == "lbm"

            # Imaging section header
            imgui.text_colored(_TREE_NODE_COLOR, "Imaging")
            imgui.separator()

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

                shown_keys.add(param.canonical)
                shown_keys.update(param.aliases)

                if _metadata_search_filter and not _matches_filter_recursive(param.canonical, value if value is not None else "", _metadata_search_filter):
                    continue

                imgui.text_colored(_IMAGING_COLOR, param.canonical)
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    if data_array and hasattr(data_array, "get_param_description"):
                        desc = data_array.get_param_description(param.canonical)
                        if desc:
                            imgui.text_wrapped(desc)
                            imgui.separator()
                    elif param.description:
                        imgui.text_wrapped(param.description)
                        imgui.separator()
                    if param.aliases:
                        imgui.text("Aliases:")
                        for alias in param.aliases:
                            imgui.bullet()
                            imgui.same_line()
                            imgui.text_colored(_IMAGING_COLOR, alias)
                    imgui.end_tooltip()
                if value is not None:
                    _clickable_value(param.canonical, value, value_col, unit=param.unit or "")
                elif param.canonical == "dz" and is_lbm:
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

            # ROI mode for LBM with multiple ROIs
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

            # Acquisition mode section
            acq_fields = [
                ("stack_type", "Acquisition stack type: lbm, piezo, or single_plane"),
                ("lbm_stack", "Light Beads Microscopy acquisition mode"),
                ("piezo_stack", "Piezo-driven z-stack acquisition mode"),
                ("num_color_channels", "Number of color channels (1 or 2)"),
            ]
            if is_lbm:
                acq_fields.append(("num_mrois", "Number of multi-ROI scan regions (LBM only)"))

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

            acq_aliases = {
                "stack_type": ("stackType",),
                "lbm_stack": ("is_lbm", "lbmStack"),
                "piezo_stack": ("is_piezo", "piezoStack"),
                "num_mrois": ("num_rois", "scanimage_multirois", "numROIs", "nrois", "n_rois"),
                "num_color_channels": ("color_channels", "ncolors", "num_colors"),
                "nchannels": ("num_channels", "n_channels", "channels", "C", "nc", "numChannels"),
            }
            for key, _ in acq_fields:
                shown_keys.add(key)
                for alias in acq_aliases.get(key, ()):
                    shown_keys.add(alias)
            shown_keys.add("nchannels")
            for alias in acq_aliases.get("nchannels", ()):
                shown_keys.add(alias)
            shown_keys.update({"SI.VERSION_MAJOR", "SI.VERSION_MINOR", "SI.imagingSystem",
                               "si.version_major", "si.version_minor", "si.imagingsystem",
                               "imaging_system"})

            acq_values = {k: metadata.get(k) for k, _ in acq_fields}
            has_si_version = si_version_major is not None and si_version_minor is not None
            has_acq_data = any(v is not None for v in acq_values.values()) or has_si_version or si_imaging_system

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

                if si_imaging_system:
                    if not _metadata_search_filter or _matches_filter_shallow("imaging_system", si_imaging_system, _metadata_search_filter):
                        imgui.text_colored(_ACQUISITION_COLOR, "imaging_system")
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("ScanImage imaging system identifier")
                        _clickable_value("imaging_system", si_imaging_system, value_col)

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
                    display_value = "Yes" if value is True else ("No" if value is False else value)
                    _clickable_value(key, display_value, value_col)

            # Cameras section
            cameras = metadata.get("cameras")
            if cameras and isinstance(cameras, dict):
                cameras_match = not _metadata_search_filter
                if _metadata_search_filter:
                    cameras_match = _matches_filter_recursive("cameras", cameras, _metadata_search_filter)

                if cameras_match:
                    imgui.spacing()
                    imgui.text_colored(_TREE_NODE_COLOR, "Cameras")
                    imgui.separator()
                    for cam_idx, cam_meta in sorted(cameras.items()):
                        if _metadata_search_filter and not _matches_filter_recursive(f"camera_{cam_idx}", cam_meta, _metadata_search_filter):
                            continue
                        if _colored_tree_node(f"camera_{cam_idx}"):
                            for k, v in sorted(cam_meta.items()):
                                if k == "multiscales":
                                    continue
                                if _metadata_search_filter and not _matches_filter_recursive(k, v, _metadata_search_filter):
                                    continue
                                imgui.text_colored(_NAME_COLORS[0], k)
                                imgui.same_line(value_col)
                                imgui.text_colored(_VALUE_COLOR, fmt_multivalue(v))
                            imgui.tree_pop()
                shown_keys.add("cameras")

            # Other metadata section
            remaining = {k: v for k, v in metadata.items() if k not in shown_keys}
            if _metadata_search_filter:
                remaining = {k: v for k, v in remaining.items() if _matches_filter_recursive(k, v, _metadata_search_filter)}
            if remaining:
                imgui.spacing()
                imgui.text_colored(_TREE_NODE_COLOR, "Other")
                imgui.separator()
                imaging_keys = set(IMAGING_METADATA_KEYS)
                acquisition_keys = {"stack_type", "lbm_stack", "piezo_stack", "num_mrois", "num_color_channels", "nchannels"}
                for k, v in sorted(remaining.items()):
                    key_color = _get_key_color(k, imaging_keys, acquisition_keys, ALIAS_MAP)
                    _render_item(k, v, filter_text=_metadata_search_filter, name_color=key_color)
        finally:
            imgui.pop_style_var()
