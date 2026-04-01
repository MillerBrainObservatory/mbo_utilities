"""
Global Metadata Editor Dialog.

This module provides the metadata editor popup that allows users to define
metadata fields before processing or saving data.
"""

from __future__ import annotations

import contextlib
from typing import Any

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.metadata import parse_filename_metadata, get_filename_suggestions

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
        custom_val = getattr(parent, "_custom_metadata", {}).get(canonical)
        source_val = field.get("value")
        if custom_val is None and source_val is None:
            missing.append(field)

    return missing


def _build_suggested_fields(parent: Any) -> list[dict]:
    """Build the complete list of suggested metadata fields."""
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


def draw_metadata_editor_content(parent: Any):
    """Draw metadata editor fields. can be embedded in any container (tab, popup, etc.)."""
    if not hasattr(parent, "_custom_metadata"):
        parent._custom_metadata = {}
    if not hasattr(parent, "_custom_key"):
        parent._custom_key = ""
    if not hasattr(parent, "_custom_value"):
        parent._custom_value = ""

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
                custom_val = parent._custom_metadata.get(canonical)
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
                            parent._custom_metadata[canonical] = parsed
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
    custom_entries = [(k, v) for k, v in parent._custom_metadata.items() if k not in suggested_keys]

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
            del parent._custom_metadata[to_remove]
        imgui.new_line()

    # add new custom entry row
    imgui.dummy(imgui.ImVec2(0, 2))
    imgui.set_next_item_width(hello_imgui.em_size(8))
    _, parent._custom_key = imgui.input_text("##key", parent._custom_key)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Custom key name")
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "=")
    imgui.same_line()
    imgui.set_next_item_width(hello_imgui.em_size(8))
    _, parent._custom_value = imgui.input_text("##val", parent._custom_value)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Custom value (auto-detects number vs text)")
    imgui.same_line()
    if imgui.button("+", imgui.ImVec2(hello_imgui.em_size(2), 0)) and parent._custom_key.strip():
        val = parent._custom_value
        with contextlib.suppress(ValueError):
            val = float(val) if "." in val else int(val)
        parent._custom_metadata[parent._custom_key.strip()] = val
        parent._custom_key = ""
        parent._custom_value = ""
