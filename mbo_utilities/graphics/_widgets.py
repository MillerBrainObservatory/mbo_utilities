import inspect
from collections.abc import Mapping, Sequence
from imgui_bundle import (
    imgui,
    imgui_ctx,
)
from imgui_bundle import imgui_md

_NAME_COLORS = (
    imgui.ImVec4(0.95, 0.80, 0.30, 1.0),  # gold/yellow for names
    imgui.ImVec4(0.60, 0.95, 0.40, 1.0),  # green for indices
)
_VALUE_COLOR = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
_TREE_NODE_COLOR = imgui.ImVec4(0.40, 0.80, 0.95, 1.0)  # cyan for tree nodes
_ALIAS_COLOR = imgui.ImVec4(0.5, 0.5, 0.5, 0.8)  # muted gray for aliases
_CANONICAL_COLOR = imgui.ImVec4(0.4, 0.9, 0.6, 1.0)  # bright green for canonical


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


def _matches_filter(key: str, value, filter_text: str) -> bool:
    """check if key or value matches the search filter (case-insensitive)."""
    if not filter_text:
        return True
    filter_lower = filter_text.lower()
    if filter_lower in str(key).lower():
        return True
    if filter_lower in str(value).lower():
        return True
    return False


def _collect_matching_keys(metadata: dict, filter_text: str) -> set:
    """recursively collect all keys that match the filter."""
    matches = set()
    if not filter_text:
        return matches

    def _check_item(key, value, parent_keys=None):
        parent_keys = parent_keys or []
        if _matches_filter(key, value, filter_text):
            matches.add(key)
            for pk in parent_keys:
                matches.add(pk)

        if isinstance(value, Mapping):
            for k, v in value.items():
                _check_item(k, v, parent_keys + [key])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for i, v in enumerate(value):
                _check_item(f"[{i}]", v, parent_keys + [key])

    for k, v in metadata.items():
        _check_item(k, v)

    return matches


def draw_metadata_inspector(metadata: dict):
    """draw metadata with canonical params first, then other fields."""
    global _metadata_search_filter, _metadata_search_active
    from mbo_utilities.metadata import METADATA_PARAMS

    with imgui_ctx.begin_child("Metadata Viewer"):
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4))

        try:
            shown_keys = set()
            value_col = 180

            # section: imaging parameters with search icon
            imgui.text_colored(_TREE_NODE_COLOR, "Imaging")
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
                imgui.text_unformatted(
                    "Hover parameter names to see aliases.\n\n"
                    "These imaging parameters have standardized aliases for "
                    "compatibility with ImageJ, OME-TIFF, OME-Zarr, and other systems.\n\n"
                    "imread/imwrite handles all alias conversions automatically."
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()

            # search icon on the right side
            avail_width = imgui.get_content_region_avail().x
            imgui.same_line(avail_width - 20)
            search_color = _TREE_NODE_COLOR if _metadata_search_active else imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
            imgui.push_style_color(imgui.Col_.text, search_color)
            if imgui.small_button("\U0001F50D"):  # magnifying glass emoji
                _metadata_search_active = not _metadata_search_active
                if not _metadata_search_active:
                    _metadata_search_filter = ""
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Search metadata (click to toggle)")

            # search input field
            if _metadata_search_active:
                imgui.set_next_item_width(-1)
                changed, _metadata_search_filter = imgui.input_text_with_hint(
                    "##metadata_search",
                    "filter by key or value...",
                    _metadata_search_filter,
                )

            imgui.separator()
            for name, param in METADATA_PARAMS.items():
                value = metadata.get(param.canonical)
                if value is None:
                    for alias in param.aliases:
                        if alias in metadata:
                            value = metadata[alias]
                            break

                if value is not None:
                    shown_keys.add(param.canonical)
                    shown_keys.update(param.aliases)

                    # skip if doesn't match filter
                    if _metadata_search_filter and not _matches_filter(param.canonical, value, _metadata_search_filter):
                        continue

                    # parameter name with alias tooltip on hover
                    imgui.text_colored(_CANONICAL_COLOR, param.canonical)
                    if param.aliases and imgui.is_item_hovered():
                        imgui.begin_tooltip()
                        imgui.text("Aliases:")
                        for alias in param.aliases:
                            imgui.bullet_text(alias)
                        imgui.end_tooltip()
                    imgui.same_line(value_col)
                    # value with optional unit
                    val_str = _fmt_multivalue(value)
                    if param.unit:
                        val_str += f" ({param.unit})"
                    imgui.text_colored(_VALUE_COLOR, val_str)

            # section: cameras (if present)
            cameras = metadata.get("cameras")
            if cameras and isinstance(cameras, dict):
                # check if any camera data matches filter
                cameras_match = not _metadata_search_filter
                if _metadata_search_filter:
                    for cam_idx, cam_meta in cameras.items():
                        if _matches_filter("cameras", cam_meta, _metadata_search_filter):
                            cameras_match = True
                            break
                        for k, v in cam_meta.items():
                            if _matches_filter(k, v, _metadata_search_filter):
                                cameras_match = True
                                break

                if cameras_match:
                    imgui.spacing()
                    imgui.text_colored(_TREE_NODE_COLOR, "Cameras")
                    imgui.separator()
                    for cam_idx, cam_meta in sorted(cameras.items()):
                        if _colored_tree_node(f"camera_{cam_idx}"):
                            for k, v in sorted(cam_meta.items()):
                                if k == "multiscales":
                                    continue  # skip verbose ome-ngff metadata
                                if _metadata_search_filter and not _matches_filter(k, v, _metadata_search_filter):
                                    continue
                                imgui.text_colored(_NAME_COLORS[0], k)
                                imgui.same_line(value_col)
                                imgui.text_colored(_VALUE_COLOR, _fmt_multivalue(v))
                            imgui.tree_pop()
                shown_keys.add("cameras")

            # section: other metadata
            remaining = {k: v for k, v in metadata.items() if k not in shown_keys}
            # filter remaining items
            if _metadata_search_filter:
                matching_keys = _collect_matching_keys(remaining, _metadata_search_filter)
                remaining = {k: v for k, v in remaining.items() if k in matching_keys}
            if remaining:
                imgui.spacing()
                imgui.text_colored(_TREE_NODE_COLOR, "Other")
                imgui.separator()
                for k, v in sorted(remaining.items()):
                    _render_item(k, v, filter_text=_metadata_search_filter)
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


def _render_item(name, val, prefix="", depth=0, filter_text=""):
    full_name = f"{prefix}{name}"
    if isinstance(val, Mapping):
        # filter out all-underscore keys and callables
        children = [
            (k, v)
            for k, v in val.items()
            if not (k.startswith("__") and k.endswith("__")) and not callable(v)
        ]
        if children:
            if _colored_tree_node(full_name):
                for k, v in children:
                    _render_item(str(k), v, prefix=full_name + ".", depth=depth + 1, filter_text=filter_text)
                imgui.tree_pop()
        else:
            imgui.text_colored(_NAME_COLORS[0], full_name)
            imgui.same_line(spacing=16)
            imgui.text_colored(_VALUE_COLOR, _fmt(val))
    elif isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray)):
        # Check if this is a list of file paths (strings that look like paths)
        is_path_list = (
            len(val) > 0
            and all(isinstance(v, str) for v in val)
            and any("\\" in v or "/" in v for v in val[:min(3, len(val))])
        )
        if is_path_list:
            # Display path lists as a collapsible tree with compact formatting
            if _colored_tree_node(f"{full_name} ({len(val)} paths)"):
                for i, path in enumerate(val):
                    # filter paths if search is active
                    if filter_text and filter_text.lower() not in path.lower():
                        continue
                    imgui.text_colored(_NAME_COLORS[1], f"[{i}]")
                    imgui.same_line(spacing=8)
                    # Truncate long paths from the left, showing the end
                    display_path = path if len(path) <= 60 else "..." + path[-57:]
                    imgui.text_colored(_VALUE_COLOR, display_path)
                    if imgui.is_item_hovered() and len(path) > 60:
                        imgui.set_tooltip(path)
                imgui.tree_pop()
        elif len(val) <= 8 and all(isinstance(v, (int, float, str, bool)) for v in val):
            imgui.text_colored(_NAME_COLORS[0], full_name)
            imgui.same_line(spacing=16)
            imgui.text_colored(_VALUE_COLOR, repr(val))
        else:
            children = [(i, v) for i, v in enumerate(val) if not callable(v)]
            if children:
                if _colored_tree_node(f"{full_name} ({len(val)} items)"):
                    for i, v in children:
                        _render_item(f"[{i}]", v, prefix=full_name, depth=depth + 1, filter_text=filter_text)
                    imgui.tree_pop()
            else:
                imgui.text_colored(_NAME_COLORS[0], full_name)
                imgui.same_line(spacing=16)
                imgui.text_colored(_VALUE_COLOR, _fmt(val))

    else:
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
                # render instance attributes
                for k, v in fields.items():
                    _render_item(k, v, prefix=full_name + ".", depth=depth + 1, filter_text=filter_text)
                # render properties by retrieving their current value
                for prop in prop_names:
                    try:
                        prop_val = getattr(val, prop)
                    except Exception:
                        continue
                    _render_item(prop, prop_val, prefix=full_name + ".", depth=depth + 1, filter_text=filter_text)
                imgui.tree_pop()
        else:
            # leaf node: display name and formatted value
            imgui.text_colored(_NAME_COLORS[0], full_name)
            imgui.same_line(spacing=16)
            imgui.text_colored(_VALUE_COLOR, _fmt(val))


def _fmt(x):
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
