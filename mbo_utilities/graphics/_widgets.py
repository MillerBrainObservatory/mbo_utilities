import inspect
from collections.abc import Mapping, Sequence
from imgui_bundle import (
    imgui,
    imgui_ctx,
)
from imgui_bundle import imgui_md

_NAME_COLORS = (
    imgui.ImVec4(0.95, 0.80, 0.30, 1.0),
    imgui.ImVec4(0.60, 0.95, 0.40, 1.0),
)
_VALUE_COLOR = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)


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

def draw_metadata_inspector(metadata: dict):
    with imgui_ctx.begin_child("Metadata Viewer"):
        imgui_md.render("# Metadata Viewer")
        imgui.separator()
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4)
        )
        try:
            for k, v in sorted(metadata.items()):
                _render_item(k, v)
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


def _render_item(name, val, prefix=""):

    full_name = f"{prefix}{name}"
    if isinstance(val, Mapping):
        # filter out all-underscore keys and callables
        children = [
            (k, v)
            for k, v in val.items()
            if not (k.startswith("__") and k.endswith("__")) and not callable(v)
        ]
        if children:
            if imgui.tree_node(full_name):
                for k, v in children:
                    _render_item(str(k), v, prefix=full_name + ".")
                imgui.tree_pop()
        else:
            imgui.text_colored(_NAME_COLORS[0], full_name)
            imgui.same_line(spacing=16)
            imgui.text_colored(_VALUE_COLOR, _fmt(val))
    elif isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray)):
        if len(val) <= 8 and all(isinstance(v, (int, float, str, bool)) for v in val):
            imgui.text_colored(_NAME_COLORS[0], full_name)
            imgui.same_line(spacing=16)
            imgui.text_colored(_VALUE_COLOR, repr(val))
        else:
            children = [(i, v) for i, v in enumerate(val) if not callable(v)]
            if children:
                if imgui.tree_node(f"{full_name} [{type(val).__name__}]"):
                    for i, v in children:
                        _render_item(f"{i}", v, prefix=full_name + "[")
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
            if imgui.tree_node(f"{full_name} ({cls.__name__})"):
                # render instance attributes
                for k, v in fields.items():
                    _render_item(k, v, prefix=full_name + ".")
                # render properties by retrieving their current value
                for prop in prop_names:
                    try:
                        prop_val = getattr(val, prop)
                    except Exception:
                        continue
                    _render_item(prop, prop_val, prefix=full_name + ".")
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
