"""
Metadata panel.

Displays metadata from loaded data arrays with search and filtering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from imgui_bundle import imgui, ImVec2

from . import BasePanel

if TYPE_CHECKING:
    from mbo_utilities.gui.viewers import BaseViewer

__all__ = ["MetadataPanel"]


# Colors for metadata display
_NAME_COLORS = {
    "imaging": imgui.ImVec4(0.4, 0.8, 1.0, 1.0),      # cyan for imaging params
    "acquisition": imgui.ImVec4(0.4, 1.0, 0.6, 1.0),  # green for acquisition
    "other": imgui.ImVec4(0.8, 0.8, 0.8, 1.0),        # light gray for other
}
_VALUE_COLOR = imgui.ImVec4(0.9, 0.9, 0.6, 1.0)       # pale yellow
_TREE_NODE_COLOR = imgui.ImVec4(0.6, 0.6, 0.9, 1.0)  # light purple
_DISABLED_COLOR = imgui.ImVec4(0.5, 0.5, 0.5, 1.0)   # gray


def _fmt(x: Any) -> str:
    """Format a value for display."""
    if isinstance(x, float):
        if abs(x) < 0.0001 or abs(x) > 10000:
            return f"{x:.4e}"
        return f"{x:.4f}"
    if isinstance(x, (list, tuple)) and len(x) > 5:
        return f"[{len(x)} items]"
    return str(x)


def _matches_filter_recursive(key: str, value: Any, filter_text: str) -> bool:
    """Check if key or any nested value matches filter."""
    filter_lower = filter_text.lower()
    if filter_lower in str(key).lower():
        return True
    if isinstance(value, dict):
        for k, v in value.items():
            if _matches_filter_recursive(k, v, filter_text):
                return True
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            if _matches_filter_recursive(str(i), item, filter_text):
                return True
    elif filter_lower in str(value).lower():
        return True
    return False


def _render_item(
    name: str,
    val: Any,
    prefix: str,
    depth: int,
    filter_text: str,
    name_color: imgui.ImVec4,
    is_disabled: bool = False,
) -> None:
    """Recursively render a metadata item."""
    # Check filter match
    if filter_text and not _matches_filter_recursive(name, val, filter_text):
        return

    color = _DISABLED_COLOR if is_disabled else name_color

    if isinstance(val, dict):
        # Nested dictionary - use tree node
        imgui.push_style_color(imgui.Col_.text, _TREE_NODE_COLOR)
        node_open = imgui.tree_node(f"{prefix}{name}")
        imgui.pop_style_color()

        if node_open:
            for k, v in sorted(val.items()):
                _render_item(k, v, "", depth + 1, filter_text, name_color, is_disabled)
            imgui.tree_pop()

    elif isinstance(val, (list, tuple)):
        if len(val) <= 5:
            # Short list - show inline
            imgui.text_colored(color, f"{prefix}{name}:")
            imgui.same_line()
            imgui.text_colored(_VALUE_COLOR, _fmt(val))
        else:
            # Long list - use tree node
            imgui.push_style_color(imgui.Col_.text, _TREE_NODE_COLOR)
            node_open = imgui.tree_node(f"{prefix}{name} [{len(val)} items]")
            imgui.pop_style_color()

            if node_open:
                for i, item in enumerate(val[:20]):  # Show first 20
                    _render_item(str(i), item, "", depth + 1, filter_text, name_color, is_disabled)
                if len(val) > 20:
                    imgui.text_disabled(f"... and {len(val) - 20} more")
                imgui.tree_pop()
    else:
        # Simple value
        imgui.text_colored(color, f"{prefix}{name}:")
        imgui.same_line()
        imgui.text_colored(_VALUE_COLOR, _fmt(val))


class MetadataPanel(BasePanel):
    """
    Panel for viewing metadata from loaded data arrays.

    Features:
    - Hierarchical display of nested metadata
    - Search/filter by key or value
    - Category-based coloring
    """

    name = "Metadata"

    def __init__(self, viewer: BaseViewer):
        super().__init__(viewer)
        self._filter_text = ""

    def draw(self) -> None:
        """Draw the metadata panel."""
        if not self._visible:
            return

        # Use absolute screen positioning
        io = imgui.get_io()
        screen_w, screen_h = io.display_size.x, io.display_size.y
        win_w, win_h = min(600, screen_w * 0.5), min(500, screen_h * 0.6)

        imgui.set_next_window_pos(
            ImVec2((screen_w - win_w) / 2, (screen_h - win_h) / 2),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(ImVec2(win_w, win_h), imgui.Cond_.first_use_ever)

        expanded, opened = imgui.begin("Metadata Viewer", self._visible)
        self._visible = opened

        if expanded:
            # Get metadata - prefer viewer's get_metadata() if available
            metadata = None
            if hasattr(self.viewer, "get_metadata"):
                metadata = self.viewer.get_metadata()

            if metadata is None:
                # Fallback to data array metadata
                data_arrays = self.viewer._get_data_arrays()
                if data_arrays:
                    data_arr = data_arrays[0]
                    if hasattr(data_arr, "metadata"):
                        metadata = data_arr.metadata

            if metadata:
                # Search filter
                imgui.set_next_item_width(200)
                _changed, self._filter_text = imgui.input_text_with_hint(
                    "##meta_search", "Search...", self._filter_text
                )

                imgui.separator()

                # Scrollable content
                if imgui.begin_child("##metadata_content"):
                    if isinstance(metadata, dict):
                        for key, value in sorted(metadata.items()):
                            # Determine color based on key
                            if any(k in key.lower() for k in ["imaging", "scan", "pixel", "zoom", "resolution"]):
                                color = _NAME_COLORS["imaging"]
                            elif any(k in key.lower() for k in ["acq", "frame", "channel", "stack"]):
                                color = _NAME_COLORS["acquisition"]
                            else:
                                color = _NAME_COLORS["other"]

                            _render_item(key, value, "", 0, self._filter_text, color)
                    else:
                        imgui.text(f"Metadata: {metadata}")
                    imgui.end_child()
            else:
                imgui.text("No metadata available")
                data_arrays = self.viewer._get_data_arrays()
                if data_arrays:
                    data_arr = data_arrays[0]
                    imgui.text(f"Data type: {type(data_arr).__name__}")
                    if hasattr(data_arr, "shape"):
                        imgui.text(f"Shape: {data_arr.shape}")

        imgui.end()
