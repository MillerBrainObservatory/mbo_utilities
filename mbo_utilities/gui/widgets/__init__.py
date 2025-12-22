"""
widget registry with auto-discovery.

widgets are automatically discovered from this directory.
each widget module should define a class that inherits from Widget.

to add a new widget:
1. create a new file in widgets/ (e.g. my_widget.py)
2. define a class inheriting from Widget
3. implement is_supported() and draw()
4. the widget will be auto-discovered and shown when is_supported() returns True
"""

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

from imgui_bundle import imgui

from mbo_utilities.gui.widgets._base import Widget

if TYPE_CHECKING:
    from mbo_utilities.gui.imgui import PreviewDataWidget

# registry of all discovered widget classes
_WIDGET_CLASSES: list[type[Widget]] = []


def _discover_widgets() -> None:
    """auto-discover widget classes from this package."""
    global _WIDGET_CLASSES

    if _WIDGET_CLASSES:
        return  # already discovered

    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        # skip private modules
        if module_info.name.startswith('_'):
            continue

        try:
            module = importlib.import_module(f".{module_info.name}", package=__name__)

            # find Widget subclasses in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Widget)
                    and attr is not Widget
                ):
                    _WIDGET_CLASSES.append(attr)
        except Exception as e:
            # log but don't crash on import errors
            print(f"warning: failed to import widget module {module_info.name}: {e}")

    # sort by priority
    _WIDGET_CLASSES.sort(key=lambda w: w.priority)


def get_supported_widgets(parent: "PreviewDataWidget") -> list[Widget]:
    """
    get all widgets that are supported for the given parent.

    returns instantiated widgets sorted by priority.
    """
    _discover_widgets()

    supported = []
    for widget_cls in _WIDGET_CLASSES:
        try:
            if widget_cls.is_supported(parent):
                supported.append(widget_cls(parent))
        except Exception as e:
            # Log the error for debugging
            import traceback
            print(f"Warning: Widget {widget_cls.__name__} support check failed: {e}")
            traceback.print_exc()

    # sort by priority (lower = first)
    supported.sort(key=lambda w: w.priority)
    return supported


def draw_all_widgets(parent: "PreviewDataWidget", widgets: list[Widget]) -> None:
    """draw all supported widgets."""
    for widget in widgets:
        try:
            widget.draw()
        except Exception as e:
            # log error but don't crash the ui
            import traceback
            error_msg = f"Error in {widget.name}: {e}"
            parent.logger.error(error_msg)
            parent.logger.error(traceback.format_exc())
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                error_msg
            )


def cleanup_all_widgets(widgets: list[Widget]) -> None:
    """clean up all widgets when the gui is closing.

    calls cleanup() on each widget to release resources.
    """
    for widget in widgets:
        try:
            widget.cleanup()
        except Exception as e:
            # log but don't crash during cleanup
            print(f"Warning: cleanup failed for {widget.name}: {e}")


__all__ = [
    "Widget",
    "get_supported_widgets",
    "draw_all_widgets",
    "cleanup_all_widgets",
]
