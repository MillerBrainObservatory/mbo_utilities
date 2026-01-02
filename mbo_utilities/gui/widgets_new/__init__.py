"""
Widget classes - generic, composable UI building blocks.

Widgets are reusable UI components that:
- Are configured with data, not subclassed
- Can be composed together
- Are not capability-based (that's Features)

Examples:
- ListSelectorWidget: Select from a list (z-planes, arrays, channels)
- CheckboxGridWidget: Grid of checkboxes for multi-selection
- SliderGroupWidget: Group of related sliders
"""

from __future__ import annotations

__all__ = [
    "ListSelectorWidget",
    "CheckboxGridWidget",
    "SliderGroupWidget",
    "SliderConfig",
]


# Lazy imports
def __getattr__(name: str):
    if name == "ListSelectorWidget":
        from .list_selector import ListSelectorWidget
        return ListSelectorWidget
    if name == "CheckboxGridWidget":
        from .checkbox_grid import CheckboxGridWidget
        return CheckboxGridWidget
    if name in ("SliderGroupWidget", "SliderConfig"):
        from .slider_group import SliderGroupWidget, SliderConfig
        if name == "SliderGroupWidget":
            return SliderGroupWidget
        return SliderConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
