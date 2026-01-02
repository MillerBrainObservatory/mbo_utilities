"""
Slider group widget.

A composable UI component for grouped sliders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from imgui_bundle import imgui

__all__ = ["SliderGroupWidget", "SliderConfig"]


@dataclass
class SliderConfig:
    """
    Configuration for a single slider.

    Parameters
    ----------
    label : str
        Display label for the slider.
    min_value : float
        Minimum allowed value.
    max_value : float
        Maximum allowed value.
    default : float
        Initial value.
    format : str
        Printf-style format string for display.
    is_int : bool
        If True, use integer slider instead of float.
    tooltip : str, optional
        Tooltip text to show on hover.
    """

    label: str
    min_value: float
    max_value: float
    default: float = 0.0
    format: str = "%.2f"
    is_int: bool = False
    tooltip: str = ""


class SliderGroupWidget:
    """
    Group of related sliders.

    This is a composable widget: you configure it with data, not subclass it.

    Parameters
    ----------
    sliders : Sequence[SliderConfig]
        Configuration for each slider.
    title : str
        Header text shown above the group.
    on_change : Callable, optional
        Called when any value changes: on_change(label, new_value).
    collapsible : bool
        If True, render as a collapsing header.
    default_open : bool
        If collapsible, whether to start open.

    Examples
    --------
    >>> # Image processing sliders
    >>> processing_sliders = SliderGroupWidget(
    ...     sliders=[
    ...         SliderConfig("Brightness", -1.0, 1.0, 0.0, "%.2f"),
    ...         SliderConfig("Contrast", 0.0, 2.0, 1.0, "%.2f"),
    ...         SliderConfig("Blur", 0, 10, 0, "%d", is_int=True),
    ...     ],
    ...     title="Image Processing",
    ...     on_change=lambda label, val: print(f"{label} = {val}"),
    ... )
    """

    def __init__(
        self,
        sliders: Sequence[SliderConfig],
        title: str = "",
        on_change: Callable[[str, float], None] | None = None,
        collapsible: bool = False,
        default_open: bool = True,
    ):
        self.sliders = list(sliders)
        self.title = title
        self.on_change = on_change
        self.collapsible = collapsible
        self.default_open = default_open
        self._values = {s.label: s.default for s in sliders}

    @property
    def values(self) -> dict[str, float]:
        """Get all current values as a dictionary."""
        return self._values.copy()

    def get(self, label: str) -> float:
        """Get the current value for a slider."""
        return self._values[label]

    def set(self, label: str, value: float) -> None:
        """Set the value for a slider."""
        self._values[label] = value

    def draw(self) -> None:
        """Render the slider group."""
        if self.collapsible:
            flags = imgui.TreeNodeFlags_.default_open if self.default_open else 0
            if not imgui.collapsing_header(self.title, flags):
                return
        elif self.title:
            imgui.text(self.title)

        for slider in self.sliders:
            current = self._values[slider.label]

            if slider.is_int:
                changed, new_val = imgui.slider_int(
                    slider.label,
                    int(current),
                    int(slider.min_value),
                    int(slider.max_value),
                )
                new_val = float(new_val)
            else:
                changed, new_val = imgui.slider_float(
                    slider.label,
                    current,
                    slider.min_value,
                    slider.max_value,
                    slider.format,
                )

            # Show tooltip if provided
            if slider.tooltip and imgui.is_item_hovered():
                imgui.set_tooltip(slider.tooltip)

            if changed:
                self._values[slider.label] = new_val
                if self.on_change:
                    self.on_change(slider.label, new_val)

    def reset(self) -> None:
        """Reset all sliders to their default values."""
        for slider in self.sliders:
            self._values[slider.label] = slider.default
