"""
Generic list selector widget.

A composable UI component for selecting from a list of items.
Use for z-plane selection, array selection, channel selection, etc.
"""

from __future__ import annotations

from typing import Callable, Sequence

from imgui_bundle import imgui

__all__ = ["ListSelectorWidget"]


class ListSelectorWidget:
    """
    Generic list selector - works for z-planes, arrays, channels, etc.

    This is a composable widget: you configure it with data, not subclass it.

    Parameters
    ----------
    items : Sequence[str]
        Display labels for each item.
    title : str
        Header text shown above the selector.
    multi_select : bool
        If True, allows multiple selections.
    on_change : Callable, optional
        Called when selection changes: on_change(selected_indices).
    default_selected : int | set[int] | None
        Initial selection.
    show_select_buttons : bool
        If True and multi_select, show Select All/None buttons.

    Examples
    --------
    >>> # Single-select z-plane selector
    >>> plane_selector = ListSelectorWidget(
    ...     items=[f"Plane {i}" for i in range(10)],
    ...     title="Z-Plane",
    ...     multi_select=False,
    ...     on_change=lambda idx: print(f"Selected plane {idx}"),
    ... )

    >>> # Multi-select array selector
    >>> array_selector = ListSelectorWidget(
    ...     items=["Array 1", "Array 2", "Array 3"],
    ...     title="Arrays",
    ...     multi_select=True,
    ...     default_selected={0, 1},
    ... )
    """

    def __init__(
        self,
        items: Sequence[str],
        title: str = "Select",
        multi_select: bool = False,
        on_change: Callable[[int | set[int]], None] | None = None,
        default_selected: int | set[int] | None = None,
        show_select_buttons: bool = True,
    ):
        self.items = list(items)
        self.title = title
        self.multi_select = multi_select
        self.on_change = on_change
        self.show_select_buttons = show_select_buttons

        # Initialize selection
        if multi_select:
            if default_selected is None:
                self._selected: set[int] = set()
            elif isinstance(default_selected, set):
                self._selected = default_selected.copy()
            else:
                self._selected = {default_selected}
        else:
            if default_selected is None:
                self._selected: int | None = None
            elif isinstance(default_selected, set):
                self._selected = next(iter(default_selected)) if default_selected else None
            else:
                self._selected = default_selected

    @property
    def selected(self) -> int | set[int] | None:
        """Current selection (index or set of indices)."""
        return self._selected

    @selected.setter
    def selected(self, value: int | set[int] | None) -> None:
        """Set the current selection."""
        if self.multi_select:
            if value is None:
                self._selected = set()
            elif isinstance(value, set):
                self._selected = value.copy()
            else:
                self._selected = {value}
        else:
            if isinstance(value, set):
                self._selected = next(iter(value)) if value else None
            else:
                self._selected = value

    def draw(self) -> None:
        """Render the selector UI."""
        if self.title:
            imgui.text(self.title)

        # Select All/None buttons for multi-select
        if self.multi_select and self.show_select_buttons:
            if imgui.small_button(f"All##{self.title}"):
                self.select_all()
            imgui.same_line()
            if imgui.small_button(f"None##{self.title}"):
                self.select_none()

        changed = False

        if self.multi_select:
            for i, item in enumerate(self.items):
                is_checked = i in self._selected
                clicked, new_checked = imgui.checkbox(f"{item}##{self.title}_{i}", is_checked)
                if clicked:
                    if new_checked:
                        self._selected.add(i)
                    else:
                        self._selected.discard(i)
                    changed = True
        else:
            for i, item in enumerate(self.items):
                is_selected = self._selected == i
                clicked, _ = imgui.selectable(f"{item}##{self.title}_{i}", is_selected)
                if clicked:
                    self._selected = i
                    changed = True

        if changed and self.on_change:
            self.on_change(self._selected)

    def select_all(self) -> None:
        """Select all items (multi_select only)."""
        if self.multi_select:
            self._selected = set(range(len(self.items)))
            if self.on_change:
                self.on_change(self._selected)

    def select_none(self) -> None:
        """Deselect all items."""
        if self.multi_select:
            self._selected = set()
        else:
            self._selected = None
        if self.on_change:
            self.on_change(self._selected)

    def update_items(self, items: Sequence[str]) -> None:
        """
        Update the list of items.

        Clears selection if items have changed.
        """
        if list(items) != self.items:
            self.items = list(items)
            self.select_none()
