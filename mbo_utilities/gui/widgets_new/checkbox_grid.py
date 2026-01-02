"""
Checkbox grid widget.

A composable UI component for multi-selection in a grid layout.
"""

from __future__ import annotations

from typing import Callable, Sequence

from imgui_bundle import imgui

__all__ = ["CheckboxGridWidget"]


class CheckboxGridWidget:
    """
    Grid of checkboxes for multi-selection.

    This is a composable widget: you configure it with data, not subclass it.

    Parameters
    ----------
    items : Sequence[str]
        Display labels for each item.
    title : str
        Header text shown above the grid.
    columns : int
        Number of columns in the grid.
    on_change : Callable, optional
        Called when selection changes: on_change(selected_indices).
    default_selected : set[int] | None
        Initially selected indices.
    show_select_buttons : bool
        If True, show Select All/None buttons.

    Examples
    --------
    >>> # Plane selection grid
    >>> plane_grid = CheckboxGridWidget(
    ...     items=[f"Z{i}" for i in range(20)],
    ...     title="Select Planes",
    ...     columns=5,
    ...     default_selected={0, 1, 2},
    ... )
    """

    def __init__(
        self,
        items: Sequence[str],
        title: str = "",
        columns: int = 4,
        on_change: Callable[[set[int]], None] | None = None,
        default_selected: set[int] | None = None,
        show_select_buttons: bool = True,
    ):
        self.items = list(items)
        self.title = title
        self.columns = columns
        self.on_change = on_change
        self.show_select_buttons = show_select_buttons
        self._selected: set[int] = set(default_selected) if default_selected else set()

    @property
    def selected(self) -> set[int]:
        """Current selection (set of indices)."""
        return self._selected.copy()

    @selected.setter
    def selected(self, value: set[int]) -> None:
        """Set the current selection."""
        self._selected = set(value)

    def draw(self) -> None:
        """Render the checkbox grid."""
        if self.title:
            imgui.text(self.title)

        # Select All/None buttons
        if self.show_select_buttons:
            if imgui.small_button(f"All##{self.title}_all"):
                self.select_all()
            imgui.same_line()
            if imgui.small_button(f"None##{self.title}_none"):
                self.select_none()
            imgui.same_line()
            imgui.text_disabled(f"({len(self._selected)}/{len(self.items)})")

        changed = False

        # Use table for grid layout
        table_flags = imgui.TableFlags_.sizing_fixed_fit
        if imgui.begin_table(f"##{self.title}_grid", self.columns, table_flags):
            for i, item in enumerate(self.items):
                imgui.table_next_column()
                is_checked = i in self._selected
                clicked, new_checked = imgui.checkbox(f"{item}##{self.title}_{i}", is_checked)
                if clicked:
                    if new_checked:
                        self._selected.add(i)
                    else:
                        self._selected.discard(i)
                    changed = True
            imgui.end_table()

        if changed and self.on_change:
            self.on_change(self._selected)

    def select_all(self) -> None:
        """Select all items."""
        self._selected = set(range(len(self.items)))
        if self.on_change:
            self.on_change(self._selected)

    def select_none(self) -> None:
        """Deselect all items."""
        self._selected = set()
        if self.on_change:
            self.on_change(self._selected)

    def update_items(self, items: Sequence[str]) -> None:
        """
        Update the list of items.

        Clears selection if items have changed.
        """
        if list(items) != self.items:
            self.items = list(items)
            self._selected = set()
