"""
Base classes for array features.

Following fastplotlib's pattern, features are composable properties that
manage their own state and emit events when changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import warn

if TYPE_CHECKING:
    from typing import Callable


class ArrayFeatureEvent:
    """
    Event emitted when an array feature changes.

    Attributes
    ----------
    type : str
        name of the feature that changed (e.g., "dim_labels")
    info : dict
        event info with 'value' key and any additional context
    """

    def __init__(self, type: str, info: dict):
        self.type = type
        self.info = info

    def __repr__(self) -> str:
        return f"ArrayFeatureEvent(type={self.type!r}, info={self.info})"


class ArrayFeature:
    """
    Base class for array features.

    Features are properties that can be attached to array classes.
    They manage their own state and emit events when changed.

    Subclasses must implement:
    - value property (getter)
    - set_value(array, value) method

    Parameters
    ----------
    property_name : str
        name used for event emission and identification
    """

    def __init__(self, property_name: str, **kwargs):
        self._property_name = property_name
        self._event_handlers: list[Callable] = []
        self._block_events = False

    @property
    def value(self) -> Any:
        """feature value, must be implemented in subclass"""
        raise NotImplementedError

    def set_value(self, array, value: Any) -> None:
        """set feature value, must be implemented in subclass"""
        raise NotImplementedError

    def block_events(self, val: bool) -> None:
        """block or unblock event emission"""
        self._block_events = val

    def add_event_handler(self, handler: Callable) -> None:
        """
        Add an event handler called when this feature changes.

        Parameters
        ----------
        handler : callable
            function accepting an ArrayFeatureEvent as argument
        """
        if not callable(handler):
            raise TypeError("event handler must be callable")

        if handler in self._event_handlers:
            warn(f"Event handler {handler} is already registered.")
            return

        self._event_handlers.append(handler)

    def remove_event_handler(self, handler: Callable) -> None:
        """remove a registered event handler"""
        if handler not in self._event_handlers:
            raise KeyError(f"event handler {handler} not registered")
        self._event_handlers.remove(handler)

    def clear_event_handlers(self) -> None:
        """clear all event handlers"""
        self._event_handlers.clear()

    def _call_event_handlers(self, event: ArrayFeatureEvent) -> None:
        if self._block_events:
            return
        for func in self._event_handlers:
            try:
                func(event)
            except Exception as e:
                warn(f"Error in {self.__class__.__name__} event handler: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._property_name!r})"
