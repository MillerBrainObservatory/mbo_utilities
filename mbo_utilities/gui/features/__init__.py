"""
Feature classes - auto-discovered capability-based controls.

A Feature is a small UI control that:
- Appears conditionally based on data capabilities
- Is auto-discovered from the features/ directory
- Has a priority for render ordering
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from imgui_bundle import imgui

if TYPE_CHECKING:
    from ..viewers import BaseViewer

__all__ = [
    "BaseFeature",
    "get_supported_features",
    "draw_all_features",
    "cleanup_all_features",
]


class BaseFeature(ABC):
    """
    Base class for capability-based features.

    A Feature is a small UI control that:
    - Appears conditionally based on data capabilities
    - Is auto-discovered from the features/ directory
    - Has a priority for render ordering

    Attributes
    ----------
    name : str
        Human-readable name for this feature.
    priority : int
        Render order priority. Lower values are rendered first.
    viewer : BaseViewer
        The parent viewer containing this feature.

    Notes
    -----
    Subclasses must implement:
    - is_supported(): Check if feature applies to the viewer's data
    - draw(): Render the feature UI
    """

    name: str = "Base Feature"
    priority: int = 50  # Lower = rendered first

    def __init__(self, viewer: "BaseViewer"):
        """
        Initialize the feature.

        Parameters
        ----------
        viewer : BaseViewer
            The parent viewer containing this feature.
        """
        self.viewer = viewer

    @classmethod
    @abstractmethod
    def is_supported(cls, viewer: "BaseViewer") -> bool:
        """
        Check if this feature applies to the viewer's data.

        Parameters
        ----------
        viewer : BaseViewer
            The viewer to check.

        Returns
        -------
        bool
            True if the feature should be shown.
        """
        ...

    @abstractmethod
    def draw(self) -> None:
        """Render the feature UI."""
        ...

    def cleanup(self) -> None:
        """
        Clean up resources when the feature is destroyed.

        Override in subclasses to release resources.
        """
        pass


# Registry of discovered feature classes
_FEATURE_CLASSES: list[type[BaseFeature]] = []


def _discover_features() -> None:
    """Auto-discover feature classes from this package."""
    global _FEATURE_CLASSES

    if _FEATURE_CLASSES:
        return  # Already discovered

    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        # Skip private modules
        if module_info.name.startswith("_"):
            continue

        # Skip subdirectories for now (pipelines handled separately)
        if module_info.ispkg:
            continue

        try:
            module = importlib.import_module(f".{module_info.name}", package=__name__)

            # Find BaseFeature subclasses in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseFeature)
                    and attr is not BaseFeature
                ):
                    _FEATURE_CLASSES.append(attr)
        except Exception as e:
            # Log but don't crash on import errors
            print(f"Warning: failed to import feature module {module_info.name}: {e}")

    # Sort by priority
    _FEATURE_CLASSES.sort(key=lambda f: f.priority)


def get_supported_features(viewer: "BaseViewer") -> list[BaseFeature]:
    """
    Get all features that are supported for the given viewer.

    Parameters
    ----------
    viewer : BaseViewer
        The viewer to check features for.

    Returns
    -------
    list[BaseFeature]
        Instantiated features sorted by priority.
    """
    _discover_features()

    supported = []
    for feature_cls in _FEATURE_CLASSES:
        try:
            if feature_cls.is_supported(viewer):
                supported.append(feature_cls(viewer))
        except Exception as e:
            import traceback
            print(f"Warning: Feature {feature_cls.__name__} support check failed: {e}")
            traceback.print_exc()

    # Sort by priority (lower = first)
    supported.sort(key=lambda f: f.priority)
    return supported


def draw_all_features(viewer: "BaseViewer", features: list[BaseFeature]) -> None:
    """
    Draw all supported features.

    Parameters
    ----------
    viewer : BaseViewer
        The parent viewer (used for error logging).
    features : list[BaseFeature]
        The features to draw.
    """
    for feature in features:
        try:
            feature.draw()
        except Exception as e:
            import traceback
            error_msg = f"Error in {feature.name}: {e}"
            if hasattr(viewer, "logger"):
                viewer.logger.error(error_msg)
                viewer.logger.error(traceback.format_exc())
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), error_msg)


def cleanup_all_features(features: list[BaseFeature]) -> None:
    """
    Clean up all features when the viewer is closing.

    Parameters
    ----------
    features : list[BaseFeature]
        The features to clean up.
    """
    for feature in features:
        try:
            feature.cleanup()
        except Exception as e:
            print(f"Warning: cleanup failed for {feature.name}: {e}")
