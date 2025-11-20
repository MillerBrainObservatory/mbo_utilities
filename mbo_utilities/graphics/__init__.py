"""
Graphics module with lazy imports to avoid loading heavy dependencies
(torch, cupy, suite2p) until actually needed.
"""

__all__ = [
    "PreviewDataWidget",
    "run_gui",
    "MboImageProcessor",
    "MultiSessionImageProcessor",
]


def __getattr__(name):
    """Lazy import heavy graphics modules only when accessed."""
    if name == "run_gui":
        from .run_gui import run_gui
        return run_gui
    elif name == "PreviewDataWidget":
        from .imgui import PreviewDataWidget
        return PreviewDataWidget
    elif name == "MboImageProcessor":
        from ._processors import MboImageProcessor
        return MboImageProcessor
    elif name == "MultiSessionImageProcessor":
        from ._processors import MultiSessionImageProcessor
        return MultiSessionImageProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
