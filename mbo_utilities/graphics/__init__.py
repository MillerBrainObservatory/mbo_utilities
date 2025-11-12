"""
Graphics module with lazy imports to avoid loading heavy dependencies
(torch, cupy, suite2p) until actually needed.
"""

__all__ = [
    "PreviewDataWidget",
    "run_gui",
]


def __getattr__(name):
    """Lazy import heavy graphics modules only when accessed."""
    if name == "run_gui":
        from .run_gui import run_gui
        return run_gui
    elif name == "PreviewDataWidget":
        from .imgui import PreviewDataWidget
        return PreviewDataWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
