"""Analysis tools for mbo_utilities.

Lazy imports keep heavy dependencies (numpy/scipy/scikit-image) out of
import-time cost for the lightweight CLI commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mbo_utilities.analysis.phasecorr import bidir_phasecorr as bidir_phasecorr
    from mbo_utilities.analysis.scanphase import (
        run_scanphase_analysis as run_scanphase_analysis,
    )

__all__ = [
    "_patch_qt_checkbox",
    "bidir_phasecorr",
    "run_scanphase_analysis",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "bidir_phasecorr": (".phasecorr", "bidir_phasecorr"),
    "run_scanphase_analysis": (".scanphase", "run_scanphase_analysis"),
}

_loaded: dict[str, object] = {}


def _patch_qt_checkbox():
    """Patch QCheckBox for Qt5/Qt6 compatibility with cellpose."""
    try:
        from qtpy.QtWidgets import QCheckBox
        if not hasattr(QCheckBox, "checkStateChanged"):
            QCheckBox.checkStateChanged = QCheckBox.stateChanged
    except ImportError:
        pass


def __getattr__(name: str) -> object:
    if name in _loaded:
        return _loaded[name]
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_name, package="mbo_utilities.analysis")
        obj = getattr(module, attr_name)
        _loaded[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
