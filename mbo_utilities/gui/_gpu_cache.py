"""Process-wide cache of ``fastplotlib.enumerate_adapters()``.

Calling ``fpl.enumerate_adapters()`` initializes wgpu, which on Windows
creates and clears an OpenGL context via WGL. Doing that from inside the
imgui rendering loop clobbers the GLFW backend's current WGL context and
emits ``Glfw Error 65544: WGL: Failed to clear current context``; the
visible symptom is the host window flickering black/unblack the moment
the Options popup opens.

This module enumerates adapters once, at startup (before ``immapp.run``
takes over the GL context), and hands the list to anyone who needs it
later. Subsequent calls return the cached list without touching wgpu.
"""
from __future__ import annotations

from typing import Any

_ADAPTERS: list[Any] | None = None


def prime() -> None:
    """Enumerate adapters and cache them. Safe to call multiple times."""
    global _ADAPTERS
    if _ADAPTERS is not None:
        return
    try:
        import fastplotlib as fpl
        _ADAPTERS = list(fpl.enumerate_adapters())
    except Exception:
        _ADAPTERS = []


def get_adapters() -> list[Any]:
    """Return the cached adapter list, priming the cache on first call.

    Callers inside the imgui frame should ensure :func:`prime` was called
    earlier (from a pre-GL context) — but calling this from inside the
    frame is still safe; it's just no-op after the first call.
    """
    if _ADAPTERS is None:
        prime()
    return _ADAPTERS or []
