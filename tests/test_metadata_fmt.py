"""Metadata viewer value-formatting regressions.

The metadata inspector renders each leaf via ``fmt_value``; failures
here become walls of numbers in the GUI. The cases covered below are
the ones that have actually surfaced as user-visible bugs:
suite2p ``ops.npy`` packs ``Vmap`` as an ``object`` ndarray whose
elements are themselves 2-D float arrays (one per detection pyramid
level). The original ``fmt_value`` had a ``size <= 8`` fast path that
called ``repr(x.tolist())``; on an object array that expands every
inner array, dumping pages of numbers into the inspector instead of
a one-line summary.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbo_utilities.gui._imgui_helpers import fmt_value


class TestSmallScalarArraysStillInline:
    """The ``size <= 8`` fast path remains useful for tiny scalar
    arrays (a 3-element float vector reads better as ``[1.0, 2.0, 3.0]``
    than as ``<shape=(3,), dtype=float64>``)."""

    def test_small_int_array_inlines(self):
        out = fmt_value(np.array([1, 2, 3]))
        assert out == "[1, 2, 3]"

    def test_small_float_array_inlines(self):
        out = fmt_value(np.array([1.5, 2.5]))
        assert out == "[1.5, 2.5]"

    def test_small_bool_array_inlines(self):
        out = fmt_value(np.array([True, False, True]))
        assert out == "[True, False, True]"

    def test_size_exactly_8_inlines(self):
        out = fmt_value(np.arange(8))
        assert out.startswith("[") and out.endswith("]")


class TestLargeScalarArraysSummarize:
    """Anything beyond the inline budget must collapse to a one-line
    shape/dtype summary regardless of dtype kind."""

    def test_large_2d_float_summarizes(self):
        out = fmt_value(np.zeros((100, 100), dtype=np.float32))
        assert out == "<shape=(100, 100), dtype=float32>"

    def test_size_9_summarizes(self):
        # one element past the inline budget
        out = fmt_value(np.arange(9))
        assert "shape=(9,)" in out


class TestObjectArrayNeverDumpsInner:
    """Object-dtype arrays must always summarize, never inline.
    Each "element" of an object ndarray can itself be an arbitrary
    Python object (including a multi-megabyte array), so
    ``repr(tolist())`` is unbounded."""

    def test_suite2p_vmap_summarizes(self):
        # exact shape produced by suite2p detection pyramid (5 levels)
        vmap = np.empty(5, dtype=object)
        vmap[0] = np.zeros((436, 440), dtype=np.float32)
        vmap[1] = np.zeros((218, 220), dtype=np.float32)
        vmap[2] = np.zeros((109, 110), dtype=np.float32)
        vmap[3] = np.zeros((55, 55), dtype=np.float32)
        vmap[4] = np.zeros((28, 28), dtype=np.float32)
        out = fmt_value(vmap)
        assert out == "<shape=(5,), dtype=object>"
        # critical: inner array values must NOT appear in the summary
        assert "0." not in out
        assert "float32" not in out  # only the OUTER dtype shows

    def test_object_array_size_below_8_still_summarizes(self):
        # even within the size budget, object dtype must summarize
        a = np.empty(3, dtype=object)
        a[0] = "hello"
        a[1] = [1, 2, 3, 4, 5, 6]
        a[2] = np.zeros((50, 50))
        out = fmt_value(a)
        assert out == "<shape=(3,), dtype=object>"
        assert "hello" not in out
        assert "0." not in out


class TestDtypelessFallback:
    """If the array-like has neither shape+dtype nor falls into other
    branches, the catch-all unknown-type formatter applies."""

    def test_class_without_shape_uses_type_name(self):
        class Thing:
            pass
        out = fmt_value(Thing())
        assert out == "<Thing>"
