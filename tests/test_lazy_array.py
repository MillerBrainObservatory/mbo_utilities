"""
tests for the v4 LazyArray base and imread dispatch registry (phase 1).

phase 1 is additive: LazyArray carries the 5D accessors and the
register/dispatch plumbing. imread() still uses the legacy if/elif chain,
so these tests only cover the new surface.
"""

import numpy as np
import pytest

import mbo_utilities
from mbo_utilities.lazy_array import (
    LazyArray,
    register_array_class,
    _dispatch,
    _REGISTRY,
)
from mbo_utilities.arrays._base import Shape5DMixin
from mbo_utilities.arrays import NumpyArray


def test_lazyarray_exported_at_top_level():
    assert mbo_utilities.LazyArray is LazyArray
    assert mbo_utilities.register_array_class is register_array_class


def test_shape5dmixin_is_lazyarray_subclass():
    # back-compat: existing `class Foo(..., Shape5DMixin)` declarations
    # become LazyArray instances without edits.
    assert issubclass(Shape5DMixin, LazyArray)


def test_existing_array_is_lazyarray_instance():
    arr = NumpyArray(np.zeros((4, 8, 8), dtype="uint16"))
    assert isinstance(arr, LazyArray)
    # behavior unchanged: still 5D shape5d.
    assert arr.shape5d == (4, 1, 1, 8, 8)


def test_default_can_open_is_false(tmp_path):
    assert LazyArray.can_open(tmp_path) is False


def test_register_rejects_non_lazyarray():
    class NotAnArray:
        pass

    with pytest.raises(TypeError):
        register_array_class(NotAnArray)


def test_register_and_dispatch_by_priority():
    class _FakeArray(LazyArray):
        PRIORITY = 999

        @classmethod
        def can_open(cls, path):
            return True

    before = list(_REGISTRY)
    try:
        register_array_class(_FakeArray)
        # idempotent
        register_array_class(_FakeArray)
        assert _REGISTRY.count(_FakeArray) == 1
        # highest PRIORITY with can_open() True wins
        assert _dispatch("anything.fake") is _FakeArray
    finally:
        if _FakeArray in _REGISTRY and _FakeArray not in before:
            _REGISTRY.remove(_FakeArray)


def test_priority_override_on_register():
    class _LowArray(LazyArray):
        PRIORITY = 1

        @classmethod
        def can_open(cls, path):
            return True

    before = list(_REGISTRY)
    try:
        register_array_class(_LowArray, priority=500)
        assert _LowArray.PRIORITY == 500
    finally:
        if _LowArray in _REGISTRY and _LowArray not in before:
            _REGISTRY.remove(_LowArray)
