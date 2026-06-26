"""Canonical selection conversion shared by GUI / lbm-suite2p-python / isoview."""

from mbo_utilities.arrays.features import (
    canonical_axis_sizes,
    selection_to_canonical,
    to_isoview_kwargs,
    to_lsp_kwargs,
)


class _Arr:
    def __init__(self, shape):
        self.shape = shape


def test_canonical_axis_sizes():
    assert canonical_axis_sizes(_Arr((10, 4, 14, 256, 256))) == {"T": 10, "C": 4, "Z": 14}


def test_selection_to_canonical_aliases_and_one_based():
    arr = _Arr((10, 4, 14, 256, 256))
    canon = selection_to_canonical(arr, {"View": "1:2", "Z": "1:14:2"})
    assert canon["C"] == [0, 1]
    assert canon["Z"] == [0, 2, 4, 6, 8, 10, 12]
    assert "T" not in canon  # omitted axis is not emitted


def test_selection_to_canonical_zero_based_list():
    arr = _Arr((10, 4, 14, 256, 256))
    canon = selection_to_canonical(arr, {"Cam": [0, 2]}, one_based=False)
    assert canon["C"] == [0, 2]


def test_none_selection_skipped():
    arr = _Arr((10, 4, 14, 256, 256))
    assert selection_to_canonical(arr, {"T": None, "Z": "1:2"}) == {"Z": [0, 1]}


def test_to_lsp_kwargs_is_one_based():
    canon = {"T": [0, 1, 2], "Z": [0, 4], "C": [1]}
    assert to_lsp_kwargs(canon) == {
        "timepoints": [1, 2, 3],
        "planes": [1, 5],
        "channels": [2],
    }


def test_to_isoview_kwargs_is_zero_based():
    canon = {"T": [0, 5], "C": [0, 2]}
    assert to_isoview_kwargs(canon) == {"timepoints": [0, 5], "cameras": [0, 2]}


def test_omitted_axis_not_emitted():
    assert to_lsp_kwargs({"Z": [0, 1]}) == {"planes": [1, 2]}
    assert to_isoview_kwargs({"Z": [0, 1]}) == {}
