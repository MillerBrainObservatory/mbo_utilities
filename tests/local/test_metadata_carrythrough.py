"""Metadata carry-through: raw ScanImage tiff -> zarr -> axial -> tiff -> bin.

Pins that the dim/metadata contract survives every hop after the LazyArray
consolidation. Needs a real ScanImage source; set MBO_PIPELINE_TIFF or place
data at one of the known paths, else the module skips.

    MBO_PIPELINE_TIFF=D:/2026-06-02_demo_lbm/raw \
        uv run pytest tests/local/test_metadata_carrythrough.py -v -s

The suite2p step itself is NOT asserted here: suite2p's ROI fitting can raise
LinAlgError on short/degenerate subsets independent of mbo. We assert the bin
+ ops.npy handed to suite2p carry the right metadata.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mbo_utilities as mbo

NT = 32
PLANE = 8  # 1-based

# physically meaningful keys that must survive every hop
CARRY = ["dx", "dy", "fs", "pixel_resolution", "num_planes"]

_CANDIDATES = [
    os.environ.get("MBO_PIPELINE_TIFF"),
    "D:/2026-06-02_demo_lbm/raw",
    str(Path.home() / ".mbo" / "tests" / "lbm" / "mbo_utilities" / "test_input.tif"),
]


def _find_source():
    for c in _CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


@pytest.fixture(scope="module")
def source():
    src = _find_source()
    if src is None:
        pytest.skip("no ScanImage source (set MBO_PIPELINE_TIFF)")
    return mbo.imread(src)


@pytest.fixture(scope="module")
def work():
    d = Path(tempfile.mkdtemp(prefix="carrythrough_"))
    yield d


def _assert_carry(md, tag):
    missing = [k for k in CARRY if k not in md or md[k] is None]
    assert not missing, f"[{tag}] missing carry keys: {missing}"


class TestMetadataCarryThrough:
    def test_raw_is_5d_tczyx(self, source):
        assert source.ndim == 5
        assert source.dims == ("T", "C", "Z", "Y", "X")
        _assert_carry(source.metadata, "raw")

    def test_zarr_hop_with_register_z(self, source, work):
        zdir = work / "zarr"
        mbo.imwrite(source, zdir, ext=".zarr", num_timepoints=NT,
                    register_z=True, overwrite=True, show_progress=False)
        zpath = next(p for p in zdir.rglob("*.zarr") if (p / "zarr.json").is_file())
        za = mbo.imread(zpath)
        assert za.dims == ("T", "C", "Z", "Y", "X")
        md = za.metadata
        _assert_carry(md, "zarr")
        ps = md.get("plane_shifts")
        assert ps is not None and len(ps) == za.shape[2], "register_z must populate plane_shifts (one row per z)"
        work.joinpath("_zpath.txt").write_text(str(zpath))

    def test_axial_then_tiff_bakes_and_hides_shifts(self, work):
        zpath = Path(work.joinpath("_zpath.txt").read_text())
        za = mbo.imread(zpath)
        view = mbo.with_axial_shifts(za)
        assert view.dims == ("T", "C", "Z", "Y", "X")
        # shift application pads the canvas, so spatial dims grow
        assert view.shape[-2] >= za.shape[-2] and view.shape[-1] >= za.shape[-1]

        tdir = work / "tiff"
        mbo.imwrite(view, tdir, ext=".tiff", overwrite=True, show_progress=False)
        tps = sorted(tdir.rglob("*.tif*"))
        ta = mbo.imread(tdir if len(tps) > 1 else tps[0])
        assert ta.dims == ("T", "C", "Z", "Y", "X")
        assert ta.shape[-2:] == view.shape[-2:]
        _assert_carry(ta.metadata, "tiff")
        # baked output must NOT advertise plane_shifts (else a reader re-shifts)
        assert "plane_shifts" not in ta.metadata

    def test_bin_hop_is_suite2p_ready(self, work):
        tdir = work / "tiff"
        tps = sorted(tdir.rglob("*.tif*"))
        ta = mbo.imread(tdir if len(tps) > 1 else tps[0])
        bdir = work / "bin"
        mbo.imwrite(ta, bdir, ext=".bin", planes=[PLANE], overwrite=True, show_progress=False)
        pdir = next(p for p in bdir.iterdir() if p.is_dir() and (p / "data_raw.bin").exists())
        ops = np.load(pdir / "ops.npy", allow_pickle=True).item()
        for k in ("fs", "Lx", "Ly", "nframes"):
            assert ops.get(k), f"ops.npy missing {k}"
        assert ops["Ly"] == ta.shape[-2] and ops["Lx"] == ta.shape[-1]
