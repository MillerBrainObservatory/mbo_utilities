"""
Suite2p readiness across output formats + imread array wrapping.

Every format you can feed to suite2p (tiff/zarr/h5/bin) must read back as a
correctly-sized single plane, and the .bin writer must stamp ops.npy with the
right Ly/Lx/nframes. One real run_plane registration smoke test confirms the
end-to-end path. Synthetic data, CI-runnable; the real run is marked slow and
skips if lbm_suite2p_python is absent.
"""

from pathlib import Path

import numpy as np
import pytest

import mbo_utilities as mbo

SUITE2P_FORMATS = [".tiff", ".zarr", ".h5", ".bin"]


def _read_back(out_dir, ext):
    if ext == ".zarr":
        return mbo.imread(next(out_dir.rglob("*.zarr")))
    if ext == ".h5":
        return mbo.imread(next(out_dir.rglob("*.h5")))
    if ext in (".tiff", ".tif"):
        return mbo.imread(next(out_dir.rglob("*.tif")))
    return mbo.imread(out_dir)  # .bin volume dir


class TestSuite2pReadiness:
    """Each format yields a correctly-sized single plane for suite2p."""

    @pytest.mark.parametrize("ext", SUITE2P_FORMATS)
    def test_single_plane_size(self, synthetic_4d_data, output_dir, ext):
        nt, nz, ny, nx = synthetic_4d_data.shape
        src = mbo.imread(synthetic_4d_data, dims="TZYX")
        mbo.imwrite(src, output_dir, ext=ext, overwrite=True)

        back = _read_back(output_dir, ext)
        assert back.shape == (nt, 1, nz, ny, nx)

        plane = np.asarray(back[:, 0, 0])  # plane 0 -> (T, Y, X)
        assert plane.shape == (nt, ny, nx)
        assert np.array_equal(plane, synthetic_4d_data[:, 0])

    def test_bin_ops_sized_for_suite2p(self, synthetic_4d_data, output_dir):
        nt, nz, ny, nx = synthetic_4d_data.shape
        src = mbo.imread(synthetic_4d_data, dims="TZYX")
        mbo.imwrite(src, output_dir, ext=".bin", overwrite=True)

        ops_files = sorted(output_dir.rglob("ops.npy"))
        assert len(ops_files) == nz
        for ops_file in ops_files:
            assert (ops_file.parent / "data_raw.bin").exists()
            ops = np.load(ops_file, allow_pickle=True).item()
            assert (ops["Ly"], ops["Lx"], ops["nframes"]) == (ny, nx, nt)


class TestSuite2pRun:
    """One real registration pass over a written .bin (detection off)."""

    @pytest.mark.slow
    def test_run_plane_registration(self, output_dir):
        pytest.importorskip("lbm_suite2p_python")
        import os

        os.environ.setdefault("MBO_GPU", "0")
        from lbm_suite2p_python import run_plane

        rng = np.random.RandomState(1)
        mov = rng.randint(0, 4096, size=(64, 64, 64), dtype=np.int16)  # T, Y, X
        arr = mbo.imread(mov, dims="TYX")
        mbo.imwrite(arr, output_dir, ext=".bin", overwrite=True, metadata={"fs": 10.0})

        plane_dir = next(p.parent for p in output_dir.rglob("data_raw.bin"))
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
        ops.update({
            "do_registration": 1, "roidetect": 0, "do_detection": 0,
            "nonrigid": False, "two_step_registration": False,
        })

        result = Path(run_plane(
            str(plane_dir / "data_raw.bin"), save_path=str(plane_dir), ops=ops,
            keep_raw=True, keep_reg=True, replot=False,
        ))

        assert result.exists()
        assert (plane_dir / "data.bin").exists()  # registered output

        final = np.load(result, allow_pickle=True).item()
        assert (final["Ly"], final["Lx"], final["nframes"]) == (64, 64, 64)


class TestImreadWrapping:
    """Wrap arbitrary arrays via imread, declaring dims and metadata."""

    def test_ndarray_with_dims_and_metadata(self):
        data = np.zeros((4, 2, 3, 8, 8), dtype=np.uint16)
        arr = mbo.imread(data, dims="TCZYX", metadata={"dx": 2.0, "fs": 5.0})

        assert arr.shape == (4, 2, 3, 8, 8)
        assert arr.dx == 2.0
        assert arr.fs == 5.0
        assert (arr.nc, arr.nz) == (2, 3)

    def test_dims_remap_places_channel(self):
        # 4D source declared TCYX -> C on axis 1, Z stays singleton
        data = np.zeros((10, 2, 32, 32), dtype=np.uint16)
        arr = mbo.imread(data, dims="TCYX")
        assert arr.shape == (10, 2, 1, 32, 32)

    def test_already_loaded_array_passes_through(self):
        arr = mbo.imread(np.zeros((4, 8, 8), dtype=np.uint16), dims="TYX")
        assert mbo.imread(arr) is arr

    def test_channel_selection_returns_4d_view(self):
        data = np.arange(4 * 2 * 1 * 8 * 8, dtype=np.uint16).reshape(4, 2, 1, 8, 8)
        view = mbo.imread(data, dims="TCZYX", channel=1)  # 0-based channel

        assert view.ndim == 4
        assert view.shape == (4, 1, 8, 8)
        assert np.array_equal(np.asarray(view[:]), data[:, 1])
