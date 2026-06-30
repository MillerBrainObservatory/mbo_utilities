"""
Format round-trip integrity.

Write synthetic data to each filetype, read it back, and verify shape, dtype,
and pixel values against the source. Documents the lossy formats (.npy, .bin
dtype) explicitly. Synthetic data, CI-runnable.
"""

import numpy as np
import pytest
import tifffile

import mbo_utilities as mbo


def _synthetic_5d():
    """Deterministic (T,C,Z,Y,X) int16 volume."""
    rng = np.random.RandomState(7)
    return rng.randint(0, 4096, size=(8, 2, 3, 32, 40), dtype=np.int16)


class TestVolumeRoundtrip:
    """tiff/zarr preserve a full 5D TCZYX volume bit-exactly."""

    @pytest.mark.parametrize("ext", [".tiff", ".zarr"])
    def test_5d_exact(self, output_dir, ext):
        data = _synthetic_5d()
        arr = mbo.imread(data, dims="TCZYX")
        mbo.imwrite(arr, output_dir, ext=ext, overwrite=True)

        path = next(output_dir.rglob("*.zarr")) if ext == ".zarr" else next(output_dir.rglob("*.tif"))
        back = mbo.imread(path)

        assert back.shape == data.shape
        assert back.dtype == data.dtype
        assert np.array_equal(np.asarray(back[:]), data)


class TestSingleChannelRoundtrip:
    """h5/bin preserve a single-channel volume (bin forces int16)."""

    def test_h5_exact(self, synthetic_4d_data, output_dir):
        arr = mbo.imread(synthetic_4d_data, dims="TZYX")
        mbo.imwrite(arr, output_dir, ext=".h5", overwrite=True)

        back = mbo.imread(next(output_dir.rglob("*.h5")))
        assert back.shape == (synthetic_4d_data.shape[0], 1, *synthetic_4d_data.shape[1:])
        assert np.array_equal(np.asarray(back[:])[:, 0], synthetic_4d_data)

    def test_bin_exact_int16(self, synthetic_4d_data, output_dir):
        arr = mbo.imread(synthetic_4d_data, dims="TZYX")
        mbo.imwrite(arr, output_dir, ext=".bin", overwrite=True)

        back = mbo.imread(output_dir)
        assert back.dtype == np.int16
        assert np.array_equal(
            np.asarray(back[:])[:, 0], synthetic_4d_data.astype(np.int16)
        )


class TestNpyRoundtrip:
    """.npy writes one file per plane; each reads back exactly."""

    def test_per_plane_files(self, synthetic_4d_data, output_dir):
        arr = mbo.imread(synthetic_4d_data, dims="TZYX")
        mbo.imwrite(arr, output_dir, ext=".npy", overwrite=True)

        npy_files = sorted(output_dir.glob("*.npy"))
        nz = synthetic_4d_data.shape[1]
        assert len(npy_files) == nz

        # each plane file reads back as (T,1,1,Y,X) == its source plane
        for z, npy_file in enumerate(npy_files):
            back = mbo.imread(npy_file)
            assert np.array_equal(np.asarray(back[:])[:, 0, 0], synthetic_4d_data[:, z])


class TestStridedSubset:
    """Two non-contiguous selection axes at once read back exactly (regression)."""

    def test_strided_t_and_z(self, output_dir):
        T, Z, Y, X = 10, 6, 8, 8
        data = np.empty((T, Z, Y, X), dtype=np.int16)
        for t in range(T):
            for z in range(Z):
                data[t, z] = t * 100 + z
        arr = mbo.imread(data, dims="TZYX")
        mbo.imwrite(arr, output_dir, ext=".zarr",
                    planes=[1, 3, 5], timepoints=[1, 3, 5, 7, 9], overwrite=True)

        back = np.asarray(mbo.imread(next(output_dir.rglob("*.zarr")))[:])  # (5,1,3,Y,X)
        assert back.shape == (5, 1, 3, Y, X)

        sel_t, sel_z = [0, 2, 4, 6, 8], [0, 2, 4]
        expected = np.array(
            [[t * 100 + z for z in sel_z] for t in sel_t], dtype=np.int16
        )[..., None, None] * np.ones((1, 1, Y, X), dtype=np.int16)
        assert np.array_equal(back[:, 0], expected)


class TestCrossFormat:
    """A -> B conversion preserves single-plane data."""

    @pytest.mark.parametrize("source_ext", [".zarr", ".h5", ".bin"])
    def test_to_tiff(self, synthetic_3d_data, output_dir, source_ext):
        src_dir = output_dir / "src"
        src_dir.mkdir()
        arr = mbo.imread(synthetic_3d_data, dims="TYX")
        mbo.imwrite(arr, src_dir, ext=source_ext, overwrite=True)

        if source_ext == ".bin":
            intermediate = mbo.imread(src_dir)
        else:
            glob = "*.zarr" if source_ext == ".zarr" else "*.h5"
            intermediate = mbo.imread(next(src_dir.rglob(glob)))

        dst_dir = output_dir / "dst"
        dst_dir.mkdir()
        mbo.imwrite(intermediate, dst_dir, ext=".tiff", overwrite=True)

        readback = tifffile.imread(next(dst_dir.rglob("*.tif")))
        expected = synthetic_3d_data.astype(np.int16) if source_ext == ".bin" else synthetic_3d_data
        assert np.array_equal(np.asarray(readback).squeeze(), expected)


class TestDtypePreservation:
    """Lossless formats keep the source dtype."""

    @pytest.mark.parametrize("ext", [".tiff", ".zarr", ".h5"])
    def test_dtype(self, synthetic_4d_data, output_dir, ext):
        arr = mbo.imread(synthetic_4d_data, dims="TZYX")
        mbo.imwrite(arr, output_dir, ext=ext, overwrite=True)

        path = next(output_dir.rglob("*.zarr")) if ext == ".zarr" else (
            next(output_dir.rglob("*.h5")) if ext == ".h5" else next(output_dir.rglob("*.tif"))
        )
        assert mbo.imread(path).dtype == synthetic_4d_data.dtype


class TestDataIntegrity:
    """No frame is silently zeroed by a round-trip."""

    def test_no_zero_frames_h5(self, synthetic_4d_data, output_dir):
        arr = mbo.imread(synthetic_4d_data, dims="TZYX")
        mbo.imwrite(arr, output_dir, ext=".h5", overwrite=True)

        back = np.asarray(mbo.imread(next(output_dir.rglob("*.h5")))[:])
        for t in range(back.shape[0]):
            assert back[t].max() > 0, f"frame {t} is all zeros"


class TestSourceArrayTypes:
    """imwrite accepts both a raw ndarray and an already-wrapped LazyArray."""

    def test_from_raw_ndarray(self, synthetic_3d_data, output_dir):
        mbo.imwrite(synthetic_3d_data, output_dir, ext=".tiff", dim_order="TYX", overwrite=True)
        readback = tifffile.imread(next(output_dir.rglob("*.tif")))
        assert np.array_equal(np.asarray(readback).squeeze(), synthetic_3d_data)

    def test_from_lazy_array(self, synthetic_4d_data, output_dir):
        arr = mbo.imread(synthetic_4d_data, dims="TZYX")
        mbo.imwrite(arr, output_dir, ext=".tiff", num_timepoints=5, overwrite=True)
        back = mbo.imread(next(output_dir.rglob("*.tif")))
        assert back.nt == 5
        assert np.array_equal(np.asarray(back[:])[:, 0], synthetic_4d_data[:5])
