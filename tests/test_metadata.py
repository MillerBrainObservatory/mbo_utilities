"""
Metadata I/O across filetypes.

Verifies that scientific metadata (dx/dy/dz/fs + free-form keys) survives a
write -> read round-trip for each format, that subset writes rescale dz/fs,
and that ScanImage source metadata is read when local data is present.

Synthetic data, CI-runnable. The single local-data test skips cleanly.
"""

import numpy as np
import pytest

import mbo_utilities as mbo

# scientific metadata stamped on every synthetic write
SAMPLE = {"dx": 1.5, "dy": 1.5, "dz": 4.0, "fs": 9.6, "objective": "16x", "comment": "unit-test"}

# formats that round-trip a single-channel volume losslessly with metadata
VOLUME_FORMATS = [".tiff", ".zarr", ".h5"]


def _wrap(data, dims=None, metadata=None):
    """imread an ndarray, declaring its source axes and metadata."""
    return mbo.imread(data, dims=dims, metadata=metadata)


def _read_back(out_dir, ext):
    """Locate and imread the written output for a format."""
    if ext == ".zarr":
        path = next(out_dir.rglob("*.zarr"))
    elif ext in (".h5", ".hdf5"):
        path = next(out_dir.rglob("*.h5"))
    elif ext in (".tiff", ".tif"):
        path = next(out_dir.rglob("*.tif"))
    else:
        path = out_dir
    return mbo.imread(path)


class TestMetadataReadback:
    """dx/dy/dz/fs + free-form keys survive a write -> read round-trip."""

    @pytest.mark.parametrize("ext", VOLUME_FORMATS)
    def test_scientific_keys_survive(self, synthetic_4d_data, output_dir, ext):
        arr = _wrap(synthetic_4d_data, dims="TZYX", metadata=SAMPLE)
        mbo.imwrite(arr, output_dir, ext=ext, metadata=SAMPLE, overwrite=True)

        back = _read_back(output_dir, ext)

        assert back.dx == SAMPLE["dx"]
        assert back.dy == SAMPLE["dy"]
        assert back.dz == SAMPLE["dz"]
        assert back.fs == SAMPLE["fs"]

    @pytest.mark.parametrize("ext", VOLUME_FORMATS)
    def test_freeform_keys_survive(self, synthetic_4d_data, output_dir, ext):
        arr = _wrap(synthetic_4d_data, dims="TZYX", metadata=SAMPLE)
        mbo.imwrite(arr, output_dir, ext=ext, metadata=SAMPLE, overwrite=True)

        md = _read_back(output_dir, ext).metadata
        assert md["objective"] == "16x"
        assert md["comment"] == "unit-test"

    @pytest.mark.parametrize("ext", VOLUME_FORMATS)
    def test_pixels_exact_after_roundtrip(self, synthetic_4d_data, output_dir, ext):
        arr = _wrap(synthetic_4d_data, dims="TZYX", metadata=SAMPLE)
        mbo.imwrite(arr, output_dir, ext=ext, metadata=SAMPLE, overwrite=True)

        back = np.asarray(_read_back(output_dir, ext)[:])  # 5D (T,1,Z,Y,X)
        assert np.array_equal(back[:, 0], synthetic_4d_data)


class TestBinOpsMetadata:
    """Suite2p .bin writes carry sizing + fs into per-plane ops.npy."""

    def test_ops_has_sizing_and_fs(self, synthetic_4d_data, output_dir):
        arr = _wrap(synthetic_4d_data, dims="TZYX", metadata=SAMPLE)
        mbo.imwrite(arr, output_dir, ext=".bin", metadata=SAMPLE, overwrite=True)

        nt, _, nz, ny, nx = arr.shape
        ops_files = sorted(output_dir.rglob("ops.npy"))
        assert len(ops_files) == nz, f"expected one ops.npy per plane, got {len(ops_files)}"

        ops = np.load(ops_files[0], allow_pickle=True).item()
        assert ops["Ly"] == ny
        assert ops["Lx"] == nx
        assert ops["nframes"] == nt
        assert ops["fs"] == SAMPLE["fs"]
        assert ops["dz"] == SAMPLE["dz"]


class TestGetMetadata:
    """mbo.get_metadata reads scientific keys back off disk."""

    @pytest.mark.parametrize("ext", VOLUME_FORMATS)
    def test_get_metadata_keys(self, synthetic_4d_data, output_dir, ext):
        arr = _wrap(synthetic_4d_data, dims="TZYX", metadata=SAMPLE)
        mbo.imwrite(arr, output_dir, ext=ext, metadata=SAMPLE, overwrite=True)

        if ext == ".zarr":
            path = next(output_dir.rglob("*.zarr"))
        elif ext == ".h5":
            path = next(output_dir.rglob("*.h5"))
        else:
            path = next(output_dir.rglob("*.tif"))

        md = mbo.get_metadata(path)
        assert md["dx"] == SAMPLE["dx"]
        assert md["dz"] == SAMPLE["dz"]
        assert float(md["fs"]) == SAMPLE["fs"]


class TestNpyEmbeddedMetadata:
    """.npy carries metadata inline (no sidecar .json)."""

    def test_no_json_sidecar_and_reactive_keys(self, synthetic_3d_data, output_dir):
        arr = _wrap(synthetic_3d_data, dims="TYX", metadata=SAMPLE)
        mbo.imwrite(arr, output_dir, ext=".npy", metadata=SAMPLE, overwrite=True)

        assert list(output_dir.glob("*.json")) == []

        back = mbo.imread(output_dir)
        md = back.metadata
        assert md["num_timepoints"] == synthetic_3d_data.shape[0]
        assert "dimension_names" in md
        # scientific keys are embedded inline (no sidecar)
        assert md["dx"] == SAMPLE["dx"]
        assert md["dz"] == SAMPLE["dz"]
        assert md["fs"] == SAMPLE["fs"]


class TestSubsetScaling:
    """Subset writes rescale stride-aware physical metadata (dz, fs)."""

    def test_plane_stride_scales_dz(self, synthetic_4d_data, output_dir):
        # source Z=3, dz=4.0; planes=[1,3] is stride 2 -> dz doubles
        arr = _wrap(synthetic_4d_data, dims="TZYX", metadata=SAMPLE)
        mbo.imwrite(arr, output_dir, ext=".zarr", planes=[1, 3], metadata=SAMPLE, overwrite=True)

        back = next(output_dir.rglob("*.zarr"))
        a = mbo.imread(back)
        assert a.nz == 2
        assert a.dz == SAMPLE["dz"] * 2

    def test_timepoint_stride_scales_fs(self, synthetic_4d_data, output_dir):
        # source T=10, fs=9.6; timepoints=[1,3,5,7,9] is stride 2 -> fs halves
        arr = _wrap(synthetic_4d_data, dims="TZYX", metadata=SAMPLE)
        mbo.imwrite(arr, output_dir, ext=".zarr", timepoints=[1, 3, 5, 7, 9],
                    metadata=SAMPLE, overwrite=True)

        a = mbo.imread(next(output_dir.rglob("*.zarr")))
        assert a.nt == 5
        assert a.fs == SAMPLE["fs"] / 2


class TestSourceMetadata:
    """Read real ScanImage metadata (skips without local data)."""

    def test_read_scanimage_keys(self, source_tiff_path):
        md = mbo.get_metadata(source_tiff_path)
        for key in ("frame_rate", "pixel_resolution"):
            assert key in md, f"missing ScanImage key: {key}"
