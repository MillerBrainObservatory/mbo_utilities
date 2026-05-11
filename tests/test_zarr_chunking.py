"""Zarr chunk + shard layout tests.

The defaults for OME-Zarr writes target the GUI's "fixed (c, z), scan T"
access pattern: one chunk per Y×X plane, one shard per (c, z) at all T.
This keeps interactive scrubbing at ~5 ms/frame and keeps the on-disk
file count to ``C * Z`` rather than ``T * C * Z``.

This module verifies the writer actually produces that layout, that
data round-trips, that the ``target_chunk_mb`` cap splits T across
multiple shards when needed, and that ``sharded=False`` falls back to
the legacy per-chunk layout.

See ``D:/demo/zarr_chunking_benchmark/benchmark.py`` for the
performance evidence behind the chosen defaults.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

# import order matters: tifffile must be imported before mbo for the
# package's pre-existing import wiring (matches tests/test_roundtrip.py)
import tifffile  # noqa: F401

import mbo_utilities as mbo
from mbo_utilities.arrays import NumpyArray


def _make_tmp() -> Path:
    """Allocate a unique tmpdir; the caller is responsible for rmtree."""
    return Path(tempfile.mkdtemp(prefix="mbo_zarr_chunking_"))


def _write_5d(data: np.ndarray, out_dir: Path, **kwargs) -> Path:
    """Write a 5D TCZYX numpy array via the public imwrite API.

    Returns the path to the resulting .zarr group.
    """
    wrapped = NumpyArray(data, dim_order="TCZYX")
    mbo.imwrite(wrapped, out_dir, ext=".zarr", ome=True, overwrite=True, **kwargs)
    written = [p for p in out_dir.iterdir() if p.suffix == ".zarr"]
    assert len(written) == 1, f"expected exactly one .zarr output, got {written}"
    return written[0]


def _write_4d(data: np.ndarray, out_dir: Path, **kwargs) -> Path:
    """Write a 4D TZYX numpy array via the public imwrite API."""
    wrapped = NumpyArray(data, dim_order="TZYX")
    mbo.imwrite(wrapped, out_dir, ext=".zarr", ome=True, overwrite=True, **kwargs)
    written = [p for p in out_dir.iterdir() if p.suffix == ".zarr"]
    assert len(written) == 1, f"expected exactly one .zarr output, got {written}"
    return written[0]


def _open_array(zarr_path: Path):
    """Open the OME-Zarr group and return its level-0 array."""
    z = zarr.open(str(zarr_path), mode="r")
    if hasattr(z, "shape"):
        return z
    assert "0" in z, f"OME-Zarr group missing '/0' array: {list(z)}"
    return z["0"]


def _on_disk_files(zarr_path: Path) -> int:
    return sum(1 for f in zarr_path.rglob("*") if f.is_file())


class TestVolumetricZarrDefaults:
    """The default chunk/shard layout for the GUI scrub pattern."""

    def test_5d_chunks_are_one_yx_plane(self):
        """5D TCZYX: inner chunk should be (1, 1, 1, Y, X) — one Y×X plane.

        Anything larger forces zarr to decompress extra data per frame
        fetch. The benchmark winner.
        """
        data = (np.random.rand(8, 2, 4, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_5d(data, out)
            arr = _open_array(zpath)
            # zarr v3 sharded array: .chunks reports the INNER chunk
            assert arr.chunks == (1, 1, 1, 64, 48)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_5d_shards_pack_all_t_per_cz(self):
        """5D TCZYX: outer shard should be (T, 1, 1, Y, X) — one shard per (c, z).

        That means a fixed-(c, z) T-scrub opens exactly one file and
        reads chunks from it sequentially.
        """
        data = (np.random.rand(8, 2, 4, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_5d(data, out)
            arr = _open_array(zpath)
            # zarr v3 sharded array: .shards reports the OUTER shard shape
            assert arr.shards == (8, 1, 1, 64, 48)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_5d_file_count_is_c_times_z_plus_metadata(self):
        """5D TCZYX with 8 timepoints / 2 channels / 4 z-planes should
        produce one shard file per (c, z) pair (= 8) plus zarr metadata
        (zarr.json at root + at /0 = 2). That's an order of magnitude
        fewer than the unsharded 64 chunk files.
        """
        data = (np.random.rand(8, 2, 4, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_5d(data, out)
            n_files = _on_disk_files(zpath)
            # 2 channels × 4 z-planes = 8 shard files. Plus zarr metadata.
            # The exact metadata count depends on zarr version (group +
            # array zarr.json), so we assert a tight upper bound rather
            # than equality.
            assert 8 <= n_files <= 12, (
                f"expected ~8 shard files + 2 metadata, got {n_files}"
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_4d_chunks_are_one_yx_plane(self):
        """4D TZYX: inner chunk should be (1, 1, Y, X)."""
        data = (np.random.rand(12, 5, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_4d(data, out)
            arr = _open_array(zpath)
            assert arr.chunks == (1, 1, 64, 48)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_4d_shards_pack_all_t_per_z(self):
        """4D TZYX: outer shard should be (T, 1, Y, X) — one shard per z."""
        data = (np.random.rand(12, 5, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_4d(data, out)
            arr = _open_array(zpath)
            assert arr.shards == (12, 1, 64, 48)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_4d_file_count_is_z_plus_metadata(self):
        """4D TZYX with 12 t / 5 z should produce 5 shard files + metadata."""
        data = (np.random.rand(12, 5, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_4d(data, out)
            n_files = _on_disk_files(zpath)
            assert 5 <= n_files <= 9, (
                f"expected 5 shard files + ~2 metadata, got {n_files}"
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestRoundtrip:
    """Data integrity: write → read → byte-exact match across (t, c, z)."""

    def test_5d_roundtrip_exact(self):
        rng = np.random.default_rng(0)
        data = (rng.random((6, 3, 5, 32, 24)) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_5d(data, out)
            arr = _open_array(zpath)
            # full read
            assert np.array_equal(np.asarray(arr[:]), data)
            # spot-check single (t, c, z) frames — the GUI's hot path
            for t, c, z in [(0, 0, 0), (3, 1, 2), (5, 2, 4)]:
                got = np.asarray(arr[t, c, z, :, :])
                assert np.array_equal(got, data[t, c, z, :, :]), (
                    f"mismatch at (t={t}, c={c}, z={z})"
                )
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_4d_roundtrip_exact(self):
        rng = np.random.default_rng(0)
        data = (rng.random((10, 4, 32, 24)) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_4d(data, out)
            arr = _open_array(zpath)
            assert np.array_equal(np.asarray(arr[:]), data)
            for t, z in [(0, 0), (5, 2), (9, 3)]:
                got = np.asarray(arr[t, z, :, :])
                assert np.array_equal(got, data[t, z, :, :])
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestTargetChunkMbCap:
    """The shard byte budget caps shard_t when (T × Y × X × itemsize) is large."""

    def test_small_target_splits_t_across_shards(self):
        """A small ``target_chunk_mb`` should split T across multiple shards.

        With shape (50, 2, 2, 256, 256) uint16 (128 KiB per Y×X plane in
        bytes_per_yx terms) and target_chunk_mb=1 (=1 MiB budget), we
        expect shard_t < 50 — i.e. T splits into multiple shards along T.
        Multi-channel + multi-plane keeps the writer in the 5D path
        (singleton C and Z get squeezed out otherwise).
        """
        data = (np.random.rand(50, 2, 2, 256, 256) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_5d(data, out, target_chunk_mb=1)
            arr = _open_array(zpath)
            # inner chunk unchanged
            assert arr.chunks == (1, 1, 1, 256, 256)
            shard_t = arr.shards[0]
            assert shard_t < 50, (
                f"target_chunk_mb=1 should cap shard_t below T=50, got {shard_t}"
            )
            # sanity: shard byte budget should be <= 2x of the 1 MiB cap
            shard_bytes = shard_t * 256 * 256 * 2
            assert shard_bytes <= 2 * 1024 * 1024
            # round-trip still exact
            assert np.array_equal(np.asarray(arr[:]), data)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_large_target_keeps_one_shard_per_cz(self):
        """The default 2 GB budget easily fits typical recordings into a
        single shard per (c, z) — confirms the default isn't accidentally
        truncating.
        """
        data = (np.random.rand(20, 2, 3, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_5d(data, out)  # use default target_chunk_mb
            arr = _open_array(zpath)
            assert arr.shards == (20, 1, 1, 64, 48), (
                f"default target should fit T=20 in one shard, "
                f"got shard_t={arr.shards[0]}"
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestUnshardedFallback:
    """``sharded=False`` should produce per-chunk files (legacy behavior)."""

    def test_sharded_false_has_no_outer_shard(self):
        # multi-channel + multi-plane to stay in the 5D path (C=1 / Z=1
        # singletons get squeezed out by the writer).
        data = (np.random.rand(6, 2, 2, 32, 24) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_5d(data, out, sharded=False)
            arr = _open_array(zpath)
            # Without sharding, .chunks is the actual chunk shape and
            # .shards (if the attribute exists) should be None.
            assert arr.chunks == (1, 1, 1, 32, 24)
            assert getattr(arr, "shards", None) is None
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_sharded_false_creates_one_file_per_chunk(self):
        """T=6, C=2, Z=2 → ~24 chunk files + metadata, vs ~5 sharded.

        Confirms unsharded falls back to the (T*C*Z)-file count layout.
        """
        data = (np.random.rand(6, 2, 2, 32, 24) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            zpath = _write_5d(data, out, sharded=False)
            n_files = _on_disk_files(zpath)
            # 6 t × 2 c × 2 z = 24 chunk files + metadata
            assert n_files >= 24, (
                f"expected at least 24 chunk files unsharded, got {n_files}"
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestGenericWriterFallback:
    """The `_try_generic_writers` zarr branch (raw numpy, no `_imwrite`)
    should produce the same optimal layout as the volumetric writer."""

    def test_5d_numpy_through_generic_writer(self):
        """Raw numpy 5D array through `imwrite` (no `_imwrite` attr) should
        end up sharded with `(T,1,1,Y,X)` shards and `(1,1,1,Y,X)` chunks.
        """
        from mbo_utilities._writers import _try_generic_writers

        data = (np.random.rand(8, 2, 4, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            target = out / "raw_numpy_5d.zarr"
            _try_generic_writers(data, target, overwrite=True)
            arr = _open_array(target)
            assert arr.chunks == (1, 1, 1, 64, 48)
            assert arr.shards == (8, 1, 1, 64, 48)
            assert np.array_equal(np.asarray(arr[:]), data)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_3d_numpy_through_generic_writer(self):
        """Raw numpy 3D TYX array through generic writer should produce
        chunks=(1,Y,X) and shards=(T,Y,X)."""
        from mbo_utilities._writers import _try_generic_writers

        data = (np.random.rand(20, 64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            target = out / "raw_numpy_3d.zarr"
            _try_generic_writers(data, target, overwrite=True)
            arr = _open_array(target)
            assert arr.chunks == (1, 64, 48)
            assert arr.shards == (20, 64, 48)
            assert np.array_equal(np.asarray(arr[:]), data)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_2d_numpy_through_generic_writer_no_shards(self):
        """Raw numpy 2D YX has no T axis to scrub — sharding adds no
        value and shouldn't be applied. Single chunk is correct."""
        from mbo_utilities._writers import _try_generic_writers

        data = (np.random.rand(64, 48) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            target = out / "raw_numpy_2d.zarr"
            _try_generic_writers(data, target, overwrite=True)
            arr = _open_array(target)
            assert arr.chunks == (64, 48)
            assert getattr(arr, "shards", None) is None
            assert np.array_equal(np.asarray(arr[:]), data)
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestPyramidLevels:
    """When `pyramid=True`, every level — not just level 0 — should
    use the same chunk/shard recipe so a viewer falling back to a
    lower-resolution level still gets per-frame zarr reads."""

    def test_pyramid_levels_share_layout_recipe(self):
        # 5D with enough Y, X to make multiple pyramid levels meaningful.
        # PyramidConfig.min_size defaults to 64; 256 → 128 → 64 = 3 levels.
        data = (np.random.rand(8, 2, 2, 256, 256) * 1000).astype(np.uint16)
        out = _make_tmp()
        try:
            wrapped = NumpyArray(data, dim_order="TCZYX")
            mbo.imwrite(
                wrapped, out, ext=".zarr", ome=True, overwrite=True,
                pyramid=True, pyramid_max_layers=2,
            )
            written = [p for p in out.iterdir() if p.suffix == ".zarr"]
            assert len(written) == 1
            z = zarr.open(str(written[0]), mode="r")

            level_paths = sorted(
                k for k in z if k.isdigit()
            )
            # we should have at least level 0 and one downsampled level
            assert len(level_paths) >= 2, (
                f"expected pyramid >= 2 levels, got {level_paths}"
            )

            for lvl in level_paths:
                a = z[lvl]
                Y, X = a.shape[-2], a.shape[-1]
                # every level uses one Y×X plane per chunk
                assert a.chunks == (1, 1, 1, Y, X), (
                    f"level {lvl}: chunks {a.chunks} != (1,1,1,{Y},{X})"
                )
                # every level uses (T, 1, 1, Y, X) shard
                assert a.shards == (data.shape[0], 1, 1, Y, X), (
                    f"level {lvl}: shards {a.shards} != "
                    f"({data.shape[0]},1,1,{Y},{X})"
                )
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestMergeZarrZplanes:
    """``merge_zarr_zplanes`` should also emit the sharded TZYX layout."""

    def test_merge_outputs_sharded_layout(self):
        """Build per-z-plane zarrs, merge them, and confirm the result has
        ``shards=(T, 1, Y, X)`` so a T-scrub at one z opens one file.
        """
        from mbo_utilities.arrays.zarr import merge_zarr_zplanes

        T, Y, X = 8, 32, 24
        n_z = 3
        rng = np.random.default_rng(1)

        # synthesize 3 per-plane zarrs with shape (T, Y, X)
        out = _make_tmp()
        try:
            plane_paths = []
            for z_idx in range(n_z):
                plane = (rng.random((T, Y, X)) * 1000).astype(np.uint16)
                p = out / f"plane{z_idx:02d}.zarr"
                root = zarr.open_group(str(p), mode="w", zarr_format=3)
                a = zarr.create(
                    store=root.store,
                    path="0",
                    shape=plane.shape,
                    chunks=plane.shape,
                    dtype=plane.dtype,
                    overwrite=True,
                )
                a[:] = plane
                plane_paths.append((p, plane))

            merged_path = out / "merged.zarr"
            merge_zarr_zplanes(
                [pp for pp, _ in plane_paths],
                merged_path,
                overwrite=True,
            )

            arr = _open_array(merged_path)
            assert arr.shape == (T, n_z, Y, X)
            assert arr.chunks == (1, 1, Y, X)
            assert arr.shards == (T, 1, Y, X)

            # round-trip per plane
            for z_idx, (_, plane) in enumerate(plane_paths):
                got = np.asarray(arr[:, z_idx, :, :])
                assert np.array_equal(got, plane), f"z={z_idx} mismatch"
        finally:
            shutil.rmtree(out, ignore_errors=True)
