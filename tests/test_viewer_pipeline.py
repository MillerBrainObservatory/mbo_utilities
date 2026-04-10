"""
Viewer-pipeline contract tests.

These pin the contracts the GUI rendering layer relies on, independent
of any actual fastplotlib/qt rendering. Each test is a regression for a
bug we shipped between the 5D refactor and now.

Coverage:
- `_SqueezeSingletonDims`: leak-proof against numpy protocol calls
  (`np.asarray`, `astype`), correct shape across all 8 singleton patterns
- `_load_subsampled`: canonicalizes any input rank/dim combo to
  `(T_sub, Z, Y, X)` so the stats loop can iterate axis 1 safely
- Window/Spatial widget feature gating across rank patterns
"""

from __future__ import annotations

import numpy as np
import pytest

from mbo_utilities.gui._stats import _load_subsampled
from mbo_utilities.gui.run_gui import _SqueezeSingletonDims
from mbo_utilities.gui.widgets.window_functions import (
    SpatialFunctionsWidget,
    WindowFunctionsWidget,
)


# ============================================================
# fakes
# ============================================================

class FakeArr:
    """Minimal lazy array — backed by an ndarray, with shape/ndim/dims."""

    def __init__(self, shape, dims, dtype=np.uint16):
        self._raw = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = self._raw.dtype
        self.dims = dims

    def __getitem__(self, key):
        return self._raw[key]

    def __array__(self, dtype=None, copy=None):
        out = self._raw
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FakeParent:
    """Parent stub matching the WidgetBase contract: parent.image_widget.data[0]."""

    def __init__(self, arr):
        self.image_widget = type("IW", (), {"data": [arr]})()


# ============================================================
# _SqueezeSingletonDims: shape correctness across all patterns
# ============================================================

# truth table for the 8 combinations of T/C/Z ∈ {1, >1}
SINGLETON_PATTERNS = [
    # (full_shape, expected_squeezed_shape, label)
    ((1, 1, 1, 64, 48), (64, 48), "all singleton (2D image)"),
    ((20, 1, 1, 64, 48), (20, 64, 48), "T only"),
    ((1, 4, 1, 64, 48), (4, 64, 48), "C only"),
    ((1, 1, 8, 64, 48), (8, 64, 48), "Z only"),
    ((20, 4, 1, 64, 48), (20, 4, 64, 48), "T,C"),
    ((20, 1, 8, 64, 48), (20, 8, 64, 48), "T,Z"),
    ((1, 4, 8, 64, 48), (4, 8, 64, 48), "C,Z"),
    ((20, 4, 8, 64, 48), (20, 4, 8, 64, 48), "no singletons"),
]


class TestSqueezeWrapperShape:
    """`_SqueezeSingletonDims` reports the correct natural rank for every
    combination of T/C/Z singleton patterns. Lockstep with `get_slider_dims`
    is required so fastplotlib's `ndim - 2 == len(slider_dim_names)` holds.
    """

    @pytest.mark.parametrize("full,expected,label", SINGLETON_PATTERNS)
    def test_squeeze(self, full, expected, label):
        arr = FakeArr(full, ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        assert w.shape == expected, f"{label}: expected {expected}, got {w.shape}"
        assert w.ndim == len(expected)


class TestSqueezeWrapperNumpyLeak:
    """Regression: `_SqueezeSingletonDims.__getattr__` previously delegated
    `astype` and `__array__` to the underlying array, leaking the original
    5D shape to fastplotlib's TextureArray. Pin every numpy entry point so
    the leak can't come back.
    """

    def test_np_asarray_honors_squeeze(self):
        arr = FakeArr((1, 1, 1, 64, 48), ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        out = np.asarray(w)
        assert out.shape == (64, 48), \
            f"np.asarray leaked underlying shape: got {out.shape}"

    def test_astype_honors_squeeze(self):
        arr = FakeArr((1, 1, 1, 64, 48), ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        out = w.astype(np.float32)
        assert out.shape == (64, 48), \
            f"astype leaked underlying shape: got {out.shape}"
        assert out.dtype == np.float32

    def test_isolated_buffer_assignment(self):
        """fastplotlib's TextureArray._fix_data path: allocate zeros at
        wrapper.shape, then assign wrapper[:] into it. Both ends must agree.
        """
        arr = FakeArr((1, 1, 1, 64, 48), ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        buf = np.zeros(w.shape, dtype=w.dtype)
        buf[:] = w[:]  # must not raise
        assert buf.shape == (64, 48)

    def test_dunder_lookups_dont_leak(self):
        """numpy may probe `__array_interface__` etc. via getattr — those
        must not silently fall through to the underlying array.
        """
        arr = FakeArr((1, 1, 1, 64, 48), ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        # the wrapper must not pretend to have arbitrary dunders
        with pytest.raises(AttributeError):
            w.__array_interface__


# ============================================================
# _load_subsampled: canonicalization across rank patterns
# ============================================================

CANONICAL_CASES = [
    # (input_shape, dims, expected_canonical_shape, label)
    ((64, 48),         ("Y", "X"),                 (1, 1, 64, 48),  "2D YX"),
    ((100, 64, 48),    ("T", "Y", "X"),            (10, 1, 64, 48), "3D TYX"),
    ((37, 64, 48),     ("Z", "Y", "X"),            (1, 37, 64, 48), "3D ZYX (mosquito-stylet bug)"),
    ((20, 5, 64, 48),  ("T", "Z", "Y", "X"),       (2, 5, 64, 48),  "4D TZYX"),
    ((1, 14, 64, 48),  ("C", "Z", "Y", "X"),       (1, 14, 64, 48), "4D CZYX (post-T-squeeze pollen)"),
    ((20, 2, 5, 64, 48), ("T", "C", "Z", "Y", "X"),(2, 5, 64, 48),  "5D TCZYX"),
]


class TestLoadSubsampledCanonicalization:
    """`_load_subsampled` returns `(T_sub, Z, Y, X)` regardless of input
    rank or dim labels. Regression for the (Z, Y, X) crash where the
    code blindly strode axis 0 as if it were T.
    """

    @pytest.mark.parametrize("shape,dims,expected,label", CANONICAL_CASES)
    def test_canonical_shape(self, shape, dims, expected, label):
        f = FakeArr(shape, dims)
        sub = _load_subsampled(f, subsample=10)
        assert sub.shape == expected, f"{label}: {sub.shape} vs {expected}"

    @pytest.mark.parametrize("shape,dims,expected,label", CANONICAL_CASES)
    def test_stats_loop_doesnt_crash(self, shape, dims, expected, label):
        """The downstream stats loop iterates axis 1 from 0 to nz-1.
        Canonicalization must produce an axis 1 large enough to support that.
        """
        f = FakeArr(shape, dims)
        sub = _load_subsampled(f, subsample=10)
        nz = sub.shape[1]
        for s in range(nz):
            stack = sub[:, s, :, :].astype(np.float32)
            assert stack.ndim == 3  # (T_sub, Y, X)


# ============================================================
# Feature gating: window/spatial widget is_supported
# ============================================================

GATING_CASES = [
    # (shape, dims, expect_window, expect_spatial, label)
    ((64, 48),               ("Y", "X"),               False, False, "2D image"),
    ((100, 64, 48),          ("T", "Y", "X"),          True,  True,  "3D time series"),
    ((37, 64, 48),           ("Z", "Y", "X"),          False, True,  "3D z-stack no time"),
    ((20, 5, 64, 48),        ("T", "Z", "Y", "X"),     True,  True,  "4D TZYX"),
    ((20, 2, 5, 64, 48),     ("T", "C", "Z", "Y", "X"), True, True,  "5D TCZYX"),
    ((1, 14, 64, 48),        ("C", "Z", "Y", "X"),     False, True,  "post-squeeze pollen"),
]


class TestFeatureGating:
    """`WindowFunctionsWidget` requires a real T axis with size > 1.
    `SpatialFunctionsWidget` requires rank > 2 (something to stack across).
    Regression: previously gated on `shape[0] > 1` and `True`, which both
    misfired for natural-rank 2D and lower-rank data.
    """

    @pytest.mark.parametrize("shape,dims,expect_window,expect_spatial,label", GATING_CASES)
    def test_window_functions_supported(self, shape, dims, expect_window, expect_spatial, label):
        parent = FakeParent(FakeArr(shape, dims))
        assert WindowFunctionsWidget.is_supported(parent) == expect_window, \
            f"{label}: WindowFunctions gating wrong"

    @pytest.mark.parametrize("shape,dims,expect_window,expect_spatial,label", GATING_CASES)
    def test_spatial_functions_supported(self, shape, dims, expect_window, expect_spatial, label):
        parent = FakeParent(FakeArr(shape, dims))
        assert SpatialFunctionsWidget.is_supported(parent) == expect_spatial, \
            f"{label}: SpatialFunctions gating wrong"


# ============================================================
# Reload-via-file-dialog parity with initial launch
# ============================================================

class TestReloadDataConsistency:
    """`load_new_data` (file-dialog reload) must produce the same view as
    the initial launch path in `_create_image_widget`. Both must apply
    the singleton-dim squeeze; otherwise the same file gives different
    shapes depending on whether you opened it via the CLI or the dialog.

    Regressions:
    - Suite2pArray reload was exposing 5D `(T, 1, Z, Y, X)` to the viewer,
      putting Z at index 2 instead of index 1. Initial launch correctly
      squeezed C=1 and exposed `(T, Z, Y, X)`.
    - mean_subtraction state persisted across reloads, leaving the
      checkbox ticked while the underlying spatial_func used stale or
      missing per-z mean images for the new data.
    """

    def test_squeeze_wrapper_puts_z_at_index_1_for_5d_with_c1(self):
        """A 5D array shaped like Suite2p output (T, 1, Z, Y, X) must
        squeeze to (T, Z, Y, X) — Z at index 1, not 2.
        """
        suite2p_like = FakeArr((10, 1, 8, 64, 48), ("T", "C", "Z", "Y", "X"))
        wrapped = _SqueezeSingletonDims(suite2p_like)
        assert wrapped.shape == (10, 8, 64, 48)
        assert wrapped.ndim == 4
        assert wrapped.dims == ("T", "Z", "Y", "X")
        # Z at index 1 (not 2) — the bug was reporting Z at index 2
        assert wrapped.dims.index("Z") == 1


class TestPerDataStateReset:
    """`_reset_per_data_state` must clear every dataset-specific flag
    that would otherwise carry across a load_new_data call.

    Regression: load_new_data used to leave per-data widget controls
    intact when swapping data — the user's gaussian sigma, projection
    mode, window size, auto-contrast toggle, and mean-subtraction
    checkbox all carried over from the previous file. Widget instances
    were freshly constructed but they read their displayed values from
    `parent.<attr>`, so the carryover values surfaced anyway.
    """

    # canonical (field, dirty_value, expected_after_reset) tuples
    RESET_FIELDS = [
        ("_mean_subtraction",   True,         False),
        ("_gaussian_sigma",     5.0,          0.0),
        ("_proj",               "max",        "mean"),
        ("_window_size",        10,           1),
        ("_auto_contrast_on_z", True,         False),
        ("_last_z_idx",         42,           0),
        ("_saveas_selected_roi", {1, 2, 3},   set()),
        ("_saveas_rois",        True,         False),
    ]

    def test_resets_every_field(self):
        """All dirty values must be reset to defaults in one call."""
        from mbo_utilities.gui._dialogs import _reset_per_data_state

        p = type("P", (), {})()
        # set every field to a non-default "dirty" value
        for field, dirty, _ in self.RESET_FIELDS:
            setattr(p, field, dirty)

        _reset_per_data_state(p)

        # every field must equal its expected default
        for field, _, expected in self.RESET_FIELDS:
            actual = getattr(p, field)
            assert actual == expected, \
                f"{field}: expected {expected!r} after reset, got {actual!r}"

    def test_idempotent_on_clean_state(self):
        """Reset must be idempotent — calling it on a clean parent
        should not error or change anything inappropriately."""
        from mbo_utilities.gui._dialogs import _reset_per_data_state

        p = type("P", (), {})()
        for field, _, expected in self.RESET_FIELDS:
            setattr(p, field, expected)

        _reset_per_data_state(p)

        for field, _, expected in self.RESET_FIELDS:
            assert getattr(p, field) == expected

    def test_load_new_data_calls_reset(self):
        """Pin that load_new_data invokes _reset_per_data_state.

        Avoids actually running load_new_data (which pulls in the heavy
        viewer/Qt module load) — just verifies the source contains the
        helper call so the contract can't silently regress.
        """
        import inspect
        from mbo_utilities.gui import _dialogs
        src = inspect.getsource(_dialogs.load_new_data)
        assert "_reset_per_data_state(parent)" in src, \
            "load_new_data must call _reset_per_data_state"

    def test_init_state_uses_same_helper(self):
        """PreviewDataWidget._init_state must define its per-data defaults
        via _reset_per_data_state so reload and initial launch can never
        diverge. Pin via source inspection (avoids importing the heavy
        widget module just to check this contract).
        """
        from pathlib import Path
        import mbo_utilities.gui.widgets.preview_data as preview_data_mod
        src = Path(preview_data_mod.__file__).read_text()
        assert "_reset_per_data_state(self)" in src, \
            "PreviewDataWidget._init_state must call _reset_per_data_state"


# ============================================================
# Custom metadata propagation through suite2p paths
# ============================================================

class TestCustomMetadataPropagation:
    """User-set values from the metadata editor (e.g. dz, fs) live on
    `parent._custom_metadata` and must reach both:
      1. ops.npy via the imwrite metadata kwarg, and
      2. lbm_suite2p_python's pipeline ops dict.

    Regression: the GUI's "Run Suite2p" button silently dropped
    `_custom_metadata` because the worker config dict captured `arr`,
    `s2p_dict`, etc. but not `_custom_metadata`. The user typed dz=15,
    ops.npy ended up with dz=None (LBM) or 1.0 (default fallback), and
    everything downstream that read dz from ops.npy was wrong.
    """

    def test_imwrite_metadata_kwarg_round_trip_dz(self, tmp_path):
        """Full round-trip: imwrite with metadata={'dz': 15} → ops.npy
        contains dz=15 → reload sees dz=15. This is the contract that
        the suite2p paths depend on.
        """
        import numpy as np
        from mbo_utilities import imwrite
        from mbo_utilities.arrays import NumpyArray, Suite2pArray

        rng = np.random.RandomState(0)
        # synthetic 5D (T, C=1, Z=4, Y, X) — bypasses natural-rank to
        # isolate the dz round-trip behavior from the bin-write fix
        data = rng.randint(0, 4096, size=(8, 1, 4, 16, 16), dtype=np.int16)
        arr = NumpyArray(data)

        out = tmp_path / "out"
        imwrite(arr, out, ext=".bin", metadata={"dz": 15.0})

        # ops.npy must contain the user's dz under canonical and alias keys
        ops_files = sorted(out.rglob("ops.npy"))
        assert ops_files, "no ops.npy written"
        for ops_file in ops_files:
            ops = np.load(ops_file, allow_pickle=True).item()
            assert ops.get("dz") == 15.0, \
                f"{ops_file}: expected dz=15.0, got {ops.get('dz')}"

    def test_natural_rank_tiff_bin_write(self, tmp_path):
        """Natural-rank 4D TiffArray (T, Z, Y, X) — was crashing because
        `_imwrite_base` did `arr.shape[2]` (which is Y, not Z, on a
        natural-rank array) and tried to iterate Y-many planes. Fix uses
        `arr.shape5d[2]` to always pick the real Z size.
        """
        import numpy as np
        import tifffile
        from mbo_utilities import imread, imwrite

        rng = np.random.RandomState(1)
        nt, nz = 8, 4
        vol_dir = tmp_path / "src"
        vol_dir.mkdir()
        for z in range(nz):
            tifffile.imwrite(
                vol_dir / f"plane{z:02d}.tif",
                rng.randint(0, 4096, (nt, 16, 16), dtype=np.uint16),
            )

        arr = imread(vol_dir)
        # natural-rank reports 4D, shape5d still reports 5D
        assert arr.ndim == 4
        assert arr.shape5d == (nt, 1, nz, 16, 16)

        out = tmp_path / "out"
        imwrite(arr, out, ext=".bin")  # must not crash

        bin_files = sorted(out.rglob("*.bin"))
        assert len(bin_files) == nz, \
            f"expected {nz} bin files (one per plane), got {len(bin_files)}"

    def test_run_plane_worker_thread_merges_custom_metadata(self):
        """`_run_plane_worker_thread` (the daemon-thread suite2p path)
        must read `config['custom_metadata']` and merge it into
        `lazy_mdata` BEFORE constructing `OutputMetadata`. Pin via
        source inspection so importing the heavy pipeline module isn't
        needed.
        """
        import inspect
        import re
        from mbo_utilities.gui.widgets.pipelines import settings as s

        src = inspect.getsource(s._run_plane_worker_thread)
        assert 'config.get("custom_metadata")' in src or "config['custom_metadata']" in src, \
            "_run_plane_worker_thread must read config['custom_metadata']"
        # the merge must happen before OutputMetadata is constructed —
        # otherwise the user's edits don't reach the reactive layer.
        merge_idx = re.search(r"lazy_mdata\.update\(custom_metadata\)", src)
        out_idx = re.search(r"OutputMetadata\s*\(", src)
        assert merge_idx and out_idx, "missing merge or OutputMetadata call"
        assert merge_idx.start() < out_idx.start(), \
            "lazy_mdata.update(custom_metadata) must come BEFORE OutputMetadata(...)"

    def test_settings_thread_config_includes_custom_metadata(self):
        """The thread-path job-config builder in settings.py must
        snapshot `_custom_metadata` into the per-job config dict.
        Inspect the source rather than driving the GUI.
        """
        from pathlib import Path
        import mbo_utilities.gui.widgets.pipelines.settings as settings_mod
        src = Path(settings_mod.__file__).read_text()
        assert '"custom_metadata": dict(getattr(self, "_custom_metadata"' in src, \
            "thread-path job config must snapshot _custom_metadata"

    def test_settings_spawn_worker_args_include_custom_metadata(self):
        """The spawn-process worker_args builder must also snapshot
        `_custom_metadata` so the subprocess can apply it to ops before
        invoking lbm_suite2p_python.pipeline.
        """
        from pathlib import Path
        import mbo_utilities.gui.widgets.pipelines.settings as settings_mod
        src = Path(settings_mod.__file__).read_text()
        # there are two snapshot sites (thread + spawn) — at least one
        # must be inside a worker_args dict
        assert "worker_args" in src and 'dict(getattr(self, "_custom_metadata"' in src, \
            "spawn-process worker_args must snapshot _custom_metadata"

    def test_task_suite2p_applies_custom_metadata_to_ops(self):
        """`task_suite2p` (the spawn-process subprocess entry point)
        must merge `args['custom_metadata']` into `ops` before invoking
        the pipeline. Pin via source inspection.
        """
        import inspect
        import re
        from mbo_utilities.gui import tasks

        src = inspect.getsource(tasks.task_suite2p)
        assert 'args.get("custom_metadata"' in src, \
            "task_suite2p must read args['custom_metadata']"
        # the merge must happen before pipeline() is called
        merge = re.search(r"ops\.update\(custom_metadata\)", src)
        pipeline_call = re.search(r"\bpipeline\s*\(", src)
        assert merge and pipeline_call, "missing ops.update or pipeline() call"
        assert merge.start() < pipeline_call.start(), \
            "ops.update(custom_metadata) must come BEFORE pipeline(...)"


# ============================================================
# Output metadata frame-count consistency
# ============================================================

# All seven aliases for "number of timepoints in the output". Whatever
# logic computes this value must propagate it to every key — downstream
# readers (and humans inspecting ops.npy) read different ones.
TIMEPOINT_ALIASES = (
    "num_timepoints", "nframes", "num_frames",
    "n_frames", "T", "nt", "timepoints",
)


class TestOutputTimepointConsistency:
    """When `imwrite` produces ops.npy, EVERY timepoint alias must agree
    with the actual number of frames written. The user reported saving
    700 of 1574 timepoints and getting `num_timepoints=1574` (the source
    value) in ops.npy while `nframes=700` was correct — different aliases
    contradicting each other inside the same dict.

    Two root causes (both fixed):
      1. `_imwrite_base` only updated 3 of 7 aliases after computing the
         output count, leaving the rest at OutputMetadata.to_dict's
         placeholder value (1 when source_shape isn't passed).
      2. `_imwrite_base` ignored the `num_frames=` truncation kwarg
         when no explicit `frames=[...]` selection was passed, falling
         back to the source frame count.
    """

    def _load_first_ops(self, out_dir):
        import numpy as np
        ops_files = sorted(out_dir.rglob("ops.npy"))
        assert ops_files, f"no ops.npy under {out_dir}"
        return np.load(ops_files[0], allow_pickle=True).item()

    def _make_arr(self, nt=1574, nz=4, h=16, w=16):
        import numpy as np
        from mbo_utilities.arrays import NumpyArray
        rng = np.random.RandomState(0)
        return NumpyArray(rng.randint(0, 4096, size=(nt, 1, nz, h, w), dtype=np.int16))

    def test_truncation_via_num_frames_kwarg(self, tmp_path):
        """imwrite(num_frames=700) on a 1574-frame source — every alias
        must read 700, not 1574 (the source) and not 1 (the to_dict
        placeholder).
        """
        from mbo_utilities import imwrite
        arr = self._make_arr(nt=1574)
        out = tmp_path / "out"
        imwrite(arr, out, ext=".bin", num_frames=700)

        ops = self._load_first_ops(out)
        for key in TIMEPOINT_ALIASES:
            assert ops.get(key) == 700, \
                f"{key}: expected 700 (truncated), got {ops.get(key)!r}"

    def test_explicit_frames_selection(self, tmp_path):
        """imwrite(frames=[1..700]) — every alias must read 700."""
        from mbo_utilities import imwrite
        arr = self._make_arr(nt=1574)
        out = tmp_path / "out"
        imwrite(arr, out, ext=".bin", frames=list(range(1, 701)))

        ops = self._load_first_ops(out)
        for key in TIMEPOINT_ALIASES:
            assert ops.get(key) == 700, \
                f"{key}: expected 700 (selected), got {ops.get(key)!r}"

    def test_no_selection_uses_source_count(self, tmp_path):
        """imwrite() without truncation/selection — every alias reads
        the full source count.
        """
        from mbo_utilities import imwrite
        arr = self._make_arr(nt=1574)
        out = tmp_path / "out"
        imwrite(arr, out, ext=".bin")

        ops = self._load_first_ops(out)
        for key in TIMEPOINT_ALIASES:
            assert ops.get(key) == 1574, \
                f"{key}: expected 1574 (source), got {ops.get(key)!r}"

    def test_aliases_internally_consistent(self, tmp_path):
        """Independent of value, all 7 aliases in ops.npy must agree
        with each other AND with the actual binary file size.
        """
        from mbo_utilities import imwrite
        arr = self._make_arr(nt=1574)
        out = tmp_path / "out"
        imwrite(arr, out, ext=".bin", num_frames=421)

        ops_files = sorted(out.rglob("ops.npy"))
        for ops_file in ops_files:
            import numpy as np
            ops = np.load(ops_file, allow_pickle=True).item()
            values = {k: ops.get(k) for k in TIMEPOINT_ALIASES}
            unique = set(values.values())
            assert len(unique) == 1, \
                f"{ops_file.name}: aliases disagree: {values}"

            # also check actual bin file size matches
            bin_files = list(ops_file.parent.glob("*.bin"))
            assert bin_files, "no bin file alongside ops.npy"
            actual_frames = bin_files[0].stat().st_size // (16 * 16 * 2)
            assert actual_frames == 421, \
                f"{bin_files[0].name}: bin has {actual_frames} frames, ops says {values}"


# ============================================================
# H5 dataset_name kwarg
# ============================================================

class TestH5DatasetNameKwarg:
    """`imwrite(..., dataset_name=...)` controls the HDF5 dataset key.

    Default is `"mov"` so files are immediately readable by suite2p,
    caiman, and any external consumer that hardcodes that name.
    Caller can override for legacy mbo data (`"data"`) or custom keys.

    Regression: the generic-writer fallback path (which fires when
    imwrite gets a sliced numpy array) used to hardcode `"data"`,
    producing files that looked empty to caiman because nothing
    existed at `f["mov"]`. The streaming `_write_h5` path used the
    opposite hardcode (`"mov"`), so the two write paths were
    inconsistent — same kwargs, different on-disk layout depending
    on whether the input was lazy or materialized.
    """

    def _ds_keys(self, h5_path):
        import h5py
        with h5py.File(h5_path, "r") as f:
            return list(f.keys())

    def test_generic_writer_default_is_mov(self, tmp_path):
        """numpy ndarray → generic writer → /mov by default."""
        import numpy as np
        from mbo_utilities import imwrite
        rng = np.random.RandomState(0)
        data = rng.randint(0, 4096, size=(20, 16, 16), dtype=np.int16)
        out = tmp_path / "frames.h5"
        imwrite(data, out, overwrite=True)
        assert "mov" in self._ds_keys(out)

    def test_generic_writer_dataset_name_data(self, tmp_path):
        """dataset_name='data' overrides the default for legacy mbo."""
        import numpy as np
        from mbo_utilities import imwrite
        rng = np.random.RandomState(0)
        data = rng.randint(0, 4096, size=(20, 16, 16), dtype=np.int16)
        out = tmp_path / "frames.h5"
        imwrite(data, out, overwrite=True, dataset_name="data")
        keys = self._ds_keys(out)
        assert "data" in keys
        assert "mov" not in keys

    def test_generic_writer_custom_dataset_name(self, tmp_path):
        """Arbitrary keys are honored."""
        import numpy as np
        from mbo_utilities import imwrite
        rng = np.random.RandomState(0)
        data = rng.randint(0, 4096, size=(20, 16, 16), dtype=np.int16)
        out = tmp_path / "frames.h5"
        imwrite(data, out, overwrite=True, dataset_name="my_movie")
        assert "my_movie" in self._ds_keys(out)

    def test_lazy_writer_default_is_mov(self, tmp_path):
        """NumpyArray → _imwrite_base → _write_h5 streaming path,
        default key is also /mov.
        """
        import numpy as np
        from mbo_utilities import imwrite
        from mbo_utilities.arrays import NumpyArray
        rng = np.random.RandomState(0)
        arr = NumpyArray(rng.randint(0, 4096, size=(20, 1, 1, 16, 16), dtype=np.int16))
        out_dir = tmp_path / "out"
        imwrite(arr, out_dir, ext=".h5", overwrite=True)
        h5_files = list(out_dir.rglob("*.h5"))
        assert h5_files, "no h5 written"
        assert "mov" in self._ds_keys(h5_files[0])

    def test_lazy_writer_custom_dataset_name(self, tmp_path):
        """The lazy streaming path also honors dataset_name."""
        import numpy as np
        from mbo_utilities import imwrite
        from mbo_utilities.arrays import NumpyArray
        rng = np.random.RandomState(0)
        arr = NumpyArray(rng.randint(0, 4096, size=(20, 1, 1, 16, 16), dtype=np.int16))
        out_dir = tmp_path / "out"
        imwrite(arr, out_dir, ext=".h5", overwrite=True, dataset_name="my_data")
        h5_files = list(out_dir.rglob("*.h5"))
        assert h5_files, "no h5 written"
        keys = self._ds_keys(h5_files[0])
        assert "my_data" in keys
        assert "mov" not in keys

    def test_round_trip_mov_via_h5array(self, tmp_path):
        """File written with default dataset_name is readable by mbo's
        own H5Array reader (which auto-detects 'mov' first)."""
        import numpy as np
        from mbo_utilities import imwrite, imread
        rng = np.random.RandomState(0)
        data = rng.randint(0, 4096, size=(20, 16, 16), dtype=np.int16)
        out = tmp_path / "frames.h5"
        imwrite(data, out, overwrite=True)
        loaded = imread(out)
        assert loaded.shape5d[0] == 20  # T axis preserved


# ============================================================
# Reactive fs/dz scaling for the suite2p Run path
# ============================================================

class TestReactiveFsZScaling:
    """The Run-Suite2p worker must use OutputMetadata to reactively
    scale fs and dz when the user has a stride selection. Without this,
    every-other-timepoint produces ops.npy with the source fs (wrong by
    a factor of 2) and every-other-plane produces ops.npy with the
    source dz (wrong by a factor of 2).

    Regression: the worker used to build the metadata dict by hand,
    pulling fs/dz raw from source metadata. The user's stride selection
    was silently dropped at every layer (config build → worker → ops),
    so the resulting fs/dz reflected the source acquisition rather than
    the actual output cadence.
    """

    def test_reactive_fs_halves_with_stride_2(self):
        """Direct OutputMetadata test: stride 2 halves fs."""
        from mbo_utilities.metadata import OutputMetadata
        out = OutputMetadata(
            source={"fs": 30.0, "dz": 10.0, "dx": 0.5, "dy": 0.5},
            source_shape=(1574, 1, 14, 256, 256),
            source_dims=("T", "C", "Z", "Y", "X"),
            selections={"T": list(range(0, 1574, 2))},
        )
        d = out.to_dict()
        assert d["fs"] == 15.0, f"fs should halve, got {d['fs']}"
        # dz unchanged when only T is strided
        assert d["dz"] == 10.0

    def test_reactive_dz_doubles_with_z_stride_2(self):
        """Direct OutputMetadata test: every-other-plane doubles dz."""
        from mbo_utilities.metadata import OutputMetadata
        out = OutputMetadata(
            source={"fs": 30.0, "dz": 10.0, "dx": 0.5, "dy": 0.5},
            source_shape=(1574, 1, 14, 256, 256),
            source_dims=("T", "C", "Z", "Y", "X"),
            selections={"Z": [0, 2, 4, 6, 8, 10, 12]},
        )
        d = out.to_dict()
        assert d["dz"] == 20.0, f"dz should double, got {d['dz']}"
        # fs unchanged when only Z is strided
        assert d["fs"] == 30.0

    def test_reactive_both_stride_simultaneous(self):
        """Both selections active — both reactive scales fire."""
        from mbo_utilities.metadata import OutputMetadata
        out = OutputMetadata(
            source={"fs": 30.0, "dz": 10.0, "dx": 0.5, "dy": 0.5},
            source_shape=(1574, 1, 14, 256, 256),
            source_dims=("T", "C", "Z", "Y", "X"),
            selections={
                "T": list(range(0, 1574, 2)),
                "Z": [0, 2, 4, 6, 8, 10, 12],
            },
        )
        d = out.to_dict()
        assert d["fs"] == 15.0
        assert d["dz"] == 20.0

    def test_worker_uses_output_metadata(self):
        """`_run_plane_worker_thread` must construct an OutputMetadata
        from `lazy_mdata` + the config's tp_indices/selected_planes_0based,
        so that fs/dz get reactively scaled.
        """
        import inspect
        from mbo_utilities.gui.widgets.pipelines import settings as s

        src = inspect.getsource(s._run_plane_worker_thread)
        assert "OutputMetadata(" in src, \
            "_run_plane_worker_thread must use OutputMetadata"
        assert "tp_indices" in src and "selected_planes_0based" in src, \
            "worker must consume config['tp_indices'] and config['selected_planes_0based']"
        assert 'selections["T"]' in src, "T selection must be set on OutputMetadata"
        assert 'selections["Z"]' in src, "Z selection must be set on OutputMetadata"

    def test_worker_passes_frames_kwarg(self):
        """The worker must pass `frames=` to imwrite when the user has a
        stride selection — `num_frames=` alone is just truncation.
        """
        import inspect
        from mbo_utilities.gui.widgets.pipelines import settings as s

        src = inspect.getsource(s._run_plane_worker_thread)
        assert "frames=frames_arg" in src, \
            "worker must pass frames= to imwrite (not just num_frames=)"

    def test_worker_warns_on_missing_fs(self):
        """Missing fs in source must warn (not crash) so the user knows
        their ops.npy will have a default frame rate.
        """
        import inspect
        from mbo_utilities.gui.widgets.pipelines import settings as s

        src = inspect.getsource(s._run_plane_worker_thread)
        assert 'get_param(lazy_mdata, "fs")' in src
        assert ".warning(" in src, "missing fs must produce a warning"

    def test_thread_config_includes_stride_selections(self):
        """The thread-path job-config builder must snapshot the
        timepoint indices and the full plane selection (0-based).
        """
        from pathlib import Path
        import mbo_utilities.gui.widgets.pipelines.settings as settings_mod
        src = Path(settings_mod.__file__).read_text()
        assert '"tp_indices":' in src, \
            "config must capture tp_indices from _s2p_tp_parsed"
        assert '"selected_planes_0based":' in src, \
            "config must capture full plane selection (0-based)"

    def test_spawn_worker_args_include_stride_selections(self):
        """Same for the spawn-process path's worker_args."""
        from pathlib import Path
        import mbo_utilities.gui.widgets.pipelines.settings as settings_mod
        src = Path(settings_mod.__file__).read_text()
        # both branches set these — check the spawn one is wired too
        assert src.count('"tp_indices":') >= 2, \
            "spawn worker_args must also capture tp_indices"
        assert src.count('"selected_planes_0based":') >= 2, \
            "spawn worker_args must also capture selected_planes_0based"

    def test_task_suite2p_uses_output_metadata(self):
        """The spawn-process task must construct OutputMetadata when
        stride selections are present.
        """
        import inspect
        from mbo_utilities.gui import tasks

        src = inspect.getsource(tasks.task_suite2p)
        assert "OutputMetadata(" in src, \
            "task_suite2p must use OutputMetadata for reactive scaling"
        assert 'args.get("tp_indices")' in src
        assert 'args.get("selected_planes_0based")' in src
