"""
Offscreen contract battery for the NDWidget-backed viewer adapter.

``MboNDViewer`` (mbo_utilities/gui/_ndviewer.py) wraps fastplotlib's
``NDWidget`` behind the exact ImageWidget surface the Studio GUI was written
against. These tests pin that contract end-to-end on a real offscreen
figure: construction, the name/letter/positional ``indices`` semantics,
data swaps (same-rank and rank-changing), window/spatial func routing
through the per-graphic slicers, both contrast resets, the playback-bar
adapter (fps seeding, loop fan-out, the space-bar toggle), and offscreen
``close()``.

The offscreen backend renders headlessly but graphic data still updates
asynchronously — index changes schedule fetches on the rendercanvas loop,
which only steps inside ``loop.run()``. ``drain()`` pumps it manually (see
its docstring for why ``_stop_when_no_canvases`` is held off during the
pump). Property setters (window funcs, frame_apply, data swaps) render
synchronously and need no drain.

``RENDERCANVAS_FORCE_OFFSCREEN=1`` must be set before *any* fastplotlib
import in the process — tests/conftest.py does this for the whole suite;
the module skips itself if another backend won the race anyway.
"""

from __future__ import annotations

import contextlib
import os

os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import time

import numpy as np
import pytest


def _offscreen_selected() -> bool:
    try:
        from rendercanvas.auto import RenderCanvas
    except Exception:
        return False
    return "offscreen" in RenderCanvas.__module__


pytestmark = pytest.mark.skipif(
    not _offscreen_selected(),
    reason="offscreen rendercanvas backend not selected (another backend "
    "was imported before RENDERCANVAS_FORCE_OFFSCREEN took effect)",
)


class LazyStandIn:
    """Array-protocol lazy stand-in: dtype/shape/ndim/__getitem__ only.

    Tracks how it was indexed so tests can prove the scalar-key sampling
    path (mbo's lazy readers service single 2D frames; a strided
    multi-dim slice would be pathological).
    """

    def __init__(self, base: np.ndarray):
        self._base = base
        self.scalar_key_reads = 0
        self.other_reads = 0

    @property
    def shape(self):
        return self._base.shape

    @property
    def ndim(self):
        return self._base.ndim

    @property
    def dtype(self):
        return self._base.dtype

    def __getitem__(self, key):
        if isinstance(key, tuple) and all(
            isinstance(k, (int, np.integer)) for k in key
        ):
            self.scalar_key_reads += 1
        else:
            self.other_reads += 1
        return self._base[key]


def drain(passes: int = 30):
    """Pump the offscreen rendercanvas loop until async fetches settle.

    Offscreen canvases never register with the loop, so after ~0.1s of
    cumulative running the loop-task would see "no canvases" and stop —
    cancelling just-created fetch tasks before their first step, which
    skips ``_fetch_request``'s finally block and permanently wedges
    ``ReferenceIndices._fetch_request_active`` for that graphic. Holding
    ``_stop_when_no_canvases`` off during the pump avoids that (test-only
    concern; GUI loops always have a registered canvas).
    """
    from rendercanvas.auto import loop

    loop._stop_when_no_canvases = False
    try:
        for _ in range(passes):
            loop.run()
            time.sleep(0.01)
    finally:
        loop._stop_when_no_canvases = True


def make_base_5d(shape=(40, 2, 3, 48, 48), seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(500, 100, size=shape).astype(np.float32)
    # give frame t a recognizable mean so window funcs are verifiable
    for t in range(shape[0]):
        base[t] += t * 10.0
    return base


class _StubLogger:
    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


class _StubParent:
    """Just enough PreviewDataWidget surface for _keyboard.toggle_playback."""

    def __init__(self, iw):
        self.image_widget = iw
        self.logger = _StubLogger()


NAMES = ("Timepoint", "Channel", "Z-plane")


@pytest.fixture(scope="module")
def base5d():
    return make_base_5d()


@pytest.fixture(scope="module")
def viewer5d(base5d):
    """One shared 5D viewer for the read-mostly contract tests.

    Tests using it set whatever state they need and must not assume
    another test left indices/funcs untouched. Swap tests build their own.
    """
    from mbo_utilities.gui._ndviewer import MboNDViewer

    arr = LazyStandIn(base5d)
    iw = MboNDViewer(
        data=arr,
        slider_dim_names=NAMES,
        window_funcs=(np.mean, None, None),
        window_sizes=(1, None, None),
        cmap="gnuplot2",
        histogram_widget=True,
        figure_kwargs={"size": (300, 300)},
        graphic_kwargs={"vmin": -100, "vmax": 4000},
    )
    assert isinstance(iw, MboNDViewer)
    iw.show()
    yield iw
    iw.close()


class TestConstruction:
    def test_factory_returns_adapter(self, viewer5d):
        from mbo_utilities.gui._ndviewer import MboNDViewer

        assert isinstance(viewer5d, MboNDViewer)

    def test_basic_structure(self, viewer5d, base5d):
        iw = viewer5d
        assert iw.n_sliders == 3
        assert len(iw.graphics) == 1
        assert iw.graphics[0] is not None
        assert iw._slider_dim_names == NAMES
        # positional letters still reported for the legacy window-funcs gate
        assert iw.slider_dims == ["t", "z", "c"]
        assert bool(iw.data) and len(list(iw.data)) == 1

    def test_data_returns_original_object(self, viewer5d):
        arr = viewer5d.data[0]
        assert isinstance(arr, LazyStandIn)

    def test_graphic_kwargs_and_cmap_applied(self, viewer5d):
        g = viewer5d.graphics[0]
        assert g.cmap == "gnuplot2"
        assert (g.vmin, g.vmax) == (-100.0, 4000.0)

    def test_figure_is_real_imgui_figure(self, viewer5d):
        fig = viewer5d.figure
        assert type(fig).__name__ == "ImguiFigure"
        assert hasattr(fig, "renderer") and hasattr(fig, "imgui_renderer")
        assert fig.canvas is not None

    def test_colorbar_attached(self, viewer5d):
        cb = viewer5d.ndgraphics[0].histogram_widget
        assert cb is not None

    def test_construction_histogram_used_scalar_keys(self, viewer5d):
        # the initial histogram must be built from bounded scalar-key frame
        # reads (mbo's lazy readers reject/pathologically service strided
        # multi-dim slices). The only non-scalar keys allowed are the couple
        # of slice-keyed display fetches the configured window funcs make.
        arr = viewer5d.data[0]
        assert 0 < arr.scalar_key_reads <= 72
        assert arr.other_reads <= 4

    def test_offscreen_draw(self, viewer5d):
        frame = viewer5d.figure.canvas.draw()
        assert getattr(frame, "shape", None) is not None


class TestIndices:
    def test_len_and_value_iteration(self, viewer5d):
        iw = viewer5d
        iw.indices = [0, 0, 0]
        assert len(iw.indices) == 3
        snapshot = list(iw.indices)
        assert snapshot == [0, 0, 0]
        assert all(isinstance(v, int) for v in snapshot)

    def test_name_keyed_set_get(self, viewer5d):
        iw = viewer5d
        iw.indices["Timepoint"] = 5
        assert iw.indices["Timepoint"] == 5
        # positional letter aliases resolve to the same axes
        assert iw.indices["t"] == 5

    def test_letter_set_positional(self, viewer5d):
        iw = viewer5d
        iw.indices["z"] = 1  # letter for axis 1, displayed as 'Channel'
        assert iw.indices["Channel"] == 1

    def test_positional_list_set(self, viewer5d):
        iw = viewer5d
        iw.indices = [2, 0, 2]
        assert list(iw.indices) == [2, 0, 2]

    def test_dict_set(self, viewer5d):
        iw = viewer5d
        iw.indices = {"Z-plane": 1}
        assert iw.indices["Z-plane"] == 1

    def test_short_list_sets_leading_dims(self, viewer5d):
        # pollen_calibration does `iw.indices = [channel]` on a >1-slider
        # viewer: only the leading positions are set
        iw = viewer5d
        iw.indices = [0, 0, 0]
        iw.indices = [3]
        assert iw.indices["Timepoint"] == 3

    def test_unknown_name_raises_keyerror(self, viewer5d):
        with pytest.raises(KeyError):
            viewer5d.indices["nope"]

    def test_refresh_idiom(self, viewer5d):
        # `iw.indices = list(iw.indices)` is the GUI's force-refresh idiom
        iw = viewer5d
        iw.indices = [4, 1, 2]
        iw.indices = list(iw.indices)
        assert list(iw.indices) == [4, 1, 2]

    def test_async_fetch_lands_after_drain(self, viewer5d):
        iw = viewer5d
        iw.indices = [10, 0, 0]
        drain()
        shown = iw.ndgraphics[0].indices_displayed
        assert shown is not None
        # indices_displayed is in the 1-based reference space
        assert int(round(shown["Timepoint"])) == 10 + 1

    def test_current_index_dict(self, viewer5d):
        iw = viewer5d
        assert set(iw.current_index.keys()) == set(iw.ndwidget.indices.dims)


class TestWindowFuncs:
    def test_legacy_dict_routed_with_window_order(self, viewer5d):
        # preview_data's legacy path sets {"t": (func, size)}; funcs for
        # dims absent from window_order are silently ignored by fastplotlib,
        # so BOTH must land on the slicer
        iw = viewer5d
        iw.indices = [20, 0, 0]
        drain()
        iw.window_funcs = {"t": (np.mean, 10)}
        proc = iw.ndgraphics[0].slicer
        assert proc.window_funcs["Timepoint"][0] is np.mean
        assert proc.window_order == ("Timepoint",)

    def test_window_func_actually_applied(self, viewer5d, base5d):
        iw = viewer5d
        iw.indices = [20, 0, 0]
        iw.window_funcs = {"t": (np.mean, 10)}
        drain()
        proc = iw.ndgraphics[0].slicer
        got = np.asarray(iw.ndgraphics[0].graphic.data.value)
        # the indexer takes 1-based reference values
        idx = proc._get_slider_dims_indexer(
            {"Timepoint": 21, "Channel": 1, "Z-plane": 1}
        )
        tsel = idx["Timepoint"]
        # the window is real: more than one frame averaged
        assert (tsel.stop - tsel.start) > 1
        expect = base5d[tsel, 0:1, 0:1].mean(axis=0)[0, 0]
        assert np.allclose(got, expect, atol=1e-3)

    def test_window_funcs_clear(self, viewer5d):
        iw = viewer5d
        iw.window_funcs = {"t": (np.mean, 10)}
        iw.window_funcs = None
        proc = iw.ndgraphics[0].slicer
        assert proc.window_order == tuple()
        assert all(v == (None, None) for v in proc.window_funcs.values())
        drain()

    def test_frame_apply_routes_to_spatial_func(self, viewer5d):
        iw = viewer5d
        iw.frame_apply = {0: lambda a: a * 0 + 7.0}
        proc = iw.ndgraphics[0].slicer
        assert proc.spatial_func is not None
        # frame_apply renders synchronously — no drain needed
        got = np.asarray(iw.ndgraphics[0].graphic.data.value)
        assert np.allclose(got[..., 0] if got.ndim == 3 else got, 7.0)

    def test_frame_apply_clear(self, viewer5d):
        iw = viewer5d
        iw.frame_apply = {0: lambda a: a * 0 + 7.0}
        iw.frame_apply = None
        assert iw.ndgraphics[0].slicer.spatial_func is None
        assert iw.frame_apply == {}
        drain()

    def test_extended_surface_present(self, viewer5d):
        assert all(
            hasattr(viewer5d, a)
            for a in ("window_funcs", "window_sizes", "spatial_func", "frame_apply")
        )


class TestContrastResets:
    def test_reset_vmin_vmax_full_sample(self, viewer5d, base5d):
        iw = viewer5d
        arr = iw.data[0]
        arr.scalar_key_reads = 0
        arr.other_reads = 0
        iw.reset_vmin_vmax()
        cb = iw.ndgraphics[0].histogram_widget
        # bounded sample must still land near the true full-data range
        assert abs(cb.vmin - float(base5d.min())) < 150
        assert abs(cb.vmax - float(base5d.max())) < 150
        # lazy-friendly: scalar-key frame reads only, and bounded
        assert 0 < arr.scalar_key_reads <= 72
        assert arr.other_reads == 0
        # graphic follows the colorbar
        g = iw.graphics[0]
        assert abs(g.vmin - cb.vmin) < 1e-6 and abs(g.vmax - cb.vmax) < 1e-6

    def test_reset_vmin_vmax_frame(self, viewer5d):
        iw = viewer5d
        iw.reset_vmin_vmax_frame()
        cb = iw.ndgraphics[0].histogram_widget
        fr = np.asarray(iw.ndgraphics[0].graphic.data.value)
        assert abs(cb.vmin - float(fr.min())) < 1e-3
        assert abs(cb.vmax - float(fr.max())) < 1e-3

    def test_cmap_set_get(self, viewer5d):
        iw = viewer5d
        iw.cmap = "viridis"
        assert iw.graphics[0].cmap == "viridis"
        assert iw.cmap == ["viridis"]
        iw.cmap = "gnuplot2"


class TestSlidersUI:
    def test_loop_fans_out_per_dim(self, viewer5d):
        sl = viewer5d._sliders_ui
        sl._loop = True
        assert all(viewer5d.ndwidget.ui_sliders._loop.values())
        assert sl._loop is True

    def test_seed_fps_semantics(self, viewer5d):
        sl = viewer5d._sliders_ui
        ndui = viewer5d.ndwidget.ui_sliders
        sl.seed_fps("t", 15.7)
        assert sl._fps["t"] == 16
        assert abs(ndui._frame_time["Timepoint"] - 1 / 16) < 1e-9
        # junk values are ignored
        sl.seed_fps("t", float("nan"))
        sl.seed_fps("t", float("inf"))
        sl.seed_fps("t", -3)
        sl.seed_fps("t", None)
        assert sl._fps["t"] == 16
        # clamps to the bar's [1, 50]
        sl.seed_fps("t", 500)
        assert sl._fps["t"] == 50

    def test_seed_fps_never_overrides_user_typed(self, viewer5d):
        sl = viewer5d._sliders_ui
        ndui = viewer5d.ndwidget.ui_sliders
        ndui._fps["Timepoint"] = 33  # simulate the user typing into the bar
        sl.seed_fps("t", 10)
        assert sl._fps["t"] == 33
        assert "Timepoint" in sl._user_fps

    def test_playing_accepts_int_and_name_keys(self, viewer5d):
        sl = viewer5d._sliders_ui
        ndui = viewer5d.ndwidget.ui_sliders
        sl._playing[0] = True
        assert ndui._playing["Timepoint"] is True
        sl._playing["Timepoint"] = False
        assert ndui._playing["Timepoint"] is False
        sl._last_frame_time[0] = 0
        assert ndui._last_frame_time["Timepoint"] == 0

    def test_space_toggle_playback(self, viewer5d):
        """The space-bar path: _keyboard.toggle_playback must flip the T
        dim's play state on the NDWidgetUI (it was a silent no-op when it
        indexed the str-keyed state by int position).
        """
        from mbo_utilities.gui._keyboard import toggle_playback

        parent = _StubParent(viewer5d)
        ndui = viewer5d.ndwidget.ui_sliders
        ndui._playing["Timepoint"] = False
        ndui._last_frame_time["Timepoint"] = 123.0

        toggle_playback(parent)
        assert ndui._playing["Timepoint"] is True
        assert ndui._last_frame_time["Timepoint"] == 0

        toggle_playback(parent)
        assert ndui._playing["Timepoint"] is False

    def test_rebind_space_to_playback(self, viewer5d):
        """preview_data's space rebind must succeed against the adapter's
        real-figure passthrough: it reaches figure.renderer and installs
        its own key_down handler.
        """
        from mbo_utilities.gui._keyboard import rebind_space_to_playback

        parent = _StubParent(viewer5d)
        rebind_space_to_playback(parent)
        assert getattr(parent, "_space_rebound", False) is True

    def test_toggle_playback_out_of_range_is_noop(self, viewer5d):
        from mbo_utilities.gui._keyboard import toggle_playback

        parent = _StubParent(viewer5d)
        before = dict(viewer5d.ndwidget.ui_sliders._playing)
        toggle_playback(parent, dim_index=99)
        assert dict(viewer5d.ndwidget.ui_sliders._playing) == before


class TestTogglePlaybackState:
    """toggle_playback against the old ImageWidget sliders' state shape (str-keyed
    defaultdicts) — no figure needed.
    """

    @staticmethod
    def _parent(n_dims=2, prepopulate=False):
        from collections import defaultdict

        class Sliders:
            def __init__(self):
                self._playing = defaultdict(bool)
                self._last_frame_time = defaultdict(float)

        class IW:
            def __init__(self):
                self._sliders_ui = Sliders()
                self.slider_dims = ["t", "z", "c"][:n_dims]

        parent = _StubParent(IW())
        if prepopulate:
            # what a drawn frame does: update() reads every dim's state
            for d in parent.image_widget.slider_dims:
                parent.image_widget._sliders_ui._playing[d]
        return parent

    def test_toggles_the_letter_key_not_a_phantom_int(self):
        from mbo_utilities.gui._keyboard import toggle_playback

        parent = self._parent()
        sliders = parent.image_widget._sliders_ui
        toggle_playback(parent)
        assert sliders._playing["t"] is True
        assert 0 not in sliders._playing  # the old bug's phantom key
        assert sliders._last_frame_time["t"] == 0

    def test_populated_mapping_resolves_positionally(self):
        from mbo_utilities.gui._keyboard import toggle_playback

        parent = self._parent(n_dims=3, prepopulate=True)
        sliders = parent.image_widget._sliders_ui
        toggle_playback(parent, dim_index=1)
        assert sliders._playing["z"] is True
        assert sliders._playing["t"] is False

    def test_no_sliders_is_noop(self):
        from mbo_utilities.gui._keyboard import toggle_playback

        parent = _StubParent(None)
        toggle_playback(parent)  # must not raise


def _make_viewer(data, **kwargs):
    from mbo_utilities.gui._ndviewer import MboNDViewer

    defaults = dict(
        histogram_widget=True,
        figure_kwargs={"size": (300, 300)},
    )
    defaults.update(kwargs)
    iw = MboNDViewer(data=data, **defaults)
    iw.show()
    return iw


class TestSameDimsSwap:
    def test_full_swap_contract(self):
        rng = np.random.default_rng(3)
        base = make_base_5d((20, 2, 3, 32, 32), seed=3)
        arr = LazyStandIn(base)
        iw = _make_viewer(
            arr,
            slider_dim_names=NAMES,
            cmap="gnuplot2",
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )
        try:
            iw.window_funcs = {"t": (np.mean, 5)}
            iw.cmap = "viridis"
            iw.indices = [5, 1, 2]
            drain()

            old_graphic = iw.graphics[0]
            old_executor = iw.ndgraphics[0].slicer._executor
            arr2 = LazyStandIn(rng.normal(100, 10, size=base.shape).astype(np.float32))
            iw.data[0] = arr2

            assert iw.data[0] is arr2
            assert iw.n_sliders == 3
            # indices reset, stale closures cleared, graphic rebuilt
            assert list(iw.indices) == [0, 0, 0]
            assert iw.window_funcs is None and iw.frame_apply == {}
            assert iw.graphics[0] is not None and iw.graphics[0] is not old_graphic
            # replaced slicer's executor must not leak
            assert old_executor._shutdown
            # exactly one live ndgraphic remains registered
            assert len(iw.ndwidget.ndgraphics) == 1
            # cmap carries across the swap
            assert iw.graphics[0].cmap == "viridis"

            drain()
            frame = iw.figure.canvas.draw()
            assert getattr(frame, "shape", None) is not None
        finally:
            iw.close()


class TestDimsChangingSwap:
    def test_5d_to_4d_and_back(self):
        base5 = make_base_5d((20, 2, 3, 32, 32), seed=5)
        arr5 = LazyStandIn(base5)
        iw = _make_viewer(arr5, slider_dim_names=NAMES)
        try:
            base4 = np.arange(12 * 3 * 24 * 24, dtype=np.float32).reshape(12, 3, 24, 24)
            arr4 = LazyStandIn(base4)
            iw.data[0] = arr4

            assert iw.data[0] is arr4
            assert iw.n_sliders == 2
            assert list(iw.indices) == [0, 0]
            assert iw.slider_dims == ["t", "z"]
            ranges = iw.ndwidget.ranges
            assert sorted(int(r.stop - r.start) for r in ranges.values()) == [3, 12]

            # letters resolve positionally on the new dim space and the
            # right frame is displayed after the async fetch drains
            iw.indices["t"] = 3
            iw.indices["z"] = 1
            assert list(iw.indices) == [3, 1]
            drain()
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert got.shape == base4[3, 1].shape
            assert np.allclose(got, base4[3, 1])
            frame = iw.figure.canvas.draw()
            assert getattr(frame, "shape", None) is not None

            # the legacy window-funcs guard still passes on the new rank
            assert "t" in iw.slider_dims
            iw.window_funcs = {"t": (np.mean, 3)}
            drain()

            # and back up to 5D
            iw.data[0] = arr5
            assert iw.n_sliders == 3 and iw.data[0] is arr5
            assert list(iw.indices) == [0, 0, 0]
            drain()
        finally:
            iw.close()


class TestFailedSwapRollsBack:
    """A replacement array whose reads raise must not leave the viewer
    half-torn-down: the old array is reinstalled and scrubbing keeps
    working (the failure is still raised to the caller).
    """

    class _Boom(LazyStandIn):
        def __getitem__(self, key):
            raise RuntimeError("boom")

    def test_swap_failure_restores_previous_data(self):
        base = np.full((10, 3, 16, 16), 7.0, dtype=np.float32)
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint", "Z-plane"))
        try:
            with pytest.raises(RuntimeError, match="boom"):
                iw.data[0] = self._Boom(np.zeros_like(base))
            # old data back in place, graphic alive, viewer scrubbable
            assert iw.data[0] is not None
            assert np.asarray(iw.data[0].shape) is not None
            assert iw.graphics[0] is not None
            iw.indices = [2, 1]
            drain()
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert np.allclose(got, 7.0)
            frame = iw.figure.canvas.draw()
            assert getattr(frame, "shape", None) is not None
        finally:
            iw.close()


class TestSwapRefreshesSliderDimNames:
    def test_dims_changing_swap_rekeys_labels(self):
        base5 = make_base_5d((8, 2, 3, 16, 16), seed=11)
        iw = _make_viewer(LazyStandIn(base5), slider_dim_names=NAMES)
        try:
            assert iw._slider_dim_names == NAMES
            iw.data[0] = LazyStandIn(base5[:, 0])  # (T, Z, Y, X)
            # stale display labels would keep resolving/naming 3 dims
            assert iw._slider_dim_names == ("t", "z")
            # still a plain writable attr
            iw._slider_dim_names = ("Timepoint", "Z-plane")
            assert iw._slider_dim_names == ("Timepoint", "Z-plane")
        finally:
            iw.close()


class TestTeardownFetchRace:
    """A queued serial fetch scheduled right before a data swap must not
    KeyError inside the rendercanvas task (whose finally used to re-insert
    a dead bookkeeping key forever, pinning the torn-down graphic).
    """

    @staticmethod
    @contextlib.contextmanager
    def _capture_task_errors():
        import logging

        errors: list[str] = []

        class _H(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    errors.append(record.getMessage())

        handler = _H()
        for name in ("", "rendercanvas"):
            logging.getLogger(name).addHandler(handler)
        try:
            yield errors
        finally:
            for name in ("", "rendercanvas"):
                logging.getLogger(name).removeHandler(handler)

    def test_swap_after_scheduled_fetch_has_no_task_error(self):
        base = np.zeros((30, 16, 16), dtype=np.float32)
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            with self._capture_task_errors() as errors:
                iw.indices = [5]
                iw.data[0] = LazyStandIn(base + 1)  # immediate swap
                drain()
            assert not errors, errors
        finally:
            iw.close()

    def test_dead_fetch_keys_do_not_grow_over_swap_cycles(self):
        base = np.zeros((30, 16, 16), dtype=np.float32)
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            ri = iw.ndwidget.indices
            with self._capture_task_errors() as errors:
                for k in range(10):
                    iw.indices = [3]
                    iw.data[0] = LazyStandIn(base + k)
                    drain()
            assert not errors, errors
            live = set(iw.ndwidget.ndgraphics)
            dead_queues = [g for g in ri._fetch_request_queue if g not in live]
            dead_active = [g for g in ri._fetch_request_active if g not in live]
            # at most the LAST cycle's flag may still await its sweep
            assert len(dead_queues) == 0, dead_queues
            assert len(dead_active) <= 1, dead_active
        finally:
            iw.close()


class TestProtocolOnly2D:
    """2D protocol-only arrays (no __array__): ``np.asarray(data)`` yields a
    useless 0-d object array, so the histogram sample must read through the
    protocol (``data[:]``) — construction used to TypeError.
    """

    def test_sample_array_reads_through_protocol(self):
        from mbo_utilities.gui._ndviewer import _sample_array

        base = np.arange(64, dtype=np.int16).reshape(8, 8)
        sample = _sample_array(LazyStandIn(base))
        assert sample.dtype == np.int16
        assert np.array_equal(sample, base)

    def test_construction_and_swap(self):
        base2d = np.arange(256, dtype=np.int16).reshape(16, 16)
        iw = _make_viewer(LazyStandIn(base2d))
        try:
            assert iw.n_sliders == 0
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert np.array_equal(got, base2d.astype(got.dtype))
            # swap 2D -> 2D protocol-only stays fully usable
            iw.data[0] = LazyStandIn(base2d[::-1].copy())
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert np.array_equal(got, base2d[::-1].astype(got.dtype))
        finally:
            iw.close()

    def test_non_numeric_sample_degrades_to_no_histogram(self):
        from mbo_utilities.gui._ndviewer import MboNDImageSlicer

        proc = object.__new__(MboNDImageSlicer)
        proc._compute_histogram = True
        proc._data = np.array([[object(), object()]], dtype=object)
        proc._recompute_histogram()  # must not raise
        assert proc._histogram is None


class TestWindowSpanContract:
    """Odd window size s covers exactly s frames centered on t (the
    old ImageWidget contract). fastplotlib's banker's-rounding identity transform
    covered s-1 or s+1 depending on the parity of t - s//2.
    """

    @staticmethod
    def _ramp(n_t=40, side=8):
        # frame t = t + a fractional per-pixel gradient in exact eighths, so
        # a 5-frame mean is exactly representable in float32
        grad = (np.arange(side * side, dtype=np.float32) / 8.0).reshape(side, side)
        base = np.empty((n_t, side, side), dtype=np.float32)
        for t in range(n_t):
            base[t] = t + grad
        return base

    @pytest.mark.parametrize("size", [3, 5, 7, 11])
    def test_odd_sizes_cover_exactly_size_frames(self, size):
        base = self._ramp()
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            iw.window_funcs = {"t": (np.mean, size)}
            proc = iw.ndgraphics[0].slicer
            # ref value 11 = array frame 10
            sel = proc._get_slider_dims_indexer({"Timepoint": 11})["Timepoint"]
            half = (size - 1) // 2
            assert (sel.start, sel.stop) == (10 - half, 10 + half + 1)
        finally:
            iw.close()

    @pytest.mark.parametrize("size", [1, 3, 5])
    def test_no_position_windows_to_an_empty_slice(self, size):
        """fastplotlib clamps a window's *exclusive* stop to ``shape - 1``, so the
        last position of a windowed dim collapsed to ``slice(n-1, n-1)``.
        Every reference position must select at least one in-bounds frame.
        """
        base = self._ramp(n_t=12)
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            iw.window_funcs = {"t": (np.mean, size)}
            proc = iw.ndgraphics[0].slicer
            for ref in range(1, 13):
                sel = proc._get_slider_dims_indexer({"Timepoint": ref})["Timepoint"]
                assert 0 <= sel.start < sel.stop <= 12, (ref, sel)
        finally:
            iw.close()

    def test_last_frame_is_not_all_nan(self):
        """The empty end-of-range slice rendered a NaN frame from a numpy array
        and killed the fetch outright on a lazy reader
        ("windowed_slice.ndim != len(display_dims): 0 != 2").
        """
        base = self._ramp(n_t=12)
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            iw.window_funcs = {"t": (np.mean, 1)}
            iw.indices = [11]
            drain()
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert np.isfinite(got).all()
            assert np.array_equal(got, base[11].astype(got.dtype))
        finally:
            iw.close()

    def test_t_mean_5_is_pixel_exact(self):
        base = self._ramp()
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            iw.window_funcs = {"t": (np.mean, 5)}
            iw.indices = [10]
            drain()
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            expect = base[8:13].mean(axis=0)  # frames 8..12 inclusive
            assert got.dtype.kind == "f"
            assert np.array_equal(got, expect.astype(got.dtype))
        finally:
            iw.close()


class TestOffscreenIndexDelivery:
    """Programmatic index changes must reach the displayed pixels on an
    offscreen canvas WITHOUT property-setter renders. (The rendercanvas
    StubLoop's no-canvases self-stop can cancel a scheduled fetch task
    before its first step, wedging the serial-fetch bookkeeping so no
    fetch would ever run again — the adapter now renders synchronously
    for offscreen canvases.)
    """

    def test_scrub_delivers_pixels_without_property_setters(self):
        base = np.zeros((40, 16, 16), dtype=np.float32)
        for t in range(40):
            base[t] = float(t)
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            iw.figure.canvas.draw()
            iw.indices = [10]
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert np.allclose(got, 10.0)
            # name-keyed setter path too, and again after plain loop passes
            # (the naive-consumer drain that used to wedge the queue)
            from rendercanvas.auto import loop

            iw.indices["Timepoint"] = 30
            for _ in range(3):
                loop.run()
                time.sleep(0.01)
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert np.allclose(got, 30.0)
            iw.indices = [17]
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert np.allclose(got, 17.0)
        finally:
            iw.close()


class TestIntegerTextureUpgrade:
    """Window/spatial funcs over integer data must not have their float
    output truncated into the original integer texture.
    """

    @staticmethod
    def _alternating_int16(n_t=40, side=8):
        base = np.empty((n_t, side, side), dtype=np.int16)
        for t in range(n_t):
            base[t] = t % 2
        return base

    def test_fractional_window_mean_survives_to_buffer(self):
        base = self._alternating_int16()
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            assert np.asarray(iw.ndgraphics[0].graphic.data.value).dtype.kind in "iu"
            iw.window_funcs = {"t": (np.mean, 5)}
            iw.indices = [10]
            drain()
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            # frames 8..12 of t % 2 -> (0, 1, 0, 1, 0) -> exactly 0.4
            assert got.dtype.kind == "f"
            assert np.allclose(got, 0.4)
        finally:
            iw.close()

    def test_frame_apply_float_output_on_int_data(self):
        base = self._alternating_int16()
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            iw.frame_apply = {0: lambda a: a + 0.25}
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert got.dtype.kind == "f"
            assert np.allclose(got, base[0] + 0.25)
        finally:
            iw.close()

    def test_swap_restores_native_dtype(self):
        base = self._alternating_int16()
        iw = _make_viewer(LazyStandIn(base), slider_dim_names=("Timepoint",))
        try:
            iw.window_funcs = {"t": (np.mean, 5)}
            drain()
            assert np.asarray(iw.ndgraphics[0].graphic.data.value).dtype.kind == "f"
            iw.data[0] = LazyStandIn(base.copy())  # funcs cleared by swap
            assert np.asarray(iw.ndgraphics[0].graphic.data.value).dtype.kind in "iu"
        finally:
            iw.close()


class TestIndexBounds:
    """The old ImageWidget raised on out-of-range and negative indices; the
    fastplotlib's ReferenceIndices silently clamps. The adapter keeps the raise.
    """

    def test_out_of_range_raises_naming_dim_and_size(self, viewer5d):
        with pytest.raises(IndexError, match=r"2500.*Timepoint.*40"):
            viewer5d.current_index = {"t": 2500}
        with pytest.raises(IndexError, match=r"Timepoint"):
            viewer5d.indices["Timepoint"] = 40
        with pytest.raises(IndexError):
            viewer5d.indices = [0, 5, 0]

    def test_negative_raises_like_the_old_widget(self, viewer5d):
        with pytest.raises(IndexError, match="negative"):
            viewer5d.indices["t"] = -1
        with pytest.raises(IndexError, match="negative"):
            viewer5d.indices = [-3, 0, 0]

    def test_in_range_edges_still_work(self, viewer5d):
        viewer5d.indices = [39, 0, 0]
        assert viewer5d.indices["Timepoint"] == 39
        viewer5d.indices = [0, 0, 0]


class TestShowPassthrough:
    def test_show_returns_ndwidget_show_result(self, viewer5d):
        # offscreen Figure.show() returns None in fastplotlib (as the old ImageWidget did),
        # so pin the pass-through with a sentinel
        sentinel = object()
        original = viewer5d._ndw.show
        viewer5d._ndw.show = lambda **kw: sentinel
        try:
            assert viewer5d.show() is sentinel
        finally:
            viewer5d._ndw.show = original


class _WithLabels(LazyStandIn):
    slider_dim_labels = ("Timepoint", "Channel", "Z-plane")


class _WithFs(LazyStandIn):
    fs = 4.915


class TestBareDimNames:
    """With no slider_dim_names the letters must follow mbo's canonical
    axis order — 5D data is (T, C, Z, Y, X), so 'z' must drive axis 2 (the
    real Z), not the singleton C axis the old ImageWidget positional order gave.
    """

    def test_bare_5d_letters_are_canonical(self):
        base = np.zeros((6, 1, 5, 16, 16), dtype=np.float32)
        for z in range(5):
            base[:, :, z] = z * 10.0
        iw = _make_viewer(LazyStandIn(base))
        try:
            assert tuple(iw._dim_names) == ("t", "c", "z")
            iw.indices["z"] = 3
            drain()
            got = np.asarray(iw.ndgraphics[0].graphic.data.value)
            assert np.allclose(got, base[0, 0, 3])
        finally:
            iw.close()

    def test_bare_4d_and_3d_letters(self):
        iw = _make_viewer(LazyStandIn(np.zeros((4, 3, 8, 8), dtype=np.float32)))
        try:
            assert tuple(iw._dim_names) == ("t", "z")
        finally:
            iw.close()
        iw = _make_viewer(LazyStandIn(np.zeros((4, 8, 8), dtype=np.float32)))
        try:
            assert tuple(iw._dim_names) == ("t",)
        finally:
            iw.close()

    def test_bare_path_prefers_full_arr_labels(self):
        base = np.zeros((4, 2, 3, 8, 8), dtype=np.float32)
        iw = _make_viewer(_WithLabels(base))
        try:
            assert tuple(iw._dim_names) == ("Timepoint", "Channel", "Z-plane")
        finally:
            iw.close()

    def test_partial_labels_fall_back_to_letters(self):
        # labels only for non-singleton axes cannot be
        # mapped positionally -> canonical letters instead
        class _Partial(LazyStandIn):
            slider_dim_labels = ("Timepoint", "Z-plane")

        base = np.zeros((4, 1, 3, 8, 8), dtype=np.float32)
        iw = _make_viewer(_Partial(base))
        try:
            assert tuple(iw._dim_names) == ("t", "c", "z")
        finally:
            iw.close()


class TestFpsSeedingFromData:
    """The bare factory path must seed playback fps from the array's frame
    rate at construction AND when data is swapped in (preview_data's
    seeding only runs inside the full GUI).
    """

    def test_construction_seeds_from_fs(self):
        iw = _make_viewer(_WithFs(np.zeros((10, 8, 8), dtype=np.float32)))
        try:
            assert iw._sliders_ui._fps["t"] == 5
        finally:
            iw.close()

    def test_swap_seeds_from_new_array(self):
        base = np.zeros((10, 8, 8), dtype=np.float32)
        iw = _make_viewer(LazyStandIn(base))
        try:
            assert iw._sliders_ui._fps["t"] == 20  # no fs -> untouched default
            iw.data[0] = _WithFs(base.copy())
            assert iw._sliders_ui._fps["t"] == 5
        finally:
            iw.close()

    def test_user_typed_fps_survives_swap_seed(self):
        base = np.zeros((10, 8, 8), dtype=np.float32)
        iw = _make_viewer(_WithFs(base))
        try:
            t_dim = iw._dim_names[0]
            iw.ndwidget.ui_sliders._fps[t_dim] = 33  # user typed into the bar
            iw._sliders_ui.seed_fps("t", 10)  # flags the user override
            iw.data[0] = _WithFs(base.copy())
            assert iw._sliders_ui._fps["t"] == 33
        finally:
            iw.close()


class TestFindSliderNameAliases:
    def test_descriptive_and_letter_labels_resolve_to_z(self):
        from mbo_utilities.arrays.features._dim_labels import find_slider_name

        # original case is returned
        assert find_slider_name(("Tile", "Cam", "Zplane"), "z") == "Zplane"
        assert find_slider_name(("t", "c", "z"), "z") == "z"
        assert find_slider_name(("Timepoint", "Channel"), "z") is None


class TestMultiArray:
    def test_two_arrays(self, base5d):
        a1 = LazyStandIn(base5d[:, 0, 0])  # (T, Y, X)
        a2 = LazyStandIn(base5d[:, 1, 0])
        iw = _make_viewer(
            [a1, a2],
            names=["ROI 1", "ROI 2"],
            slider_dim_names=("Timepoint",),
            cmap="gray",
            figure_shape=(1, 2),
            # roomy enough that each subplot keeps a usable viewport next to
            # its right-dock colorbar; tiny viewports make fpl's axes update
            # crash in fastplotlib (map_screen_to_world returns None)
            figure_kwargs={"size": (640, 400)},
            graphic_kwargs={"vmin": 0, "vmax": 1000},
        )
        try:
            assert len(iw.graphics) == 2
            assert iw.n_sliders == 1
            assert iw.data[0] is a1 and iw.data[1] is a2
            iw.indices = [4]
            drain()
            assert list(iw.indices) == [4]
            iw.figure.canvas.draw()
        finally:
            iw.close()


class TestClose:
    def test_close_offscreen_shuts_executors_and_is_idempotent(self, base5d):
        arr = LazyStandIn(base5d[:, 0])  # (T, Z, Y, X)
        iw = _make_viewer(arr, slider_dim_names=("Timepoint", "Z-plane"))
        executors = [ndg.slicer._executor for ndg in iw.ndgraphics]
        iw.close()  # offscreen: Figure._output is None; must not raise
        assert all(ex._shutdown for ex in executors)
        iw.close()  # idempotent


class TestSidePanelShortcut:
    def test_the_shortcut_collapses_and_restores_the_side_panel(self, tmp_path):
        from mbo_utilities import imread
        from mbo_utilities.gui._keyboard import toggle_side_panel
        from mbo_utilities.gui.run_gui import _create_image_widget

        np.save(tmp_path / "movie.npy", make_base_5d())
        iw = _create_image_widget(
            imread(tmp_path / "movie.npy"),
            widget=True,
            figure_kwargs_override={"size": (900, 600)},
        )
        panel = iw.figure.imgui_windows["right"]
        width = panel.size
        toggle_side_panel(panel)
        assert panel.collapsed
        assert panel.size < width
        toggle_side_panel(panel)
        assert not panel.collapsed
        assert panel.size == width
        # let signal quality finish so its reads don't land in the next test
        deadline = time.time() + 60
        while any(panel._zstats_running) and time.time() < deadline:
            iw.figure.canvas.draw()
            time.sleep(0.02)
        iw.close()
