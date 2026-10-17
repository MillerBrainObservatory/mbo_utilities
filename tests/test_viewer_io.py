"""The viewer reads only what it shows.

Every read the viewer makes of a recording is counted in frames, one (Y, X)
plane per T, C, Z position. Opening a file reads one frame and one bounded
contrast sample; moving a slider reads one frame (a window's span with a
window function on); nothing else reads.
"""

import time
from functools import partialmethod
from itertools import product

import numpy as np
import pytest
from mbo_utilities import imread
from mbo_utilities.arrays.features._summary_stats import SubsampleBudget
from mbo_utilities.arrays.numpy import NumpyArray
from mbo_utilities.gui._dialogs import load_new_data
from mbo_utilities.gui._ndviewer import VMINMAX_SAMPLE_COUNTS
from mbo_utilities.gui.run_gui import _create_image_widget

pytestmark = pytest.mark.critical

SHAPE = (120, 1, 3, 32, 40)
OTHER = (90, 1, 2, 32, 40)
# the most frames one contrast sample reads, whatever the rank
SAMPLE_FRAMES = max(int(np.prod(c)) for c in VMINMAX_SAMPLE_COUNTS.values())
STATS_FRAMES = SubsampleBudget().avg_max_samples


def frames(key, shape) -> int:
    """How many (Y, X) planes ``arr[key]`` reads from an array of ``shape``."""
    key = key if isinstance(key, tuple) else (key,)
    n = 1
    for axis, size in enumerate(shape[:-2]):
        k = key[axis] if axis < len(key) else slice(None)
        if isinstance(k, slice):
            n *= len(range(*k.indices(size)))
        elif np.ndim(k):
            n *= len(k)
    return n


def kind(key) -> str:
    """``sample`` (scalar key), ``stats`` (Signal Quality, strided T) or ``display``."""
    key = key if isinstance(key, tuple) else (key,)
    if all(isinstance(k, (int, np.integer)) for k in key[:3]):
        return "sample"
    t = key[0]
    if isinstance(t, slice) and (t.step or 1) > 1:
        return "stats"
    return "display"


def record(arr, reads, key):
    reads.append((kind(key), frames(key, arr.shape)))
    return NumpyArray._unrecorded_getitem(arr, key)


def of(reads, wanted) -> list[int]:
    return [n for k, n in reads if k == wanted]


def wait_for_stats(iw, panel):
    deadline = time.time() + 60
    while not all(panel._zstats_done) and time.time() < deadline:
        iw.figure.canvas.draw()
        time.sleep(0.02)
    assert all(panel._zstats_done)


@pytest.fixture
def reads(monkeypatch):
    log = []
    monkeypatch.setattr(
        NumpyArray, "_unrecorded_getitem", NumpyArray.__getitem__, raising=False
    )
    monkeypatch.setattr(NumpyArray, "__getitem__", partialmethod(record, log))
    return log


@pytest.fixture
def viewer(tmp_path, reads):  # noqa: ARG001  reads records from the first frame on
    data = np.random.default_rng(0).integers(0, 1000, SHAPE, dtype=np.int16)
    np.save(tmp_path / "movie.npy", data)
    np.save(tmp_path / "other.npy", data[: OTHER[0], :, : OTHER[2]])
    iw = _create_image_widget(
        imread(tmp_path / "movie.npy"),
        widget=True,
        figure_kwargs_override={"size": (900, 600)},
    )
    panel = iw.figure.imgui_windows["right"]
    yield iw, panel
    iw.close()


def test_opening_a_file_reads_one_frame_and_one_sample(viewer, reads):
    iw, panel = viewer
    wait_for_stats(iw, panel)
    assert of(reads, "display") == [1]
    assert 0 < len(of(reads, "sample")) <= SAMPLE_FRAMES
    # Signal Quality: one strided read per z-plane, within its budget
    assert len(of(reads, "stats")) == SHAPE[2]
    assert max(of(reads, "stats")) <= STATS_FRAMES


def test_a_slider_step_reads_one_frame(viewer, reads):
    iw, panel = viewer
    wait_for_stats(iw, panel)
    reads.clear()
    steps = list(product(range(10, 15), range(SHAPE[2])))
    for t, z in steps:
        iw.indices = [t, z]
    assert reads == [("display", 1)] * len(steps)


def test_a_window_reads_its_span_once_per_step(viewer, reads):
    iw, panel = viewer
    wait_for_stats(iw, panel)
    iw.indices = [50, 0]
    reads.clear()
    panel.window_size = 5
    assert reads == [("display", 5)]
    panel.window_size = 5
    assert len(reads) == 1
    iw.indices = [60, 0]
    assert reads == [("display", 5)] * 2
    panel.window_size = 1
    assert reads[-1] == ("display", 1)


def test_contrast_from_the_frame_on_screen_reads_nothing(viewer, reads):
    iw, panel = viewer
    wait_for_stats(iw, panel)
    reads.clear()
    iw.reset_vmin_vmax_frame()
    iw.figure.canvas.draw()
    assert reads == []


def test_opening_another_file_reads_one_frame_and_one_sample(viewer, reads, tmp_path):
    iw, panel = viewer
    wait_for_stats(iw, panel)
    panel.window_size = 5
    reads.clear()
    load_new_data(panel, str(tmp_path / "other.npy"))
    wait_for_stats(iw, panel)
    assert of(reads, "display") == [1]
    assert 0 < len(of(reads, "sample")) <= SAMPLE_FRAMES
    assert len(of(reads, "stats")) == OTHER[2]
    assert max(of(reads, "stats")) <= STATS_FRAMES
