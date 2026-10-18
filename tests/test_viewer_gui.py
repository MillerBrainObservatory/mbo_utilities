"""The viewer and its side panel on fastplotlib's NDWidget, drawn offscreen.

``RENDERCANVAS_FORCE_OFFSCREEN`` is set in conftest; no window opens.
"""

import logging
import time
from functools import partial

import numpy as np
import pytest
from imgui_bundle import imgui

from mbo_utilities import imread
from mbo_utilities.gui import _stats
from mbo_utilities.gui._colormaps import listed_name
from mbo_utilities.gui._dialogs import load_new_data
from mbo_utilities.gui._ndviewer import MboNDViewer
from mbo_utilities.gui.run_gui import _create_image_widget
from mbo_utilities.gui.viewers import time_series
from mbo_utilities.gui.widgets import preview_data
from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

FIGURE = {"size": (1100, 760)}
LOGGERS = ("mbo", "rendercanvas", "fastplotlib", "wgpu")


class ErrorLog(logging.Handler):
    """Every ERROR record of the loggers it is attached to."""

    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def movie(path, shape):
    data = (np.random.default_rng(0).random(shape) * 1000).astype(np.int16)
    np.save(path, data)
    return path


def draw(iw, frames=3):
    for _ in range(frames):
        iw.figure.canvas.draw()


def select_tab(real, wanted, label, p_open=None, flags=0):
    """``imgui.begin_tab_item`` that selects ``wanted`` the way a click does."""
    if label == wanted:
        flags = flags | imgui.TabItemFlags_.set_selected
    return real(label, p_open, flags)


@pytest.fixture
def viewer(request, tmp_path, monkeypatch):
    on = getattr(request, "param", False)
    for module in (_stats, preview_data, time_series):
        monkeypatch.setattr(module, "signal_quality_enabled", lambda: on)
    arr = imread(movie(tmp_path / "movie.npy", (12, 1, 3, 48, 64)))
    iw = _create_image_widget(arr, widget=True, figure_kwargs_override=FIGURE)
    panel = iw.figure.imgui_windows["right"]
    errors = ErrorLog()
    attached = [
        logging.getLogger(name)
        for name in list(logging.root.manager.loggerDict)
        if name.split(".")[0] in LOGGERS
    ]
    for logger in attached:
        logger.addHandler(errors)
    yield iw, panel, errors
    for logger in attached:
        logger.removeHandler(errors)
    iw.close()


def test_the_viewer_opens_with_its_side_panel(viewer):
    iw, panel, errors = viewer
    assert isinstance(iw, MboNDViewer) and isinstance(panel, PreviewDataWidget)
    # C is a singleton, so the sliders are T and Z
    assert iw.dim_names == ("t", "z") and iw.n_sliders == 2
    assert panel.image_widget is iw and panel._figure is iw.figure
    draw(iw)
    assert errors.messages == []


def test_the_colormap_is_listed_by_name(viewer):
    iw, _panel, _errors = viewer
    # a graphic's cmap is a cmap.Colormap, not a string
    assert not isinstance(iw.graphics[0].cmap, str)
    assert listed_name(iw.graphics[0].cmap) == "gnuplot2"
    iw.cmap = "viridis"
    assert listed_name(iw.graphics[0].cmap) == "viridis"


def test_window_and_spatial_functions_reach_the_viewer(viewer):
    iw, panel, errors = viewer
    panel.window_size = 5
    assert iw.window_funcs == {"t": (np.mean, 5)}
    panel.proj = "max"
    assert iw.window_funcs == {"t": (np.max, 5)}
    # an even size is made odd so the window is centred on the frame
    panel.window_size = 4
    assert iw.window_funcs == {"t": (np.max, 5)}
    panel.window_size = 1
    assert iw.window_funcs is None
    panel.gaussian_sigma = 1.5
    assert callable(iw.spatial_func[0])
    draw(iw)
    panel.gaussian_sigma = 0.0
    assert iw.spatial_func is None
    draw(iw)
    assert errors.messages == []


def test_the_panels_popups_are_drawn_every_frame(viewer):
    iw, panel, errors = viewer
    draw(iw, 2)
    # set by draw_keybinds_popup, which only the per-frame draw() reaches
    panel._show_keybinds_popup = True
    draw(iw, 2)
    assert panel._keybinds_popup_actually_open is True
    for flag in ("_saveas_popup_open", "_show_metadata_popup", "show_metadata_viewer"):
        setattr(panel, flag, True)
        draw(iw, 2)
        setattr(panel, flag, False)
    assert errors.messages == []


@pytest.mark.parametrize("viewer", [True], indirect=True)
def test_the_signal_quality_tab_plots(viewer, monkeypatch):
    iw, panel, errors = viewer
    deadline = time.time() + 60
    while time.time() < deadline and not all(panel._zstats_done):
        draw(iw, 1)
        time.sleep(0.05)
    assert all(panel._zstats_done)
    monkeypatch.setattr(
        imgui,
        "begin_tab_item",
        partial(select_tab, imgui.begin_tab_item, "Signal Quality"),
    )
    draw(iw, 4)
    assert errors.messages == []


def test_opening_another_file_swaps_the_data(viewer, tmp_path):
    iw, panel, errors = viewer
    other = movie(tmp_path / "other.npy", (9, 1, 1, 40, 56))
    load_new_data(panel, str(other))
    # T only now: one slider, back at the first frame
    assert iw.data[0].shape == (9, 40, 56)
    assert iw.dim_names == ("t",) and list(iw.indices) == [0]
    assert panel.shape == (9, 40, 56)
    draw(iw)
    assert errors.messages == []
