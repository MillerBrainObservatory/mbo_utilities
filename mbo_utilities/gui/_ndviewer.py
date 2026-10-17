"""The viewer: fastplotlib's NDWidget with the indexing, contrast and data-swap
behavior the GUI uses.

Every read is one frame per slider axis (a window function's span on a
windowed axis); contrast comes from one bounded sample of scalar-key frames.
Indices are 0-based; the sliders show 1-based values.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Sequence
from itertools import product
from time import perf_counter

import numpy as np
from fastplotlib.utils import calculate_figure_shape
from fastplotlib.widgets.nd_widget import (
    NDImage,
    NDImageSlicer,
    NDWidget,
    RangeContinuous,
)
from fastplotlib.widgets.nd_widget._async import run_sync
from fastplotlib.widgets.nd_widget._ui import NDWidgetUI
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from mbo_utilities.arrays.features._dim_labels import default_dim_letters
from mbo_utilities.gui import _fpl_config  # noqa: F401

__all__ = ["MboNDViewer", "MboNDImageSlicer", "_sample_array", "sliders_height"]


def sliders_height(n_sliders: int) -> int:
    """Height of the slider window for ``n_sliders`` sliders."""
    return 57 + 50 * n_sliders


# positional names for slider axes 0, 1, 2, accepted wherever a dim name is
_SLIDER_LETTERS = ("t", "z", "c")

_default_dim_letters = default_dim_letters

# spatial dim names, kept apart from any slider label
_ROW, _COL, _RGB = "__row__", "__col__", "__rgb__"


# positions sampled per slider axis for contrast, outermost first: ~64 frames
# whatever the rank or length
VMINMAX_SAMPLE_COUNTS = {1: (64,), 2: (24, 3), 3: (12, 3, 2)}


def _sample_array(data) -> np.ndarray:
    """An even sample of ``data`` across every slider axis, read one 2D frame
    per scalar key (``VMINMAX_SAMPLE_COUNTS``); 2D data is returned whole.
    """
    shape = tuple(getattr(data, "shape", ()) or ())
    if len(shape) <= 2:
        if not shape:
            return np.asarray(data)
        # through __getitem__: np.asarray of an array without __array__ is 0-d
        return np.asarray(data[:])
    scroll = shape[:-2]
    if any(n == 0 for n in scroll):
        return np.empty(0)
    counts = VMINMAX_SAMPLE_COUNTS.get(len(scroll), (8,) * len(scroll))
    grids = [
        np.unique(np.linspace(0, n - 1, min(int(n), c)).astype(int))
        for n, c in zip(scroll, counts)
    ]
    blocks = [
        np.asarray(data[tuple(int(i) for i in combo)]).ravel()
        for combo in product(*grids)
    ]
    return np.concatenate(blocks) if blocks else np.empty(0)


def _histogram(block):
    """``np.histogram`` of ``block``'s finite values over [min, max], or
    [min, min + 1] when flat; None when there are none.
    """
    block = np.asarray(block).ravel()
    if block.dtype.kind not in "biuf":
        return None
    if block.dtype.kind == "f":
        block = block[np.isfinite(block)]
    if block.size == 0:
        return None
    lo, hi = float(block.min()), float(block.max())
    return np.histogram(block, bins=100, range=(lo, hi if hi > lo else lo + 1.0))


def _round_half_up(x) -> int:
    """Round .5 up, so an odd window of size ``s`` covers exactly ``s`` frames
    centered on its index (``round`` rounds half to even).
    """
    return int(math.floor(float(x) + 0.5))


def _ref_to_index(x) -> int:
    """1-based reference value -> 0-based array index (half-up)."""
    return _round_half_up(x) - 1


def _with_float32_cast(func):
    """``func`` (or identity) with float32 output, so funcs over integer data
    are not truncated into an integer texture.
    """
    if func is None:

        def _cast(a):
            return np.asarray(a, dtype=np.float32)
    else:

        def _cast(a):
            return np.asarray(func(a), dtype=np.float32)

    _cast._mbo_float_cast = True
    _cast._mbo_user_func = func
    return _cast


async def _noop_indices(*_args, **_kwargs):
    """``_set_indices_`` of a removed graphic: a scheduled fetch does nothing."""
    return None


class MboNDImageSlicer(NDImageSlicer):
    """``NDImageSlicer`` whose histogram reads ``_sample_array``, not one
    strided slice across every axis, which lazy readers serve slowly or not at all.
    """

    def _recompute_histogram(self):
        if not self._compute_histogram or self.data is None:
            self._histogram = None
            return
        self._histogram = _histogram(_sample_array(self.data))


class _NDDataList(list):
    """The viewer's arrays; ``data[i] = arr`` swaps one."""

    def __init__(self, viewer, items):
        super().__init__(items)
        self._viewer = viewer

    def __setitem__(self, i, new_array):
        self._viewer._replace_data(i, new_array)


class _MboIndicesView:
    """0-based slider indices, keyed by dim name; iterates values in slider order."""

    def __init__(self, viewer: MboNDViewer):
        self._viewer = viewer

    def __getitem__(self, name):
        dim = self._viewer._resolve_dim(name)
        return int(round(float(self._viewer._ndw.indices[dim]))) - 1

    def __setitem__(self, name, value):
        dim = self._viewer._resolve_dim(name)
        value = self._viewer._check_index(dim, value)
        self._viewer._ndw.indices.set_dim_index(dim, value + 1)
        self._viewer._flush_index_fetches()

    def __iter__(self):
        ri = self._viewer._ndw.indices
        return iter([int(round(float(ri[d]))) - 1 for d in self._viewer._dim_names])

    def __len__(self):
        return len(self._viewer._dim_names)

    def __repr__(self):
        ri = self._viewer._ndw.indices
        return repr({d: int(round(float(ri[d]))) - 1 for d in self._viewer._dim_names})


class _DimStateView:
    """An ``NDWidgetUI`` per-dim dict keyed by position, name, label or letter."""

    def __init__(self, viewer: MboNDViewer, store: dict):
        self._viewer = viewer
        self._store = store

    def _key(self, key) -> str:
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            dims = self._viewer._dim_names
            if not 0 <= int(key) < len(dims):
                raise IndexError(key)
            return dims[int(key)]
        return self._viewer._resolve_dim(key)

    def __getitem__(self, key):
        return self._store[self._key(key)]

    def __setitem__(self, key, value):
        self._store[self._key(key)] = value

    def __contains__(self, key):
        try:
            return self._key(key) in self._store
        except (KeyError, IndexError):
            return False

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def __iter__(self):
        return iter(self._viewer._dim_names)

    def __len__(self):
        return len(self._viewer._dim_names)

    def __repr__(self):
        return repr({d: self._store.get(d) for d in self._viewer._dim_names})


class _MboSlidersUI:
    """Playback state of the slider window: one loop switch for every dim, and
    an fps seeded from the data (``NDWidgetUI`` starts every dim at 20) that
    never overrides one the user typed.
    """

    def __init__(self, viewer: MboNDViewer):
        self._viewer = viewer
        self._user_fps: set[str] = set()
        # what seed_fps last wrote per dim; anything else came from the user
        self._seeded_fps: dict[str, int] = {}
        self._loop_all = False

    @property
    def _ndui(self):
        return self._viewer._ndw.ui_sliders

    @property
    def _playing(self) -> _DimStateView:
        return _DimStateView(self._viewer, self._ndui._playing)

    @property
    def _fps(self) -> _DimStateView:
        return _DimStateView(self._viewer, self._ndui._fps)

    @property
    def _frame_time(self) -> _DimStateView:
        return _DimStateView(self._viewer, self._ndui._frame_time)

    @property
    def _last_frame_time(self) -> _DimStateView:
        return _DimStateView(self._viewer, self._ndui._last_frame_time)

    @property
    def _loop(self) -> bool:
        return self._loop_all

    @_loop.setter
    def _loop(self, value: bool):
        self._loop_all = bool(value)
        for d in self._viewer._dim_names:
            if d in self._ndui._loop:
                self._ndui._loop[d] = self._loop_all

    def seed_fps(self, dim: str, fps) -> None:
        """Set ``dim``'s playback fps to ``fps`` clamped to [1, 50], unless the
        user set one or ``fps`` is not a positive number.
        """
        if fps is None:
            return
        try:
            d = self._viewer._resolve_dim(dim)
        except KeyError:
            return
        if d in self._user_fps:
            return
        current = self._ndui._fps.get(d)
        if current is not None and current != 20 and current != self._seeded_fps.get(d):
            self._user_fps.add(d)
            return
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return
        if fps <= 0 or not math.isfinite(fps):
            return
        value = min(max(int(round(fps)), 1), 50)
        self._ndui._fps[d] = value
        self._ndui._frame_time[d] = 1 / value
        self._seeded_fps[d] = value

    def _snapshot_fps(self) -> tuple[dict, dict]:
        return dict(self._ndui._fps), dict(self._ndui._frame_time)

    def _after_dims_changed(self, fps_snapshot: tuple[dict, dict] | None):
        # push_dims reset every dim's fps to 20; keep the ones that survived
        if fps_snapshot is not None:
            fps, ftime = fps_snapshot
            for d in self._viewer._dim_names:
                if d in fps and d in self._ndui._fps:
                    self._ndui._fps[d] = fps[d]
                    self._ndui._frame_time[d] = ftime[d]
        for d in self._viewer._dim_names:
            if d in self._ndui._loop:
                self._ndui._loop[d] = self._loop_all


class _MboNDWidgetUI(NDWidgetUI):
    """``NDWidgetUI`` with integer sliders for integer ranges."""

    def update(self):
        if len(self._ndwidget.indices) < 1 and len(self._update_calls) < 2:
            if self.size != 0:
                self.size = 0
            self.collapsed = True
            return

        now = perf_counter()

        for dim, current_index in self._ndwidget.indices:
            imgui.push_id(f"{self._id_counter}_{dim}")
            rr = self._ndwidget.ranges[dim]

            if self._playing[dim]:
                if imgui.button(label=fa.ICON_FA_PAUSE):
                    self._playing[dim] = False
                if now - self._last_frame_time[dim] >= self._frame_time[dim]:
                    self._set_index(dim, current_index + rr.step)
                    self._last_frame_time[dim] = now
            else:
                if imgui.button(label=fa.ICON_FA_PLAY):
                    self._last_frame_time[dim] = 0
                    self._playing[dim] = True

            imgui.same_line()
            if imgui.button(label=fa.ICON_FA_BACKWARD_STEP) and not self._playing[dim]:
                self._set_index(dim, current_index - rr.step)

            imgui.same_line()
            if imgui.button(label=fa.ICON_FA_FORWARD_STEP) and not self._playing[dim]:
                self._set_index(dim, current_index + rr.step)

            imgui.same_line()
            if imgui.button(label=fa.ICON_FA_STOP):
                self._playing[dim] = False
                self._last_frame_time[dim] = 0
                self._ndwidget.indices.set_dim_index(dim, rr.start)

            imgui.same_line()
            _, self._loop[dim] = imgui.checkbox(
                label=fa.ICON_FA_ROTATE, v=self._loop[dim]
            )
            if imgui.is_item_hovered(0):
                imgui.set_tooltip("loop playback")

            imgui.same_line()
            imgui.text("framerate :")
            imgui.same_line()
            imgui.set_next_item_width(100)
            fps_changed, value = imgui.input_int(
                label="fps", v=self._fps[dim], step_fast=5
            )
            if imgui.is_item_hovered(0):
                imgui.set_tooltip(
                    "framerate is approximate and less reliable as it approaches your monitor refresh rate"
                )
            if fps_changed:
                value = min(max(value, 1), 100)
                self._fps[dim] = value
                self._frame_time[dim] = 1 / value

            imgui.text(str(dim))
            imgui.same_line()
            imgui.set_next_item_width(self.width * 0.85)

            if isinstance(rr, RangeContinuous):
                if all(float(v).is_integer() for v in (rr.start, rr.stop, rr.step)):
                    changed, new_index = imgui.slider_int(
                        v=int(round(current_index)),
                        v_min=int(rr.start),
                        v_max=int(rr.stop - rr.step),
                        label=f"##{dim}",
                    )
                else:
                    changed, new_index = imgui.slider_float(
                        v=current_index,
                        v_min=rr.start,
                        v_max=rr.stop - rr.step,
                        label=f"##{dim}",
                    )

                if changed:
                    if now - self._last_slider_movement[dim] > rr.throttle:
                        self._ndwidget.indices.set_dim_index(
                            dim, new_index, cancel_awaiting=True
                        )
                        self._last_slider_movement[dim] = now
                elif imgui.is_item_hovered():
                    if imgui.is_key_pressed(imgui.Key.right_arrow):
                        self._set_index(dim, current_index + rr.step)
                    elif imgui.is_key_pressed(imgui.Key.left_arrow):
                        self._set_index(dim, current_index - rr.step)

            imgui.pop_id()

        if not self._collapsed:
            height = round(
                imgui.get_cursor_screen_pos().y
                - self.y
                + imgui.get_style().window_padding.y
            )
            if height != self.size:
                self.size = height


class MboNDViewer:
    """An NDWidget showing one array per subplot, with shared sliders."""

    def __init__(
        self,
        data,
        names: Sequence[str] | None = None,
        slider_dim_names: Sequence[str] | None = None,
        window_funcs=None,
        window_sizes=None,
        frame_apply=None,
        figure_shape: tuple[int, int] | None = None,
        figure_kwargs: dict | None = None,
        histogram_widget: bool = True,
        rgb: bool | Sequence[bool] | None = None,
        cmap: str = "plasma",
        graphic_kwargs: dict | None = None,
    ):
        if isinstance(data, (list, tuple)):
            arrays = list(data)
        else:
            arrays = [data]
        if not arrays:
            raise ValueError("`data` must contain at least one array")
        for arr in arrays:
            for attr in ("shape", "ndim", "dtype", "__getitem__"):
                if not hasattr(arr, attr):
                    raise TypeError(
                        f"data arrays must be array-like with `{attr}`, got: {type(arr)}"
                    )

        # per-array rgb flags
        if rgb is None:
            self._rgb = [bool(getattr(a, "rgb", False)) for a in arrays]
        elif isinstance(rgb, bool):
            self._rgb = [rgb] * len(arrays)
        else:
            self._rgb = [bool(r) for r in rgb]
            if len(self._rgb) != len(arrays):
                raise ValueError("`rgb` must match the number of data arrays")

        self._arrays = arrays
        self._histogram_widget = bool(histogram_widget)
        self._closed = False

        self._slider_dim_names = tuple(slider_dim_names) if slider_dim_names else None

        counts = [self._n_slider_dims(a, r) for a, r in zip(arrays, self._rgb)]
        n_dims = max(counts) if counts else 0
        if self._slider_dim_names and len(self._slider_dim_names) == n_dims:
            requested = self._slider_dim_names
        else:
            requested = None
        if requested is None:
            requested = self._labels_from_arrays(arrays, counts, n_dims)
        self._dim_names: list[str] = list(self._make_dim_names(n_dims, requested))

        # sliders show 1..n; _ref_to_index maps them to 0-based indices
        ref_ranges = {
            name: RangeContinuous(1, self._dim_size(j) + 1, 1)
            for j, name in enumerate(self._dim_names)
        }

        fig_kwargs = {"controller_ids": "sync", "names": list(names) if names else None}
        fig_kwargs.update(figure_kwargs or {})
        fig_kwargs["shape"] = (
            tuple(figure_shape)
            if figure_shape is not None
            else calculate_figure_shape(len(arrays))
        )

        self._ndw = NDWidget(ranges=ref_ranges, **fig_kwargs)
        # fastplotlib titles an unnamed subplot with its grid position
        if not names:
            for subplot in self._ndw.figure:
                subplot.title = ""

        # replaces NDWidget's slider window; ui_sliders has no setter
        ui = _MboNDWidgetUI(self._ndw)
        self._ndw.figure.add_imgui_window(
            ui,
            location="bottom",
            size=sliders_height(len(self._ndw.indices)),
            title="NDWidget controls",
        )
        self._ndw._sliders_ui = ui

        # offscreen there is no loop to run fetches: index changes render now
        self._offscreen = "offscreen" in type(self._ndw.figure.canvas).__module__

        gk = {k: v for k, v in (graphic_kwargs or {}).items() if v is not None}
        gk.setdefault("cmap", cmap)

        self._window_funcs_state = window_funcs
        self._window_sizes_state = tuple(window_sizes) if window_sizes else None
        self._frame_apply_state: dict[int, Callable] = {}
        self._spatial_func_state = None
        # each graphic's spatial func as given; its slicer may hold a float32 wrapper
        self._routed_spatial: dict[int, Callable | None] = {}

        # window funcs go in at construction, so the first fetch is the only one
        window = self._translate_window_funcs(window_funcs)
        self._ndgraphics: list[NDImage] = []
        subplots = [self._ndw[s] for s in self._ndw.figure]
        for i, (arr, nd_subplot) in enumerate(zip(arrays, subplots)):
            gname = None
            if names is not None and i < len(names):
                gname = str(names[i])
            ndg = self._add_image(
                nd_subplot,
                arr,
                self._rgb[i],
                name=gname,
                graphic_kwargs=gk,
                window_funcs=window,
            )
            self._ndgraphics.append(ndg)
            self._apply_spatial(i)
        if frame_apply is not None:
            self.frame_apply = frame_apply

        self._sliders = _MboSlidersUI(self)
        self._seed_fps_from_data()

    @staticmethod
    def _n_slider_dims(arr, rgb: bool) -> int:
        return max(int(arr.ndim) - 2 - (1 if rgb else 0), 0)

    def _dim_size(self, j: int) -> int:
        """Largest size of slider axis ``j`` across the arrays."""
        size = 1
        for arr, rgb in zip(self._arrays, self._rgb):
            if self._n_slider_dims(arr, rgb) > j:
                size = max(size, int(arr.shape[j]))
        return size

    @staticmethod
    def _labels_from_arrays(arrays, counts, n_dims: int):
        """``slider_dim_labels`` of a widest array when they name every slider."""
        if n_dims == 0:
            return None
        for arr, cnt in zip(arrays, counts):
            if cnt != n_dims:
                continue
            try:
                labels = getattr(arr, "slider_dim_labels", None)
            except Exception:
                continue
            if labels:
                labels = tuple(str(x) for x in labels)
                if len(labels) == n_dims:
                    return labels
        return None

    def _seed_fps_from_data(self):
        """Seed the t playback fps from the first array's ``fs``."""
        try:
            arr = self._arrays[0]
            fs = getattr(arr, "fs", None)
            if fs is None:
                from mbo_utilities.metadata import get_param

                fs = get_param(getattr(arr, "metadata", None), "fs")
            self._sliders.seed_fps("t", fs)
        except Exception:
            pass

    def _make_dim_names(
        self, n: int, requested: Sequence[str] | None
    ) -> tuple[str, ...]:
        """``requested`` when it names ``n`` axes, else the per-rank letters;
        deduplicated and kept off the spatial names.
        """
        if requested is not None and len(requested) == n:
            base = [str(x) for x in requested]
        else:
            base = list(_default_dim_letters(n))
        out, seen = [], set()
        for name in base:
            if name in (_ROW, _COL, _RGB):
                name = name + "_"
            while name in seen:
                name = name + "_"
            seen.add(name)
            out.append(name)
        return tuple(out)

    def _add_image(
        self,
        nd_subplot,
        arr,
        rgb: bool,
        name: str | None = None,
        graphic_kwargs=None,
        window_funcs: dict | None = None,
    ):
        k = self._n_slider_dims(arr, rgb)
        spatial = (_ROW, _COL) + ((_RGB,) if rgb else ())
        dims = tuple(self._dim_names[:k]) + spatial
        wf = self._graphic_window_funcs(window_funcs or {}, dims[:k])
        return nd_subplot.add_nd_image(
            data=arr,
            dims=dims,
            display_dims=spatial,
            rgb_dim=_RGB if rgb else None,
            window_funcs=dict(wf) or None,
            window_order=tuple(wf) or None,
            compute_histogram=self._histogram_widget,
            slicer_type=MboNDImageSlicer,
            # fresh dicts: fastplotlib mutates both
            slider_maps={d: _ref_to_index for d in dims[:k]} or None,
            graphic_kwargs=dict(graphic_kwargs) if graphic_kwargs else None,
            name=name or "mbo_image",
        )

    def _resolve_dim(self, name) -> str:
        """The dim name for a dim name, label or positional letter."""
        name = str(name)
        dims = self._dim_names
        if name in dims:
            return name
        labels = tuple(self._slider_dim_names or ())
        if name in labels:
            pos = labels.index(name)
            if pos < len(dims):
                return dims[pos]
        if name in _SLIDER_LETTERS:
            pos = _SLIDER_LETTERS.index(name)
            if pos < len(dims):
                return dims[pos]
        raise KeyError(name)

    def _check_index(self, dim: str, value) -> int:
        """``value`` as an int, IndexError when negative or past ``dim``'s end
        (``ReferenceIndices`` would clamp it).
        """
        value = int(value)
        if value < 0:
            raise IndexError(
                f"negative indexing is not supported (got {value} for dim {dim!r})"
            )
        rr = self._ndw.indices.ref_ranges.get(dim)
        if isinstance(rr, RangeContinuous):
            size = int(rr.stop - rr.start)
            if value >= size:
                raise IndexError(
                    f"index {value} out of bounds for dim {dim!r} with size {size}"
                )
        return value

    @property
    def ndwidget(self) -> NDWidget:
        """The wrapped NDWidget."""
        return self._ndw

    @property
    def ndgraphics(self) -> tuple:
        return tuple(self._ndgraphics)

    @property
    def figure(self):
        """The NDWidget's figure."""
        return self._ndw.figure

    @property
    def data(self) -> _NDDataList:
        return _NDDataList(self, self._arrays)

    @property
    def graphics(self) -> list:
        """One ImageGraphic per array."""
        return [ndg.graphic for ndg in self._ndgraphics]

    @property
    def n_sliders(self) -> int:
        return len(self._dim_names)

    @property
    def dim_names(self) -> tuple[str, ...]:
        """Slider names in order: the arrays' labels, else per-rank letters."""
        return tuple(self._dim_names)

    @property
    def slider_dims(self) -> list[str]:
        """Positional letters for the slider axes (``t``, ``z``, ``c``)."""
        return [
            _SLIDER_LETTERS[j] if j < len(_SLIDER_LETTERS) else f"dim{j}"
            for j in range(len(self._dim_names))
        ]

    @property
    def indices(self) -> _MboIndicesView:
        return _MboIndicesView(self)

    @indices.setter
    def indices(self, value):
        if isinstance(value, dict):
            items = {self._resolve_dim(k): v for k, v in value.items()}
        else:
            items = dict(zip(self._dim_names, value))
        items = {d: self._check_index(d, v) for d, v in items.items()}
        # every set renders, so `iw.indices = list(iw.indices)` refreshes
        self._ndw.indices.set({d: v + 1 for d, v in items.items()})
        self._flush_index_fetches()

    @property
    def current_index(self) -> dict[str, int]:
        ri = self._ndw.indices
        return {d: int(round(float(ri[d]))) - 1 for d in self._dim_names}

    @current_index.setter
    def current_index(self, value: dict):
        self.indices = dict(value)

    @property
    def _sliders_ui(self) -> _MboSlidersUI:
        return self._sliders

    @property
    def cmap(self) -> list[str]:
        return [getattr(g, "cmap", None) for g in self.graphics]

    @cmap.setter
    def cmap(self, value):
        if isinstance(value, str):
            values = [value] * len(self._ndgraphics)
        else:
            values = list(value)
        for ndg, name in zip(self._ndgraphics, values):
            g = ndg.graphic
            if g is None or getattr(g, "cmap", None) is None:
                continue
            with contextlib.suppress(Exception):
                g.cmap = name

    def _fold_positional(self, funcs, sizes) -> dict:
        folded = {}
        funcs = tuple(funcs)
        sizes = tuple(sizes) if sizes else ()
        for j, dim in enumerate(self._dim_names):
            if j < len(funcs) and funcs[j] is not None:
                size = 1
                if j < len(sizes) and sizes[j]:
                    size = sizes[j]
                folded[dim] = (funcs[j], size)
        return folded

    @property
    def window_funcs(self):
        return self._window_funcs_state

    @window_funcs.setter
    def window_funcs(self, value):
        """``{dim: (func, size)}``, ``(func, ...)`` positional with ``window_sizes``,
        or None. Sizes are in frames. Setting what is already set reads nothing.
        """
        self._window_funcs_state = value
        translated = self._translate_window_funcs(value)
        changed = []
        for i, ndg in enumerate(self._ndgraphics):
            wf = self._graphic_window_funcs(translated, ndg.slicer.slider_dims)
            if wf == self._graphic_window_funcs(
                ndg.slicer.window_funcs or {}, ndg.slicer.window_order or ()
            ):
                continue
            # fresh dict per graphic: the fastplotlib setter mutates its argument
            ndg.slicer.window_funcs = dict(wf) or None
            ndg.slicer.window_order = tuple(wf) or None
            changed.append(i)
        rebuilt = {i for i in changed if self._apply_spatial(i)}
        self._force_render([i for i in changed if i not in rebuilt])

    def _translate_window_funcs(self, value) -> dict:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {self._resolve_dim(k): tuple(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return self._fold_positional(value, self._window_sizes_state)
        raise TypeError(
            "window_funcs must be None, a {dim: (func, size)} dict, or a "
            f"positional tuple of funcs; got: {type(value)}"
        )

    @staticmethod
    def _graphic_window_funcs(funcs: dict, dims) -> dict:
        """The entries of ``funcs`` with a func, for ``dims``, in ``dims`` order."""
        return {
            d: tuple(funcs[d])
            for d in dims
            if d in funcs and funcs[d] and funcs[d][0] is not None
        }

    @property
    def window_sizes(self):
        return self._window_sizes_state

    @window_sizes.setter
    def window_sizes(self, value):
        self._window_sizes_state = tuple(value) if value else None
        if isinstance(self._window_funcs_state, (tuple, list)):
            self.window_funcs = self._window_funcs_state

    @property
    def frame_apply(self) -> dict:
        return self._frame_apply_state

    @frame_apply.setter
    def frame_apply(self, value):
        """``{array index: func}`` or one func for every array, applied to each
        displayed frame; the histogram is left as it is.
        """
        if value is None:
            mapping = {}
        elif callable(value):
            mapping = {i: value for i in range(len(self._ndgraphics))}
        elif isinstance(value, dict):
            mapping = dict(value)
        else:
            raise TypeError(
                "frame_apply must be a callable or a {data_index: callable} dict"
            )
        self._frame_apply_state = mapping
        self._route_spatial({i: mapping.get(i) for i in range(len(self._ndgraphics))})

    @property
    def spatial_func(self):
        return self._spatial_func_state

    @spatial_func.setter
    def spatial_func(self, value):
        """One func, or one per array; same as ``frame_apply``."""
        self._spatial_func_state = value
        if isinstance(value, (list, tuple)):
            funcs = list(value)
        else:
            funcs = [value] * len(self._ndgraphics)
        self._route_spatial(dict(enumerate(funcs[: len(self._ndgraphics)])))

    def _route_spatial(self, funcs: dict):
        """Give each graphic in ``funcs`` its spatial func and render the ones
        that changed, once.
        """
        changed = [i for i, f in funcs.items() if self._routed_spatial.get(i) is not f]
        rebuilt = set()
        for i in changed:
            self._routed_spatial[i] = funcs[i]
            if self._apply_spatial(i):
                rebuilt.add(i)
        self._force_render([i for i in changed if i not in rebuilt])

    def _apply_spatial(self, i: int) -> bool:
        """Put graphic ``i``'s spatial func on its slicer, cast to float32 over
        integer data; True when that rebuilt the graphic, which renders it.
        """
        ndg = self._ndgraphics[i]
        func = self._routed_spatial.get(i)
        if self._needs_float(i):
            ndg.slicer.spatial_func = _with_float32_cast(func)
            return self._ensure_float_texture(ndg)
        ndg.slicer.spatial_func = func
        return False

    def _needs_float(self, i: int) -> bool:
        """True when graphic ``i`` shows func output of integer data."""
        ndg = self._ndgraphics[i]
        data = ndg.data
        if data is None:
            return False
        try:
            kind = np.dtype(data.dtype).kind
        except Exception:
            return False
        if kind not in "bui":
            return False
        if self._routed_spatial.get(i) is not None:
            return True
        wf = ndg.slicer.window_funcs or {}
        order = ndg.slicer.window_order or ()
        return any(
            d in order and (wf.get(d) or (None, None))[0] is not None for d in wf
        )

    def _ensure_float_texture(self, ndg) -> bool:
        """Rebuild ``ndg``'s graphic with a float texture when it has an integer
        one, keeping contrast, camera and stacking; True when rebuilt.
        """
        g = ndg.graphic
        if g is None:
            return False
        try:
            kind = np.asarray(g.data.value).dtype.kind
        except Exception:
            return False
        if kind not in "bui":
            return False
        vmin = getattr(g, "vmin", None)
        vmax = getattr(g, "vmax", None)
        # _create_graphic resets the range and the camera
        subplot = ndg._nd_subplot.subplot
        camera_state = None
        with contextlib.suppress(Exception):
            camera_state = subplot.camera.get_state()
        # graphics stack in z by the order they were added; re-adding the image
        # would put it over every overlay added after it
        z_before = {id(gr): float(gr.offset[2]) for gr in subplot.graphics}
        old_z = float(g.offset[2])

        with contextlib.suppress(Exception):
            run_sync(ndg._create_graphic())

        new = ndg.graphic
        with contextlib.suppress(Exception):
            if new is not None:
                new.offset = (*new.offset[:2], old_z)
            for gr in subplot.graphics:
                z = z_before.get(id(gr))
                if z is not None and gr is not new:
                    gr.offset = (*gr.offset[:2], z)
        if camera_state is not None:
            with contextlib.suppress(Exception):
                subplot.camera.set_state(camera_state)
        cb = ndg.histogram_widget
        target = cb if cb is not None else ndg.graphic
        if target is not None and vmin is not None and vmax is not None:
            with contextlib.suppress(Exception):
                target.vmax = float(vmax)
                target.vmin = float(vmin)
        return True

    def _force_render(self, which=None):
        """Fetch and draw graphics ``which`` (default all) at the current indices."""
        for i, ndg in enumerate(self._ndgraphics):
            if which is not None and i not in which:
                continue
            if ndg.data is None or ndg.graphic is None:
                continue
            with contextlib.suppress(Exception):
                run_sync(ndg._set_indices_())

    def _flush_index_fetches(self):
        """Offscreen: render the queued index change now and drop it from the
        queue (cleared in place; a scheduled task still looks it up).
        """
        if not getattr(self, "_offscreen", False):
            return
        ri = self._ndw.indices
        for ndg in self._ndgraphics:
            queue = ri._fetch_request_queue.get(ndg)
            if queue is not None:
                queue.clear()
        self._force_render()

    @property
    def compute_histogram(self) -> bool:
        return self._histogram_widget

    @compute_histogram.setter
    def compute_histogram(self, value: bool):
        self._histogram_widget = bool(value)
        for ndg in self._ndgraphics:
            ndg.compute_histogram = self._histogram_widget

    def _set_contrast(self, ndg, histogram):
        """Set one graphic's range and colorbar from ``(counts, edges)``."""
        if histogram is None:
            return
        edges = histogram[1]
        vmin, vmax = float(edges[0]), float(edges[-1])
        cb = ndg.histogram_widget
        if cb is not None:
            # set first: it re-derives the axis the handles sit on
            cb.histogram = histogram
            # vmax first, so a new vmin above the old vmax never inverts
            cb.vmax = vmax
            cb.vmin = vmin
        else:
            g = ndg.graphic
            if g is not None:
                g.vmin, g.vmax = vmin, vmax

    def reset_vmin_vmax(self):
        """Contrast from a sample of every array (``_sample_array``)."""
        for ndg, arr in zip(self._ndgraphics, self._arrays):
            self._set_contrast(ndg, _histogram(_sample_array(arr)))

    def reset_vmin_vmax_frame(self):
        """Contrast from the frames on screen; reads nothing."""
        for ndg in self._ndgraphics:
            g = ndg.graphic
            if g is None:
                continue
            self._set_contrast(ndg, _histogram(g.data.value))

    def _teardown_ndgraphic(self, ndg):
        """Remove one NDGraphic: its fetches, executor, graphic, colorbar and
        registrations (``delete_nd_graphic`` is not implemented in fastplotlib 0.7).
        """
        ndg._set_indices_ = _noop_indices
        ndg.pause = True
        with contextlib.suppress(Exception):
            ndg.slicer.close()
        subplot = ndg._nd_subplot.subplot
        if ndg.graphic is not None:
            with contextlib.suppress(Exception):
                subplot.delete_graphic(ndg.graphic)
            ndg._graphic = None
        if getattr(ndg, "_histogram_widget", None) is not None:
            with contextlib.suppress(Exception):
                subplot.remove_imgui_window("right")
            ndg._histogram_widget = None
        with contextlib.suppress(ValueError):
            ndg._nd_subplot._nd_graphics.remove(ndg)
        ri = self._ndw.indices
        ri._fetch_rev.pop(ndg, None)
        # a scheduled fetch task looks its queue up: clear it in place, and drop
        # the keys only when no task is pending (else _sweep_dead_fetch_state)
        queue = ri._fetch_request_queue.get(ndg)
        if queue is not None:
            queue.clear()
        if not ri._fetch_request_active.get(ndg, False):
            ri._fetch_request_queue.pop(ndg, None)
            ri._fetch_request_active.pop(ndg, None)

    def _sweep_dead_fetch_state(self):
        """Drop removed graphics' fetch state once their tasks have run."""
        ri = self._ndw.indices
        live = set(self._ndw.ndgraphics)
        for store in (ri._fetch_request_active, ri._fetch_request_queue):
            for ndg in [g for g in store if g not in live]:
                if ri._fetch_request_active.get(ndg, False):
                    queue = ri._fetch_request_queue.get(ndg)
                    if queue is not None:
                        queue.clear()
                    continue
                ri._fetch_request_active.pop(ndg, None)
                ri._fetch_request_queue.pop(ndg, None)

    def _replace_data(self, i, new_array):
        """``data[i] = new_array``: rebuild graphic ``i`` and the sliders for the
        new array, clear window and spatial funcs, go to index 0. When the new
        array fails to install the old one is put back and the error raised.
        """
        i = int(i)
        if not 0 <= i < len(self._arrays):
            raise IndexError(i)
        for attr in ("shape", "ndim", "dtype", "__getitem__"):
            if not hasattr(new_array, attr):
                raise TypeError(f"replacement data must be array-like with `{attr}`")

        self._sweep_dead_fetch_state()

        old_ndg = self._ndgraphics[i]
        nd_subplot = old_ndg._nd_subplot
        old_array = self._arrays[i]
        old_rgb = self._rgb[i]

        new_rgb = getattr(new_array, "rgb", None)
        if isinstance(new_rgb, bool):
            self._rgb[i] = new_rgb

        old_cmap = None
        if old_ndg.graphic is not None:
            old_cmap = getattr(old_ndg.graphic, "cmap", None)

        self._teardown_ndgraphic(old_ndg)
        try:
            self._install_array(i, new_array, nd_subplot, old_ndg.name, old_cmap)
        except Exception:
            self._rgb[i] = old_rgb
            with contextlib.suppress(Exception):
                self._install_array(i, old_array, nd_subplot, old_ndg.name, old_cmap)
            raise

    def _install_array(self, i, new_array, nd_subplot, name, cmap):
        """Put ``new_array`` in slot ``i``: new sliders, no funcs, a new graphic."""
        self._arrays[i] = new_array

        counts = [self._n_slider_dims(a, r) for a, r in zip(self._arrays, self._rgb)]
        need = max(counts) if counts else 0
        ri = self._ndw.indices
        ndui = self._ndw.ui_sliders
        fps_snapshot = (
            self._sliders._snapshot_fps() if hasattr(self, "_sliders") else None
        )

        old = list(self._dim_names)
        if len(self._arrays) == 1:
            # one array: its own labels name the sliders
            self._dim_names = list(self._derive_swap_names(new_array, need))
        else:
            # shared positional sliders: never rename one
            del self._dim_names[need:]
            while len(self._dim_names) < need:
                j = len(self._dim_names)
                dim = self._make_dim_names(j + 1, None)[j]
                while dim in self._dim_names or dim in ri.dims:
                    dim = dim + "_"
                self._dim_names.append(dim)
        # the new sliders, at index 0, go in before the stale ones come out:
        # pop_dims tells index subscribers, who must find them
        ri.push_dims(
            {
                dim: RangeContinuous(1, self._dim_size(j) + 1, 1)
                for j, dim in enumerate(self._dim_names)
            }
        )
        stale = [d for d in old if d not in self._dim_names]
        if stale:
            ri.pop_dims(*stale)

        self._slider_dim_names = tuple(self._dim_names) or None

        if hasattr(self, "_sliders"):
            self._sliders._after_dims_changed(fps_snapshot)
            self._seed_fps_from_data()

        self._window_funcs_state = None
        self._window_sizes_state = None
        self._frame_apply_state = {}
        self._spatial_func_state = None
        self._routed_spatial = {}
        for j, ndg in enumerate(self._ndgraphics):
            if j == i:
                continue
            ndg.slicer.window_funcs = None
            ndg.slicer.window_order = None
            ndg.slicer.spatial_func = None

        # redraw the other arrays at index 0; the new one reads its frame when added
        self._force_render([j for j in range(len(self._ndgraphics)) if j != i])

        new_ndg = self._add_image(
            nd_subplot,
            new_array,
            self._rgb[i],
            name=name,
            graphic_kwargs={"cmap": cmap} if cmap is not None else None,
        )
        self._ndgraphics[i] = new_ndg

        # contrast from the sample the histogram already read
        histogram = new_ndg.slicer.histogram
        with contextlib.suppress(Exception):
            if histogram is None:
                histogram = _histogram(_sample_array(new_array))
            self._set_contrast(new_ndg, histogram)

        # a slider window collapsed for 2D data reopens
        with contextlib.suppress(Exception):
            ndui.size = sliders_height(len(self._dim_names))
            ndui.collapsed = False

        # every graphic is drawn at index 0; tell index subscribers without a fetch
        ri._indices_changed()

    def _derive_swap_names(self, arr, n: int) -> tuple[str, ...]:
        """Slider names for a new array: its labels, else the current names
        when the count is unchanged, else letters.
        """
        labels = getattr(arr, "slider_dim_labels", None)
        if labels:
            labels = tuple(str(x) for x in labels)
            if len(labels) == n:
                return self._make_dim_names(n, labels)
        if len(self._dim_names) == n:
            return self._make_dim_names(n, self._dim_names)
        return self._make_dim_names(n, None)

    def show(self, **kwargs):
        return self._ndw.show(**kwargs)

    def close(self):
        """Close the figure and every slicer's executor (fastplotlib does not)."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._sweep_dead_fetch_state()
        for ndg in self._ndgraphics:
            ndg._set_indices_ = _noop_indices
            with contextlib.suppress(Exception):
                ndg.slicer.close()
        fig = self._ndw.figure
        if getattr(fig, "_output", None) is not None:
            with contextlib.suppress(Exception):
                fig.close()
        else:
            # offscreen: Figure.close() fails without an output
            with contextlib.suppress(Exception):
                fig.canvas.close()

    def __repr__(self):
        shapes = [tuple(getattr(a, "shape", ())) for a in self._arrays]
        return (
            f"{type(self).__name__}(n_data={len(self._arrays)}, "
            f"dims={tuple(self._dim_names)}, shapes={shapes})"
        )
