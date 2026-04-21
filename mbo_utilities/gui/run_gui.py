"""
CLI entry point for mbo_utilities GUI.

This module is designed for fast startup - heavy imports are deferred until needed.
Operations like --download-notebook and --check-install should be near-instant.
"""
import sys
from pathlib import Path
from typing import Any

import click
import contextlib

# Set AppUserModelID immediately for Windows
try:
    import ctypes
    import sys
    if sys.platform == "win32":
        myappid = "mbo.utilities.gui.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass


def _set_qt_icon():
    """Set the Qt application window icon.

    Must be called AFTER the canvas/window is created and shown.
    Sets the icon on QApplication and all top-level windows including
    native window handles for proper Windows taskbar display.
    """
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QIcon
        from mbo_utilities.gui._setup import get_package_assets_path
        from mbo_utilities.preferences import get_mbo_dirs

        app = QApplication.instance()
        if app is None:
            return

        # try package assets first, then user assets
        icon_path = get_package_assets_path() / "app_settings" / "icon.png"
        if not icon_path.exists():
            icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"

        if not icon_path.exists():
            return

        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)

        # set on all top-level windows including native handles
        for window in app.topLevelWidgets():
            window.setWindowIcon(icon)
            handle = window.windowHandle()
            if handle:
                handle.setIcon(icon)
            app.processEvents()
    except Exception:
        pass


class SplashScreen:
    """Simple splash screen using tkinter (always available on windows)."""

    def __init__(self):
        self.root = None
        self._closed = False

    def show(self):
        """Show splash screen with loading indicator."""
        try:
            import tkinter as tk

            self.root = tk.Tk()
            self.root.overrideredirect(True)  # no title bar
            self.root.attributes("-topmost", True)

            # center on screen
            width, height = 300, 120
            x = (self.root.winfo_screenwidth() - width) // 2
            y = (self.root.winfo_screenheight() - height) // 2
            self.root.geometry(f"{width}x{height}+{x}+{y}")

            # styling
            self.root.configure(bg="#1e1e2e")

            # title
            title = tk.Label(
                self.root,
                text="MBO Utilities",
                font=("Segoe UI", 16, "bold"),
                fg="#89b4fa",
                bg="#1e1e2e",
            )
            title.pack(pady=(20, 5))

            # loading text
            self.loading_label = tk.Label(
                self.root,
                text="Loading...",
                font=("Segoe UI", 10),
                fg="#a6adc8",
                bg="#1e1e2e",
            )
            self.loading_label.pack(pady=(0, 10))

            # simple progress animation
            self.progress_frame = tk.Frame(self.root, bg="#1e1e2e")
            self.progress_frame.pack(pady=5)

            self.dots = []
            for _i in range(5):
                dot = tk.Label(
                    self.progress_frame,
                    text="●",
                    font=("Segoe UI", 12),
                    fg="#45475a",
                    bg="#1e1e2e",
                )
                dot.pack(side=tk.LEFT, padx=3)
                self.dots.append(dot)

            self.current_dot = 0
            self._animate()
            self.root.update()

        except Exception:
            self.root = None

    def _animate(self):
        """Animate the loading dots."""
        if self.root is None or self._closed:
            return
        try:
            for i, dot in enumerate(self.dots):
                dot.configure(fg="#89b4fa" if i == self.current_dot else "#45475a")
            self.current_dot = (self.current_dot + 1) % len(self.dots)
            self.root.after(200, self._animate)
        except Exception:
            pass

    def close(self):
        """Close the splash screen."""
        self._closed = True
        if self.root is not None:
            with contextlib.suppress(Exception):
                self.root.destroy()
            self.root = None


def _get_version() -> str:
    """Get the current mbo_utilities version."""
    try:
        import mbo_utilities
        return getattr(mbo_utilities, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _check_for_upgrade() -> tuple[str, str | None]:
    """check pypi for newer version of mbo_utilities (cached for 1 hour).

    returns (current_version, latest_version) or (current_version, None) if check fails.
    """
    import urllib.request
    import json

    current = _get_version()

    # check cache first (1 hour expiry)
    try:
        from mbo_utilities.env_cache import get_cached_pypi_version, update_pypi_cache
        cached = get_cached_pypi_version(max_age_hours=1)
        if cached:
            return current, cached
    except Exception:
        pass

    # fetch from pypi
    try:
        url = "https://pypi.org/pypi/mbo-utilities/json"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            latest = data["info"]["version"]
            # update cache
            try:
                update_pypi_cache(latest)
            except Exception:
                pass
            return current, latest
    except Exception:
        return current, None


def _print_upgrade_status():
    """Print upgrade status to console."""
    current, latest = _check_for_upgrade()

    click.echo(f"Current version: {current}")

    if latest is None:
        click.secho("Could not check for updates (network error or package not on PyPI)", fg="yellow")
        return

    click.echo(f"Latest version:  {latest}")

    if current == "unknown":
        click.secho("Could not determine current version", fg="yellow")
    elif current == latest:
        click.secho("You are running the latest version!", fg="green")
    else:
        # simple version comparison (works for semver)
        try:
            from packaging.version import parse
            if parse(current) < parse(latest):
                click.secho("\nUpgrade available! Run:", fg="cyan")
                click.secho("  uv pip install --upgrade mbo-utilities", fg="cyan", bold=True)
                click.echo("  or")
                click.secho("  pip install --upgrade mbo-utilities", fg="cyan", bold=True)
            else:
                click.secho("You are running a newer version than PyPI (dev build)", fg="green")
        except ImportError:
            # no packaging module, do string comparison
            if current != latest:
                click.secho("\nDifferent version on PyPI. To upgrade:", fg="cyan")
                click.secho("  uv pip install --upgrade mbo-utilities", fg="cyan", bold=True)


def _download_notebook_file(
    output_path: str | Path | None = None,
    notebook_url: str | None = None,
):
    """Download a Jupyter notebook from a URL to a local file.

    Parameters
    ----------
    output_path : str, Path, optional
        Directory or file path to save the notebook. If None or '.', saves to current directory.
        If a directory, saves using the notebook's filename from the URL.
        If a file path, uses that exact filename.
    notebook_url : str, optional
        URL to the notebook file. If None, downloads the default user guide notebook.
        Supports GitHub blob URLs (automatically converted to raw URLs).

    Examples
    --------
    # Download default user guide
    _download_notebook_file()

    # Download specific notebook from GitHub
    _download_notebook_file(
        output_path="./notebooks",
        notebook_url="https://github.com/org/repo/blob/main/demos/example.ipynb"
    )
    """
    import urllib.request

    default_url = "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/demos/user_guide.ipynb"
    url = notebook_url or default_url

    # Convert GitHub blob URLs to raw URLs
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    # Extract filename from URL
    url_filename = url.split("/")[-1]
    if not url_filename.endswith(".ipynb"):
        url_filename = "notebook.ipynb"

    # Determine output file path
    if output_path is None or output_path == ".":
        output_file = Path.cwd() / url_filename
    else:
        output_file = Path(output_path)
        if output_file.is_dir():
            output_file = output_file / url_filename
        elif output_file.suffix != ".ipynb":
            # If it's a directory that doesn't exist yet, create it and use url filename
            output_file.mkdir(parents=True, exist_ok=True)
            output_file = output_file / url_filename

    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        click.echo(f"Downloading notebook from:\n  {url}")
        click.echo(f"Saving to:\n  {output_file.resolve()}")

        # Download the file
        urllib.request.urlretrieve(url, output_file)

        click.secho(f"\nSuccessfully downloaded notebook to: {output_file.resolve()}", fg="green")
        click.echo("\nTo use the notebook:")
        click.echo(f"  jupyter lab {output_file.resolve()}")

    except Exception as e:
        click.secho(f"\nFailed to download notebook: {e}", fg="red")
        click.echo(f"\nYou can manually download from: {url}")
        sys.exit(1)

    return output_file


def download_notebook(
    output_path: str | Path | None = None,
    notebook_url: str | None = None,
) -> Path:
    """Download a Jupyter notebook from a URL to a local file.

    This is the public API for downloading notebooks programmatically.

    Parameters
    ----------
    output_path : str, Path, optional
        Directory or file path to save the notebook. If None, saves to current directory.
        If a directory, saves using the notebook's filename from the URL.
        If a file path, uses that exact filename.
    notebook_url : str, optional
        URL to the notebook file. If None, downloads the default user guide notebook.
        Supports GitHub blob URLs (automatically converted to raw URLs).

    Returns
    -------
    Path
        Path to the downloaded notebook file.

    Examples
    --------
    >>> from mbo_utilities.gui import download_notebook

    # Download default user guide to current directory
    >>> download_notebook()

    # Download specific notebook from GitHub
    >>> download_notebook(
    ...     output_path="./notebooks",
    ...     notebook_url="https://github.com/org/repo/blob/main/demos/example.ipynb"
    ... )

    # Download to specific filename
    >>> download_notebook(
    ...     output_path="./my_notebook.ipynb",
    ...     notebook_url="https://github.com/org/repo/blob/main/nb.ipynb"
    ... )
    """
    return _download_notebook_file(output_path=output_path, notebook_url=notebook_url)


def _check_installation():
    """Verify that mbo_utilities and key dependencies are properly installed."""
    from mbo_utilities.install import check_installation, print_status_cli
    status = check_installation()
    print_status_cli(status)
    return status.all_ok


def _select_file(runner_params: Any | None = None) -> tuple[Any, Any, Any, bool, str]:
    """Show file selection dialog and return user choices."""
    from mbo_utilities.gui.widgets.file_dialog import FileDialog  # triggers _setup import
    from mbo_utilities.gui._setup import get_default_ini_path
    from imgui_bundle import immapp, hello_imgui

    dlg = FileDialog()

    if runner_params is None:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = "Miller Brain Suite – Data Selection"
        params.app_window_params.window_geometry.size = (340, 720)
        params.app_window_params.window_geometry.size_auto = False
        params.app_window_params.resizable = True
        params.ini_filename = get_default_ini_path("file_dialog")
    else:
        params = runner_params

    # always override the gui callback to render our dialog
    params.callbacks.show_gui = dlg.render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)

    # Get selected mode
    mode = dlg.gui_modes[dlg.selected_mode_index]

    return (
        dlg.selected_path,
        dlg.split_rois,
        dlg.widget_enabled,
        dlg.metadata_only,
        mode,
    )


def _show_metadata_viewer(metadata: dict) -> None:
    """Show metadata in an ImGui window."""
    from imgui_bundle import immapp, hello_imgui
    from mbo_utilities.gui._metadata import draw_metadata_inspector
    from mbo_utilities.gui._setup import get_default_ini_path

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Miller Brain Suite – Metadata"
    params.app_window_params.window_geometry.size = (800, 800)
    params.ini_filename = get_default_ini_path("metadata_viewer")
    params.callbacks.show_gui = lambda: draw_metadata_inspector(metadata)

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)


class _SqueezeSingletonDims:
    """wraps a 5D TCZYX array, dropping every singleton non-spatial dim.

    fastplotlib's ImageWidget derives its slider count from ndim-2 and
    doesn't handle singleton sliders, so any T/C/Z with size 1 must be
    removed before handoff. get_slider_dims filters by the same
    size>1 predicate; this wrapper keeps the shape in lockstep so
    len(slider_dim_names) always equals wrapped.ndim - 2.
    """

    def __init__(self, arr):
        self._arr = arr
        full = tuple(arr.shape)
        if len(full) != 5:
            raise ValueError(f"expected 5D TCZYX array, got shape {full}")
        # keep non-spatial dims only if size > 1, always keep Y, X
        self._kept = [i for i in range(3) if full[i] > 1] + [3, 4]
        self._dropped = [i for i in range(3) if full[i] <= 1]
        self._shape = tuple(full[i] for i in self._kept)
        orig_dims = getattr(arr, "dims", None)
        if orig_dims is not None and len(orig_dims) == 5:
            self._dims = tuple(orig_dims[i] for i in self._kept)
        else:
            self._dims = None

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def dims(self):
        return self._dims

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # pad short keys to wrapped ndim
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))
        # rebuild a 5D key: 0 at dropped axes, user key at kept axes
        full_key = [None] * 5
        for d in self._dropped:
            full_key[d] = 0
        it = iter(key)
        for i in self._kept:
            full_key[i] = next(it)
        return self._arr[tuple(full_key)]

    def __len__(self):
        return self._shape[0]

    def __array__(self, dtype=None, copy=None):
        # materialize through __getitem__ so the squeezed shape is honored.
        # without this, np.asarray() / .astype() / etc. fall through
        # __getattr__ to the underlying array and leak the unsqueezed rank.
        import numpy as np
        arr = np.asarray(self[tuple(slice(None) for _ in range(self.ndim))])
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def astype(self, dtype, *args, **kwargs):
        # fastplotlib's TextureArray._fix_data calls data.astype(np.float32).
        # route through __array__ so the cast sees the squeezed shape.
        import numpy as np
        return np.asarray(self).astype(dtype, *args, **kwargs)

    def __getattr__(self, name):
        # IMPORTANT: do not forward numpy-array-protocol methods here
        # (__array__, astype, reshape, etc.) — those must be defined
        # explicitly above so they respect the squeezed shape. this
        # fallback exists only for domain-specific attributes like
        # `metadata`, `stack_type`, `num_beamlets`, etc.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._arr, name)


def _create_image_widget(data_array, widget: bool = True):
    """Create fastplotlib ImageWidget with optional PreviewDataWidget."""
    import copy
    import numpy as np
    import fastplotlib as fpl

    try:
        from rendercanvas.pyqt6 import RenderCanvas
    except (ImportError, RuntimeError): # RuntimeError if qt is already selected
        RenderCanvas = None

    if RenderCanvas is not None:
        figure_kwargs = {
            "canvas": "pyqt6",
            "canvas_kwargs": {"present_method": "bitmap"},
            "size": (800, 800)
        }
    else:
        figure_kwargs = {"size": (800, 800)}

    # Determine slider dimension names from array's dims property if available
    from mbo_utilities.arrays.features import get_slider_dims

    slider_dim_names = get_slider_dims(data_array)

    # window_funcs/window_sizes must match slider_dim_names length
    if slider_dim_names:
        n_sliders = len(slider_dim_names)
        # apply mean to first dim (usually t), None for rest
        window_funcs = (np.mean,) + (None,) * (n_sliders - 1)
        window_sizes = (1,) + (None,) * (n_sliders - 1)
    else:
        window_funcs = None
        window_sizes = None

    def _squeeze_for_viewer(arr):
        """drop every singleton non-spatial dim so fastplotlib's ndim-2
        slider count matches get_slider_dims output.

        fastplotlib expects ndim == len(slider_dim_names) + 2. any
        T/C/Z with size 1 must be squeezed, not just C.
        """
        if not hasattr(arr, "shape") or len(arr.shape) != 5:
            return arr
        if any(arr.shape[i] == 1 for i in range(3)):
            return _SqueezeSingletonDims(arr)
        return arr

    # Handle multi-ROI data (duck typing: check for roi_mode attribute)
    if hasattr(data_array, "roi_mode") and hasattr(data_array, "iter_rois"):
        arrays = []
        names = []
        # get name from first filename if available, truncate if too long
        base_name = None
        if hasattr(data_array, "filenames") and data_array.filenames:
            from pathlib import Path
            first_file = Path(data_array.filenames[0])
            base_name = first_file.stem
            # for suite2p arrays (data.bin), use parent folder name instead
            if base_name in ("data", "data_raw"):
                base_name = first_file.parent.name
            if len(base_name) > 24:
                base_name = base_name[:21] + "..."
        for r in data_array.iter_rois():
            arr = copy.copy(data_array)
            arr.fix_phase = False
            arr.roi = r
            arrays.append(_squeeze_for_viewer(arr))
            names.append(f"ROI {r}" if r else (base_name or "Full Image"))

        iw = fpl.ImageWidget(
            data=arrays,
            names=names,
            slider_dim_names=slider_dim_names,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
            cmap="gnuplot2",
            histogram_widget=True,
            figure_kwargs=figure_kwargs,
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )
    else:
        iw = fpl.ImageWidget(
            data=_squeeze_for_viewer(data_array),
            slider_dim_names=slider_dim_names,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
            cmap="gnuplot2",
            histogram_widget=True,
            figure_kwargs=figure_kwargs,
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )

    iw.show()

    # set qt window title and icon after canvas is created
    canvas = iw.figure.canvas
    if hasattr(canvas, "set_title"):
        canvas.set_title("Miller Brain Suite")
    _set_qt_icon()

    # Add PreviewDataWidget if requested
    if widget:
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        gui = PreviewDataWidget(
            iw=iw,
            fpath=data_array.source_path,
            size=300,
        )
        iw.figure.add_gui(gui)

    return iw


def _is_jupyter() -> bool:
    """Check if running in Jupyter environment."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def _run_gui_impl(
    data_in: str | Path | None = None,
    roi: int | tuple[int, ...] | None = None,
    widget: bool = True,
    metadata_only: bool = False,
    select_only: bool = False,
    show_splash: bool = False,
    runner_params: Any | None = None,
    mode: str = "Standard Viewer",
):
    """Internal implementation of run_gui with all heavy imports."""
    # show splash screen while loading (only for desktop shortcut launches)
    splash = None
    if show_splash and not _is_jupyter():
        splash = SplashScreen()
        splash.show()

    try:
        # Import heavy dependencies only when actually running GUI

        # close splash before showing file dialog
        if splash:
            splash.close()
            splash = None

        # Handle file selection if no path provided
        if data_in is None:
            data_in, roi_from_dialog, widget, metadata_only, mode = _select_file(runner_params=runner_params)
            if not data_in:
                return None
            # Use ROI from dialog if not specified in function call
            if roi is None:
                roi = roi_from_dialog

        # If select_only, just return the path
        if select_only:
            return data_in


        # Dispatch based on Mode
        # Note: pollen calibration is auto-detected in Standard Viewer via get_viewer_class()
        if mode == "Standard Viewer":
            return _launch_standard_viewer(data_in, roi, widget, metadata_only)
        if mode == "Napari":
            return _launch_napari(data_in, roi)
        return _launch_standard_viewer(data_in, roi, widget, metadata_only)

    finally:
        # ensure splash is closed on any exit path
        if splash:
            splash.close()


def _launch_standard_viewer(data_in, roi, widget, metadata_only):
    from mbo_utilities.reader import imread
    from mbo_utilities.arrays import normalize_roi

    roi = normalize_roi(roi)
    data_array = imread(data_in, roi=roi)

    if metadata_only:
        metadata = data_array.metadata
        if not metadata:
            return None
        _show_metadata_viewer(metadata)
        return None

    import fastplotlib as fpl
    iw = _create_image_widget(data_array, widget=widget)

    if _is_jupyter():
        return iw
    fpl.loop.run()
    return None


class _NapariArray:
    """Wrapper that prevents napari from eagerly materializing lazy arrays.

    mbo_utilities lazy arrays define __array__() returning the first frame.
    Napari calls np.asarray() on the object when added, triggering materialization
    of just that 2D frame instead of the full ND dataset, breaking the viewer.
    By not defining __array__, we force napari to use __getitem__ to access slices.

    When `c_index` is set, the wrapper presents the underlying 5D TCZYX
    array as a 4D TZYX view by pinning C to that index. This is what
    `_launch_napari` uses when the data is single-channel — passing the
    raw 5D array makes napari see 5 dimensions and reject 4-element
    `axis_labels`, while `channel_axis=1` would create a stray
    one-channel split. Squeezing C here gets us a clean 4D view that
    napari treats as a single (T, Z, Y, X) image layer.
    """
    def __init__(self, arr, c_index: int | None = None):
        self._arr = arr
        self._c_index = c_index
        if c_index is not None:
            # advertise 4D (T, Z, Y, X) by stripping the C axis at index 1
            full = tuple(arr.shape)
            if len(full) != 5:
                raise ValueError(
                    f"_NapariArray(c_index=...) expects 5D TCZYX input, got shape {full}"
                )
            self._shape = (full[0], full[2], full[3], full[4])
            self._ndim = 4
        else:
            self._shape = tuple(arr.shape)
            self._ndim = arr.ndim

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ndim(self):
        return self._ndim

    def __getitem__(self, key):
        import numpy as np

        if self._c_index is None:
            return np.asarray(self._arr[key])

        # rebuild a 5D TCZYX key from the user's 4D TZYX key by inserting
        # the pinned c_index at position 1. napari's slicing protocol
        # passes a tuple of slices/ints; pad short keys with full slices
        # the same way numpy does so partial indexes still work.
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < 4:
            key = key + (slice(None),) * (4 - len(key))
        full_key = (key[0], int(self._c_index), key[1], key[2], key[3])
        return np.asarray(self._arr[full_key])


# napari layer defaults — used for every layer added by `_launch_napari`,
# including the napari-ome-zarr plugin path (we set them post-add there).
# If you want to change them globally, change them here, not at every
# add_image call site.
_NAPARI_DEFAULTS = {
    "colormap": "gnuplot2",
    "contrast_limits": (0, 400),
}


def _prompt_for_dz(default: float = 16.0) -> float | None:
    """Pop a Qt input dialog asking the user for the z-spacing.

    Used by `_launch_napari` when the source metadata doesn't supply `dz`
    (LBM TIFFs are the common case — `dz` has to come from the user). The
    napari Viewer is already running so its QApplication is alive; we
    just need a modal QInputDialog on top of it.

    Returns the entered value in micrometers, or None if the user
    cancelled. The dialog uses QInputDialog.getDouble which validates
    range and decimals natively, so we don't need a separate validator.

    Range: 0.1 - 1000.0 µm. Anything outside that is almost certainly a
    typo (sub-100nm pixels are nanoscale; >1mm is not light microscopy).
    """
    try:
        from qtpy.QtWidgets import QInputDialog, QApplication
    except ImportError:
        try:
            from PyQt6.QtWidgets import QInputDialog, QApplication
        except ImportError:
            return None  # no Qt available — caller falls back to default

    # need a QApplication instance to parent the dialog. napari's already
    # running one by the time we get here, but be defensive.
    app = QApplication.instance()
    if app is None:
        return None

    value, ok = QInputDialog.getDouble(
        None,
        "Z-spacing required",
        (
            "No `dz` (z-step) found in source metadata.\n\n"
            "Enter the z-spacing in micrometers (\u00b5m):"
        ),
        float(default),  # initial value
        0.1,             # min
        1000.0,          # max
        3,               # decimals
    )
    return float(value) if ok else None


def _launch_napari(data_in, roi=None):
    from mbo_utilities.log import get as get_logger
    logger = get_logger("gui.napari")

    try:
        import napari
    except ImportError:
        logger.warning("napari not installed. Please install it using `uv pip install napari pyqt6`")
        return None

    try:
        from mbo_utilities.reader import imread
        from mbo_utilities.arrays import normalize_roi
        from mbo_utilities.arrays.features._dim_labels import get_dims
        from mbo_utilities.metadata.output import OutputMetadata
        from pathlib import Path

        path_str = str(data_in)

        # Probe the source up front so we can pick the right viewer mode
        # (3D when there's a real Z axis) and log the spatial scale we'll
        # be passing to napari. We re-load below for the actual layer when
        # falling out of the ome-zarr plugin path; the probe is cheap.
        probe_arr = None
        probe_dims = None
        probe_z_size = 1
        try:
            probe_arr = imread(data_in, roi=normalize_roi(roi))
            probe_dims = get_dims(probe_arr)
            if "Z" in probe_dims:
                probe_z_size = int(probe_arr.shape[probe_dims.index("Z")])
        except Exception as e:
            logger.debug(f"napari probe failed (will still try plugin path): {e}")

        # 3D display when the data actually has multiple z-planes; otherwise
        # 2D so the canvas isn't constantly switching back and forth.
        ndisplay = 3 if probe_z_size > 1 else 2
        viewer = napari.Viewer(ndisplay=ndisplay)
        logger.info(f"napari viewer launched (ndisplay={ndisplay})")

        def _apply_layer_defaults(layer):
            """Force colormap + contrast on a layer, ignoring failures."""
            try:
                layer.colormap = _NAPARI_DEFAULTS["colormap"]
            except Exception:
                pass
            try:
                layer.contrast_limits = _NAPARI_DEFAULTS["contrast_limits"]
            except Exception:
                pass

        # Try napari-ome-zarr plugin first for .zarr files
        loaded = False
        if path_str.endswith(".zarr"):
            try:
                added = viewer.open(path_str, plugin="napari-ome-zarr")
                # `viewer.open` returns the list of layers it created
                if added:
                    for layer in added:
                        _apply_layer_defaults(layer)
                loaded = True
            except Exception as e:
                # OME-Zarr plugin failed, fall back to mbo_utilities
                logger.debug(f"napari-ome-zarr failed: {e}")

        if not loaded:
            # Load via mbo_utilities and add as layer
            try:
                arr = probe_arr if probe_arr is not None else imread(
                    data_in, roi=normalize_roi(roi)
                )
                dims = probe_dims if probe_dims is not None else get_dims(arr)

                # 5D TCZYX contract:
                #   - real multi-channel (C > 1) → use napari's channel_axis,
                #     leave the array 5D, napari splits per channel
                #   - singleton C (the common case for LBM) → tell the
                #     wrapper to squeeze C, so napari sees a clean 4D
                #     (T, Z, Y, X) layer with no stray "channels" slider
                # We must NOT just drop C from axis_labels while leaving
                # the underlying array 5D — napari rejects that with
                # `axis_labels must have length ndim=5`.
                channel_axis = None
                squeeze_c_index = None
                if "C" in dims:
                    c_idx = list(dims).index("C")
                    c_size = int(arr.shape[c_idx]) if c_idx < len(arr.shape) else 1
                    if c_size > 1:
                        channel_axis = c_idx
                    else:
                        squeeze_c_index = 0

                # Build scale via OutputMetadata.to_napari_scale, which
                # already pulls dx/dy/dz off the source metadata via the
                # voxel_size feature. Log the resolved values so the user
                # can confirm the metadata was actually picked up rather
                # than silently falling back to (1, 1, 1).
                scale = None
                resolved_dz = None
                try:
                    om = OutputMetadata(
                        source=arr.metadata,
                        source_shape=arr.shape,
                        source_dims=dims,
                    )
                    vs = om.voxel_size
                    logger.info(
                        f"napari scale source: dx={vs.dx}, dy={vs.dy}, dz={vs.dz}"
                    )
                    scale = list(om.to_napari_scale(dims))
                    resolved_dz = vs.dz
                except Exception as e:
                    logger.debug(f"Failed to build scale: {e}")

                # If the source didn't supply a dz and we actually have
                # multiple z-planes, prompt the user. Otherwise napari
                # silently falls back to dz=1.0 and the 3D view is
                # squashed in the wrong proportion. The dialog is the
                # same one the metadata editor would show, so the value
                # entered here is treated as authoritative for this
                # session and written into the array's scale tuple.
                if (
                    resolved_dz is None
                    and "Z" in dims
                    and probe_z_size > 1
                    and scale is not None
                ):
                    user_dz = _prompt_for_dz()
                    if user_dz is not None:
                        z_idx = list(dims).index("Z")
                        if z_idx < len(scale):
                            scale[z_idx] = float(user_dz)
                        logger.info(
                            f"napari scale: applied user-entered dz={user_dz} um"
                        )
                    else:
                        logger.warning(
                            "napari scale: user cancelled dz prompt; "
                            "falling back to dz=1.0 (z-axis will be squashed)"
                        )

                axis_labels = list(dims)
                # Drop the C axis from scale and labels regardless of whether
                # it's a real channel split — napari's `channel_axis` strips
                # it from the displayed dims for us, and for the singleton
                # case we collapsed it to None and want it gone anyway.
                if "C" in dims:
                    c_idx_for_strip = list(dims).index("C")
                    if scale and len(scale) > c_idx_for_strip:
                        scale.pop(c_idx_for_strip)
                    if len(axis_labels) > c_idx_for_strip:
                        axis_labels.pop(c_idx_for_strip)

                layer_name = Path(path_str).stem

                def _add_to_viewer(layer_arr, name):
                    # squeeze the singleton C axis when there isn't a
                    # real channel split — napari then sees a 4D
                    # (T, Z, Y, X) view with rank matching axis_labels.
                    n_arr = _NapariArray(layer_arr, c_index=squeeze_c_index)
                    kwargs = {
                        "name": name,
                        "colormap": _NAPARI_DEFAULTS["colormap"],
                        "contrast_limits": _NAPARI_DEFAULTS["contrast_limits"],
                    }
                    if scale:
                        kwargs["scale"] = tuple(scale)
                    if channel_axis is not None:
                        kwargs["channel_axis"] = channel_axis

                    try:
                        added_layers = viewer.add_image(
                            n_arr, axis_labels=tuple(axis_labels), **kwargs
                        )
                    except TypeError:
                        # Older napari without axis_labels kwarg
                        added_layers = viewer.add_image(n_arr, **kwargs)

                    # add_image returns either a single layer or a list when
                    # channel_axis is set; normalize and re-apply the
                    # defaults in case napari overrode them after add.
                    if added_layers is None:
                        return
                    if not isinstance(added_layers, (list, tuple)):
                        added_layers = [added_layers]
                    for layer in added_layers:
                        _apply_layer_defaults(layer)

                # Handle multi-ROI
                if hasattr(arr, "roi_mode") and hasattr(arr, "iter_rois"):
                    import copy
                    for r in arr.iter_rois():
                        r_arr = copy.copy(arr)
                        r_arr.fix_phase = False
                        r_arr.roi = r
                        name = f"ROI {r}" if r else layer_name
                        _add_to_viewer(r_arr, name)
                else:
                    _add_to_viewer(arr, layer_name)

                loaded = True
            except Exception as e:
                logger.error(f"Failed to load via mbo_utilities: {e}")

        if not loaded:
            # Last resort: let napari try to open it directly
            try:
                added = viewer.open(path_str)
                if added:
                    for layer in added:
                        _apply_layer_defaults(layer)
            except Exception as e:
                logger.error(f"Failed to open via napari default fallback: {e}")

        napari.run()
    except Exception as e:
        logger.error(f"Error launching napari: {e}")



def run_gui(
    data_in: str | Path | None = None,
    roi: int | tuple[int, ...] | None = None,
    widget: bool = True,
    metadata_only: bool = False,
    select_only: bool = False,
    runner_params: Any | None = None,
):
    """
    Open a GUI to preview data of any supported type.

    Works both as a CLI command and as a Python function for Jupyter/scripts.
    In Jupyter, returns the ImageWidget so you can interact with it.
    In standalone mode, runs the event loop (blocking).

    Parameters
    ----------
    data_in : str, Path, optional
        Path to data file or directory. If None, shows file selection dialog.
    roi : int, tuple of int, optional
        ROI index(es) to display. None shows all ROIs for raw files.
    widget : bool, default True
        Enable PreviewDataWidget for raw ScanImage tiffs.
    metadata_only : bool, default False
        If True, only show metadata inspector (no image viewer).
    select_only : bool, default False
        If True, only show file selection dialog and return the selected path.
        Does not load data or open the image viewer.
    runner_params : Any, optional
        hello_imgui.RunnerParams instance for custom window configuration.

    Returns
    -------
    ImageWidget, Path, or None
        In Jupyter: returns the ImageWidget (already shown via iw.show()).
        In standalone: returns None (runs event loop until closed).
        With select_only=True: returns the selected path (str or Path).

    Examples
    --------
    From Python/Jupyter:
    >>> from mbo_utilities.gui import run_gui
    >>> # Option 1: Just show the GUI
    >>> run_gui("path/to/data.tif")
    >>> # Option 2: Get reference to manipulate it
    >>> iw = run_gui("path/to/data.tif", roi=1, widget=False)
    >>> iw.cmap = "viridis"  # Change colormap
    >>> # Option 3: Just get file path from dialog
    >>> path = run_gui(select_only=True)
    >>> print(f"Selected: {path}")

    From command line:
    $ mbo path/to/data.tif
    $ mbo path/to/data.tif --roi 1 --no-widget
    $ mbo path/to/data.tif --metadata-only
    $ mbo --select-only  # Just open file dialog
    """
    return _run_gui_impl(
        data_in=data_in,
        roi=roi,
        widget=widget,
        metadata_only=metadata_only,
        select_only=select_only,
        runner_params=runner_params,
    )


@click.command()
@click.version_option(version=_get_version(), prog_name="mbo-utilities")
@click.option(
    "--check-upgrade",
    is_flag=True,
    help="Check if a newer version is available on PyPI.",
)
@click.option(
    "--roi",
    multiple=True,
    type=int,
    help="ROI index (can pass multiple, e.g. --roi 0 --roi 2). Leave empty for None."
    " If 0 is passed, all ROIs will be shown (only for Raw files).",
    default=None,
)
@click.option(
    "--widget/--no-widget",
    default=True,
    help="Enable or disable PreviewDataWidget for Raw ScanImge tiffs.",
)
@click.option(
    "--metadata-only/--full-preview",
    default=False,
    help="If enabled, only show extracted metadata.",
)
@click.option(
    "--select-only",
    is_flag=True,
    help="Only show file selection dialog and print selected path. Does not open viewer.",
)
@click.option(
    "--download-notebook",
    is_flag=True,
    help="Download a Jupyter notebook and exit. Uses --notebook-url if provided, else downloads user guide.",
)
@click.option(
    "--notebook-url",
    type=str,
    default=None,
    help="URL of notebook to download. Supports GitHub blob URLs (auto-converted to raw). Use with --download-notebook.",
)
@click.option(
    "--check-install",
    is_flag=True,
    help="Verify the installation of mbo_utilities and dependencies.",
)
@click.option(
    "--splash",
    is_flag=True,
    hidden=True,
    help="Show splash screen during startup (used by desktop shortcut).",
)
@click.argument("data_in", required=False)
def _cli_entry(data_in=None, widget=None, roi=None, metadata_only=False, select_only=False, download_notebook=False, notebook_url=None, check_install=False, check_upgrade=False, splash=False):
    """CLI entry point for mbo-gui command."""
    # Handle upgrade check first (light operation)
    if check_upgrade:
        _print_upgrade_status()
        return

    # Handle installation check (light operation)
    if check_install:
        _check_installation()
        if download_notebook:
            click.echo("\n")
            _download_notebook_file(output_path=data_in, notebook_url=notebook_url)
        return

    # Handle download notebook option (light operation)
    if download_notebook:
        _download_notebook_file(output_path=data_in, notebook_url=notebook_url)
        return

    # Run the GUI (heavy imports happen here)
    result = _run_gui_impl(
        data_in=data_in,
        roi=roi if roi else None,
        widget=widget,
        metadata_only=metadata_only,
        select_only=select_only,
        show_splash=splash,
    )

    # If select_only, print the selected path
    if select_only and result:
        click.echo(result)


if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
