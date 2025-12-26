"""
CLI entry point for mbo_utilities GUI.

This module is designed for fast startup - heavy imports are deferred until needed.
Operations like --download-notebook and --check-install should be near-instant.
"""
import sys
import os
import importlib.util
from pathlib import Path
from typing import Any, Optional, Union

import click


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
            for i in range(5):
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
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None


def _get_version() -> str:
    """Get the current mbo_utilities version."""
    try:
        import mbo_utilities
        return getattr(mbo_utilities, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _check_for_upgrade() -> tuple[str, str | None]:
    """Check PyPI for newer version of mbo_utilities.

    Returns (current_version, latest_version) or (current_version, None) if check fails.
    """
    import urllib.request
    import json

    current = _get_version()

    try:
        url = "https://pypi.org/pypi/mbo-utilities/json"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            latest = data["info"]["version"]
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
                click.secho(f"\nUpgrade available! Run:", fg="cyan")
                click.secho(f"  uv pip install --upgrade mbo-utilities", fg="cyan", bold=True)
                click.echo("  or")
                click.secho(f"  pip install --upgrade mbo-utilities", fg="cyan", bold=True)
            else:
                click.secho("You are running a newer version than PyPI (dev build)", fg="green")
        except ImportError:
            # no packaging module, do string comparison
            if current != latest:
                click.secho(f"\nDifferent version on PyPI. To upgrade:", fg="cyan")
                click.secho(f"  uv pip install --upgrade mbo-utilities", fg="cyan", bold=True)


def _download_notebook_file(
    output_path: Optional[Union[str, Path]] = None,
    notebook_url: Optional[str] = None,
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
    output_path: Optional[Union[str, Path]] = None,
    notebook_url: Optional[str] = None,
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
    from mbo_utilities.install_checker import check_installation, print_status_cli
    status = check_installation()
    print_status_cli(status)
    return status.all_ok


def _select_file() -> tuple[Any, Any, Any, bool]:
    """Show file selection dialog and return user choices."""
    from mbo_utilities.gui._file_dialog import FileDialog  # triggers _setup import
    from mbo_utilities.gui._setup import get_default_ini_path
    from imgui_bundle import immapp, hello_imgui

    dlg = FileDialog()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Utilities – Data Selection"
    params.app_window_params.window_geometry.size = (420, 680)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False  # fixed size to prevent scrollbar issues
    params.ini_filename = get_default_ini_path("file_dialog")
    params.callbacks.show_gui = dlg.render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)

    return (
        dlg.selected_path,
        dlg.split_rois,
        dlg.widget_enabled,
        dlg.metadata_only,
    )


def _show_metadata_viewer(metadata: dict) -> None:
    """Show metadata in an ImGui window."""
    from imgui_bundle import immapp, hello_imgui
    from mbo_utilities.gui._widgets import draw_metadata_inspector
    from mbo_utilities.gui._setup import get_default_ini_path

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Metadata Viewer"
    params.app_window_params.window_geometry.size = (800, 800)
    params.ini_filename = get_default_ini_path("metadata_viewer")
    params.callbacks.show_gui = lambda: draw_metadata_inspector(metadata)

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)


def _create_image_widget(data_array, widget: bool = True):
    """Create fastplotlib ImageWidget with optional PreviewDataWidget."""
    import copy
    import numpy as np
    import fastplotlib as fpl
    from mbo_utilities.array_types import iter_rois

    try:
        from rendercanvas.pyside6 import RenderCanvas
    except (ImportError, RuntimeError): # RuntimeError if qt is already selected
        RenderCanvas = None

    if RenderCanvas is not None:
        figure_kwargs = {
            "canvas": "pyside6",
            "canvas_kwargs": {"present_method": "bitmap"},
            "size": (800, 800)
        }
    else:
        figure_kwargs = {"size": (800, 800)}

    # Determine slider dimension names from array's dims property if available
    # otherwise fall back to defaults based on ndim
    from mbo_utilities.arrays.features import get_slider_dims

    slider_dim_names = get_slider_dims(data_array)
    ndim = data_array.ndim

    # window_funcs tuple must match slider_dim_names length
    if slider_dim_names:
        n_sliders = len(slider_dim_names)
        # apply mean to first dim (usually t), None for rest
        window_funcs = (np.mean,) + (None,) * (n_sliders - 1)
        window_sizes = (1,) + (None,) * (n_sliders - 1)
    else:
        window_funcs = None
        window_sizes = None

    # Handle multi-ROI data
    if hasattr(data_array, "rois"):
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
        for r in iter_rois(data_array):
            arr = copy.copy(data_array)
            arr.fix_phase = False
            arr.roi = r
            arrays.append(arr)
            names.append(f"ROI {r}" if r else (base_name or "Full Image"))

        iw = fpl.ImageWidget(
            data=arrays,
            names=names,
            slider_dim_names=slider_dim_names,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
            histogram_widget=True,
            figure_kwargs=figure_kwargs,
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )
    else:
        iw = fpl.ImageWidget(
            data=data_array,
            slider_dim_names=slider_dim_names,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
            histogram_widget=True,
            figure_kwargs=figure_kwargs,
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )

    iw.show()

    # set qt window icon after canvas is created
    from mbo_utilities.gui._setup import set_qt_icon
    set_qt_icon()

    # Add PreviewDataWidget if requested
    if widget:
        from mbo_utilities.gui.imgui import PreviewDataWidget

        gui = PreviewDataWidget(
            iw=iw,
            fpath=data_array.filenames,
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
    data_in: Optional[Union[str, Path]] = None,
    roi: Optional[Union[int, tuple[int, ...]]] = None,
    widget: bool = True,
    metadata_only: bool = False,
    select_only: bool = False,
    show_splash: bool = False,
):
    """Internal implementation of run_gui with all heavy imports."""
    # show splash screen while loading (only for desktop shortcut launches)
    splash = None
    if show_splash and not _is_jupyter():
        splash = SplashScreen()
        splash.show()

    try:
        # Import heavy dependencies only when actually running GUI
        from mbo_utilities.array_types import normalize_roi
        from mbo_utilities.gui import _setup  # triggers setup on import

        # close splash before showing file dialog
        if splash:
            splash.close()
            splash = None

        # Handle file selection if no path provided
        if data_in is None:
            data_in, roi_from_dialog, widget, metadata_only = _select_file()
            if not data_in:
                print("No file selected, exiting.")
                return None
            # Use ROI from dialog if not specified in function call
            if roi is None:
                roi = roi_from_dialog

        # If select_only, just return the path without loading data or opening viewer
        if select_only:
            return data_in

        # Normalize ROI to standard format
        roi = normalize_roi(roi)

        # Load data
        from mbo_utilities.lazy_array import imread
        data_array = imread(data_in, roi=roi)

        # Show metadata viewer if requested
        if metadata_only:
            metadata = data_array.metadata
            if not metadata:
                print("No metadata found.")
                return None
            _show_metadata_viewer(metadata)
            return None

        # Create and show image viewer
        import fastplotlib as fpl
        iw = _create_image_widget(data_array, widget=widget)

        # In Jupyter, just return the widget (user can interact immediately)
        # In standalone, run the event loop
        if _is_jupyter():
            return iw
        else:
            fpl.loop.run()
            return None

    finally:
        # ensure splash is closed on any exit path
        if splash:
            splash.close()


def run_gui(
    data_in: Optional[Union[str, Path]] = None,
    roi: Optional[Union[int, tuple[int, ...]]] = None,
    widget: bool = True,
    metadata_only: bool = False,
    select_only: bool = False,
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
