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
    >>> from mbo_utilities.graphics import download_notebook

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
    click.echo("Checking mbo_utilities installation...\n")

    # Core package check
    try:
        import mbo_utilities
        version = getattr(mbo_utilities, "__version__", "unknown")
        click.secho(f"mbo_utilities {version} installed", fg="green")
    except ImportError as e:
        click.secho(f"mbo_utilities import failed: {e}", fg="red")
        return False

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.secho(f"Python {py_version}", fg="green")

    # Check critical dependencies
    dependencies = {
        "imgui_bundle": "ImGui Bundle - required for GUI",
        "fastplotlib": "FastPlotLib - required for GUI visualization",
    }

    optional_dependencies = {
        "cupy": "CuPy - required for Suite3D z-registration",
        "torch": "PyTorch - required for Suite2p processing",
        "suite2p": "Suite2p - calcium imaging processing pipeline",
    }

    all_good = True
    click.echo("\nCore Dependencies:")
    for module, desc in dependencies.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "installed")
            click.secho(f"  {desc}: {ver}", fg="green")
        except ImportError:
            click.secho(f"  {desc}: not installed", fg="red")
            all_good = False

    click.echo("\nOptional Dependencies:")
    cupy_installed = False
    for module, desc in optional_dependencies.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "installed")
            click.secho(f"  {desc}: {ver}", fg="green")
            if module == "cupy":
                cupy_installed = True
        except ImportError:
            click.secho(f"  {desc}: not installed (optional)", fg="yellow")

    # Check CUDA/GPU configuration if CuPy is installed
    cuda_path = None
    suggested_cuda_path = None
    if cupy_installed:
        click.echo("\nGPU/CUDA Configuration:")
        cuda_path = os.environ.get("CUDA_PATH")

        # Try to find CUDA installation
        if not cuda_path:
            # Common CUDA installation paths
            if sys.platform == "win32":
                base_path = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
                if base_path.exists():
                    # Find the latest version
                    versions = sorted([d for d in base_path.iterdir() if d.is_dir()], reverse=True)
                    if versions:
                        suggested_cuda_path = str(versions[0])
            else:
                # Linux/Mac common paths
                for path in ["/usr/local/cuda", "/opt/cuda"]:
                    if Path(path).exists():
                        suggested_cuda_path = path
                        break

        if cuda_path:
            click.secho(f"  CUDA_PATH environment variable set: {cuda_path}", fg="green")
        else:
            click.secho("  CUDA_PATH environment variable not set", fg="yellow")
            if suggested_cuda_path:
                click.secho(f"  Found CUDA installation at: {suggested_cuda_path}", fg="cyan")
            all_good = False

        try:
            import cupy as cp
            # Try to create a simple array to test CUDA functionality
            test_array = cp.array([1, 2, 3])

            # Test NVRTC compilation (required for Suite3D)
            # This will fail if nvrtc64_*.dll is missing
            try:
                kernel = cp.ElementwiseKernel(
                    'float32 x', 'float32 y', 'y = x * 2', 'test_kernel'
                )
                # Actually execute the kernel to trigger compilation
                test_in = cp.array([1.0], dtype='float32')
                test_out = cp.empty_like(test_in)
                kernel(test_in, test_out)
                click.secho("  NVRTC (CUDA JIT compiler) working", fg="green")
            except Exception as nvrtc_err:
                click.secho("  NVRTC compilation failed", fg="red")
                click.echo(f"         Error: {str(nvrtc_err)[:100]}")
                click.echo("         Suite3D z-registration will NOT work without NVRTC.")
                click.echo("         Install CUDA Toolkit 12.0 runtime libraries to fix this.")
                all_good = False

            # Get CUDA runtime version
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            cuda_major = cuda_version // 1000
            cuda_minor = (cuda_version % 1000) // 10
            click.secho(f"  CUDA Runtime Version: {cuda_major}.{cuda_minor}", fg="green")

            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                device = cp.cuda.Device()
                device_name = device.attributes.get("Name", "Unknown")
                click.secho(f"  CUDA device available: {device_name} (Device {device.id})", fg="green")
                click.secho(f"  Total CUDA devices: {device_count}", fg="green")

                # Detect likely CUDA installation path from version
                if not suggested_cuda_path and sys.platform == "win32":
                    suggested_cuda_path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_major}.{cuda_minor}"
            else:
                click.secho("  No CUDA devices found", fg="red")
                all_good = False
        except Exception as e:
            click.secho(f"  CuPy CUDA initialization failed: {e}", fg="red")
            click.echo("         This will prevent GPU-accelerated operations like Suite3D z-registration.")
            click.echo("         Action required: Set CUDA_PATH environment variable to your CUDA installation directory.")
            all_good = False

    # Check for mbo directories
    click.echo("\nConfiguration:")
    try:
        from mbo_utilities.file_io import get_mbo_dirs
        dirs = get_mbo_dirs()
        click.secho(f"  Config directory: {dirs.get('base', 'unknown')}", fg="green")
    except Exception as e:
        click.secho(f"  Failed to get config directories: {e}", fg="red")
        all_good = False

    # Summary
    click.echo("\n" + "=" * 50)
    if all_good:
        click.secho("Installation check passed!", fg="green", bold=True)
        click.echo("\nYou can now:")
        click.echo("  - Run 'uv run mbo' to open the GUI")
        click.echo("  - Run 'uv run mbo /path/to/file' to directly open any supported file")
        click.echo("  - Run 'uv run mbo --download-notebook' to get the user guide")
        return True
    else:
        click.secho("Installation check failed!", fg="red", bold=True)
        click.echo("\nIssues detected:")
        if not cuda_path and cupy_installed:
            click.echo("  - CUDA_PATH not set: GPU operations (Suite3D z-registration) will fail")
            click.echo("    Fix: Set CUDA_PATH environment variable to your CUDA installation")
            if suggested_cuda_path:
                if sys.platform == "win32":
                    click.secho("\n    Run this command (then restart terminal):", fg="cyan")
                    click.secho(f"      setx CUDA_PATH \"{suggested_cuda_path}\"", fg="cyan", bold=True)
                    click.echo("\n    Or set for current session only:")
                    click.secho(f"      $env:CUDA_PATH = \"{suggested_cuda_path}\"", fg="cyan")
                else:
                    click.echo("\n    Add this to your ~/.bashrc or ~/.zshrc:")
                    click.secho(f"      export CUDA_PATH={suggested_cuda_path}", fg="cyan", bold=True)
                    click.echo("\n    Or set for current session only:")
                    click.secho(f"      export CUDA_PATH={suggested_cuda_path}", fg="cyan")
            else:
                click.echo("    Example (Windows): setx CUDA_PATH \"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\"")
                click.echo("    Example (Linux/Mac): export CUDA_PATH=/usr/local/cuda")
        click.echo("\nFor other issues, try reinstalling with: uv sync")
        return False


def _setup_qt_backend():
    """Set up Qt backend for rendercanvas if PySide6 is available.

    This must happen BEFORE importing fastplotlib to avoid glfw selection.
    """
    if importlib.util.find_spec("PySide6") is not None:
        os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
        import PySide6  # noqa: F401 - Must be imported before rendercanvas.qt can load


def _select_file() -> tuple[Any, Any, Any, bool]:
    """Show file selection dialog and return user choices."""
    from mbo_utilities.file_io import get_mbo_dirs
    from mbo_utilities.graphics._file_dialog import FileDialog, setup_imgui
    from imgui_bundle import immapp, hello_imgui

    setup_imgui()  # ensure assets (fonts + icons) are available
    dlg = FileDialog()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Utilities – Data Selection"
    params.app_window_params.window_geometry.size = (420, 480)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = True
    params.ini_filename = str(
        Path(get_mbo_dirs()["settings"], "fd_settings.ini").expanduser()
    )
    params.callbacks.show_gui = dlg.render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    hello_imgui.set_assets_folder(str(get_mbo_dirs()["assets"]))
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
    from mbo_utilities.graphics._widgets import draw_metadata_inspector

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Metadata Viewer"
    params.app_window_params.window_geometry.size = (800, 800)
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

    # Determine slider dimension names and window functions based on data dimensionality
    # MBO data is typically TZYX (4D) or TYX (3D)
    # window_funcs tuple must match slider_dim_names: (t_func, z_func) for 4D, (t_func,) for 3D
    ndim = data_array.ndim
    if ndim == 4:
        slider_dim_names = ("t", "z")
        # Apply mean to t-dim only, None for z-dim
        window_funcs = (np.mean, None)
        window_sizes = (1, None)
    elif ndim == 3:
        slider_dim_names = ("t",)
        window_funcs = (np.mean,)
        window_sizes = (1,)
    else:
        slider_dim_names = None
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
            base_name = Path(data_array.filenames[0]).stem
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

    # Add PreviewDataWidget if requested
    if widget:
        from mbo_utilities.graphics.imgui import PreviewDataWidget

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
):
    """Internal implementation of run_gui with all heavy imports."""
    # Set up Qt backend before any GUI imports
    _setup_qt_backend()

    # Import heavy dependencies only when actually running GUI
    from mbo_utilities.array_types import normalize_roi
    from mbo_utilities.graphics._file_dialog import setup_imgui

    setup_imgui()  # ensure assets (fonts + icons) are available

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
    >>> from mbo_utilities.graphics import run_gui
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
@click.argument("data_in", required=False)
def _cli_entry(data_in=None, widget=None, roi=None, metadata_only=False, select_only=False, download_notebook=False, notebook_url=None, check_install=False):
    """CLI entry point for mbo-gui command."""
    # Handle installation check first (light operation)
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
    result = run_gui(
        data_in=data_in,
        roi=roi if roi else None,
        widget=widget,
        metadata_only=metadata_only,
        select_only=select_only,
    )

    # If select_only, print the selected path
    if select_only and result:
        click.echo(result)


if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
