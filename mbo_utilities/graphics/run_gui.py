import copy
import sys
import webbrowser
from pathlib import Path
from typing import Any, Optional, Union
import click
import numpy as np
from mbo_utilities.array_types import iter_rois, normalize_roi
from mbo_utilities.graphics._file_dialog import FileDialog, setup_imgui


def _check_installation():
    """Verify that mbo_utilities and key dependencies are properly installed."""
    click.echo("Checking mbo_utilities installation...\n")

    # Core package check
    try:
        import mbo_utilities
        version = getattr(mbo_utilities, "__version__", "unknown")
        click.secho(f"[OK] mbo_utilities {version} installed", fg="green")
    except ImportError as e:
        click.secho(f"[FAIL] mbo_utilities import failed: {e}", fg="red")
        return False

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.secho(f"[OK] Python {py_version}", fg="green")

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
            click.secho(f"  [OK] {desc}: {ver}", fg="green")
        except ImportError:
            click.secho(f"  [FAIL] {desc}: not installed", fg="red")
            all_good = False

    click.echo("\nOptional Dependencies:")
    cupy_installed = False
    for module, desc in optional_dependencies.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "installed")
            click.secho(f"  [OK] {desc}: {ver}", fg="green")
            if module == "cupy":
                cupy_installed = True
        except ImportError:
            click.secho(f"  [SKIP] {desc}: not installed (optional)", fg="yellow")

    # Check CUDA/GPU configuration if CuPy is installed
    cuda_path = None
    suggested_cuda_path = None
    if cupy_installed:
        click.echo("\nGPU/CUDA Configuration:")
        import os
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
            click.secho(f"  [OK] CUDA_PATH environment variable set: {cuda_path}", fg="green")
        else:
            click.secho(f"  [WARN] CUDA_PATH environment variable not set", fg="yellow")
            if suggested_cuda_path:
                click.secho(f"  [INFO] Found CUDA installation at: {suggested_cuda_path}", fg="cyan")
            all_good = False

        try:
            import cupy as cp
            # Try to create a simple array to test CUDA functionality
            test_array = cp.array([1, 2, 3])

            # Get CUDA runtime version
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            cuda_major = cuda_version // 1000
            cuda_minor = (cuda_version % 1000) // 10
            click.secho(f"  [OK] CUDA Runtime Version: {cuda_major}.{cuda_minor}", fg="green")

            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                device = cp.cuda.Device()
                device_name = device.attributes.get("Name", "Unknown")
                click.secho(f"  [OK] CUDA device available: {device_name} (Device {device.id})", fg="green")
                click.secho(f"  [OK] Total CUDA devices: {device_count}", fg="green")

                # Detect likely CUDA installation path from version
                if not suggested_cuda_path and sys.platform == "win32":
                    suggested_cuda_path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_major}.{cuda_minor}"
            else:
                click.secho(f"  [FAIL] No CUDA devices found", fg="red")
                all_good = False
        except Exception as e:
            click.secho(f"  [FAIL] CuPy CUDA initialization failed: {e}", fg="red")
            click.echo("         This will prevent GPU-accelerated operations like Suite3D z-registration.")
            click.echo("         Action required: Set CUDA_PATH environment variable to your CUDA installation directory.")
            all_good = False

    # Check for mbo directories
    click.echo("\nConfiguration:")
    try:
        from mbo_utilities.file_io import get_mbo_dirs
        dirs = get_mbo_dirs()
        click.secho(f"  [OK] Config directory: {dirs.get('base', 'unknown')}", fg="green")
    except Exception as e:
        click.secho(f"  [FAIL] Failed to get config directories: {e}", fg="red")
        all_good = False

    # Summary
    click.echo("\n" + "=" * 50)
    if all_good:
        click.secho("Installation check passed!", fg="green", bold=True)
        click.echo("\nYou can now:")
        click.echo("  - Run 'uv run mbo' to open the GUI")
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
                    click.secho(f"\n    Run this command (then restart terminal):", fg="cyan")
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


def _select_file() -> tuple[Any, Any, Any, bool]:
    """Show file selection dialog and return user choices."""
    from mbo_utilities.file_io import get_mbo_dirs
    from imgui_bundle import immapp, hello_imgui

    dlg = FileDialog()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Utilities – Data Selection"
    params.app_window_params.window_geometry.size = (1400, 950)
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
    import fastplotlib as fpl

    # Handle multi-ROI data
    if hasattr(data_array, "rois"):
        arrays = []
        names = []
        for r in iter_rois(data_array):
            arr = copy.copy(data_array)
            arr.fix_phase = False
            arr.roi = r
            arrays.append(arr)
            names.append(f"ROI {r}" if r else "Full Image")

        iw = fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=True,
            figure_kwargs={"size": (800, 800)},
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )
    else:
        iw = fpl.ImageWidget(
            data=data_array,
            histogram_widget=True,
            figure_kwargs={"size": (800, 800)},
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
):
    """Internal implementation of GUI launcher."""
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

    Returns
    -------
    ImageWidget or None
        In Jupyter: returns the ImageWidget (already shown via iw.show()).
        In standalone: returns None (runs event loop until closed).

    Examples
    --------
    From Python/Jupyter:
    >>> from mbo_utilities.graphics import run_gui
    >>> # Option 1: Just show the GUI
    >>> run_gui("path/to/data.tif")
    >>> # Option 2: Get reference to manipulate it
    >>> iw = run_gui("path/to/data.tif", roi=1, widget=False)
    >>> iw.cmap = "viridis"  # Change colormap

    From command line:
    $ mbo path/to/data.tif
    $ mbo path/to/data.tif --roi 1 --no-widget
    $ mbo path/to/data.tif --metadata-only
    """
    return _run_gui_impl(
        data_in=data_in,
        roi=roi,
        widget=widget,
        metadata_only=metadata_only,
    )


# Create CLI wrapper
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
    "--download-notebook",
    is_flag=True,
    help="Download the user guide Jupyter notebook and exit.",
)
@click.option(
    "--check-install",
    is_flag=True,
    help="Verify the installation of mbo_utilities and dependencies.",
)
@click.argument("data_in", required=False)
def _cli_entry(data_in=None, widget=None, roi=None, metadata_only=False, download_notebook=False, check_install=False):
    """CLI entry point for mbo-gui command."""
    # Handle installation check first
    if check_install:
        _check_installation()
        if download_notebook:
            click.echo("\n")
            url = "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/demos/user_guide.ipynb"
            click.echo(f"Opening browser to download user guide notebook from: {url}")
            webbrowser.open(url)
        return

    # Handle download notebook option
    if download_notebook:
        url = "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/demos/user_guide.ipynb"
        click.echo(f"Opening browser to download user guide notebook from: {url}")
        webbrowser.open(url)
        return

    # Run the GUI
    run_gui(
        data_in=data_in,
        roi=roi if roi else None,
        widget=widget,
        metadata_only=metadata_only,
    )


if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
