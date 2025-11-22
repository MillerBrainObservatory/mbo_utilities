import copy
import sys
from pathlib import Path
from typing import Any, Optional, Union
import urllib.request
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
            click.secho(f"  CUDA_PATH environment variable set: {cuda_path}", fg="green")
        else:
            click.secho(f"  CUDA_PATH environment variable not set", fg="yellow")
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
                click.secho(f"  NVRTC (CUDA JIT compiler) working", fg="green")
            except Exception as nvrtc_err:
                click.secho(f"  NVRTC compilation failed", fg="red")
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
                click.secho(f"  No CUDA devices found", fg="red")
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


def _download_notebook_file(output_path: Optional[Union[str, Path]] = None):
    """Download the user guide notebook to a local file.

    Parameters
    ----------
    output_path : str, Path, optional
        Directory or file path to save the notebook. If None or '.', saves to current directory.
        If a directory, saves as 'user_guide.ipynb' in that directory.
        If a file path, uses that exact filename.
    """
    url = "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/demos/user_guide.ipynb"

    # Determine output file path
    if output_path is None or output_path == ".":
        output_file = Path.cwd() / "user_guide.ipynb"
    else:
        output_file = Path(output_path)
        if output_file.is_dir():
            output_file = output_file / "user_guide.ipynb"

    try:
        click.echo(f"Downloading user guide notebook from:\n  {url}")
        click.echo(f"Saving to:\n  {output_file.resolve()}")

        # Download the file
        urllib.request.urlretrieve(url, output_file)

        click.secho(f"\n✓ Successfully downloaded notebook to: {output_file.resolve()}", fg="green")
        click.echo("\nTo use the notebook:")
        click.echo(f"  jupyter lab {output_file.resolve()}")

    except Exception as e:
        click.secho(f"\n✗ Failed to download notebook: {e}", fg="red")
        click.echo(f"\nYou can manually download from: {url}")
        sys.exit(1)


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
    from mbo_utilities.graphics._processors import MboImageProcessor

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
        processors = []
        for r in iter_rois(data_array):
            arr = copy.copy(data_array)
            arr.fix_phase = False
            arr.roi = r
            arrays.append(arr)
            names.append(f"ROI {r}" if r else "Full Image")
            processors.append(MboImageProcessor)

        iw = fpl.ImageWidget(
            data=arrays,
            processors=processors,
            names=names,
            slider_dim_names=slider_dim_names,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
            histogram_widget=True,
            figure_kwargs={"size": (800, 800)},
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )
    else:
        iw = fpl.ImageWidget(
            data=data_array,
            processors=MboImageProcessor,
            slider_dim_names=slider_dim_names,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
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
            _download_notebook_file(data_in)
        return

    # Handle download notebook option
    if download_notebook:
        _download_notebook_file(data_in)
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
