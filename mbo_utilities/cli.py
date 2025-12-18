"""
CLI entry point for mbo_utilities.

This module handles command-line operations with minimal imports.
GUI-related imports are deferred until actually needed.

Usage patterns:
  mbo                           # Open GUI with file dialog
  mbo /path/to/data             # Open GUI with specific file
  mbo /path/to/data --metadata  # Show only metadata
  mbo convert INPUT OUTPUT      # Convert with CLI args
  mbo info INPUT                # Show array info (CLI only)
"""
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Union

import click


class PathAwareGroup(click.Group):
    """Custom click Group that routes file paths to the 'view' command.

    This allows `mbo /path/to/data` to work the same as `mbo view /path/to/data`.
    """

    def resolve_command(self, ctx, args):
        """Override to check if first arg looks like a path instead of a command."""
        if args:
            first_arg = args[0]
            # First check if it's a known command
            if first_arg in self.commands:
                return super().resolve_command(ctx, args)

            # Not a known command - check if it looks like a file path
            # (contains path separators, has file extension, or exists on disk)
            if (
                "/" in first_arg
                or "\\" in first_arg
                or "." in first_arg
                or Path(first_arg).exists()
            ):
                # Route to 'view' command with this path as argument
                view_cmd = self.commands.get("view")
                if view_cmd:
                    return "view", view_cmd, args

        return super().resolve_command(ctx, args)


def download_file(
    url: str,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Download a file from a URL to a local path.

    Parameters
    ----------
    url : str
        URL to the file. Supports GitHub blob URLs (automatically converted to raw URLs).
    output_path : str, Path, optional
        Directory or file path to save the file. If None or '.', saves to current directory.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    import urllib.request

    # Convert GitHub blob URLs to raw URLs
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    # Extract filename from URL
    url_filename = url.split("/")[-1]
    if "?" in url_filename:
        url_filename = url_filename.split("?")[0]

    # Determine output file path
    if output_path is None or output_path == ".":
        output_file = Path.cwd() / url_filename
    else:
        output_file = Path(output_path).expanduser().resolve()
        if output_file.is_dir() or (not output_file.suffix and not output_file.exists()):
            output_file.mkdir(parents=True, exist_ok=True)
            output_file = output_file / url_filename

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        click.echo(f"Downloading from:\n  {url}")
        click.echo(f"Saving to:\n  {output_file.resolve()}")
        urllib.request.urlretrieve(url, output_file)
        click.secho(f"\nSuccessfully downloaded: {output_file.resolve()}", fg="green")
    except Exception as e:
        click.secho(f"\nFailed to download: {e}", fg="red")
        click.echo(f"\nYou can manually download from: {url}")
        sys.exit(1)

    return output_file


def download_notebook(
    output_path: Optional[Union[str, Path]] = None,
    notebook_url: Optional[str] = None,
) -> Path:
    """Download a Jupyter notebook from a URL to a local file."""
    default_url = "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/demos/user_guide.ipynb"
    url = notebook_url or default_url
    output_file = download_file(url, output_path)
    click.echo("\nTo use the notebook:")
    click.echo(f"  jupyter lab {output_file.resolve()}")
    return output_file


def _get_marker_path() -> Path:
    """get path to first-run marker file"""
    from mbo_utilities import get_mbo_dirs
    return get_mbo_dirs()["base"] / ".initialized"


def _is_first_run() -> bool:
    """check if this is the first run (no marker file exists)"""
    try:
        return not _get_marker_path().exists()
    except Exception:
        return False


def _mark_initialized() -> None:
    """create marker file to indicate successful initialization"""
    try:
        marker = _get_marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except Exception:
        pass


class LoadingSpinner:
    """simple terminal spinner for loading feedback"""

    def __init__(self, message: str = "Loading"):
        self.message = message
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def _spin(self):
        # ascii spinner for windows compatibility (cp1252 can't encode unicode spinners)
        chars = "|/-\\"
        i = 0
        while self._running:
            sys.stdout.write(f"\r{chars[i % len(chars)]} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1


@click.group(cls=PathAwareGroup, invoke_without_command=True)
@click.option(
    "--download-notebook",
    is_flag=True,
    help="Download the user guide notebook and exit.",
)
@click.option(
    "--notebook-url",
    type=str,
    default=None,
    help="URL of notebook to download.",
)
@click.option(
    "--download-file",
    "download_file_url",
    type=str,
    default=None,
    help="Download a file from URL (e.g. GitHub).",
)
@click.option(
    "-o", "--output",
    "output_path",
    type=str,
    default=None,
    help="Output path for --download-file or --download-notebook.",
)
@click.option(
    "--check-install",
    is_flag=True,
    help="Verify the installation of mbo_utilities and dependencies.",
)
@click.pass_context
def main(
    ctx,
    download_notebook=False,
    notebook_url=None,
    download_file_url=None,
    output_path=None,
    check_install=False,
):
    """
    MBO Utilities CLI - data preview and processing tools.

    \b
    GUI Mode:
      mbo                            Open file selection dialog
      mbo /path/to/data              Open specific file in GUI
      mbo /path/to/data --metadata   Show only metadata

    \b
    Commands:
      mbo convert INPUT OUTPUT       Convert between formats
      mbo info INPUT                 Show array information (CLI)
      mbo download URL               Download file from GitHub
      mbo formats                    List supported formats

    \b
    Utilities:
      mbo --download-notebook             Download user guide notebook
      mbo --check-install                 Verify installation
    """
    if download_file_url:
        download_file(download_file_url, output_path)
        return

    if download_notebook:
        download_notebook_func = globals()["download_notebook"]
        download_notebook_func(output_path=output_path, notebook_url=notebook_url)
        return

    if check_install:
        from mbo_utilities.graphics.run_gui import _check_installation
        _check_installation()
        return

    # If a subcommand is invoked, skip main logic
    if ctx.invoked_subcommand is not None:
        return

    # show first-run warning
    first_run = _is_first_run()
    if first_run:
        click.secho("First run detected - initial startup may take longer while caches are built.", fg="yellow")

    # show loading spinner while importing heavy dependencies
    spinner = LoadingSpinner("Loading GUI")
    spinner.start()
    try:
        from mbo_utilities.graphics.run_gui import run_gui
        spinner.stop()
    except Exception as e:
        spinner.stop()
        raise e

    # mark as initialized after successful import
    if first_run:
        _mark_initialized()

    run_gui(data_in=None, roi=None, widget=True, metadata_only=False)


@main.command()
@click.argument("data_in", required=False, type=click.Path())
@click.option(
    "--roi",
    multiple=True,
    type=int,
    help="ROI index (can pass multiple: --roi 0 --roi 2).",
)
@click.option(
    "--widget/--no-widget",
    default=True,
    help="Enable/disable PreviewDataWidget for Raw ScanImage tiffs.",
)
@click.option(
    "--metadata",
    is_flag=True,
    help="Show only metadata (no image viewer).",
)
def view(data_in=None, roi=None, widget=True, metadata=False):
    """
    Open imaging data in the GUI viewer.

    \b
    Examples:
      mbo view                       Open file selection dialog
      mbo view /data/raw.tiff        Open specific file
      mbo view /data/raw --metadata  Show only metadata
      mbo view /data --roi 0 --roi 2 View specific ROIs
    """
    # show first-run warning
    first_run = _is_first_run()
    if first_run:
        click.secho("First run detected - initial startup may take longer while caches are built.", fg="yellow")

    # show loading spinner while importing
    spinner = LoadingSpinner("Loading GUI")
    spinner.start()
    try:
        from mbo_utilities.graphics.run_gui import run_gui
        spinner.stop()
    except Exception as e:
        spinner.stop()
        raise e

    if first_run:
        _mark_initialized()

    run_gui(
        data_in=data_in,
        roi=roi if roi else None,
        widget=widget,
        metadata_only=metadata,
    )


@main.command()
@click.argument("input_path", required=False, type=click.Path())
@click.argument("output_path", required=False, type=click.Path())
@click.option(
    "-e", "--ext",
    type=click.Choice([".tiff", ".tif", ".zarr", ".bin", ".h5", ".npy"], case_sensitive=False),
    default=None,
    help="Output format extension.",
)
@click.option(
    "-p", "--planes",
    multiple=True,
    type=int,
    help="Z-planes to export (1-based): -p 1 -p 7 -p 14",
)
@click.option(
    "-n", "--num-frames",
    type=int,
    default=None,
    help="Number of frames to export.",
)
@click.option(
    "--roi",
    type=str,
    default=None,
    help="ROI: None=stitch, 0=split, N=specific, '1,3'=multiple.",
)
@click.option(
    "--register-z/--no-register-z",
    default=False,
    help="Z-plane registration using Suite3D.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing output files.",
)
@click.option(
    "--chunk-mb",
    type=int,
    default=100,
    help="Chunk size in MB for streaming writes.",
)
@click.option(
    "--fix-phase/--no-fix-phase",
    default=None,
    help="Bidirectional phase correction (MboRawArray).",
)
@click.option(
    "--phasecorr-method",
    type=click.Choice(["mean", "median", "max"]),
    default=None,
    help="Phase correction method.",
)
@click.option(
    "--ome/--no-ome",
    default=True,
    help="Write OME-Zarr metadata (zarr only).",
)
@click.option(
    "--output-name",
    type=str,
    default=None,
    help="Output filename for binary format.",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Verbose debug logging.",
)
def convert(
    input_path,
    output_path,
    ext,
    planes,
    num_frames,
    roi,
    register_z,
    overwrite,
    chunk_mb,
    fix_phase,
    phasecorr_method,
    ome,
    output_name,
    debug,
):
    """
    Convert imaging data between formats.

    If INPUT_PATH and OUTPUT_PATH are provided, runs conversion directly.
    If omitted, opens a GUI for interactive conversion.

    \b
    Examples:
      mbo convert                                    # Open conversion GUI
      mbo convert /data/raw output/ -e .zarr        # Convert to Zarr
      mbo convert /data/raw output/ -e .npy -p 1 -p 7   # Export planes as NPY
      mbo convert /data/raw output/ --fix-phase     # With phase correction
    """
    # If no input provided, could open a conversion GUI in the future
    if input_path is None:
        click.echo("Conversion GUI not yet implemented. Please provide INPUT_PATH and OUTPUT_PATH.")
        click.echo("\nUsage: mbo convert INPUT_PATH OUTPUT_PATH [OPTIONS]")
        click.echo("\nRun 'mbo convert --help' for all options.")
        return

    if output_path is None:
        click.secho("Error: OUTPUT_PATH is required when INPUT_PATH is provided.", fg="red")
        return

    from mbo_utilities import imread, imwrite

    # Parse ROI argument
    parsed_roi = None
    if roi is not None:
        roi = roi.strip()
        if roi.lower() == "none":
            parsed_roi = None
        elif "," in roi:
            parsed_roi = [int(x.strip()) for x in roi.split(",")]
        else:
            parsed_roi = int(roi)

    parsed_planes = list(planes) if planes else None

    click.echo(f"Reading: {input_path}")

    # Build imread kwargs
    imread_kwargs = {}
    if fix_phase is not None:
        imread_kwargs["fix_phase"] = fix_phase
    if phasecorr_method:
        imread_kwargs["phasecorr_method"] = phasecorr_method
    if parsed_roi is not None and parsed_roi != 0:
        imread_kwargs["roi"] = parsed_roi

    # Read data
    data = imread(input_path, **imread_kwargs)
    click.echo(f"  Shape: {data.shape}, dtype: {data.dtype}")

    # Configure array-specific options
    if hasattr(data, "fix_phase") and fix_phase is not None:
        data.fix_phase = fix_phase
    if hasattr(data, "phasecorr_method") and phasecorr_method:
        data.phasecorr_method = phasecorr_method

    # Determine output extension
    output_ext = ext or ".tiff"
    click.echo(f"Writing: {output_path} (format: {output_ext})")

    # Build imwrite kwargs
    imwrite_kwargs = {
        "ext": output_ext,
        "overwrite": overwrite,
        "target_chunk_mb": chunk_mb,
        "debug": debug,
    }

    if parsed_planes:
        imwrite_kwargs["planes"] = parsed_planes
    if num_frames:
        imwrite_kwargs["num_frames"] = num_frames
    if parsed_roi is not None:
        imwrite_kwargs["roi"] = parsed_roi
    if register_z:
        imwrite_kwargs["register_z"] = True
    if output_ext.lower() == ".zarr":
        imwrite_kwargs["ome"] = ome
    if output_name:
        imwrite_kwargs["output_name"] = output_name

    result = imwrite(data, output_path, **imwrite_kwargs)
    click.secho(f"\nDone! Output saved to: {result}", fg="green")

@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--metadata/--no-metadata",
    default=True,
    help="Show metadata.",
)
def info(input_path, metadata):
    """
    Show information about an imaging dataset.

    \b
    Examples:
      mbo info /data/raw.tiff
      mbo info /data/volume.zarr
      mbo info /data/suite2p/plane0
    """
    from mbo_utilities import imread

    click.echo(f"Loading: {input_path}")
    data = imread(input_path)

    click.echo(f"\nArray Information:")
    click.echo(f"  Type:  {type(data).__name__}")
    click.echo(f"  Shape: {data.shape}")
    click.echo(f"  Dtype: {data.dtype}")
    click.echo(f"  Ndim:  {data.ndim}")

    if hasattr(data, "filenames"):
        click.echo(f"  Files: {len(data.filenames)}")
        if len(data.filenames) <= 5:
            for f in data.filenames:
                click.echo(f"    - {f}")
        else:
            for f in data.filenames[:3]:
                click.echo(f"    - {f}")
            click.echo(f"    ... and {len(data.filenames) - 3} more")

    if hasattr(data, "min") and hasattr(data, "max"):
        try:
            click.echo(f"  Min:   {data.min:.4f}")
            click.echo(f"  Max:   {data.max:.4f}")
        except Exception:
            pass

    if metadata and hasattr(data, "metadata"):
        md = data.metadata
        if md:
            click.echo(f"\nMetadata:")
            important_keys = ["nframes", "num_frames", "Ly", "Lx", "fs", "num_rois", "plane"]
            for key in important_keys:
                if key in md:
                    click.echo(f"  {key}: {md[key]}")
            other_keys = [k for k in md.keys() if k not in important_keys]
            if other_keys:
                click.echo(f"  ... and {len(other_keys)} more keys")


@main.command()
@click.argument("url", type=str)
@click.option(
    "-o", "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output directory or file path. Default: current directory.",
)
def download(url, output_path):
    """
    Download a file from a URL (supports GitHub).

    \b
    Examples:
      mbo download https://github.com/user/repo/blob/main/notebook.ipynb
      mbo download https://github.com/user/repo/blob/main/data.npy -o ./data/
    """
    download_file(url, output_path)


@main.command("formats")
def list_formats():
    """List supported file formats."""
    click.echo("Supported input formats:")
    click.echo("  .tif, .tiff  - TIFF files (BigTIFF, OME-TIFF, ScanImage)")
    click.echo("  .zarr        - Zarr v3 arrays")
    click.echo("  .bin         - Suite2p binary format (with ops.npy)")
    click.echo("  .h5, .hdf5   - HDF5 files")
    click.echo("  .npy         - NumPy arrays")
    click.echo("  .json        - Zarr array metadata (loads parent .zarr)")

    click.echo("\nSupported output formats:")
    click.echo("  .tiff        - Multi-page BigTIFF")
    click.echo("  .zarr        - Zarr v3 with optional OME-NGFF metadata")
    click.echo("  .bin         - Suite2p binary format")
    click.echo("  .h5          - HDF5 format")
    click.echo("  .npy         - NumPy array")


@main.command("scanphase")
@click.argument("input_path", required=False, type=click.Path())
@click.option(
    "-o", "--output",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Output directory for results. Default: <input>_scanphase_analysis/",
)
@click.option(
    "-n", "--num-tifs",
    "num_tifs",
    type=int,
    default=None,
    help="If input is a folder, only use the first N tiff files.",
)
@click.option(
    "--format",
    "image_format",
    type=click.Choice(["png", "pdf", "svg", "tiff"]),
    default="png",
    help="Output image format.",
)
@click.option(
    "--show/--no-show",
    default=False,
    help="Display plots interactively after analysis.",
)
def scanphase(input_path, output_dir, num_tifs, image_format, show):
    """
    Scan-phase analysis for bidirectional scanning data.

    Analyzes phase offset to determine optimal correction parameters.

    \b
    OUTPUT:
      temporal.png           - per-frame offset time series and histogram
      window_convergence.png - offset vs window size (key diagnostic)
      spatial.png            - spatial variation across FOV
      scanphase_results.npz  - all numerical data

    \b
    Examples:
      mbo scanphase                          # open file dialog
      mbo scanphase /path/to/data.tiff       # analyze specific file
      mbo scanphase ./folder/ -n 5           # use first 5 tiffs in folder
      mbo scanphase data.tiff -o ./results/  # custom output directory
      mbo scanphase data.tiff --show         # show plots interactively
    """
    from pathlib import Path
    from mbo_utilities import get_files
    from mbo_utilities.analysis.scanphase import run_scanphase_analysis

    try:
        # handle num_tifs for folder input
        actual_input = input_path
        if input_path is not None:
            input_path_obj = Path(input_path)
            if input_path_obj.is_dir() and num_tifs is not None:
                tiffs = get_files(input_path, str_contains=".tif", max_depth=1)
                if not tiffs:
                    click.secho(f"No tiff files found in {input_path}", fg="red")
                    raise click.Abort()
                tiffs = tiffs[:num_tifs]
                click.echo(f"Using {len(tiffs)} tiff files from {input_path}")
                actual_input = tiffs

        # determine output directory for display
        if input_path is not None:
            input_path_obj = Path(input_path)
            actual_output_dir = output_dir if output_dir else input_path_obj.parent / f"{input_path_obj.stem}_scanphase_analysis"
        else:
            actual_output_dir = output_dir

        results = run_scanphase_analysis(
            data_path=actual_input,
            output_dir=output_dir,
            image_format=image_format,
            show_plots=show,
        )

        if results is None:
            return  # user cancelled file selection

        # print summary
        summary = results.get_summary()
        meta = summary.get('metadata', {})

        click.echo("")
        click.secho("scan-phase analysis complete", fg="cyan", bold=True)
        click.echo("")
        click.echo(f"data: {meta.get('num_frames', 0)} frames, "
                   f"{meta.get('num_rois', 1)} ROIs, "
                   f"{meta.get('frame_shape', (0, 0))[1]}x{meta.get('frame_shape', (0, 0))[0]} px")
        click.echo(f"analysis time: {meta.get('analysis_time', 0):.1f}s")
        click.echo(f"output: {actual_output_dir}")

        # fft stats
        if 'fft' in summary:
            stats = summary['fft']
            click.echo("")
            click.secho("offset (FFT)", fg="yellow", bold=True)
            click.echo(f"  mean:   {stats.get('mean', 0):+.3f} px")
            click.echo(f"  median: {stats.get('median', 0):+.3f} px")
            click.echo(f"  std:    {stats.get('std', 0):.3f} px")
            click.echo(f"  range:  [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}] px")

        # int stats
        if 'int' in summary:
            stats = summary['int']
            click.echo("")
            click.secho("offset (integer)", fg="yellow", bold=True)
            click.echo(f"  mean:   {stats.get('mean', 0):+.3f} px")
            click.echo(f"  std:    {stats.get('std', 0):.3f} px")

        click.echo("")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        raise click.Abort()


if __name__ == "__main__":
    main()
