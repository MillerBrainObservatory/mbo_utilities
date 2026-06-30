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

# set windows appusermodelid immediately for taskbar icon grouping
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mbo.utilities.gui.1.0")
    except Exception:
        pass

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


def _get_marker_path() -> Path:
    """Get path to first-run marker file."""
    from mbo_utilities import get_mbo_dirs
    return get_mbo_dirs()["base"] / ".initialized"


def _is_first_run() -> bool:
    """Check if this is the first run (no marker file exists)."""
    try:
        return not _get_marker_path().exists()
    except Exception:
        return False


def _mark_initialized() -> None:
    """Create marker file to indicate successful initialization."""
    try:
        marker = _get_marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except Exception:
        pass


class LoadingSpinner:
    """simple terminal spinner for loading feedback."""

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


def _get_version_info() -> str:
    """get version string with install location info (cached)."""
    from mbo_utilities import __version__

    # try cache first for install type
    install_type = None
    try:
        from mbo_utilities.env_cache import get_cached_install_type
        install_type = get_cached_install_type()
    except Exception:
        pass

    if not install_type:
        # compute install type from executable path
        exe_str = sys.executable.lower()
        if ".local" in exe_str or ("uv" in exe_str and "tools" in exe_str):
            install_type = "uv tool"
        elif "envs" in exe_str or "venv" in exe_str or ".venv" in exe_str:
            install_type = "environment"
        elif "conda" in exe_str or "miniconda" in exe_str or "anaconda" in exe_str:
            install_type = "conda"
        else:
            install_type = "system"

    return f"mbo_utilities {__version__}\nPython: {sys.executable}\nInstall: {install_type}"


def _version_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Custom version callback to show extended version info."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(_get_version_info())
    ctx.exit()


@click.group(cls=PathAwareGroup, invoke_without_command=True)
@click.option(
    "-V", "--version",
    is_flag=True,
    callback=_version_callback,
    expose_value=False,
    is_eager=True,
    help="Show version and installation info.",
)
@click.option(
    "--check-install",
    is_flag=True,
    help="Verify the installation of mbo_utilities and dependencies.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Bypass environment cache (forces fresh checks).",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear environment cache and exit.",
)
@click.option(
    "--gpu",
    "gpu_index_arg",
    type=int,
    default=None,
    help="Set and persist GPU adapter index. `mbo --gpu N` saves + exits; "
         "`mbo --gpu N <path>` saves then opens. See `mbo view --list-gpus`.",
)
@click.pass_context
def main(
    ctx,
    check_install=False,
    no_cache=False,
    clear_cache=False,
    gpu_index_arg=None,
):
    """
    MBO Utilities - preview and process imaging data.

    \b
    Open the viewer:
      mbo                   file selection dialog
      mbo PATH              open a file or folder
      mbo PATH --metadata   print metadata only

    \b
    Common commands:
      mbo init [PATH]       create starter notebooks
      mbo convert IN OUT    convert between formats
      mbo info PATH         print array info
      mbo formats           list supported formats
      mbo shortcut          create a desktop icon

    \b
    Verify install:
      mbo --check-install
    """
    # handle --clear-cache early
    if clear_cache:
        from mbo_utilities.env_cache import clear_cache as do_clear, get_cache_path
        path = get_cache_path()
        if do_clear():
            click.secho(f"Cache cleared: {path}", fg="green")
        else:
            click.secho("No cache to clear.", fg="yellow")
        return

    if check_install:
        # force full cache rebuild including install status
        from mbo_utilities.env_cache import build_full_cache_with_install_status, save_cache
        cache = build_full_cache_with_install_status()
        save_cache(cache)
        from mbo_utilities.gui.run_gui import _check_installation
        _check_installation()
        return

    if gpu_index_arg is not None:
        import fastplotlib as fpl
        adapters = fpl.enumerate_adapters()
        if not 0 <= gpu_index_arg < len(adapters):
            raise click.BadParameter(
                f"--gpu {gpu_index_arg} out of range; found {len(adapters)} "
                f"adapter(s). Run `mbo view --list-gpus` to see them.",
                param_hint="--gpu",
            )
        from mbo_utilities.preferences import set_gpu_index
        set_gpu_index(gpu_index_arg)
        info = getattr(adapters[gpu_index_arg], "info", {}) or {}
        click.echo(
            f"Saved GPU {gpu_index_arg}: "
            f"{info.get('device', info.get('description', '?'))} "
            f"(persisted to ~/.mbo/settings/preferences.json)"
        )
        # run_gui.py reads the persisted index on launch, so a path that
        # routes to `view` will pick this up automatically.
        if ctx.invoked_subcommand is None:
            return

    # If a subcommand is invoked, skip main logic
    if ctx.invoked_subcommand is not None:
        return

    # show first-run warning
    first_run = _is_first_run()
    if first_run:
        click.secho("First run detected - initial startup may take longer while caches are built.", fg="yellow")

    # ensure environment cache exists (build if missing/invalid)
    if not no_cache:
        try:
            from mbo_utilities.env_cache import ensure_cache
            ensure_cache()
        except Exception:
            pass  # don't crash if cache fails

    # show loading spinner while importing heavy dependencies
    spinner = LoadingSpinner("Loading GUI")
    spinner.start()
    try:
        from mbo_utilities.gui.run_gui import run_gui
        spinner.stop()
    except Exception:
        spinner.stop()
        raise

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
@click.option(
    "--gpu",
    "gpu_index",
    type=int,
    default=None,
    help="0-based GPU adapter index (see --list-gpus). Default: wgpu auto-pick.",
)
@click.option(
    "--list-gpus",
    is_flag=True,
    help="List available GPU adapters and exit.",
)
def view(data_in=None, roi=None, widget=True, metadata=False, gpu_index=None, list_gpus=False):
    r"""
    Open imaging data in the GUI viewer.

    \b
    Examples:
      mbo view                       Open file selection dialog
      mbo view /data/raw.tiff        Open specific file
      mbo view /data/raw --metadata  Show only metadata
      mbo view /data --roi 0 --roi 2 View specific ROIs
      mbo view --list-gpus           Show available GPU adapters
      mbo view /data/raw --gpu 0     Force GPU index 0
    """
    if list_gpus:
        import fastplotlib as fpl
        adapters = fpl.enumerate_adapters()
        click.echo(f"{'idx':<4} {'type':<14} {'vendor':<10} {'name'}")
        for i, a in enumerate(adapters):
            info = getattr(a, "info", {}) or {}
            type_ = info.get("adapter_type", info.get("device_type", "?"))
            vendor = info.get("vendor", "?")
            name = info.get("device", info.get("description", "?"))
            click.echo(f"{i:<4} {str(type_):<14} {str(vendor):<10} {name}")
        return

    if gpu_index is not None:
        import fastplotlib as fpl
        adapters = fpl.enumerate_adapters()
        if not 0 <= gpu_index < len(adapters):
            raise click.BadParameter(
                f"--gpu {gpu_index} out of range; found {len(adapters)} adapter(s). "
                f"Run `mbo view --list-gpus` to see them.",
                param_hint="--gpu",
            )
        fpl.select_adapter(adapters[gpu_index])
        from mbo_utilities.preferences import set_gpu_index
        set_gpu_index(gpu_index)
        info = getattr(adapters[gpu_index], "info", {}) or {}
        click.echo(f"Using GPU {gpu_index}: {info.get('device', info.get('description', '?'))}")

    # show first-run warning
    first_run = _is_first_run()
    if first_run:
        click.secho("First run detected - initial startup may take longer while caches are built.", fg="yellow")

    # show loading spinner while importing
    spinner = LoadingSpinner("Loading GUI")
    spinner.start()
    try:
        from mbo_utilities.gui.run_gui import run_gui
        spinner.stop()
    except Exception:
        spinner.stop()
        raise

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
    "-t", "--timepoints",
    multiple=True,
    type=int,
    help="Timepoints to export (1-based): -t 1 -t 50 -t 100",
)
@click.option(
    "--num-timepoints",
    type=int,
    default=None,
    help="Number of timepoints to export (first N).",
)
@click.option(
    "--num-zplanes",
    type=int,
    default=None,
    help="Number of z-planes to export (first N).",
)
@click.option(
    "-n", "--num-frames",
    type=int,
    default=None,
    help="Deprecated, use --num-timepoints.",
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
    help="Bidirectional phase correction (ScanImageArray).",
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
    "-c", "--channels",
    multiple=True,
    type=int,
    help="Color channels to export (1-based): -c 1 -c 2",
)
@click.option(
    "--order",
    multiple=True,
    type=int,
    help="Reorder planes before writing (0-based indices into --planes).",
)
@click.option(
    "--output-suffix",
    type=str,
    default=None,
    help="Suffix appended to output filenames.",
)
@click.option(
    "--dataset-name",
    type=str,
    default=None,
    help="HDF5 dataset name (.h5 only). Default: mov.",
)
@click.option(
    "--roi-mode",
    type=click.Choice(["concat_y", "separate"]),
    default=None,
    help="Multi-ROI handling: concat_y=stitch, separate=one file per ROI.",
)
@click.option(
    "--compressor",
    type=click.Choice(["none", "gzip", "zstd", "blosc-lz4", "blosc-zstd"]),
    default=None,
    help="Zarr compressor (.zarr only).",
)
@click.option(
    "--compression-level",
    type=int,
    default=None,
    help="Zarr compression level (.zarr only).",
)
@click.option(
    "--sharded/--no-sharded",
    default=None,
    help="Zarr v3 sharding (.zarr only).",
)
@click.option(
    "--pyramid/--no-pyramid",
    default=None,
    help="Write multiscale pyramid (.zarr only).",
)
@click.option(
    "--pyramid-max-layers",
    type=int,
    default=None,
    help="Max pyramid resolution levels (.zarr only).",
)
@click.option(
    "--pyramid-method",
    type=click.Choice(["mean", "median", "mode", "gaussian", "nearest", "local_mean"]),
    default=None,
    help="Pyramid downsampling method (.zarr only).",
)
@click.option(
    "--border",
    type=int,
    default=None,
    help="Phase correction: border pixels excluded from estimation.",
)
@click.option(
    "--max-offset",
    type=int,
    default=None,
    help="Phase correction: max pixel offset to search.",
)
@click.option(
    "--use-fft/--no-use-fft",
    default=None,
    help="Phase correction: FFT-based 2D correction.",
)
@click.option(
    "--reg-max-frames",
    type=int,
    default=None,
    help="register-z: subsample frame count (default 200).",
)
@click.option(
    "--reg-chunk-frames",
    type=int,
    default=None,
    help="register-z: streaming batch size (default 10).",
)
@click.option(
    "--reg-max-xy",
    type=int,
    default=None,
    help="register-z: search radius in pixels (default 30).",
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
    timepoints,
    num_timepoints,
    num_zplanes,
    num_frames,
    roi,
    register_z,
    overwrite,
    chunk_mb,
    fix_phase,
    phasecorr_method,
    ome,
    output_name,
    channels,
    order,
    output_suffix,
    dataset_name,
    roi_mode,
    compressor,
    compression_level,
    sharded,
    pyramid,
    pyramid_max_layers,
    pyramid_method,
    border,
    max_offset,
    use_fft,
    reg_max_frames,
    reg_chunk_frames,
    reg_max_xy,
    debug,
):
    r"""
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
    parsed_timepoints = list(timepoints) if timepoints else None
    parsed_channels = list(channels) if channels else None
    parsed_order = list(order) if order else None

    if num_frames and not num_timepoints:
        click.echo("--num-frames is deprecated, use --num-timepoints.")
        num_timepoints = num_frames

    click.echo(f"Reading: {input_path}")

    # Build imread kwargs
    imread_kwargs = {}
    if fix_phase is not None:
        imread_kwargs["fix_phase"] = fix_phase
    if phasecorr_method:
        imread_kwargs["phasecorr_method"] = phasecorr_method
    if border is not None:
        imread_kwargs["border"] = border
    if max_offset is not None:
        imread_kwargs["max_offset"] = max_offset
    if use_fft is not None:
        imread_kwargs["use_fft"] = use_fft
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
    if parsed_timepoints:
        imwrite_kwargs["timepoints"] = parsed_timepoints
    if parsed_channels:
        imwrite_kwargs["channels"] = parsed_channels
    if num_timepoints:
        imwrite_kwargs["num_timepoints"] = num_timepoints
    if num_zplanes:
        imwrite_kwargs["num_zplanes"] = num_zplanes

    # ROI: any explicit --roi implies separate handling (concat_y ignores
    # roi). --roi-mode overrides. 0=split all, N=specific, "1,3"=multiple.
    roi_mode_kw = roi_mode
    if roi_mode_kw is None and parsed_roi is not None:
        roi_mode_kw = "separate"
    if parsed_roi is not None:
        imwrite_kwargs["roi"] = parsed_roi
    if roi_mode_kw is not None:
        imwrite_kwargs["roi_mode"] = roi_mode_kw

    if parsed_order:
        if not parsed_planes:
            click.secho("--order requires --planes; ignoring --order.", fg="yellow")
        else:
            imwrite_kwargs["order"] = parsed_order

    if register_z:
        imwrite_kwargs["register_z"] = True
        if reg_max_frames is not None:
            imwrite_kwargs["max_frames"] = reg_max_frames
        if reg_chunk_frames is not None:
            imwrite_kwargs["chunk_frames"] = reg_chunk_frames
        if reg_max_xy is not None:
            imwrite_kwargs["max_reg_xy"] = reg_max_xy

    if output_ext.lower() == ".zarr":
        imwrite_kwargs["ome"] = ome
        if compressor is not None:
            imwrite_kwargs["compressor"] = compressor
        if compression_level is not None:
            imwrite_kwargs["compression_level"] = compression_level
        if sharded is not None:
            imwrite_kwargs["sharded"] = sharded
        if pyramid is not None:
            imwrite_kwargs["pyramid"] = pyramid
        if pyramid_max_layers is not None:
            imwrite_kwargs["pyramid_max_layers"] = pyramid_max_layers
        if pyramid_method is not None:
            imwrite_kwargs["pyramid_method"] = pyramid_method

    if dataset_name and output_ext.lower() in (".h5", ".hdf5"):
        imwrite_kwargs["dataset_name"] = dataset_name
    if output_name:
        imwrite_kwargs["output_name"] = output_name
    if output_suffix:
        imwrite_kwargs["output_suffix"] = output_suffix

    result = imwrite(data, output_path, **imwrite_kwargs)
    click.secho(f"\nDone! Output saved to: {result}", fg="green")

def _info_num(v):
    """Compact number formatting: 30.0 -> '30', 0.5 -> '0.5'. None -> None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:  # nan
            return None
        return f"{v:g}"
    return str(v)


def _info_md(md, name):
    """get_param value, or None when the key is truly absent.

    get_param falls back to a parameter's registered default (e.g. dx=1.0),
    which would mask missing values; a distinct sentinel detects real absence.
    """
    from mbo_utilities.metadata import get_param

    sentinel = object()
    val = get_param(md, name, default=sentinel)
    return None if val is sentinel else val


def _info_dim_sizes(data):
    """Map of dim label (T/C/Z/Y/X) -> size, from shape + dims labels."""
    dims = getattr(data, "dims", None)
    shape = getattr(data, "shape", ()) or ()
    out = {}
    if dims and len(dims) == len(shape):
        for d, s in zip(dims, shape):
            out[str(d).upper()] = int(s)
    return out


def _info_find_ops_dirs(root, max_depth=2):
    """Directories at/under root (inclusive) that contain ops.npy."""
    from pathlib import Path

    skip_suffixes = {".zarr", ".corrected", ".fused", ".registered"}
    found = []
    seen = set()

    def _add(p):
        if p not in seen:
            seen.add(p)
            found.append(p)

    try:
        if (root / "ops.npy").exists():
            _add(root)

        def _walk(d, depth):
            if depth > max_depth:
                return
            for child in sorted(d.iterdir()):
                if not child.is_dir() or child.suffix.lower() in skip_suffixes:
                    continue
                if (child / "ops.npy").exists():
                    _add(child)
                _walk(child, depth + 1)

        _walk(Path(root), 1)
    except (OSError, PermissionError):
        pass
    return found


def _info_segmentation_counts(ops_dir):
    """(n_rois, n_cells) from stat.npy / iscell.npy, or (None, None)."""
    import numpy as np
    from mbo_utilities.file_io import load_npy

    stat_path = ops_dir / "stat.npy"
    if not stat_path.exists():
        return None, None
    try:
        n_rois = len(load_npy(stat_path))
    except Exception:
        return None, None
    n_cells = None
    iscell_path = ops_dir / "iscell.npy"
    if iscell_path.exists():
        try:
            iscell = np.asarray(load_npy(iscell_path))
            n_cells = int(iscell[:, 0].astype(bool).sum())
        except Exception:
            n_cells = None
    return n_rois, n_cells


def _info_kv(label, value, unit=""):
    """Print an aligned label/value line; skip when value is None/empty."""
    if value is None or value == "":
        return
    line = f"  {label:<15} {value}"
    if unit:
        line += f" {unit}"
    click.echo(line)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--metadata/--no-metadata",
    default=True,
    help="Show imaging/acquisition metadata sections.",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Also dump the full raw metadata dict.",
)
def info(input_path, metadata, show_all):
    r"""
    Show information about an imaging dataset.

    \b
    Examples:
      mbo info /data/raw.tiff
      mbo info /data/volume.zarr
      mbo info /data/suite2p/plane0
      mbo info /data/raw --all
    """
    from pathlib import Path
    from mbo_utilities import imread

    click.echo(f"Loading: {input_path}")
    data = imread(input_path)
    md = getattr(data, "metadata", None) or {}
    sizes = _info_dim_sizes(data)

    is_isoview = hasattr(data, "is_tiled")
    is_tiled = bool(getattr(data, "is_tiled", False))

    dims = getattr(data, "dims", None)
    if dims and is_isoview:
        dims = tuple("V" if d == "C" else d for d in dims)
    shape_str = str(tuple(data.shape))
    if dims and len(dims) == len(data.shape):
        shape_str += f"  [{', '.join(dims)}]"

    click.echo("")
    click.secho(str(input_path), bold=True)
    _info_kv("Type", type(data).__name__)
    _info_kv("Shape", shape_str)
    _info_kv("Dtype", str(data.dtype))

    if metadata:
        fs = _info_md(md, "fs")
        dx = _info_md(md, "dx")
        dy = _info_md(md, "dy")
        dz = _info_md(md, "dz")
        ny = sizes.get("Y", data.shape[-2] if data.ndim >= 2 else None)
        nx = sizes.get("X", data.shape[-1] if data.ndim >= 1 else None)
        if is_isoview and is_tiled:
            n_timepoints = 1
        else:
            n_timepoints = sizes.get("T") or _info_md(md, "num_timepoints")

        click.secho("\nImaging", fg="cyan")
        if n_timepoints != 1:
            _info_kv("Frame rate", _info_num(fs), "Hz")
        if dx is not None or dy is not None:
            _info_kv("Pixel size", f"{_info_num(dx)} x {_info_num(dy)}", "um")
        _info_kv("Z step", _info_num(dz), "um")
        if ny is not None and nx is not None:
            _info_kv("Frame size", f"{ny} x {nx}", "px (Y x X)")
            if dx is not None and dy is not None:
                _info_kv("FOV", f"{_info_num(dy * ny)} x {_info_num(dx * nx)}", "um")

        zplanes = sizes.get("Z") or _info_md(md, "num_zplanes")

        click.secho("\nAcquisition", fg="cyan")
        _info_kv("Stack type", _info_md(md, "stack_type"))

        if is_isoview:
            # view_names are the VW labels (VW00, VW90, ...); `views` holds
            # raw camera keys, so prefer the labels for display.
            view_labels = getattr(data, "view_names", None) or getattr(data, "views", None) or []
            view_str = ", ".join(str(v) for v in view_labels) or _info_num(
                getattr(data, "num_views", None) or sizes.get("C")
            )
            if is_tiled:
                _info_kv("Tiles", _info_num(sizes.get("T")))
                _info_kv("Timepoints", "1")
            else:
                _info_kv("Timepoints", _info_num(sizes.get("T")))
            _info_kv("Z-planes", _info_num(zplanes))
            _info_kv("Views", view_str)
            _info_kv("Color channels", _info_num(getattr(data, "num_color_channels", None)))
        else:
            timepoints = n_timepoints
            channels = sizes.get("C") or _info_md(md, "num_color_channels")
            mrois = _info_md(md, "num_mrois")

            avg = None
            if isinstance(md.get("si"), dict):
                try:
                    from mbo_utilities.metadata.scanimage import get_log_average_factor
                    avg = get_log_average_factor(md)
                except Exception:
                    avg = None

            _info_kv("Timepoints", _info_num(timepoints))
            _info_kv("Z-planes", _info_num(zplanes))
            _info_kv("Color channels", _info_num(channels))
            _info_kv("mROIs", _info_num(mrois))
            if avg and avg > 1:
                _info_kv("Averaging", f"{avg}x")
            if fs and timepoints:
                _info_kv("Duration", f"{timepoints / fs:.1f}", "s")

    try:
        vmin = data.vmin
        vmax = data.vmax
        if vmin is not None and vmax is not None:
            _info_kv("Value range", f"[{_info_num(float(vmin))}, {_info_num(float(vmax))}]")
    except Exception:
        pass

    filenames = getattr(data, "filenames", None)
    if filenames:
        names = [Path(f).name for f in filenames]
        collide = len(set(names)) < len(names)

        def _file_label(f):
            p = Path(f)
            return f"{p.parent.name}/{p.name}" if collide else p.name

        click.secho(f"\nFiles ({len(filenames)})", fg="cyan")
        shown = filenames if len(filenames) <= 5 else filenames[:3]
        for f in shown:
            click.echo(f"  - {_file_label(f)}")
        if len(filenames) > 5:
            click.echo(f"  ... and {len(filenames) - 3} more")

    if is_isoview:
        click.secho("\nIsoview", fg="cyan")
        _info_kv("Pipeline stage", _info_md(md, "stack_type"))
        _info_kv("Layout", "tiled" if is_tiled else "timelapse")
    else:
        input_obj = Path(input_path)
        scan_root = input_obj if input_obj.is_dir() else input_obj.parent
        ops_dirs = _info_find_ops_dirs(scan_root)

        click.secho("\nResults", fg="cyan")
        if ops_dirs:
            total_rois = total_cells = seg_planes = 0
            have_seg = False
            for d in ops_dirs:
                n_rois, n_cells = _info_segmentation_counts(d)
                if n_rois is not None:
                    have_seg = True
                    seg_planes += 1
                    total_rois += n_rois
                    if n_cells is not None:
                        total_cells += n_cells
            have_reg = any((d / "data.bin").exists() for d in ops_dirs)
            have_raw = any((d / "data_raw.bin").exists() for d in ops_dirs)

            _info_kv("Suite2p", f"{len(ops_dirs)} folder(s)")
            if have_reg:
                _info_kv("Registration", "data.bin present")
            elif have_raw:
                _info_kv("Registration", "data_raw.bin only (not registered)")
            if have_seg:
                _info_kv("Segmentation", f"{total_rois} ROIs ({total_cells} cells), {seg_planes} plane(s)")
            else:
                _info_kv("Segmentation", "not run (no stat.npy)")
        else:
            click.echo("  none found")

    if show_all and md:
        click.secho("\nAll metadata", fg="cyan")
        for k in sorted(md, key=str):
            v = md[k]
            if isinstance(v, dict):
                click.echo(f"  {k}: {{...{len(v)} keys}}")
            elif isinstance(v, (list, tuple)) and len(v) > 8:
                click.echo(f"  {k}: [{len(v)} items]")
            else:
                s = str(v)
                if len(s) > 80:
                    s = s[:77] + "..."
                click.echo(f"  {k}: {s}")


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
@click.option(
    "--patch",
    "patch_size",
    type=int,
    default=32,
    help="Spatial patch size in px for the distribution map. Default: 32.",
)
@click.option(
    "--docs",
    is_flag=True,
    default=False,
    help="Save output to docs/_images/scanphase/ for documentation.",
)
def scanphase(input_path, output_dir, num_tifs, image_format, show, docs, patch_size):
    """
    Scan-phase analysis for bidirectional scanning data.

    Analyzes phase offset to determine optimal correction parameters.

    \b
    OUTPUT:
      window_lengths.png        - offset vs frames averaged, with/without FFT
      spatial_distribution.png  - offset across FOV, with/without FFT
      scanphase_results.npz     - all numerical data

    \b
    Examples:
      mbo scanphase                          # open file dialog
      mbo scanphase /path/to/data.tiff       # analyze specific file
      mbo scanphase ./folder/ -n 5           # use first 5 tiffs in folder
      mbo scanphase data.tiff -o ./results/  # custom output directory
      mbo scanphase data.tiff --patch 16     # finer spatial map
      mbo scanphase data.tiff --show         # show plots interactively
    """
    from pathlib import Path
    from mbo_utilities import get_files
    from mbo_utilities.analysis.scanphase import run_scanphase_analysis

    try:
        # handle --docs flag: override output_dir to docs/_images/scanphase/
        if docs:
            # find the repo root (where docs/ folder is)
            repo_root = Path(__file__).parent.parent
            output_dir = str(repo_root / "docs" / "_images" / "scanphase")
            click.echo(f"--docs flag: saving to {output_dir}")

        # handle num_tifs for folder input
        actual_input = input_path
        if input_path is not None:
            input_path_obj = Path(input_path)
            if input_path_obj.is_dir() and num_tifs is not None:
                tiffs = get_files(input_path, str_contains=".tif", max_depth=1)
                if not tiffs:
                    click.secho(f"No tiff files found in {input_path}", fg="red")
                    raise click.Abort
                tiffs = tiffs[:num_tifs]
                click.echo(f"Using {len(tiffs)} tiff files from {input_path}")
                actual_input = tiffs

        # determine output directory for display
        if output_dir is not None:
            actual_output_dir = Path(output_dir)
        elif input_path is not None:
            input_path_obj = Path(input_path)
            actual_output_dir = input_path_obj.parent / f"{input_path_obj.stem}_scanphase_analysis"
        else:
            actual_output_dir = None

        results = run_scanphase_analysis(
            data_path=actual_input,
            output_dir=output_dir,
            image_format=image_format,
            show_plots=show,
            patch_size=patch_size,
        )

        if results is None:
            return  # user cancelled file selection

        # print summary
        summary = results.get_summary()
        meta = summary.get("metadata", {})

        click.echo("")
        click.secho("scan-phase analysis complete", fg="cyan", bold=True)
        click.echo("")
        click.echo(f"data: {meta.get('num_timepoints', meta.get('num_frames', 0))} timepoints, "
                   f"{meta.get('num_rois', 1)} ROIs, "
                   f"{meta.get('frame_shape', (0, 0))[1]}x{meta.get('frame_shape', (0, 0))[0]} px")
        click.echo(f"analysis time: {meta.get('analysis_time', 0):.1f}s")
        click.echo(f"output: {actual_output_dir}")

        # fft stats
        if "fft" in summary:
            stats = summary["fft"]
            click.echo("")
            click.secho("offset (FFT)", fg="yellow", bold=True)
            click.echo(f"  mean:   {stats.get('mean', 0):+.3f} px")
            click.echo(f"  median: {stats.get('median', 0):+.3f} px")
            click.echo(f"  std:    {stats.get('std', 0):.3f} px")
            click.echo(f"  range:  [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}] px")

        # int stats
        if "int" in summary:
            stats = summary["int"]
            click.echo("")
            click.secho("offset (integer)", fg="yellow", bold=True)
            click.echo(f"  mean:   {stats.get('mean', 0):+.3f} px")
            click.echo(f"  std:    {stats.get('std', 0):.3f} px")

        click.echo("")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        raise click.Abort


@main.command("processes")
@click.option(
    "--kill-all",
    is_flag=True,
    help="Kill all tracked background processes.",
)
@click.option(
    "--kill",
    type=int,
    default=None,
    help="Kill a specific process by PID.",
)
@click.option(
    "--cleanup",
    is_flag=True,
    help="Remove entries for finished processes.",
)
def processes(kill_all, kill, cleanup):
    r"""
    Manage background processes (suite2p, save operations, etc.).

    \b
    Examples:
      mbo processes                  # List all tracked processes
      mbo processes --cleanup        # Remove finished process entries
      mbo processes --kill 12345     # Kill specific process
      mbo processes --kill-all       # Kill all background processes
    """
    from mbo_utilities.gui.widgets.process_manager import get_process_manager

    pm = get_process_manager()

    if cleanup:
        count = pm.cleanup_finished()
        click.echo(f"Cleaned up {count} finished processes.")
        return

    if kill_all:
        running = pm.get_running()
        if not running:
            click.echo("No running processes to kill.")
            return
        count = pm.kill_all()
        click.secho(f"Killed {count} processes.", fg="yellow")
        return

    if kill is not None:
        if pm.kill(kill):
            click.secho(f"Killed process {kill}.", fg="yellow")
        else:
            click.secho(f"Process {kill} not found or could not be killed.", fg="red")
        return

    # list all processes
    all_procs = pm.get_all()
    if not all_procs:
        click.echo("No tracked background processes.")
        return

    click.echo(f"\nTracked processes ({len(all_procs)}):")
    click.echo("-" * 60)

    for p in all_procs:
        alive = p.is_alive()
        status = click.style("RUNNING", fg="green") if alive else click.style("FINISHED", fg="bright_black")
        click.echo(f"  PID {p.pid:>6}  {status}  {p.description}")
        click.echo(f"           Started: {p.elapsed_str()}")
        if p.output_path:
            click.echo(f"           Output:  {p.output_path}")

    click.echo("-" * 60)
    running = [p for p in all_procs if p.is_alive()]
    click.echo(f"Running: {len(running)}, Finished: {len(all_procs) - len(running)}")

    if len(all_procs) > len(running):
        click.echo("\nTip: Run 'mbo processes --cleanup' to remove finished entries.")


@main.command("gpu")
@click.option(
    "--watch",
    type=float,
    default=None,
    help="Refresh every N seconds until Ctrl-C.",
)
@click.option(
    "--processes",
    "show_processes",
    is_flag=True,
    help="Also list per-process VRAM usage.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit render/compute GPU + devices + processes as JSON.",
)
def gpu(watch, show_processes, as_json):
    r"""
    Show which GPU renders (fastplotlib) and which computes (suite2p /
    cellpose / cupy), plus device memory.

    \b
    Examples:
      mbo gpu               # render GPU, compute GPU, device memory
      mbo gpu --processes   # also per-process VRAM
      mbo gpu --watch 2     # refresh every 2s
      mbo gpu --json        # machine-readable
    """
    from mbo_utilities import gpu as gpu_mod

    # reflect the GPU picked in the GUI's File > Options (persisted pref)
    gpu_mod.apply_persisted_compute_gpu()

    if as_json:
        import json as _json
        click.echo(_json.dumps({
            "render_gpu": gpu_mod.render_gpu(),
            "compute_gpu": gpu_mod.compute_gpu(),
            "devices": gpu_mod.gpu_devices(),
            "processes": gpu_mod.gpu_processes() if show_processes else [],
            "compute_disabled": gpu_mod.gpu_compute_disabled(),
        }, indent=2))
        return

    if watch:
        try:
            while True:
                click.clear()
                click.echo(gpu_mod.format_gpu_report(show_processes=show_processes))
                click.echo("\n(Ctrl-C to stop)")
                time.sleep(watch)
        except KeyboardInterrupt:
            return
    else:
        click.echo(gpu_mod.format_gpu_report(show_processes=show_processes))


@main.command("init")
@click.argument("data_path", required=False, type=click.Path())
@click.option(
    "-o", "--output",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Destination directory (overrides default location).",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing notebooks.",
)
def init(data_path, output_dir, overwrite):
    r"""
    Create starter notebooks (mbo + LBM-Suite2p user guides).

    With DATA_PATH, notebooks go in <DATA_PATH>/../scripts and the data path
    is filled in. Without it, notebooks go in the current directory.

    \b
    Examples:
      mbo init                       # notebooks in current directory
      mbo init /data/raw             # notebooks in /data/scripts, path filled in
      mbo init /data/raw -o ./nb     # notebooks in ./nb
    """
    # Shipped copy (wheel / uv tool install) is generated from demos/ at build
    # time into the package. Fall back to demos/ for source/editable runs.
    pkg_dir = Path(__file__).resolve().parent
    candidates = [pkg_dir / "assets" / "notebooks", pkg_dir.parent / "demos"]
    src_dir = next(
        (c for c in candidates if (c / "mbo_user_guide.ipynb").exists()),
        candidates[0],
    )

    # source notebook -> default data-path token replaced when DATA_PATH given
    notebooks = {
        "mbo_user_guide.ipynb": "D:/demo/raw",
        "lsp_user_guide.ipynb": "D:/demo/raw",
    }

    if output_dir is not None:
        dest = Path(output_dir).expanduser()
    elif data_path is not None:
        dest = Path(data_path).expanduser().resolve().parent / "scripts"
    else:
        dest = Path.cwd()
    dest.mkdir(parents=True, exist_ok=True)

    fill = Path(data_path).expanduser().resolve().as_posix() if data_path else None

    from datetime import date
    today = date.today().isoformat()

    written = []
    for filename, token in notebooks.items():
        src = src_dir / filename
        if not src.exists():
            click.secho(f"Missing: {filename}", fg="red")
            continue
        out = dest / f"{today}_{filename}"
        if out.exists() and not overwrite:
            click.secho(f"Exists: {out}  (--overwrite to replace)", fg="yellow")
            continue
        text = src.read_text(encoding="utf-8")
        if fill and token:
            text = text.replace(token, fill)
        out.write_text(text, encoding="utf-8")
        written.append(out)
        click.secho(f"Created: {out}", fg="green")

    if not written:
        return
    if fill:
        click.echo(f"Data path: {fill}")
    click.echo(f"\nOpen with:\n  jupyter lab {dest}")


@main.command("download-models")
@click.option(
    "--force",
    is_flag=True,
    help="Re-download even if already present.",
)
def download_models(force):
    r"""
    Download/repair the cellpose model used by suite2p detection.

    \b
    Examples:
      mbo download-models           # download if missing or corrupt
      mbo download-models --force   # re-download
    """
    from mbo_utilities._cellpose_model import ensure_cellpose_model

    path = ensure_cellpose_model(force=force)
    if path:
        click.secho(f"cellpose model ready: {path}", fg="green")
    else:
        click.secho("cellpose model unavailable (pip install cellpose)", fg="red")
        raise click.Abort


@main.command("shortcut")
@click.option(
    "--name",
    default="Miller Brain Studio",
    help="Shortcut name.",
)
def shortcut(name):
    r"""
    Create a desktop icon that opens the GUI.

    Windows: a .lnk that launches the viewer with no console window.
    Linux: a .desktop entry on the Desktop.

    \b
    Examples:
      mbo shortcut
      mbo shortcut --name "MBO"
    """
    from mbo_utilities._shortcut import create_desktop_shortcut

    try:
        path = create_desktop_shortcut(name=name)
    except Exception as e:
        raise click.ClickException(str(e))
    click.secho(f"Created: {path}", fg="green")


# hpc subcommand group (light import: heavy deps load lazily inside the commands).
from mbo_utilities.hpc.cli import hpc as _hpc_group  # noqa: E402

main.add_command(_hpc_group)


if __name__ == "__main__":
    main()
