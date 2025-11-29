"""
CLI entry point for mbo_utilities.

This module handles command-line operations with minimal imports.
GUI-related imports are deferred until actually needed.
"""
import sys
from pathlib import Path
from typing import Optional, Union

import click


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
        If a directory, saves using the filename from the URL.
        If a file path, uses that exact filename.

    Returns
    -------
    Path
        Path to the downloaded file.

    Examples
    --------
    >>> from mbo_utilities.cli import download_file
    >>> download_file(
    ...     "https://github.com/org/repo/blob/main/data/example.mat",
    ...     output_path="C:/Users/RBO/repos/test/"
    ... )
    """
    import urllib.request

    # Convert GitHub blob URLs to raw URLs
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    # Extract filename from URL
    url_filename = url.split("/")[-1]
    # Remove query parameters if present
    if "?" in url_filename:
        url_filename = url_filename.split("?")[0]

    # Determine output file path
    if output_path is None or output_path == ".":
        output_file = Path.cwd() / url_filename
    else:
        output_file = Path(output_path).expanduser().resolve()
        if output_file.is_dir() or (not output_file.suffix and not output_file.exists()):
            # It's a directory (existing or to be created)
            output_file.mkdir(parents=True, exist_ok=True)
            output_file = output_file / url_filename
        # else: it's a file path, use as-is

    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        click.echo(f"Downloading from:\n  {url}")
        click.echo(f"Saving to:\n  {output_file.resolve()}")

        # Download the file
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
    """Download a Jupyter notebook from a URL to a local file.

    Parameters
    ----------
    output_path : str, Path, optional
        Directory or file path to save the notebook. If None, saves to current directory.
    notebook_url : str, optional
        URL to the notebook file. If None, downloads the default user guide notebook.

    Returns
    -------
    Path
        Path to the downloaded notebook file.
    """
    default_url = "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/demos/user_guide.ipynb"
    url = notebook_url or default_url

    output_file = download_file(url, output_path)

    # Show notebook-specific instructions
    click.echo("\nTo use the notebook:")
    click.echo(f"  jupyter lab {output_file.resolve()}")

    return output_file


@click.command()
@click.option(
    "--roi",
    multiple=True,
    type=int,
    help="ROI index (can pass multiple, e.g. --roi 0 --roi 2). Leave empty for None.",
    default=None,
)
@click.option(
    "--widget/--no-widget",
    default=True,
    help="Enable or disable PreviewDataWidget for Raw ScanImage tiffs.",
)
@click.option(
    "--metadata-only/--full-preview",
    default=False,
    help="If enabled, only show extracted metadata.",
)
@click.option(
    "--download-notebook",
    is_flag=True,
    help="Download the user guide notebook and exit.",
)
@click.option(
    "--notebook-url",
    type=str,
    default=None,
    help="URL of notebook to download (use with --download-notebook).",
)
@click.option(
    "--download-file",
    "download_file_url",
    type=str,
    default=None,
    help="Download a file from URL (e.g. GitHub) to DATA_IN path. "
         "Supports GitHub blob URLs (auto-converted to raw).",
)
@click.option(
    "--check-install",
    is_flag=True,
    help="Verify the installation of mbo_utilities and dependencies.",
)
@click.argument("data_in", required=False)
def main(
    data_in=None,
    widget=None,
    roi=None,
    metadata_only=False,
    download_notebook=False,
    notebook_url=None,
    download_file_url=None,
    check_install=False,
):
    """
    MBO Utilities CLI - data preview and processing tools.

    \b
    Examples:
      mbo                              # Open file selection dialog
      mbo /path/to/data.tif            # Open specific file
      mbo --download-file URL /path/   # Download file from GitHub
      mbo --download-notebook          # Download user guide notebook
      mbo --check-install              # Verify installation
    """
    # Handle download-file first (lightweight, no GUI imports)
    if download_file_url:
        download_file(download_file_url, data_in)
        return

    # Handle download-notebook (lightweight, no GUI imports)
    if download_notebook:
        download_notebook_func = globals()["download_notebook"]
        download_notebook_func(output_path=data_in, notebook_url=notebook_url)
        return

    # Handle check-install (imports GUI code for full check)
    if check_install:
        from mbo_utilities.graphics.run_gui import _check_installation
        _check_installation()
        return

    # Everything else requires the GUI
    from mbo_utilities.graphics.run_gui import run_gui
    run_gui(
        data_in=data_in,
        roi=roi if roi else None,
        widget=widget,
        metadata_only=metadata_only,
    )


if __name__ == "__main__":
    main()
