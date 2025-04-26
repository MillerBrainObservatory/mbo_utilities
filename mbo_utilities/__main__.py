from __future__ import annotations

import argparse
from pathlib import Path

import fastplotlib as fpl

from mbo_utilities import run_gui


def add_args(parser: argparse.ArgumentParser):
    """
    Add command-line arguments to the parser, dynamically adding arguments
    for each key in the `ops` dictionary.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which arguments are added.

    Returns
    -------
    argparse.ArgumentParser
        The parser with added arguments.
    """
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to a directory containing raw scanimage tiff files for a single session.",
    )
    parser.add_argument("--gui", action="store_true", help="Run the GUI.")
    parser.add_argument(
        "--version", action="store_true", help="Print the version of the package."
    )
    return parser


def main():
    parser = argparse.ArgumentParser(
        description="Preview a scanimage imaging session."
        "This will display 3D [Tyx] or 4D [Tzyx] data in a GUI."
        "The path must be a directory containing raw or assembled scanimage tiff files."
    )
    parser = add_args(parser)
    args = parser.parse_args()

    # Handle version
    if args.version:
        import mbo_utilities as mbo

        print("lbm_caiman_python v{}".format(mbo.__version__))
        return

    if args.gui:
        run_gui()
        return
    if not args.path or args.path == "":
        run_gui()
        return
    else:
        data_path = Path(args.path).expanduser().resolve()
        print(f"Reading data from '{data_path}'")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Path '{data_path}' does not exist as a file or directory."
            )
        if data_path.is_dir():
            run_gui(data_path)
        else:
            raise FileNotFoundError(f"Path '{data_path}' is not a directory.")


if __name__ == "__main__":
    main()
    if fpl.__version__ == "0.2.0":
        raise NotImplementedError("fastplotlib version 0.2.0 does not support GUIs.")
    elif fpl.__version__ == "0.3.0":
        fpl.loop.run()
