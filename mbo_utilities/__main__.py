from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import fastplotlib as fpl
from pprint import pprint
from mbo_utilities.lcp_io import get_metadata, read_scan
from mbo_utilities.scanreader.exceptions import PathnameError, FieldDimensionMismatch
from mbo_utilities.scanreader.core import scans, expand_wildcard


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
    parser.add_argument('--path', type=str, help='Path to a directory containing raw scanimage tiff files for a single session.')
    parser.add_argument('--version', action='store_true', help='Print the version of the package.')
    return parser


def main():
    parser = argparse.ArgumentParser(description="Preview a scanimage imaging session.")
    parser = add_args(parser)
    args = parser.parse_args()

    # Handle version
    if args.version:
        import mbo_utilities as mbo
        print("lbm_caiman_python v{}".format(mbo.__version__))
        return

    # Handle help
    data_path = Path(args.path).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Path '{data_path}' does not exist as a file or directory.")
    if data_path.is_dir():
        files = [str(f) for f in data_path.glob('*.tif*')]
        scan = read_scan(files, join_contiguous=True)
        metadata = get_metadata(files[args.index])
        pprint(metadata)
        iw = fpl.ImageWidget(scan, histogram_widget=False)
        iw.show()
    else:
        raise FileNotFoundError(f"Path '{data_path}' is not a directory.")

    # pprint(metadata)
    iw = fpl.ImageWidget(scan, histogram_widget=False)
    iw.show()
    if fpl.__version__ == "0.2.0":
        fpl.run()
    elif fpl.__version__ == "0.3.0":
        fpl.loop.run()


if __name__ == '__main__':
    main()
