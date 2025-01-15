import argparse
from pathlib import Path

import numpy as np
import fastplotlib as fpl
import scanreader
from mbo_utilities import get_metadata


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

def update_scan(scan):
    pass

def main():
    parser = argparse.ArgumentParser(description="MBO Utilities")
    parser = add_args(parser)
    args = parser.parse_args()

    # Handle version
    if args.version:
        import mbo_utilities as mbo
        print("lbm_caiman_python v{}".format(mbo.__version__))
        return

    files = [str(f) for f in Path(args.path).glob('*.tif*')]
    metadata = get_metadata(files[0])

    scan = scanreader.read_scan(files, join_contiguous=True)
    scan[:]  # shape (600, 576, 30, 1730)

    scan.ndim = 4
    scan.shape = (600, 576, metadata['num_planes'], metadata['num_frames'])

    iw = fpl.ImageWidget(scan, histogram_widget=False)
    iw.show()
    fpl.loop.run()


if __name__ == '__main__':
    main()





