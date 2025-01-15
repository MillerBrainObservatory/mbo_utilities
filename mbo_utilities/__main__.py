from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import fastplotlib as fpl
from mbo_utilities.scanreader.exceptions import PathnameError, FieldDimensionMismatch
from mbo_utilities.scanreader.core import scans, expand_wildcard


def read_scan(pathnames, dtype=np.int16, join_contiguous=False):
    """ Reads a ScanImage scan. """
    # Expand wildcards
    filenames = expand_wildcard(pathnames)
    if len(filenames) == 0:
        error_msg = 'Pathname(s) {} do not match any files in disk.'.format(pathnames)
        raise PathnameError(error_msg)

    scan = ScanMultiROIReordered(join_contiguous=join_contiguous)

    # Read metadata and data (lazy operation)
    scan.read_data(filenames, dtype=dtype)

    return scan


class ScanMultiROIReordered(scans.ScanMultiROI):
    """
    A subclass of ScanMultiROI that ignores the num_fields dimension
    and reorders the output to [time, z, x, y].
    """

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Call the parent class's __getitem__ with the reordered key
        item = super().__getitem__((0, key[2], key[3], key[1], key[0]))
        if item.ndim == 2:
            return item
        elif item.ndim == 3:
            return np.transpose(item, (2, 0, 1))
        else:
            raise FieldDimensionMismatch('ScanMultiROIReordered.__getitem__')

    @property
    def shape(self):
        return self.num_frames, self.num_channels, self.field_heights[0], self.field_widths[0]

    @property
    def ndim(self):
        return 4


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
    parser.add_argument('--path', type=str,
                        help='Path to a directory containing raw scanimage tiff files for a single session.')
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

    files = [str(f) for f in Path(args.path).glob('*.tif*')]
    scan = read_scan(files, join_contiguous=True)
    iw = fpl.ImageWidget(scan, histogram_widget=False)
    iw.show()
    if fpl.__version__ == "0.2.0":
        fpl.run()
    elif fpl.__version__ == "0.3.0":
        fpl.loop.run()


if __name__ == '__main__':
    main()
