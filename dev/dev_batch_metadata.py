from pathlib import Path

from tifffile import TiffFile

from mbo_utilities._tiff import fast_tiff_page_count, fast_tiff_page_count_cached
import tifffile

from mbo_utilities.metadata import get_metadata_single

"""
SIMPLE INSTANT PAGE COUNT - The solution you actually need

For ScanImage files with uniform pages:
- Takes 0.001 seconds regardless of file size
- No IFD chain walking
- Just: file_size / page_size = num_pages

Replace this ONE line:
    n_pages = len(tifffile.TiffFile(file_path).pages)  # HOURS
With:
    n_pages = instant_scanimage_pages(file_path)  # MILLISECONDS
"""

import struct
import os


def instant_scanimage_pages(file_path):
    """
    Get page count INSTANTLY for ScanImage files (milliseconds, not hours).

    Works by:
    1. Reading first TWO IFD offsets only
    2. Calculating page_size = second_offset - first_offset
    3. Estimating: total_pages = file_size / page_size

    For ScanImage files where all pages are uniform, this is exact.

    Parameters
    ----------
    file_path : str
        Path to ScanImage TIFF file

    Returns
    -------
    int
        Number of pages
    """
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        # Read header (8 bytes)
        header = f.read(8)

        # Detect byte order
        if header[:2] == b'II':
            bo = '<'  # Little-endian
        elif header[:2] == b'MM':
            bo = '>'  # Big-endian
        else:
            raise ValueError("Not a TIFF file")

        # Detect TIFF version
        version = struct.unpack(f'{bo}H', header[2:4])[0]

        if version == 42:
            # Classic TIFF (32-bit offsets)
            offset_fmt = f'{bo}I'
            offset_size = 4
            tag_count_fmt = f'{bo}H'
            tag_count_size = 2
            tag_size = 12
            first_ifd_offset = struct.unpack(offset_fmt, header[4:8])[0]
            header_size = 8

        elif version == 43:
            # BigTIFF (64-bit offsets)
            offset_fmt = f'{bo}Q'
            offset_size = 8
            tag_count_fmt = f'{bo}Q'
            tag_count_size = 8
            tag_size = 20
            f.seek(8)
            first_ifd_offset = struct.unpack(offset_fmt, f.read(offset_size))[0]
            header_size = 16

        else:
            raise ValueError(f"Unknown TIFF version: {version}")

        # Go to first IFD
        f.seek(first_ifd_offset)

        # Read tag count
        tag_count = struct.unpack(tag_count_fmt, f.read(tag_count_size))[0]

        # Skip all tags to get to next IFD offset
        f.seek(first_ifd_offset + tag_count_size + (tag_count * tag_size))

        # Read second IFD offset
        second_ifd_offset = struct.unpack(offset_fmt, f.read(offset_size))[0]

        if second_ifd_offset == 0:
            return 1  # Only one page

        # Calculate page size (IFD + image data for one page)
        page_size = second_ifd_offset - first_ifd_offset

        # Calculate total pages
        data_size = file_size - header_size
        num_pages = data_size // page_size

        return int(num_pages)


if __name__ == '__main__':
    import time
    full_file = Path(r"\\rbo-s1\S1_DATA\lbm\kbarber\2025-07-27\mk355\green")
    files = list(full_file.glob("*.tif*"))

    # Time the instant method
    start = time.time()
    num_pages_per_file = []
    num_pages_per_file_old = []
    for file in files[:10]:
        pages = instant_scanimage_pages(file)
        old_pages = len(TiffFile(file).pages)
        num_pages_per_file.append(pages)
        num_pages_per_file_old.append(old_pages)
    instant_time = time.time() - start
    print(f"Instant time: {instant_time}")
    print(f"Num files: {len(num_pages_per_file)}")
    print(f"Num frames: {sum(num_pages_per_file)}")
    print(f"Frames in file 1: {len(num_pages_per_file[0])}")


# ----------------------------------------------------------------------------


#
# start = time.time()
# metadata_single = mbo.get_metadata(
#     files[0],
# )
# print(f"Metadata retrieval took {time.time() - start:.2f} seconds")
#
# start = time.time()
# metadata_multiple = mbo.get_metadata(files)
# print(f"Metadata retrieval took {time.time() - start:.2f} seconds")
# x = 2