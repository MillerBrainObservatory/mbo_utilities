import argparse
from pathlib import Path
from mbo_utilities import imread, imwrite, __version__


def add_args(parser):
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output directory or filename")
    parser.add_argument(
        "--ext", default=".tiff", help="File extension for output (e.g., .tif, .npy)"
    )
    parser.add_argument(
        "--planes", nargs="*", type=int, help="Planes to export (1-based index)"
    )
    parser.add_argument("--roi", nargs="*", type=int, help="ROI indices to include")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--target-chunk-mb", type=int, default=20, help="Target chunk size in MB"
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser


def main():
    parser = argparse.ArgumentParser(description="TODO!")
    parser = add_args(parser)
    args = parser.parse_args()

    if args.version:
        print(f"v{__version__}")
        return

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    lazy_array = imread(input_path, roi=args.roi)
    print(type(lazy_array))

    imwrite(
        lazy_array=lazy_array,
        outpath=output_path,
        planes=args.planes,
        roi=args.roi,
        overwrite=args.overwrite,
        ext=args.ext,
        target_chunk_mb=args.target_chunk_mb,
        debug=args.debug,
    )

    print(f"\nDone. Output saved to {output_path}\n")
