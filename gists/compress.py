# /// script
# requires-python = ">=3.13"
# dependencies = ["click", "numpy", "tifffile", "tqdm"]
# ///
import os
import numpy as np
import tifffile
from pathlib import Path
import click
from tqdm import tqdm


def create_compressed_downsampled(tiff_path, output_path, max_bytes=500 * 1024 * 1024, dtype="int16"):
    tiff_path = Path(tiff_path)
    data = tifffile.memmap(str(tiff_path), mode="r")
    T, Y, X = data.shape
    itemsize = np.dtype(dtype).itemsize
    frame_size = Y * X * itemsize
    max_frames = max_bytes // frame_size
    step = (T + max_frames - 1) // max_frames
    downsampled = data[::step].astype(dtype)
    new_T = downsampled.shape[0]
    raw_bytes = new_T * frame_size
    output_file = Path(output_path).with_suffix(".npz")
    np.savez_compressed(str(output_file), downsampled=downsampled)
    print(f"Saved to {output_file} ({new_T} frames, {raw_bytes / 1024 ** 2:.1f} MB raw)")


@click.command()
@click.option(
    "--input", "input_path",
    prompt=f"Path to TIFF or folder{' [use backslashes]' if os.name == "nt" else ''}",
    type=click.Path(exists=True)
)
@click.option("--unit", prompt="Size unit (MB or GB)", type=click.Choice(["MB", "GB"], case_sensitive=False))
@click.option("--size", prompt="Max size in selected unit", type=float)
def cli(input_path, unit, size):
    base = 1024 ** 2 if unit.upper() == "MB" else 1024 ** 3
    max_bytes = int(size * base)
    input_path = Path(input_path)

    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        outpath = output_dir / input_path.stem
        create_compressed_downsampled(input_path, outpath, max_bytes=max_bytes)
    else:
        # TODO: use mbo.get_files once imports are more clean, i.e. fast pip install mbo_utilities without GUI cmds
        tiffs = sorted(input_path.glob("*.tif*")) + sorted(input_path.glob("*.tiff"))
        with tqdm(total=len(tiffs), desc="Compressing TIFFs") as pbar:
            for tif in tiffs:
                outpath = output_dir / tif.stem
                create_compressed_downsampled(tif, outpath, max_bytes=max_bytes)
                pbar.update(1)


if __name__ == "__main__":
    cli()
