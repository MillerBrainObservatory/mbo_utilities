# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "tifffile",
# ]
# ///
import os
import numpy as np
import tifffile
from pathlib import Path

def create_compressed_downsampled(tiff_path, output_path, max_bytes=500*1024*1024, dtype="int16"):
    """
    Load a 3D TIFF (T, Y, X), downsample the time axis to keep raw data under max_bytes, and save as a compressed npz.
    """

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
    print(f"Downsampling step: {step}, {new_T} frames, {raw_bytes/1024**2:.1f}MB raw.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    tiff_path = r"D:\W2_DATA\kbarber\2025_03_01\mk301\assembled\plane_01_mk301.tiff"
    output_path = "../data/downsampled_movie"
    create_compressed_downsampled(tiff_path, output_path)
