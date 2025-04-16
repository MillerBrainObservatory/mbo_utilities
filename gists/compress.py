# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "fastplotlib",
#     "glfw",
# ]
# ///
import os
import math
import numpy as np
import tifffile
from pathlib import Path

def create_compressed_downsampled(tiff_path, output_path, max_bytes=500*1024*1024, dtype="int16"):
    """
    Loads a 3D TIFF file (T, Y, X), downsamples the time axis so that the total size
    (in raw binary bytes for dtype) is below max_bytes, and saves the downsampled movie
    to a compressed .npz file.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the input TIFF file.
    output_path : str or Path
        Path (without extension) to save the compressed file.
    max_bytes : int, optional
        Maximum allowed file size in bytes (default is 500MB).
    dtype : str or np.dtype, optional
        Data type to enforce.
    """
    tiff_path = Path(tiff_path)
    data = tifffile.memmap(str(tiff_path), mode="r")
    T, Y, X = data.shape
    itemsize = np.dtype(dtype).itemsize

    # Each frame (of size Y x X) occupies:
    frame_size = Y * X * itemsize  # in bytes

    # Maximum number of frames to stay below max_bytes:
    max_frames = max_bytes // frame_size

    # Calculate downsampling step (ceil to ensure we do not exceed max_bytes):
    step = math.ceil(T / max_frames)
    downsampled = data[::step].astype(dtype)

    new_T = downsampled.shape[0]
    raw_bytes = new_T * frame_size
    print(f"Original movie: {T} frames; each frame {frame_size} bytes.")
    print(f"Downsampling step: {step} (resulting in {new_T} frames, ~{raw_bytes / (1024**2):.1f}MB raw).")

    # Save as a compressed npz file:
    output_file = Path(output_path).with_suffix(".npz")
    np.savez_compressed(str(output_file), downsampled=downsampled)
    print(f"Saved compressed movie to {output_file}")

if __name__ == "__main__":
    tiff_path = r"D:\W2_DATA\kbarber\2025_03_01\mk301\assembled\plane_01_mk301.tiff"
    output_path = "../data/downsampled_movie"
    create_compressed_downsampled(tiff_path, output_path)
