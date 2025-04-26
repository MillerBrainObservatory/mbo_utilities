from pathlib import Path
import tifffile

try:
    from suite2p.io.binary import BinaryFile
except ImportError:
    BinaryFile = None


def tiff_to_binary(tiff_path, out_path, dtype="int16"):
    data = tifffile.memmap(tiff_path)
    out_path = Path(out_path).with_suffix(".bin")

    if data.ndim != 3:
        raise ValueError("Must be assembled, 3D (T, Y, X)")

    nframes, x, y = data.shape
    bf = BinaryFile(
        Ly=y, Lx=x, filename=str(Path(out_path)), n_frames=nframes, dtype=dtype
    )

    bf[:] = data
    bf.close()

    print(f"Wrote binary file '{out_path}' with {nframes} frames of shape ({x},{y}).")
