from pathlib import Path
import shutil
import numpy as np
import time
import mbo_utilities as mbo
import zarr
import tifffile
from zarr.codecs import BloscCodec
import fastplotlib as fpl
# from mbo_utilities.util import align_zplanes

# arr = mbo.imread(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_planar.zarr\plane01_stitched.zarr")
# fpl.ImageWidget(arr).show()
# fpl.loop.run()
#
# arr = mbo.imread(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_planar.zarr")
# fpl.ImageWidget(arr).show()
# fpl.loop.run()

import time
from pathlib import Path
import zarr
from numcodecs import Blosc as BloscCodec
import mbo_utilities as mbo
from numcodecs import Blosc, Zstd, GZip, LZ4


import time
from numcodecs import Blosc


def benchmark_compression(data_path, out_dir=None, clevels=(1, 3)):
    data_path = Path(data_path)
    if out_dir is None:
        out_dir = data_path.parent / "compression"
    out_dir.mkdir(exist_ok=True)

    codecs = {
        "blosc_zstd": lambda c: Blosc(cname="zstd", clevel=c, shuffle=Blosc.BITSHUFFLE),
        "blosc_lz4": lambda c: Blosc(cname="lz4", clevel=c, shuffle=Blosc.BITSHUFFLE),
        "blosc_lz4hc": lambda c: Blosc(
            cname="lz4hc", clevel=c, shuffle=Blosc.BITSHUFFLE
        ),
        "blosc_zlib": lambda c: Blosc(cname="zlib", clevel=c, shuffle=Blosc.BITSHUFFLE),
    }

    x = mbo.imread(data_path, fix_phase=False)

    for name, fn in codecs.items():
        for clevel in clevels:
            codec = fn(clevel)
            label = f"{name}-c{clevel}"
            out_file = out_dir / label

            print(f"Running {label}")
            start = time.time()
            mbo.imwrite(
                x,
                out_file,
                register_z=False,
                ext=".zarr",
                roi=None,
                filters=[codec],
                planes=[7],
            )
            end = time.time()

            with open(out_dir / f"{label}.txt", "w") as f:
                f.write(f"Codec: {name}\n")
                f.write(f"Clevel: {clevel}\n")
                f.write(f"Time: {end - start:.2f}s\n")


def benchmark_phasecorr(data_path, out_dir=None):
    """
    Run a grid search over phase correlation methods and FFT vs. correlation backends.

    Parameters
    ----------
    data_path : Path or str
        Path to a directory of .tif files.
    out_dir : Path or str, optional
        Where to save outputs. Defaults to data_path/../phasecorr.
    """
    data_path = Path(data_path)
    files = list(data_path.glob("*.tif*"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found under {data_path}")

    if out_dir is None:
        out_dir = data_path.parent / "phasecorr"
    out_dir.mkdir(exist_ok=True)

    methods = ["frame", "mean", "max", "std", "mean-sub"]
    backends = [False, True]  # use_fft flag

    for method in methods:
        for use_fft in backends:
            print(f"Running method={method}, use_fft={use_fft}")
            # load
            x = mbo.imread(data_path, use_fft=use_fft)
            x.phasecorr_method = method

            # save
            label = f"{method}-fft{int(use_fft)}"
            out_file = out_dir / label
            start = time.time()
            x = mbo.imwrite(
                x,
                out_file,
                register_z=True,
                ext=".zarr",
                roi=None,
                compressor=compressors,
            )
            end = time.time()

            # log timing
            with open(out_dir / f"{label}.txt", "w") as f:
                f.write(f"Phase correlation method: {method}\n")
                f.write(f"Use FFT: {use_fft}\n")
                f.write(f"Time taken: {end - start:.2f} seconds\n")

    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    import zarr
    # import fastplotlib as fpl
    # x = mbo.imread(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
    # fpl.ImageWidget(x).show()
    # fpl.loop.run()
    #
    # z = zarr.open(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_planar\plane01_stitched.zarr", mode='r')

    data_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
    files = list(data_path.glob("*.tif*"))
    compressors = BloscCodec(
        cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle
    )
    x = mbo.imread(data_path, fix_phase=False, use_fft=False)
    x.phasecorr_method = "mean"
    out_file = data_path.parent / "phasecorr"
    out_file.mkdir(exist_ok=True)
    # benchmark_phasecorr(data_path, out_dir=out_file)
    benchmark_compression(data_path, out_dir=out_file)
    start = time.time()
    x = mbo.imwrite(
        x,
        out_file / f"{x.phasecorr_method}-{x.use_fft}",
        register_z=True,
        ext=".zarr",
        roi=None,
    )
    end = time.time()
    with open(out_file / f"{x.phasecorr_method}-{x.use_fft}.txt", "w") as f:
        f.write(f"Phase correlation method: {x.phasecorr_method}\n")
        f.write(f"Use FFT: {x.use_fft}\n")
        f.write(f"Time taken: {end - start:.2f} seconds\n")
