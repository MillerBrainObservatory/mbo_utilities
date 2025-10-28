# /// script
# requires-python = ">=3.12.7, <3.12.10"
# dependencies = [
#     "mbo_utilities",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "dev" }

import time
from pathlib import Path
from mbo_utilities.lazy_array import imread, imwrite


def benchmark_wrapper(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    duration = time.time() - start

    # infer output directory (second positional arg or 'outpath' kwarg)
    outdir = Path(kwargs.get("outpath", args[1] if len(args) > 1 else Path.cwd()))
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = outdir / "benchmark_log.txt"
    with log_path.open("a") as f:
        f.write(f"{func.__name__} took {duration:.2f} seconds\n")

    print(f"{func.__name__} took {duration:.2f} seconds (logged to {log_path})")
    return duration


def run_with_fft(raw_data_path, outpath):
    data = imread(raw_data_path)

    data.roi = 0
    data.fix_phase = True
    data.use_fft = True

    imwrite(
        data,
        outpath,
        register_z=True,
        ext=".zarr",
        overwrite=True,
        planes=None,  # all zplanes
    )


def run_without_fft(raw_data_path, outpath):
    data = imread(raw_data_path)
    data.roi = 0
    data.fix_phase = True
    data.use_fft = False
    imwrite(data, outpath, register_z=True, ext=".zarr", overwrite=True, planes=None)


if __name__ == "__main__":
    path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green")

    fft_out = Path("D:/demo/mrois_fft")
    nofft_out = Path("D:/demo/mrois_nofft")

    fft_runtime = benchmark_wrapper(run_with_fft, path, fft_out)
    no_fft_runtime = benchmark_wrapper(run_without_fft, path, nofft_out)
