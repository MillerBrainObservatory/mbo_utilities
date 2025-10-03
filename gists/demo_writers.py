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
    result = func(*args, **kwargs)
    end = time.time()
    with open("benchmark_log.txt", "a") as f:
        f.write(f"Function {func.__name__} took {end - start:.2f} seconds\n"
                f"Args: {args}, Kwargs: {kwargs}\n\n")
    total_seconds_runtime = end - start
    print(f"Function {func.__name__} took {total_seconds_runtime:.2f} seconds")
    return total_seconds_runtime

def run_with_fft(raw_data_path):
    outpath = raw_data_path.parent.joinpath("green.processed-fft")
    outpath.mkdir(exist_ok=True)

    data = imread(raw_data_path)

    data.roi = None
    data.fix_phase = True
    data.use_fft = False

    imwrite(
        data,
        outpath,
        register_z=True,
        ext=".zarr", overwrite=True, planes=None
    )

def run_without_fft(raw_data_path):
    outpath = raw_data_path.parent.joinpath("green.processed-no-fft")
    outpath.mkdir(exist_ok=True)

    data = imread(raw_data_path)
    data.roi = None
    data.fix_phase = True
    data.use_fft = True
    imwrite(
        data,
        outpath,
        register_z=True,
        ext=".zarr", overwrite=True, planes=None
    )


if __name__ == "__main__":

    path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green")
    fft_runtime = benchmark_wrapper(run_with_fft, path)
    no_fft_runtime = benchmark_wrapper(run_without_fft, path)
