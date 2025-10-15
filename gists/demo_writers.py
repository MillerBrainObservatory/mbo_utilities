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
    total_seconds_runtime = end - start

    # infer outpath if function defines one inside
    outpath = None
    for a in args:
        if isinstance(a, (str, Path)) and "demo" in str(a):
            outpath = Path(a)
    if "outpath" in kwargs:
        outpath = Path(kwargs["outpath"])
    if outpath is None:
        # fallback to current dir
        outpath = Path.cwd()

    if not outpath.exists():
        if hasattr(func, "__name__"):
            if "fft" in func.__name__:
                outpath = Path(r"D://demo//mrois_fft")
            elif "nofft" in func.__name__:
                outpath = Path(r"D://demo//mrois_nofft")
    outpath.mkdir(parents=True, exist_ok=True)

    log_path = outpath / "benchmark_log.txt"
    with open(log_path, "a") as f:
        f.write(f"{func.__name__} took {total_seconds_runtime:.2f}s\n")
    print(f"{func.__name__} took {total_seconds_runtime:.2f}s (logged to {log_path})")
    return total_seconds_runtime

def run_with_fft(raw_data_path):
    outpath = r"D://demo//mrois_fft"
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

def run_without_fft(raw_data_path):
    outpath = r"D://demo//mrois_nofft"

    data = imread(raw_data_path)
    data.roi = 0
    data.fix_phase = True
    data.use_fft = False
    imwrite(
        data,
        outpath,
        register_z=True,
        ext=".zarr",
        overwrite=True,
        planes=None
    )


if __name__ == "__main__":

    path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green")
    fft_runtime = benchmark_wrapper(run_with_fft, path)
    no_fft_runtime = benchmark_wrapper(run_without_fft, path)
