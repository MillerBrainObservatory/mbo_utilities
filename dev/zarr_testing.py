from pathlib import Path
import mbo_utilities as mbo
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


def bench_phasecorr(data_path, out_dir=None, method="mean"):
    """
    Benchmark real I/O + phase correction performance for FFT vs non-FFT backends.
    Logs total time and per-1000-frame throughput.
    """
    data_path = Path(data_path)
    if out_dir is None:
        out_dir = data_path.parent / "phasecorr_bench"
    out_dir.mkdir(exist_ok=True)

    backends = [False, True]  # use_fft flag

    for use_fft in backends:
        print(f"--- method={method}, use_fft={use_fft} ---")

        # Load and configure reader
        start = time.time()
        x = mbo.imread(data_path, fix_phase=True, use_fft=use_fft)
        load_time = time.time() - start

        n_frames = getattr(x, "num_frames", len(x))
        start = time.time()

        # Write back with phase correction
        label = f"{method}_fft{int(use_fft)}"
        out_file = out_dir / label
        mbo.imwrite(
            x,
            out_file,
            register_z=True,
            ext=".zarr",
            roi=None,
        )
        write_time = time.time() - start
        total_time = load_time + write_time
        per_1000 = total_time / (n_frames / 1000)

        # Log
        log_file = out_dir / f"{label}.txt"
        with open(log_file, "w") as f:
            f.write(f"Method: {method}\n")
            f.write(f"Use FFT: {use_fft}\n")
            f.write(f"Frames: {n_frames}\n")
            f.write(f"Load time: {load_time:.2f}s\n")
            f.write(f"Write time: {write_time:.2f}s\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Seconds per 1000 frames: {per_1000:.2f}s\n")

        print(f"{label} finished in {total_time:.2f}s ({per_1000:.2f}s/1k frames)")

    print(f"\nLogs written to {out_dir}")


if __name__ == "__main__":
    data_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
    out_dir = data_path.parent / "phasecorr_bench"
    bench_phasecorr(data_path, out_dir)
