import time
from pathlib import Path
# import numpy as np
# from lbm_suite2p_python import load_ops

from mbo_utilities import get_files, imread, imwrite

# import lbm_suite2p_python as lsp
# from lbm_suite2p_python.run_lsp import run_plane_bin, _should_register
from mbo_utilities.file_io import MBO_SUPPORTED_FTYPES

base_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green")
out_path = Path(f"D://demo//v2.1.0//mrois")
out_path.mkdir(parents=True, exist_ok=True)
data = imread(base_path)
data.fix_phase = True
data.phasecorr_method = "mean"
data.use_fft = True
times = {}

if __name__ == "__main__":
    for ext in MBO_SUPPORTED_FTYPES:
        start = time.time()
        imwrite(
            data,
            f"{out_path.joinpath(ext)}",
            register_z=True,
            num_frames=45_000,
            roi=0,
            ext=ext,
        )
        end = time.time()
        times[ext] = f"{end - start:.2f}"

    times = {}
    data.roi = None
    out_path = Path(f"D://demo//v2.1.0//stitched")

    for ext in MBO_SUPPORTED_FTYPES:
        start = time.time()
        imwrite(
            data,
            f"{out_path.joinpath(ext)}",
            register_z=True,
            num_frames=45_000,
            roi=0,
            ext=ext,
        )
        end = time.time()
        times[ext] = f"{end - start:.2f}"

    # base_path = Path(r"D:\demo\multichannel")
    # structural = get_files(base_path / "structural")
    # functional = get_files(base_path / "functional")
    #
    # func_only_outdir = base_path / "functional_only_registration"
    # struct_only_outdir = base_path / "structural_only_registration"
    #
    # print(f"Loading structural and functional stacks from {base_path}")
    # t0 = time.time()
    # s_data = imread(structural)
    # f_data = imread(functional)
    # print(f"Loaded in {time.time() - t0:.2f}s")
    #
    # # Match frame counts and write Suite2p binaries
    # structural_nframes = s_data.shape[0]
    # functional_nframes = f_data.shape[0]
    # min_nframes = min(structural_nframes, functional_nframes)
    #
    # for outdir in (func_only_outdir, struct_only_outdir):
    #     outdir.mkdir(exist_ok=True)
    #     print(f"\nWriting binaries to {outdir}")
    #
    #     imwrite(
    #         s_data, outdir, structural=True, ext=".bin", overwrite=False,
    #         planes=None, num_frames=min_nframes
    #     )
    #     imwrite(
    #         f_data, outdir, ext=".bin", overwrite=False,
    #         planes=None, num_frames=min_nframes
    #     )
    #
    # ops_files = get_files(outdir, "ops.npy", 3)
    #
    # for ops_path in ops_files:
    #     for label, align_chan in [
    #         ("functional-only", 1),
    #         ("structural (red)", 2),
    #     ]:
    #         print(f"\n=== Running {label} registration for {ops_path} ===")
    #
    #         if not _should_register(ops_path):
    #             print(f"Skipping {label} for {ops_path} (register already exists)")
    #             continue
    #
    #         ops = load_ops(ops_path)
    #         ops["anatomical_only"] = 3
    #         ops["cellprob_threshold"] = -6
    #         ops["align_by_chan"] = align_chan
    #         ops["roidetect"] = False
    #         np.save(ops_path, ops)
    #
    #         run_plane_bin(str(ops_path))
    #
    # # Compare results
    # import matplotlib.pyplot as plt
    # from mbo_utilities import _binary
    #
    #
    # def frame_std_trace(bin_file, Ly, Lx, sample=300):
    #     with _binary.BinaryFile(Ly=Ly, Lx=Lx, filename=bin_file) as f:
    #         idx = np.linspace(0, f.shape[0] - 1, sample, dtype=int)
    #         return np.array([f[i].std() for i in idx])
    #
    # def plot_all_comparisons(func_root, struct_root, save_dir=None, sample=300):
    #     func_ops_files = get_files(func_root, "ops.npy", 3)
    #     struct_ops_files = get_files(struct_root, "ops.npy", 3)
    #     n = min(len(func_ops_files), len(struct_ops_files))
    #     if n == 0:
    #         raise FileNotFoundError("No matching ops.npy files found for comparison")
    #
    #     for i in range(n):
    #         fpath = Path(func_ops_files[i])
    #         spath = Path(struct_ops_files[i])
    #         ops_func = lsp.load_ops(fpath)
    #         ops_struct = lsp.load_ops(spath)
    #         Ly, Lx = ops_func["Ly"], ops_func["Lx"]
    #         std_func = frame_std_trace(ops_func["reg_file"], Ly, Lx, sample)
    #         std_struct = frame_std_trace(ops_struct["reg_file"], Ly, Lx, sample)
    #
    #         plt.figure(figsize=(8, 4))
    #         plt.plot(std_func, label="functional-only")
    #         plt.plot(std_struct, label="with structural channel")
    #         plt.xlabel("Frame index")
    #         plt.ylabel("Per-frame std")
    #         plt.title(f"Registration quality comparison\n{fpath.parent.name}")
    #         plt.legend()
    #         plt.tight_layout()
    #
    #         if save_dir:
    #             out = Path(save_dir) / f"{fpath.parent.name}_reg_quality.png"
    #             out.parent.mkdir(parents=True, exist_ok=True)
    #             plt.savefig(out, dpi=150)
    #             plt.close()
    #         else:
    #             plt.show()
    #
    #         print(f"{fpath.parent.name}: mean std func={np.mean(std_func):.4f}, struct={np.mean(std_struct):.4f}")
    #
    # plot_all_comparisons(func_only_outdir, struct_only_outdir, save_dir=outdir / "plots")
