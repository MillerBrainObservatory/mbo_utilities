import time

from pathlib import Path
from mbo_utilities import get_files, imread, imwrite

def run_plane_bin(ops) -> bool:
    from mbo_utilities._binary import BinaryFile
    from suite2p.run_s2p import pipeline
    from contextlib import nullcontext
    import numpy as np
    from pathlib import Path

    ops = lsp.load_ops(ops)
    Ly, Lx = ops["Ly"], ops["Lx"]

    # input functional channel (unregistered)
    raw_file = ops.get("raw_file")
    nframes_chan1 = ops.get("nframes_chan1") or ops.get("nframes") or ops.get("n_frames")
    if raw_file is None or nframes_chan1 is None:
        raise KeyError("Missing raw_file or nframes_chan1")

    # optional structural channel
    chan2_file = ops.get("chan2_file", "")
    nframes_chan2 = ops.get("nframes_chan2", 0)
    if chan2_file and nframes_chan2 > 0:
        ops["align_by_chan"] = 2

    # define registered output file for Suite2p GUI
    ops_parent = Path(ops.get("ops_path")).parent
    ops["save_path"] = ops_parent

    reg_file = ops_parent / "data.bin"
    ops["reg_file"] = str(reg_file)

    # sanity fix for diameter
    if "diameter" in ops:
        if ops["diameter"] is not None and np.isnan(ops["diameter"]):
            ops["diameter"] = 8
        if (ops["diameter"] is None or ops["diameter"] == 0) and ops.get("anatomical_only", 0) > 0:
            ops["diameter"] = 8
            print("Warning: diameter was not set, defaulting to 8.")

    with (
        BinaryFile(Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=nframes_chan1) as f_reg,
        BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file, n_frames=nframes_chan1) as f_raw,
        (BinaryFile(Ly=Ly, Lx=Lx, filename=chan2_file, n_frames=nframes_chan2)
         if chan2_file and nframes_chan2 else nullcontext()) as f_reg_chan2
    ):
        ops = pipeline(
            f_reg=f_reg,
            f_raw=f_raw,
            f_reg_chan2=f_reg_chan2,
            f_raw_chan2=f_reg_chan2,  # critical fix
            run_registration=ops.get("do_registration", True),
            ops=ops,
            stat=None,
        )

    np.save(ops["ops_path"], ops)
    return True


if __name__ == "__main__":
    import lbm_suite2p_python as lsp

    base_path = Path(r"D:\demo\multichannel")
    structural = get_files(base_path / "structural")
    functional = get_files(base_path / "functional")

    start = time.time()
    s_data = imread(structural)
    f_data = imread(functional)
    end = time.time()
    print(f"{end - start} seconds")

    outpath = structural[0].parent.joinpath("registered")
    structural_nframes = s_data.shape[0]
    functional_nframes = f_data.shape[0]
    min_nframes = min(structural_nframes, functional_nframes)
    imwrite(s_data, outpath, structural=True, ext=".bin", overwrite=True, planes=[1], num_frames=min_nframes)
    imwrite(f_data, outpath, ext=".bin", overwrite=True, planes=[1], num_frames=min_nframes)
    ops_files = get_files(outpath, "ops.npy", 3)
    _ops = lsp.load_ops(ops_files[0])
    _ops["anatomical_only"] = 3
    _ops["cellprob_threshold"] = -6
    run_plane_bin(_ops)