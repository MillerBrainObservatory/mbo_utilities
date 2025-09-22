from pathlib import Path
import numpy as np
from .lazy_array import Suite2pArray

def load_ops(ops_input: str | Path | list[str | Path]) -> dict:
    """Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning empty dict.")
    return {}


def embed_image(cropped, yrange, xrange, full_shape):
    full = np.zeros(full_shape, dtype=cropped.dtype)
    y0, y1 = yrange
    x0, x1 = xrange
    h = min(y1 - y0, cropped.shape[0])
    w = min(x1 - x0, cropped.shape[1])
    full[y0:y0 + h, x0:x0 + w] = cropped[:h, :w]
    return full


def merge_rois(roi_dirs, save_path):
    roi_dirs = [Path(d) for d in roi_dirs]
    assert len(roi_dirs) >= 2

    ops_list = []
    stat_list = []
    iscell_list = []
    F_list = []
    Fneu_list = []
    spks_list = []
    bin_paths_list = []

    for d in roi_dirs:
        ops = np.load(d / "ops.npy", allow_pickle=True).item()
        stat = np.load(d / "stat.npy", allow_pickle=True)
        iscell = np.load(d / "iscell.npy", allow_pickle=True)
        F = np.load(d / "F.npy")
        Fneu = np.load(d / "Fneu.npy")
        spks = np.load(d / "spks.npy")
        bin_path = d / "data.bin"

        ops_list.append(ops)
        stat_list.append(stat)
        iscell_list.append(iscell)
        F_list.append(F)
        Fneu_list.append(Fneu)
        spks_list.append(spks)
        bin_paths_list.append(bin_path)

    Ly = ops_list[0]["Ly"]
    for ops in ops_list:
        assert ops["Ly"] == Ly

    Lx_list = [ops["Lx"] for ops in ops_list]
    total_Lx = sum(Lx_list)

    for i, stat in enumerate(stat_list[1:], start=1):
        offset = sum(Lx_list[:i])
        for s in stat:
            s["xpix"] += offset
            s["med"][1] += offset
            if "ipix_neuropil" in s:
                s["ipix_neuropil"] += offset * Ly

    stat = np.concatenate(stat_list)
    iscell = np.concatenate(iscell_list)
    F = np.concatenate(F_list)
    Fneu = np.concatenate(Fneu_list)
    spks = np.concatenate(spks_list)
    print([p for p in bin_paths_list])

    arrays = [Suite2pArray(p) for p in bin_paths_list]
    nframes = arrays[0].nframes
    dtype = arrays[0].dtype

    with open(Path(save_path) / "data.bin", "wb") as f:
        for i in range(nframes):
            frames = [arr[i] for arr in arrays]
            f.write(np.hstack(frames).astype(dtype).tobytes())

    for arr in arrays:
        arr.close()

    merged_ops = dict(ops_list[0])
    merged_ops["Ly"] = Ly
    merged_ops["Lx"] = total_Lx
    merged_ops["yrange"] = [0, Ly]
    merged_ops["xrange"] = [0, total_Lx]
    merged_ops["reg_file"] = str((Path(save_path) / "data.bin").resolve())
    merged_ops["ops_path"] = str((Path(save_path) / "ops.npy").resolve())

    for key in ("meanImg", "meanImgE", "max_img", "Vcorr"):
        if all(key in ops for ops in ops_list):
            if key != "Vcorr":
                merged_ops[key] = np.hstack([ops[key] for ops in ops_list])
            else:
                canvases = []
                for ops in ops_list:
                    canvas = embed_image(ops[key], ops["yrange"], ops["xrange"], (Ly, total_Lx))
                    canvases.append(canvas)
                merged_ops[key] = sum(canvases)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "ops.npy", merged_ops)
    np.save(save_dir / "stat.npy", stat)
    np.save(save_dir / "iscell.npy", iscell)
    np.save(save_dir / "F.npy", F)
    np.save(save_dir / "Fneu.npy", Fneu)
    np.save(save_dir / "spks.npy", spks)