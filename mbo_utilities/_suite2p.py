import time
from pathlib import Path
import lbm_suite2p_python as lsp
from mbo_utilities.lazy_array import Suite2pArray
import numpy as np
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from skimage.segmentation import find_boundaries

def safe_delete(file_path):
    if file_path.exists():
        try:
            file_path.unlink()
        except PermissionError:
            print(f"Error: Cannot delete {file_path}, it's open elsewhere.")

def group_plane_rois(input_dir):
    input_dir = Path(input_dir)
    grouped = defaultdict(list)

    for d in input_dir.iterdir():
        if d.is_dir() and d.stem.startswith("plane") and "_roi" in d.stem:
            parts = d.stem.split("_")
            if len(parts) == 2 and parts[1].startswith("roi"):
                plane = parts[0]  # "plane01"
                grouped[plane].append(d)

    return grouped

def load_ops(ops_input: str | Path | list[str | Path]) -> dict:
    """Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning empty dict.")
    return {}

def merge_rois_auto(input_dir, output_dir, overwrite=True):
    """
    Merge Suite2p outputs from multiple ROIs into per-plane outputs.

    Parameters
    ----------
    input_dir : str or Path
        Directory with subfolders like plane01_roi1, plane01_roi2, ...
    output_dir : str or Path
        Directory where merged outputs will be saved (plane01, plane02, ...).
    overwrite : bool
        If False, skip merging if ops.npy already exists in target.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    grouped = group_plane_rois(input_dir)

    for plane, dirs in sorted(grouped.items()):
        out_dir = output_dir / plane
        out_ops = out_dir / "ops.npy"

        if out_ops.exists() and not overwrite:
            print(f"Skipping {plane}, merged outputs already exist")
            continue

        out_dir.mkdir(exist_ok=True)

        # --- Load all ROI results
        ops_list, stat_list, iscell_list = [], [], []
        F_list, Fneu_list, spks_list = [], [], []
        bin_paths = []

        for d in sorted(dirs):
            ops = np.load(d / "ops.npy", allow_pickle=True).item()
            stat = np.load(d / "stat.npy", allow_pickle=True)
            iscell = np.load(d / "iscell.npy", allow_pickle=True)
            F = np.load(d / "F.npy")
            Fneu = np.load(d / "Fneu.npy")
            spks = np.load(d / "spks.npy")

            # prefer data_raw.bin if it exists, else data.bin
            bin_path = (d / "data_raw.bin") if (d / "data_raw.bin").exists() else (d / "data.bin")

            ops_list.append(ops)
            stat_list.append(stat)
            iscell_list.append(iscell)
            F_list.append(F)
            Fneu_list.append(Fneu)
            spks_list.append(spks)
            bin_paths.append(bin_path)

        # --- Dimensions
        Ly = ops_list[0]["Ly"]
        for ops in ops_list:
            if ops["Ly"] != Ly:
                raise ValueError("Inconsistent Ly across ROIs")
        Lx_list = [ops["Lx"] for ops in ops_list]
        total_Lx = sum(Lx_list)

        # --- Shift stat coordinates by ROI offsets
        for i, stat in enumerate(stat_list):
            offset_x = sum(Lx_list[:i])

            for s in stat:
                # shift ROI into global canvas
                s["xpix"] = np.asarray(s["xpix"], dtype=int) + offset_x
                s["ypix"] = np.asarray(s["ypix"], dtype=int)

                s["med"] = [float(s["med"][0]), float(s["med"][1]) + offset_x]

                s["lam"] = np.asarray(s["lam"], dtype=float).ravel()

                if "ipix_neuropil" in s:
                    ypix, xpix = s["ypix"], s["xpix"]
                    s["ipix_neuropil"] = ypix + xpix * Ly

        stat = np.concatenate(stat_list)
        iscell = np.concatenate(iscell_list, axis=0)
        F = np.concatenate(F_list, axis=0)
        Fneu = np.concatenate(Fneu_list, axis=0)
        spks = np.concatenate(spks_list, axis=0)

        # --- Merge binary
        arrays = [Suite2pArray(p) for p in bin_paths]
        nframes = arrays[0].nframes
        dtype = arrays[0].dtype
        merged_bin = out_dir / "data.bin"
        with open(merged_bin, "wb") as f:
            for i in range(nframes):
                frames = [arr[i] for arr in arrays]
                f.write(np.hstack(frames).astype(dtype).tobytes())  # noqa
        for arr in arrays:
            arr.close()

        # --- Merge ops
        merged_ops = dict(ops_list[0])
        merged_ops.update({
            "Ly": Ly,
            "Lx": total_Lx,
            "yrange": [0, Ly],
            "xrange": [0, total_Lx],
            "reg_file": str(merged_bin.resolve()),
            "ops_path": str(out_ops.resolve()),
            "nrois": len(dirs),
        })

        # Full-FOV images: just tile ROIs horizontally
        for key in ["refImg", "meanImg", "meanImgE"]:
            if all(key in ops for ops in ops_list):
                canvas = np.zeros((Ly, total_Lx), dtype=ops_list[0][key].dtype)
                x_offset = 0
                for ops in ops_list:
                    arr = ops[key]
                    h, w = arr.shape
                    canvas[0:h, x_offset:x_offset + w] = arr
                    x_offset += ops["Lx"]
                merged_ops[key] = canvas  # noqa

        # Cropped images: place at yrange/xrange
        for key in ["max_proj", "Vcorr"]:
            if all(key in ops for ops in ops_list):
                canvas = np.zeros((Ly, total_Lx), dtype=ops_list[0][key].dtype)
                x_offset = 0
                for ops in ops_list:
                    arr = ops[key]
                    yr = np.array(ops["yrange"])
                    xr = np.array(ops["xrange"]) + x_offset
                    h, w = arr.shape
                    # Ensure array size matches expected slice
                    h_slice, w_slice = yr[1] - yr[0], xr[1] - xr[0]
                    canvas[yr[0]:yr[0] + h, xr[0]:xr[0] + w] = arr[:h_slice, :w_slice]
                    x_offset += ops["Lx"]
                merged_ops[key] = canvas  # noqa

        # --- Handle timeseries arrays (check consistency)
        timeseries_keys = ["yoff", "xoff", "corrXY", "badframes"]
        for key in timeseries_keys:
            if all(key in ops for ops in ops_list):
                arrays = [ops[key] for ops in ops_list]
                if all(np.array_equal(a, arrays[0]) for a in arrays[1:]):
                    merged_ops[key] = arrays[0]  # identical across ROIs
                else:
                    # fallback: take from first
                    merged_ops[key] = arrays[0]

        # --- Save merged results
        np.save(out_ops, merged_ops)
        np.save(out_dir / "stat.npy", stat)
        np.save(out_dir / "iscell.npy", iscell)
        np.save(out_dir / "F.npy", F)
        np.save(out_dir / "Fneu.npy", Fneu)
        np.save(out_dir / "spks.npy", spks)

        print(f"✔ Finished merging {plane} ({len(dirs)} ROIs)")
        remake_plane_figures(out_dir, run_rastermap=False)
        print(f"✔ Finished figures for {plane}")

def normalize_traces(F, mode="per_neuron"):
    """
    Normalize fluorescence traces F to [0, 1] range.
    Parameters
    ----------
    F : ndarray
        2d array of fluorescence traces (n_neurons x n_timepoints).
    mode : str
        Normalization mode, either "per_neuron" or "percentile".

    Returns
    -------
    F_norm : ndarray
        Normalized fluorescence traces in [0, 1] range.

    Notes
    -----
    - "per_neuron": scales each neuron's trace based on its own min and max.
    - "percentile": scales each neuron's trace based on its 1st and 99th percentiles.
    - If min == max for each cell, the trace is set to all zeros to avoid division by zero.
    """
    F_norm = np.zeros_like(F, dtype=float)

    if mode == "per_neuron":
        for i in range(F.shape[0]):
            f = F[i]
            fmax = np.max(f)
            fmin = np.min(f)
            if fmax > fmin:
                F_norm[i] = (f - fmin) / (fmax - fmin)
            else:
                F_norm[i] = f * 0
    elif mode == "percentile":
        for i in range(F.shape[0]):
            f = F[i]
            fmin = np.percentile(f, 1)
            fmax = np.percentile(f, 99)
            if fmax > fmin:
                F_norm[i] = (f - fmin) / (fmax - fmin)
            else:
                F_norm[i] = f * 0
    return F_norm

def remake_plane_figures(plane_dir, dff_percentile=8, dff_window_size=101, run_rastermap=False, **kwargs):
    """
    Re-generate Suite2p diagnostic figures for a merged plane.

    Parameters
    ----------
    plane_dir : Path
        Path to the planeXX output directory (with ops.npy, stat.npy, etc.).
    dff_percentile : int
        Percentile used for ΔF/F baseline.
    dff_window_size : int
        Window size for ΔF/F rolling baseline.
    kwargs : dict
        Extra keyword args (e.g. fig_label).
    """
    plane_dir = Path(plane_dir)

    expected_files = {
        "ops": plane_dir / "ops.npy",
        "stat": plane_dir / "stat.npy",
        "iscell": plane_dir / "iscell.npy",
        "registration": plane_dir / "registration.png",
        "segmentation_accepted": plane_dir / "segmentation_accepted.png",
        "segmentation_rejected": plane_dir / "segmentation_rejected.png",
        "max_proj": plane_dir / "max_projection_image.png",
        "meanImg": plane_dir / "mean_image.png",
        "meanImgE": plane_dir / "mean_image_enhanced.png",
        "traces_raw": plane_dir / "traces_raw.png",
        "traces_dff": plane_dir / "traces_dff.png",
        "traces_noise": plane_dir / "traces_noise.png",
        "noise": plane_dir / "shot_noise_distrubution.png",
        "model": plane_dir / "model.npy",
        "rastermap": plane_dir / "rastermap.png",
    }

    output_ops = load_ops(expected_files["ops"])

    # force remake of the heavy figures
    for key in ["registration", "segmentation_accepted", "segmentation_rejected", "traces_raw", "traces_dff", "traces_noise", "noise", "rastermap"]:
        if key in expected_files:
            safe_delete(expected_files[key])

    if expected_files["stat"].is_file():
        res = lsp.load_planar_results(plane_dir)
        iscell = res["iscell"]
        iscell_mask = iscell[:, 0].astype(bool) if iscell.ndim == 2 else iscell.astype(bool)

        stat = res["stat"]
        spks = res["spks"]
        F = res["F"]
        # F = F - F.min(axis=1, keepdims=True) * 0.9

        n_neurons = F.shape[0]
        if n_neurons < 10:
            print(f"Too few cells to plot traces for {plane_dir.stem}.")
            return output_ops

        # rastermap model
        F_accepted = F[iscell_mask]
        F_rejected = F[~iscell_mask]
        spks_cells = spks[iscell_mask]
        if run_rastermap:
            if expected_files["model"].is_file():
                model = np.load(expected_files["model"], allow_pickle=True).item()
            else:
                params = {
                    "n_clusters": 100 if n_neurons >= 200 else None,
                    "n_PCs": min(128, max(2, n_neurons - 1)),
                    "locality": 0.0 if n_neurons >= 200 else 0.1,
                    "time_lag_window": 15,
                    "grid_upsample": 10 if n_neurons >= 200 else 0,
                }
                import rastermap
                model = rastermap.Rastermap(**params).fit(spks_cells)
                np.save(expected_files["model"], model)

                lsp.plot_rastermap(
                    spks_cells,
                    model,
                    neuron_bin_size=0,
                    save_path=expected_files["rastermap"],
                    title_kwargs={"fontsize": 8, "y": 0.95},
                    title="Rastermap Sorted Activity",
                )

            if model is not None:
                # indices of cells relative to *all* ROIs
                isort_global = np.where(iscell_mask)[0][model.isort]
                output_ops["isort"] = isort_global

                # reorder just the cells
                F_accepted = F_accepted[model.isort]

        # compute dF/F
        f_norm_acc = normalize_traces(F_accepted, mode="per_neuron")
        f_norm_rej = normalize_traces(F_rejected, mode="per_neuron")

        dffp_acc = lsp.dff_rolling_percentile(f_norm_acc, percentile=dff_percentile, window_size=dff_window_size) * 100
        dffp_rej = lsp.dff_rolling_percentile(f_norm_rej, percentile=dff_percentile, window_size=dff_window_size) * 100

        fs = output_ops.get("fs", 1.0)
        dff_noise = lsp.dff_shot_noise(dffp_acc, fs)
        dff_noise_rejected = lsp.dff_shot_noise(dffp_rej, fs)

        if n_neurons >= 30:
            print(f"Plotting traces for {plane_dir.stem}...")
            _, colors = lsp.plot_traces(
                dffp_acc,
                save_path=expected_files["traces_dff"],
                num_neurons=output_ops.get("plot_n_traces", 30),
                signal_units="dffp",
            )
            _, colors = lsp.plot_traces(
                f_norm_acc,
                save_path=expected_files["traces_raw"],
                num_neurons=output_ops.get("plot_n_traces", 30),
                signal_units="raw",
            )

        print(f"Plotting noise distribution for {plane_dir.stem}...")
        lsp.plot_noise_distribution(dff_noise, save_path=expected_files["noise"])
        plot_masks(plane_dir)

    fig_label = kwargs.get("fig_label", plane_dir.stem)
    for key in ["meanImg", "max_proj", "meanImgE"]:
        if key in output_ops:
            lsp.plot_projection(
                output_ops,
                expected_files[key],
                fig_label=fig_label,
                display_masks=False,
                add_scalebar=True,
                proj=key,
            )

    return output_ops

def merge_zarr_rois(input_dir, output_dir=None, overwrite=True):
    """
    Concatenate roi1 + roi2 .zarr stores for each plane into a single planeXX.zarr.

    Parameters
    ----------
    input_dir : Path or str
        Directory containing planeXX_roi1, planeXX_roi2 subfolders with ops.npy + data.zarr.
    output_dir : Path or str, optional
        Where to write merged planeXX.zarr. Defaults to `input_dir`.
    overwrite : bool
        If True, existing outputs are replaced.
    """
    import dask.array as da
    z_merged = None
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir.parent / (input_dir.name + "_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    roi1_dirs = sorted(input_dir.glob("*plane*_roi1*"))
    roi2_dirs = sorted(input_dir.glob("*plane*_roi2*"))
    if not roi1_dirs or not roi2_dirs:
        print("No roi1 or roi2 in input dir")
        return None
    assert len(roi1_dirs) == len(roi2_dirs), "Mismatched ROI dirs"

    for roi1, roi2 in zip(roi1_dirs, roi2_dirs):
        start = time.time()
        zplane = roi1.stem.split("_")[0]  # "plane01"
        out_path = output_dir / f"{zplane}.zarr"
        if out_path.exists():
            if overwrite:
                import shutil
                shutil.rmtree(out_path)
            else:
                print(f"Skipping {zplane}, {out_path} exists")
                continue

        # load ops
        start_da_from_zarr = time.time()
        z1 = da.from_zarr(roi1)
        z2 = da.from_zarr(roi2)
        print(f"Loaded zarr arrays in {time.time() - start_da_from_zarr:.1f}s")

        # sanity check
        assert z1.shape[0] == z2.shape[0], "Frame count mismatch"
        assert z1.shape[1] == z2.shape[1], "Height mismatch"

        # concatenate along width (axis=2)
        start_concat = time.time()
        z_merged = da.concatenate([z1, z2], axis=2)
        print(f"Concatenated arrays in {time.time() - start_concat:.1f}s")

        # write back
        start_merge = time.time()
        z_merged.to_zarr(out_path, overwrite=overwrite)
        print(f"Wrote zarr in {time.time() - start_merge:.1f}s")
        end = time.time()
        print(f"Wrote {out_path.name} in {end - start:.1f}s")

    if z_merged:
        print(f"Merged zarrs to {output_dir}"
              f"{z_merged}")

    return None

def plot_masks(plane_dir, out_prefix="rois", outpath=None):
    """
    Make ROI overlays like Suite2p GUI: one PNG for accepted and one for rejected.

    Parameters
    ----------
    plane_dir : str or Path
        Directory containing ops.npy, stat.npy, iscell.npy
    out_prefix : str
        Prefix for output files (saved in plane_dir)
    """
    plane_dir = Path(plane_dir)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    iscell = np.load(plane_dir / "iscell.npy")

    Ly, Lx = ops["Ly"], ops["Lx"]
    meanImg = ops.get("meanImgE", ops.get("meanImg", np.zeros((Ly, Lx))))

    def draw_masks(mask_idx, fname, outpath=None):
        canvas = np.tile(
            (meanImg - meanImg.min()) / (np.ptp(meanImg) + 1e-6),
            (3, 1, 1)
        ).transpose(1, 2, 0)  # grayscale RGB

        colors = plt.cm.hsv(np.linspace(0, 1, mask_idx.sum() + 1))
        c = 0
        for n, s in enumerate(stat):
            if mask_idx[n]:
                ypix = s["ypix"]
                xpix = s["xpix"]
                lam = s["lam"] / s["lam"].max()
                col = colors[c][:3]  # RGB
                c += 1
                for k in range(3):
                    canvas[ypix, xpix, k] = (
                            0.5 * canvas[ypix, xpix, k] + 0.5 * col[k] * lam
                    )

        plt.figure(figsize=(10, 10))
        plt.imshow(canvas, interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        if outpath is None:
            plt.savefig(plane_dir / f"{out_prefix}_{fname}.png", dpi=300)
        else:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
        plt.close()

    # accepted
    draw_masks(mask_idx=iscell[:, 0] == 1, fname="accepted",outpath= plane_dir if outpath is None else Path(outpath))
    # rejected
    draw_masks(mask_idx=iscell[:, 0] == 0, fname="rejected", outpath=plane_dir if outpath is None else Path(outpath))

if __name__ == "__main__":
    fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw\anatomical_3_roi")
    merge_rois_auto(fpath, fpath.parent / "anatomical_3_merged")
    x = 2
