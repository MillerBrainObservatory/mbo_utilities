import warnings
import h5py
import tifffile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.ndimage import uniform_filter1d
from mbo_utilities import get_metadata, get_files

warnings.simplefilter(action="ignore")


def pollen_calibration_mbo(filepath, dual_cavity=False, order=None):
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    metadata = get_metadata(filepath, verbose=True)
    z_step_um = metadata["all"]["FrameData"]["SI.hStackManager.stackZStepSize"]

    fov_um_x, fov_um_y = metadata["fov"]
    nx = metadata["roi_width_px"]
    ny = metadata["roi_height_px"]
    nc = metadata["num_planes"]

    # nz comes from TIFF, not metadata
    arr = tifffile.imread(filepath)
    nz = arr.shape[0] // nc

    if order is None:
        order = list(range(nc))

    dx = fov_um_x / nx
    dy = fov_um_y / ny

    vol = load_or_read_data(filepath, ny, nx, nc, nz)

    # 1. scan offset correction
    vol, scan_corrections = correct_scan_phase(vol, filepath)

    # 2. user marked pollen
    xs, ys, Iz, III = user_pollen_selection(vol)

    # 3. power vs z
    ZZ, zoi, pp = analyze_power_vs_z(Iz, filepath, z_step_um, order)

    # 4. analyze z
    analyze_z_positions(ZZ, zoi, order, filepath)

    # 5. exponential decay
    fit_exp_decay(ZZ, zoi, order, filepath, pp)

    # 6. XY calibration
    calibrate_xy(xs, ys, III, filepath)


def load_or_read_data(filepath, ny, nx, nc, nz):
    """Read TIFF → reshape into (nz, nc, ny, nx)."""
    arr = tifffile.imread(filepath).astype(np.float32)  # (nframes, ny, nx)
    nframes = arr.shape[0]
    if nframes != nz * nc:
        raise ValueError(f"{nframes} frames not divisible by nc={nc}")

    # C-order reshape: (nz, nc, ny, nx)
    vol = arr.reshape(nz, nc, ny, nx)

    # subtract mean, normalize
    vol -= vol.mean()

    return vol


def correct_scan_phase(vol, filepath):
    """Detect and correct scan phase offsets along Y-axis."""
    scan_corrections = []
    nz, nc, ny, nx = vol.shape

    for c in range(nc):
        # take z-projection like MATLAB Iinit(:,:,c)
        Iproj = vol[:, c, :, :].max(axis=0)  # (ny, nx)
        offset = return_scan_offset(Iproj)
        scan_corrections.append(offset)

        # apply to each z-slice
        for z in range(nz):
            vol[z, c, :, :] = fix_scan_phase(vol[z, c, :, :], offset)

    # Save scan corrections
    h5_path = filepath.with_name(filepath.stem + "_pollen.h5")
    with h5py.File(h5_path, "a") as f:
        if "scan_corrections" in f:
            del f["scan_corrections"]
        f.create_dataset("scan_corrections", data=np.array(scan_corrections))

    return vol, scan_corrections


def return_scan_offset(Iin, n=8):
    """Return scan offset (along Y, rows)."""
    Iv1 = Iin[:, ::2]
    Iv2 = Iin[:, 1::2]
    min_cols = min(Iv1.shape[1], Iv2.shape[1])
    Iv1 = Iv1[:, :min_cols]
    Iv2 = Iv2[:, :min_cols]

    buffers = np.zeros((n, Iv1.shape[1]))
    Iv1 = np.vstack([buffers, Iv1, buffers]).ravel()
    Iv2 = np.vstack([buffers, Iv2, buffers]).ravel()

    Iv1 -= Iv1.mean()
    Iv2 -= Iv2.mean()
    Iv1[Iv1 < 0] = 0
    Iv2[Iv2 < 0] = 0

    r = correlate(Iv1, Iv2, mode="full")
    lag = np.arange(-len(Iv1) + 1, len(Iv1))
    return lag[np.argmax(r)]


def fix_scan_phase(frame, offset):
    """Apply scan phase correction along Y axis."""
    out = np.zeros_like(frame)
    if offset > 0:
        out[offset:, :] = frame[:-offset, :]
    elif offset < 0:
        out[:offset, :] = frame[-offset:, :]
    else:
        out = frame
    return out


def user_pollen_selection(vol, num=10):
    """
    vol : ndarray, shape (nz, nc, ny, nx)
    """
    nz, nc, ny, nx = vol.shape
    xs, ys, Iz, III = [], [], [], []

    print("Select pollen beads...")

    for c in range(nc):
        # MATLAB: imagesc(max(vol(:,:,c,:),[],4))
        img = vol[:, c, :, :].max(axis=0)  # (ny, nx)

        fig, ax = plt.subplots()
        ax.imshow(
            img,
            cmap="gray",
            vmin=np.percentile(img, 1),
            vmax=np.percentile(img, 99),
            origin="upper",
        )
        ax.set_title(f"Select pollen bead for beamlet {c + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        pts = plt.ginput(1, timeout=0)
        plt.close(fig)
        if not pts:
            continue

        x, y = pts[0]
        ix, iy = int(round(x)), int(round(y))
        xs.append(x)
        ys.append(y)

        # patch around point across z
        y0, y1 = max(0, iy - num), min(ny, iy + num + 1)
        x0, x1 = max(0, ix - num), min(nx, ix + num + 1)

        patch = vol[:, c, y0:y1, x0:x1]  # (nz, roi_y, roi_x)
        trace = patch.mean(axis=(1, 2))  # (nz,)
        Iz.append(trace)

        zoi = int(np.argmax(uniform_filter1d(trace, size=10)))
        crop = vol[zoi, c, y0:y1, x0:x1]  # 2D crop at best z
        III.append(crop)

    Iz = np.vstack(Iz) if Iz else np.zeros((0, nz))
    if III:
        max_h = max(im.shape[0] for im in III)
        max_w = max(im.shape[1] for im in III)
        pads = [
            np.pad(
                im,
                ((0, max_h - im.shape[0]), (0, max_w - im.shape[1])),
                mode="constant",
            )
            for im in III
        ]
        III = np.stack(pads, axis=-1)
    else:
        III = np.zeros((2 * num + 1, 2 * num + 1, 0))

    return np.array(xs), np.array(ys), Iz, III


def analyze_power_vs_z(Iz, filepath, DZ, order):
    nz = Iz.shape[1]
    ZZ = np.flip(np.arange(nz) * DZ)
    amt = int(10.0 / DZ)
    smoothed = uniform_filter1d(Iz, size=amt, axis=1)
    zoi = smoothed.argmax(axis=1)
    pp = smoothed.max(axis=1)

    plt.figure()
    plt.plot(ZZ, np.sqrt(smoothed[order, :]).T)
    plt.xlabel("Piezo Z (µm)")
    plt.ylabel("2p signal (a.u.)")
    plt.title("Power vs. Z-depth")
    plt.savefig(filepath.with_name("pollen_calibration_power_vs_z.png"))
    plt.close()

    return ZZ, zoi, pp


def analyze_z_positions(ZZ, zoi, order, filepath):
    Z0 = ZZ[zoi[order[0]]]
    plt.figure()
    plt.plot(range(len(order)), ZZ[zoi[order]] - Z0, "bo-")
    plt.xlabel("Beam number")
    plt.ylabel("Z position (µm)")
    plt.grid(True)
    plt.savefig(filepath.with_name("pollen_calibration_z_vs_N.png"))
    plt.close()


def fit_exp_decay(
    ZZ,
    zoi,
    order,
    filepath,
    pp,
):
    plt.figure()
    z = ZZ[zoi[order]]
    p = np.sqrt(pp[order])
    plt.plot(z, p, "bo")
    plt.xlabel("Z (µm)")
    plt.ylabel("Power (a.u.)")
    plt.grid(True)
    plt.savefig(filepath.with_name("pollen_calibration_power_decay.png"))
    plt.close()


def calibrate_xy(xs, ys, III, filepath):
    nc_total = III.shape[2]
    # nc = nc_total // 2 if dual_cavity else nc_total
    nc = nc_total
    x_shifts = np.round(xs - xs[:nc].mean()).astype(int)
    y_shifts = np.round(ys - ys[:nc].mean()).astype(int)

    h5_path = filepath.with_name(filepath.stem + "_pollen.h5")
    with h5py.File(h5_path, "a") as f:
        if "x_shifts" in f:
            del f["x_shifts"]
        if "y_shifts" in f:
            del f["y_shifts"]
        f["x_shifts"] = x_shifts
        f["y_shifts"] = y_shifts

    plt.figure()
    plt.plot(x_shifts, y_shifts, "bo")
    plt.xlabel("X (µm)")
    plt.ylabel("Y (µm)")
    plt.axis("equal")
    plt.savefig(filepath.with_name("pollen_calibration_xy_offsets.png"))
    plt.close()


if __name__ == "__main__":
    import h5py

    files = get_files(r"D:\demo\pollen", str_contains="_pollen.h5", max_depth=1)

    xshifts, yshifts = [], []
    with h5py.File(files[0], "r") as hf:
        for k in hf.keys():
            if k == "x_shifts":
                xshifts = hf[k][()]
            else:
                yshifts = hf[k][()]

            print(f"  {k}: {hf[k][()]}")
        print()
    plane_shifts = np.array([xshifts, yshifts]).T
    print(plane_shifts)
    pollen_path = r"D:\demo\pollen\07_27_2025_pollen_mounted_on_rig_07_17_2025__2xzoom_30pctpower_5umsteps_81slices_100avg_00001_00001_00001_00001_00001_00001_00001.tif"
    pollen_calibration_mbo(pollen_path, dual_cavity=False)
