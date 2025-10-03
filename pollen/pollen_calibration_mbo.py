import warnings
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate
from scipy.optimize import curve_fit

from mbo_utilities import get_metadata

from imgui_bundle import (
    imgui,
    imgui_md,
    hello_imgui,
    imgui_ctx,
)
from imgui_bundle import portable_file_dialogs as pfd


warnings.simplefilter(action="ignore")


plt.rcParams.update({
    "font.size": 12,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
})


class PollenDialog:
    def __init__(self):
        from mbo_utilities.file_io import get_mbo_dirs
        from pathlib import Path

        self.selected_path = None
        self._open_multi = None
        self._select_folder = None

        # assets/settings from MBO utilities
        self._assets_path = Path(get_mbo_dirs()["assets"])
        self._settings_path = Path(get_mbo_dirs()["settings"])

    def render(self):
        with imgui_ctx.begin_child("#pollen_fd"):
            imgui.push_id("pollen_fd")

            # header --------------------------------------------------
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.5))

            # Title + documentation
            imgui_md.render_unindented("""
            # Pollen Calibration

            This tool calibrates multi-beam two-photon systems using pollen grains.  
            Steps performed during calibration:

            1. **Scan offset correction** – corrects phase shifts in Y scanning.  
            2. **Bead selection** – user clicks pollen beads per beamlet.  
            3. **Power vs. Z** – analyzes signal strength with depth.  
            4. **Z position offsets** – measures axial offset between beams.  
            5. **Exponential decay** – characterizes power falloff.  
            6. **XY calibration** – computes lateral beamlet offsets.  

            Results are saved as `.h5` (offsets) and `.png` figures in the same folder as the TIFF.
            """)

            imgui.dummy(hello_imgui.em_to_vec2(0, 2.0))

            # Centered “Open File” button
            bsz_file = hello_imgui.em_to_vec2(18, 2.4)
            x_file = (imgui.get_window_width() - bsz_file.x) * 0.5
            imgui.set_cursor_pos_x(x_file)
            if imgui.button("Open File", bsz_file):
                self._open_multi = pfd.open_file("Select pollen TIFF", options=0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Open a TIFF file for pollen calibration.")

            # handle file selection
            if self._open_multi and self._open_multi.ready():
                result = self._open_multi.result()
                if result:
                    # result is a list, pick the first file
                    if isinstance(result, (list, tuple)):
                        self.selected_path = result[0]
                    else:
                        self.selected_path = result
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._open_multi = None

            # quit button bottom-right
            qsz = hello_imgui.em_to_vec2(10, 1.8)
            imgui.set_cursor_pos(
                imgui.ImVec2(
                    imgui.get_window_width() - qsz.x - hello_imgui.em_size(1),
                    imgui.get_window_height() - qsz.y - hello_imgui.em_size(1),
                )
            )
            if imgui.button("Quit", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True

            imgui.pop_id()


def pollen_calibration_mbo(filepath, order=None):
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    metadata = get_metadata(filepath, verbose=True)
    z_step_um = metadata["all"]["FrameData"]["SI.hStackManager.stackZStepSize"]

    nx = metadata["roi_width_px"]
    ny = metadata["roi_height_px"]
    nc = metadata["num_planes"]

    # nz comes from TIFF, not metadata
    arr = tifffile.imread(filepath)
    nz = arr.shape[0] // nc

    if order is None:
        order = list(range(nc))

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


def print_tifffile_note():
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    msg = (
            "\n"
            + "*" * 50 + "\n"
            + f"* {BOLD}{YELLOW}NOTE:{RESET}".ljust(49) + "\n"
            + f"* {YELLOW}IGNORE TIFFFILE FAILED TO RESHAPE ERRORS{RESET}".ljust(49) + "\n"
            + "*" * 50 + "\n"
    )
    print(msg)


def load_or_read_data(filepath, ny, nx, nc, nz):
    """Read TIFF → reshape into (nz, nc, ny, nx)."""
    print_tifffile_note()
    arr = tifffile.imread(filepath).astype(np.float32)  # (nframes, ny, nx)
    nframes = arr.shape[0]
    if nframes != nz * nc:
        raise ValueError(f"{nframes} frames not divisible by nc={nc}")

    vol = arr.reshape(nz, nc, ny, nx)
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

    amt = max(1, int(round(10.0 / DZ)))
    smoothed = uniform_filter1d(Iz, size=amt, axis=1, mode="nearest")

    zoi = smoothed.argmax(axis=1)
    pp = smoothed.max(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, o in enumerate(order):
        ax.plot(ZZ, np.sqrt(smoothed[o, :]), label=f"Beam {i+1}")

    ax.plot(ZZ[zoi], np.sqrt(pp), "k.", markersize=8)

    for i, o in enumerate(order):
        ax.text(
            ZZ[zoi[o]],
            np.sqrt(pp[o]) + 0.02 * np.max(np.sqrt(pp)),
            str(i + 1),
            ha="center",
            fontsize=10,
            weight="bold",
            )

    ax.set_xlabel("Piezo Z (µm)", fontweight="bold")
    ax.set_ylabel("2p signal (a.u.)", fontweight="bold")
    ax.set_title("Power vs. Z-depth", fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_power_vs_z.png"), dpi=150)
    plt.close()

    return ZZ, zoi, pp


def analyze_z_positions(ZZ, zoi, order, filepath):
    Z0 = ZZ[zoi[order[0]]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(order) + 1), ZZ[zoi[order]] - Z0, "bo-", markersize=6)

    ax.set_xlabel("Beam number", fontweight="bold")
    ax.set_ylabel("Z position (µm)", fontweight="bold")
    ax.set_title("Z Position vs. Beam Number", fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_z_vs_N.png"), dpi=150)
    plt.close()


def fit_exp_decay(ZZ, zoi, order, filepath, pp):

    def exp_func(z, a, b):
        return a * np.exp(b * z)

    z = ZZ[zoi[order]]
    p = np.sqrt(pp[order])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z, p, "bo", markersize=6, label="Data")

    try:
        popt, _ = curve_fit(exp_func, z, p, p0=(p.max(), -0.01))
        z_fit = np.linspace(z.min(), z.max(), 200)
        ax.plot(z_fit, exp_func(z_fit, *popt), "r-", label=f"Fit (ls = {1/popt[1]:.1f} µm)")
    except Exception as e:
        print("Exp fit failed:", e)

    ax.set_xlabel("Z (µm)", fontweight="bold")
    ax.set_ylabel("Power (a.u.)", fontweight="bold")
    ax.set_title("Exponential Power Decay", fontweight="bold")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_power_decay.png"), dpi=150)
    plt.close()


def calibrate_xy(xs, ys, III, filepath):
    nc_total = III.shape[2]
    x_shifts = np.round(xs - np.mean(xs[:nc_total])).astype(int)
    y_shifts = np.round(ys - np.mean(ys[:nc_total])).astype(int)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x_shifts, y_shifts, "bo", markersize=6)
    ax.set_xlabel("X (µm)", fontweight="bold")
    ax.set_ylabel("Y (µm)", fontweight="bold")
    ax.set_title("XY Offsets", fontweight="bold")
    ax.axis("equal")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_xy_offsets.png"), dpi=150)
    plt.close()


def select_pollen_file() -> str | None:
    from imgui_bundle import immapp, hello_imgui

    dlg = PollenDialog()

    def _render():
        dlg.render()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Pollen Calibration"
    params.app_window_params.window_geometry.size = (1000, 700)
    params.callbacks.show_gui = _render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    hello_imgui.set_assets_folder(str(dlg._assets_path))
    immapp.run(runner_params=params, add_ons_params=addons)

    return dlg.selected_path if dlg.selected_path else None


@click.command()
@click.option("--in", "input_path", type=click.Path(exists=True, file_okay=True, dir_okay=True),
              help="Input file or directory containing pollen data")
def main(input_path):
    """Run pollen calibration with optional input/output paths."""

    if input_path is None:
        data_in = select_pollen_file()
        if not data_in:
            click.echo("No file selected, exiting.")
            return
        input_path = data_in

    pollen_calibration_mbo(input_path)


if __name__ == "__main__":
    main()