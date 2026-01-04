"""
Pollen calibration analysis functions.

These functions are extracted from pollen/pollen_calibration_mbo.py
to be used by the PollenCalibrationWidget.
"""


import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate
from scipy.optimize import curve_fit

from mbo_utilities.metadata import get_param


# =============================================================================
# H5 File Helpers - Unified pollen calibration file structure
# =============================================================================

def get_pollen_h5_path(filepath):
    """Get the unified pollen calibration H5 file path.

    Parameters
    ----------
    filepath : Path
        Path to the source data file.

    Returns
    -------
    Path
        Path to the unified pollen calibration H5 file (*_pollen.h5).
    """
    return filepath.with_name(filepath.stem + "_pollen.h5")


def get_mode_group(h5_file, mode, create=True):
    """Get or create the group for a calibration mode.

    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 file handle.
    mode : str
        Calibration mode ('auto' or 'manual').
    create : bool
        If True, create the group if it doesn't exist.

    Returns
    -------
    h5py.Group
        The group for the specified mode.
    """
    if mode not in h5_file:
        if create:
            return h5_file.create_group(mode)
        else:
            raise KeyError(f"Mode '{mode}' not found in H5 file")
    return h5_file[mode]


# =============================================================================
# Color Palette - Consistent dark theme colors
# =============================================================================

class Colors:
    """Consistent color palette for pollen calibration plots."""

    # Background colors
    BG_DARK = "#1e1e1e"       # Main figure background
    BG_AXES = "#2d2d2d"       # Axes background

    # Text colors
    TEXT = "#e0e0e0"          # Primary text
    TEXT_DIM = "#999999"      # Dimmed/secondary text

    # Line/edge colors
    EDGE = "#555555"          # Axes edges, borders
    GRID = "#444444"          # Grid lines

    # Data colors - primary series
    AUTO = "#00bfff"          # Auto mode - cyan/sky blue
    MANUAL = "#66ff66"        # Manual mode - bright green
    CAVITY_A = "#00bfff"      # Cavity A - cyan (same as auto)
    CAVITY_B = "#66ff66"      # Cavity B - light green (same as manual)

    # Data colors - secondary/accents
    WHITE = "#ffffff"         # Markers, emphasis
    FIT_A = "#ff6666"         # Fit line for cavity A - light red
    FIT_B = "#ffaa00"         # Fit line for cavity B - orange
    FIT_COMBINED = "#ff66ff"  # Combined fit - magenta
    DIFF_X = "#ff6666"        # X difference - light red
    DIFF_Y = "#ffaa00"        # Y difference - orange

    @staticmethod
    def mode_label(mode: str) -> str:
        """Get display label for calibration mode."""
        return "Manual" if mode == "manual" else "Auto"

    @staticmethod
    def mode_color(mode: str) -> str:
        """Get primary color for calibration mode."""
        return Colors.MANUAL if mode == "manual" else Colors.AUTO


# Dark theme for matplotlib
plt.rcParams.update(
    {
        # Dark background
        "figure.facecolor": Colors.BG_DARK,
        "axes.facecolor": Colors.BG_AXES,
        "savefig.facecolor": Colors.BG_DARK,
        # Light text/lines
        "text.color": Colors.TEXT,
        "axes.labelcolor": Colors.TEXT,
        "axes.edgecolor": Colors.EDGE,
        "xtick.color": Colors.TEXT,
        "ytick.color": Colors.TEXT,
        "grid.color": Colors.GRID,
        "legend.facecolor": Colors.BG_AXES,
        "legend.edgecolor": Colors.EDGE,
        "legend.labelcolor": Colors.TEXT,
        # Font settings
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titlecolor": Colors.TEXT,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
    }
)


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


def correct_scan_phase(vol, filepath, z_step_um, metadata, mode="auto"):
    """Detect and correct scan phase offsets along Y-axis.

    Parameters
    ----------
    mode : str
        'auto' or 'manual' - calibration mode stored in separate groups
    """
    scan_corrections = []
    nz, nc, ny, nx = vol.shape

    for c in range(nc):
        Iproj = vol[:, c, :, :].max(axis=0)
        offset = return_scan_offset(Iproj)
        scan_corrections.append(offset)

    for c in range(nc):
        for z in range(nz):
            vol[z, c, :, :] = fix_scan_phase(vol[z, c, :, :], scan_corrections[c])

    # Save scan corrections to unified H5 file with mode-specific group
    h5_path = get_pollen_h5_path(filepath)
    with h5py.File(h5_path, "a") as f:
        # Store shared metadata at root level (only once)
        f.attrs["source_file"] = filepath.name
        f.attrs["pollen_calibration_version"] = "2.0"
        f.attrs["num_planes"] = nc
        f.attrs["roi_width_px"] = nx
        f.attrs["roi_height_px"] = ny
        f.attrs["z_step_um"] = z_step_um

        frame_rate = get_param(metadata, "fs")
        if frame_rate is not None:
            f.attrs["frame_rate"] = frame_rate

        dx = get_param(metadata, "dx")
        dy = get_param(metadata, "dy")
        if dx is not None and dy is not None:
            f.attrs["pixel_resolution"] = f"({dx}, {dy})"

        # Store mode-specific data in group
        grp = get_mode_group(f, mode)
        if "scan_corrections" in grp:
            del grp["scan_corrections"]
        grp.create_dataset("scan_corrections", data=np.array(scan_corrections))
        grp.attrs["calibration_mode"] = mode

    return vol, scan_corrections


def plot_beamlet_grid(vol, order, filepath, mode="auto"):
    """Plot grid of max-projected beamlet images."""
    _nz, nc, ny, nx = vol.shape
    Imax = vol.max(axis=0)

    n_cols = 6
    n_rows = int(np.ceil(nc / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()

    for idx in range(nc):
        ax = axes[idx]
        channel = order[idx] if idx < len(order) else idx
        img = Imax[channel, :, :].T
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        ax.set_xlim([0, ny])
        ax.set_ylim([0, nx])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Beam {idx+1}", fontsize=8, color=Colors.TEXT)

    for idx in range(nc, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Max-Projected Beamlet Images ({Colors.mode_label(mode)})",
                 fontweight="bold", color=Colors.TEXT)
    plt.tight_layout()
    out_name = f"pollen_{mode}_beamlet_grid.png"
    plt.savefig(filepath.with_name(out_name), dpi=150)
    plt.close()


def analyze_power_vs_z(Iz, filepath, DZ, order, nc, mode="auto"):
    """Analyze power vs Z depth."""
    nz = Iz.shape[1]
    ZZ = np.flip(np.arange(nz) * DZ)

    amt = max(1, round(10.0 / DZ))
    smoothed = uniform_filter1d(Iz, size=amt, axis=1, mode="nearest")

    zoi = smoothed.argmax(axis=1)
    pp = smoothed.max(axis=1)

    _fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(order)):
        ax.plot(ZZ, np.sqrt(smoothed[i, :]), alpha=0.7)

    ax.plot(ZZ[zoi], np.sqrt(pp), ".", color=Colors.WHITE, markersize=10)

    sqrt_pp = np.sqrt(pp)
    label_offset = 0.03 * (sqrt_pp.max() - sqrt_pp.min())
    for i in range(len(order)):
        ax.text(
            ZZ[zoi[i]],
            np.sqrt(pp[i]) + label_offset,
            str(i + 1),
            ha="center",
            fontsize=8,
            color=Colors.TEXT,
        )

    ax.set_xlabel("Piezo Z (um)", fontweight="bold")
    ax.set_ylabel("2p signal (a.u.)", fontweight="bold")
    ax.set_title(f"Power vs. Z-depth ({Colors.mode_label(mode)})", fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    out_name = f"pollen_{mode}_power_vs_z.png"
    plt.savefig(filepath.with_name(out_name), dpi=150)
    plt.close()

    return ZZ, zoi, pp


def analyze_z_positions(ZZ, zoi, order, filepath, cavity_info, mode="auto"):
    """Analyze Z positions vs beam number with cavity split."""
    n_beams = len(order)

    Z0 = ZZ[zoi[0]]
    z_rel = ZZ[zoi] - Z0

    _fig, ax = plt.subplots(figsize=(7, 5))

    cavity_a_channels = set(cavity_info["cavity_a"])
    cavity_b_channels = set(cavity_info["cavity_b"])

    cavity_a_beams = []
    cavity_b_beams = []

    for beam_idx in range(n_beams):
        channel = order[beam_idx] if beam_idx < len(order) else beam_idx
        if channel in cavity_a_channels:
            cavity_a_beams.append(beam_idx)
        elif channel in cavity_b_channels:
            cavity_b_beams.append(beam_idx)
        else:
            cavity_a_beams.append(beam_idx)

    if cavity_a_beams:
        ax.plot(
            [i+1 for i in cavity_a_beams],
            [z_rel[i] for i in cavity_a_beams],
            "o", color=Colors.CAVITY_A, markersize=6, label="Cavity A"
        )

    if cavity_b_beams:
        ax.plot(
            [i+1 for i in cavity_b_beams],
            [z_rel[i] for i in cavity_b_beams],
            "s", color=Colors.CAVITY_B, markersize=6, label="Cavity B"
        )

    beam_nums = np.arange(1, n_beams + 1)
    r_squared = None
    z_slope = None
    try:
        coeffs = np.polyfit(beam_nums, z_rel, 1)
        poly = np.poly1d(coeffs)

        y_pred = poly(beam_nums)
        ss_res = np.sum((z_rel - y_pred) ** 2)
        ss_tot = np.sum((z_rel - np.mean(z_rel)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        z_slope = coeffs[0]

        x_fit = np.linspace(0, n_beams + 1, 101)
        ax.plot(x_fit, poly(x_fit), "-", color=Colors.WHITE,
                label=f"Linear fit (r² = {r_squared:.3f})\ny = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
    except Exception:
        pass

    ax.set_xlabel("Beam number", fontweight="bold")
    ax.set_ylabel("Z position (um)", fontweight="bold")
    ax.set_title(f"Z Position vs. Beam Number ({Colors.mode_label(mode)})", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    out_name = f"pollen_{mode}_z_vs_N.png"
    plt.savefig(filepath.with_name(out_name), dpi=150)
    plt.close()

    # Save fit quality to unified H5 file
    h5_path = get_pollen_h5_path(filepath)
    with h5py.File(h5_path, "a") as f:
        grp = get_mode_group(f, mode)
        if r_squared is not None:
            grp.attrs["z_fit_r_squared"] = r_squared
        if z_slope is not None:
            grp.attrs["z_slope_um_per_beam"] = z_slope


def fit_exp_decay(ZZ, zoi, order, filepath, pp, cavity_info, DZ, nz, mode="auto"):
    """Fit exponential decay with cavity splits."""
    def exp_func(z, a, b):
        return a * np.exp(b * z)

    n_beams = len(order)

    cavity_a_channels = set(cavity_info["cavity_a"])
    cavity_b_channels = set(cavity_info["cavity_b"])

    cavity_a_beams = []
    cavity_b_beams = []

    for beam_idx in range(n_beams):
        channel = order[beam_idx] if beam_idx < len(order) else beam_idx
        if channel in cavity_a_channels:
            cavity_a_beams.append(beam_idx)
        elif channel in cavity_b_channels:
            cavity_b_beams.append(beam_idx)
        else:
            cavity_a_beams.append(beam_idx)

    z_all = ZZ[zoi]
    p_all = np.sqrt(pp)

    z1 = np.array([z_all[i] for i in cavity_a_beams]) if cavity_a_beams else np.array([])
    p1 = np.array([p_all[i] for i in cavity_a_beams]) if cavity_a_beams else np.array([])

    z2 = np.array([z_all[i] for i in cavity_b_beams]) if cavity_b_beams else np.array([])
    p2 = np.array([p_all[i] for i in cavity_b_beams]) if cavity_b_beams else np.array([])

    # Linear scale plot
    _fig, ax = plt.subplots(figsize=(8, 6))

    if len(z1) > 0:
        ax.plot(z1, p1, "o", color=Colors.CAVITY_A, markersize=6, label="Data Cavity A")
    if len(z2) > 0:
        ax.plot(z2, p2, "s", color=Colors.CAVITY_B, markersize=6, label="Data Cavity B")

    z_fit_range = DZ * np.linspace(0, nz - 1, 1001)

    # Track decay lengths for saving
    decay_length_a = None
    decay_length_b = None
    decay_length_combined = None

    # Fit each cavity
    if len(z1) > 2:
        try:
            popt1, _ = curve_fit(exp_func, z1, p1, p0=(p1.max(), -0.01), maxfev=5000)
            decay_length_a = abs(1/popt1[1])
            ax.plot(z_fit_range, exp_func(z_fit_range, *popt1), color=Colors.FIT_A,
                    label=f"Fit C1 (ls = {decay_length_a:.0f} um)")
        except Exception:
            pass

    if len(z2) > 2:
        try:
            popt2, _ = curve_fit(exp_func, z2, p2, p0=(p2.max(), -0.01), maxfev=5000)
            decay_length_b = abs(1/popt2[1])
            ax.plot(z_fit_range, exp_func(z_fit_range, *popt2), color=Colors.FIT_B,
                    label=f"Fit C2 (ls = {decay_length_b:.0f} um)")
        except Exception:
            pass

    # Combined fit
    z_combined = np.concatenate([z1, z2]) if len(z2) > 0 else z1
    p_combined = np.concatenate([p1, p2]) if len(p2) > 0 else p1

    if len(z_combined) > 2:
        try:
            popt3, _ = curve_fit(exp_func, z_combined, p_combined, p0=(p_combined.max(), -0.01), maxfev=5000)
            decay_length_combined = abs(1/popt3[1])
            ax.plot(z_fit_range, exp_func(z_fit_range, *popt3), color=Colors.FIT_COMBINED,
                    label=f"Fit both (ls = {decay_length_combined:.0f} um)")
        except Exception:
            pass

    ax.set_xlabel("Z (um)", fontweight="bold")
    ax.set_ylabel("Power (a.u.)", fontweight="bold")
    ax.set_title(f"Power vs Depth ({Colors.mode_label(mode)})", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    out_name = f"pollen_{mode}_power_linear.png"
    plt.savefig(filepath.with_name(out_name), dpi=150)
    plt.close()

    # Save decay lengths to unified H5 file
    h5_path = get_pollen_h5_path(filepath)
    with h5py.File(h5_path, "a") as f:
        grp = get_mode_group(f, mode)
        if decay_length_a is not None:
            grp.attrs["decay_length_cavity_a_um"] = decay_length_a
        if decay_length_b is not None:
            grp.attrs["decay_length_cavity_b_um"] = decay_length_b
        if decay_length_combined is not None:
            grp.attrs["decay_length_um"] = decay_length_combined


def plot_z_spacing(ZZ, zoi, order, filepath, mode="auto"):
    """Plot Z spacing between consecutive beams."""
    z_positions = ZZ[zoi]
    z_diff = np.diff(z_positions)

    _fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(z_diff) + 1), z_diff, "o", color=Colors.WHITE, markersize=10)
    ax.set_xlabel("Beam pair", fontweight="bold")
    ax.set_ylabel("dZ (um)", fontweight="bold")
    ax.set_title(f"Z Spacing Between Beams ({Colors.mode_label(mode)})", fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    out_name = f"pollen_{mode}_z_spacing.png"
    plt.savefig(filepath.with_name(out_name), dpi=150)
    plt.close()


def calibrate_xy(xs, ys, III, filepath, dx, dy, nx, ny, cavity_info, mode="auto"):
    """XY calibration with centroid refinement."""
    n_patches = III.shape[2]
    num = III.shape[0] // 2

    xx = np.arange(-num, num + 1)
    yy = np.arange(-num, num + 1)
    XX, YY = np.meshgrid(xx, yy)

    offx = np.zeros(n_patches)
    offy = np.zeros(n_patches)

    for zz in range(n_patches):
        IOI = III[:, :, zz].copy()
        IOI[IOI > 0] = 1
        IOI[IOI <= 0] = 0

        total = np.trapezoid(np.trapezoid(IOI, axis=0))
        if total > 0:
            offx[zz] = round(np.trapezoid(np.trapezoid(XX * IOI, axis=0)) / total)
            offy[zz] = round(np.trapezoid(np.trapezoid(YY * IOI, axis=0)) / total)

    xs_refined = xs + offx
    ys_refined = ys + offy

    vx = (np.arange(-nx//2, nx//2 + 1)) * dx
    vy = (np.arange(-ny//2, ny//2 + 1)) * dy

    xs_um = np.zeros(n_patches)
    ys_um = np.zeros(n_patches)

    for i in range(n_patches):
        xi = round(xs_refined[i])
        yi = round(ys_refined[i])
        xi = np.clip(xi, 0, len(vx) - 1)
        yi = np.clip(yi, 0, len(vy) - 1)
        xs_um[i] = vx[xi]
        ys_um[i] = vy[yi]

    # Plot
    _fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs_um, ys_um, "o", color=Colors.mode_color(mode), markersize=8)

    for i in range(n_patches):
        ax.annotate(str(i + 1), (xs_um[i], ys_um[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color=Colors.TEXT)

    ax.set_xlabel("X (um)", fontweight="bold")
    ax.set_ylabel("Y (um)", fontweight="bold")
    ax.set_title(f"XY Offsets ({Colors.mode_label(mode)})", fontweight="bold")
    ax.axis("equal")
    margin = 20
    ax.set_xlim([xs_um.min() - margin, xs_um.max() + margin])
    ax.set_ylim([ys_um.min() - margin, ys_um.max() + margin])
    ax.grid(True)
    plt.tight_layout()
    out_name = f"pollen_{mode}_xy_offsets.png"
    plt.savefig(filepath.with_name(out_name), dpi=150)
    plt.close()

    # Save calibration data
    n_cavity_a = len(cavity_info["cavity_a"])

    diffx = xs_um.copy()
    diffy = ys_um.copy()

    if n_patches >= n_cavity_a > 0:
        diffx = diffx - max(diffx[0] if n_patches > 0 else 0,
                           diffx[n_cavity_a-1] if n_patches >= n_cavity_a else 0)
        diffy = diffy - min(diffy[0] if n_patches > 0 else 0,
                           diffy[n_cavity_a-1] if n_patches >= n_cavity_a else 0)

    # Save to unified H5 file with mode-specific group
    h5_path = get_pollen_h5_path(filepath)
    with h5py.File(h5_path, "a") as f:
        # Store shared cavity info at root level
        f.attrs["is_lbm"] = cavity_info["is_lbm"]
        f.attrs["num_cavities"] = cavity_info["num_cavities"]

        # Store mode-specific data in group
        grp = get_mode_group(f, mode)
        for key in ["diffx", "diffy", "xs_um", "ys_um", "centroid_offx", "centroid_offy",
                    "cavity_a_channels", "cavity_b_channels"]:
            if key in grp:
                del grp[key]

        grp.create_dataset("diffx", data=diffx)
        grp.create_dataset("diffy", data=diffy)
        grp.create_dataset("xs_um", data=xs_um)
        grp.create_dataset("ys_um", data=ys_um)
        grp.create_dataset("centroid_offx", data=offx)
        grp.create_dataset("centroid_offy", data=offy)

        grp.create_dataset("cavity_a_channels", data=np.array(cavity_info["cavity_a"]))
        grp.create_dataset("cavity_b_channels", data=np.array(cavity_info["cavity_b"]))



def plot_comparison(filepath):
    """Plot comparison of auto vs manual calibration results.

    Creates a side-by-side comparison figure showing XY offsets
    and calibration values from both auto and manual modes.

    Parameters
    ----------
    filepath : Path
        Path to the source data file (used to locate h5 files).

    Returns
    -------
    bool
        True if comparison was generated, False if both modes not available.
    """
    h5_path = get_pollen_h5_path(filepath)

    # Check file exists
    if not h5_path.exists():
        return False

    # Load data from both modes
    try:
        with h5py.File(h5_path, "r") as f:
            # Check both modes exist
            if "auto" not in f or "manual" not in f:
                return False

            auto_grp = f["auto"]
            auto_xs = auto_grp["xs_um"][:]
            auto_ys = auto_grp["ys_um"][:]
            auto_dx = auto_grp["diffx"][:]
            auto_dy = auto_grp["diffy"][:]

            manual_grp = f["manual"]
            manual_xs = manual_grp["xs_um"][:]
            manual_ys = manual_grp["ys_um"][:]
            manual_dx = manual_grp["diffx"][:]
            manual_dy = manual_grp["diffy"][:]
    except Exception:
        return False

    n_beams = len(auto_xs)
    beam_nums = np.arange(1, n_beams + 1)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: XY positions comparison
    ax = axes[0, 0]
    ax.plot(auto_xs, auto_ys, "o", color=Colors.AUTO, markersize=8, label="Auto", alpha=0.8)
    ax.plot(manual_xs, manual_ys, "s", color=Colors.MANUAL, markersize=8, label="Manual", alpha=0.8)
    # Draw lines connecting corresponding points
    for i in range(n_beams):
        ax.plot([auto_xs[i], manual_xs[i]], [auto_ys[i], manual_ys[i]],
                "-", color=Colors.WHITE, alpha=0.3, linewidth=1)
    ax.set_xlabel("X (um)", fontweight="bold")
    ax.set_ylabel("Y (um)", fontweight="bold")
    ax.set_title("XY Positions: Auto vs Manual", fontweight="bold")
    ax.legend(loc="best")
    ax.axis("equal")
    ax.grid(True)

    # Top right: dX comparison
    ax = axes[0, 1]
    width = 0.35
    ax.bar(beam_nums - width/2, auto_dx, width, color=Colors.AUTO, label="Auto", alpha=0.8)
    ax.bar(beam_nums + width/2, manual_dx, width, color=Colors.MANUAL, label="Manual", alpha=0.8)
    ax.set_xlabel("Beam Number", fontweight="bold")
    ax.set_ylabel("dX (um)", fontweight="bold")
    ax.set_title("X Offset: Auto vs Manual", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, axis="y")

    # Bottom left: dY comparison
    ax = axes[1, 0]
    ax.bar(beam_nums - width/2, auto_dy, width, color=Colors.AUTO, label="Auto", alpha=0.8)
    ax.bar(beam_nums + width/2, manual_dy, width, color=Colors.MANUAL, label="Manual", alpha=0.8)
    ax.set_xlabel("Beam Number", fontweight="bold")
    ax.set_ylabel("dY (um)", fontweight="bold")
    ax.set_title("Y Offset: Auto vs Manual", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, axis="y")

    # Bottom right: Difference (manual - auto)
    ax = axes[1, 1]
    diff_x = manual_dx - auto_dx
    diff_y = manual_dy - auto_dy
    ax.bar(beam_nums - width/2, diff_x, width, color=Colors.DIFF_X, label="ΔX", alpha=0.8)
    ax.bar(beam_nums + width/2, diff_y, width, color=Colors.DIFF_Y, label="ΔY", alpha=0.8)
    ax.axhline(y=0, color=Colors.WHITE, linestyle="--", alpha=0.5)
    ax.set_xlabel("Beam Number", fontweight="bold")
    ax.set_ylabel("Difference (um)", fontweight="bold")
    ax.set_title("Manual - Auto Difference", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, axis="y")

    # Add statistics text
    rms_x = np.sqrt(np.mean(diff_x**2))
    rms_y = np.sqrt(np.mean(diff_y**2))
    stats_text = f"RMS diff: X={rms_x:.2f}um, Y={rms_y:.2f}um"
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=11, color=Colors.TEXT)

    fig.suptitle("Auto vs Manual Calibration Comparison", fontweight="bold",
                 fontsize=14, color=Colors.TEXT)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_name = "pollen_comparison.png"
    plt.savefig(filepath.with_name(out_name), dpi=150)
    plt.close()

    return True


def extract_calibration_summary(h5_path: str, mode: str = None) -> dict | None:
    """Extract key metrics from calibration H5 file for display.

    Parameters
    ----------
    h5_path : str
        Path to the pollen calibration H5 file.
    mode : str, optional
        Calibration mode to extract ('auto' or 'manual').
        If None, prefers manual if available, else auto.

    Returns
    -------
    dict or None
        Dictionary with calibration summary metrics, or None if file not found.
    """
    from pathlib import Path
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            # Get shared metadata from root
            summary = {
                "num_beamlets": f.attrs.get("num_planes", 0),
                "z_step_um": f.attrs.get("z_step_um", 0),
                "is_lbm": f.attrs.get("is_lbm", False),
                "num_cavities": f.attrs.get("num_cavities", 1),
                "source_file": f.attrs.get("source_file", ""),
            }

            # Determine which mode to load
            if mode is None:
                if "manual" in f:
                    mode = "manual"
                elif "auto" in f:
                    mode = "auto"
                else:
                    return None

            if mode not in f:
                return None

            summary["calibration_mode"] = mode
            grp = f[mode]

            # Mode-specific fit quality metrics
            summary["z_fit_r_squared"] = grp.attrs.get("z_fit_r_squared")
            summary["z_slope_um_per_beam"] = grp.attrs.get("z_slope_um_per_beam")
            summary["decay_length_um"] = grp.attrs.get("decay_length_um")
            summary["decay_length_cavity_a_um"] = grp.attrs.get("decay_length_cavity_a_um")
            summary["decay_length_cavity_b_um"] = grp.attrs.get("decay_length_cavity_b_um")

            # Load arrays from group
            if "diffx" in grp and "diffy" in grp:
                diffx = grp["diffx"][:]
                diffy = grp["diffy"][:]
                summary["rms_dx"] = float(np.sqrt(np.mean(diffx**2)))
                summary["rms_dy"] = float(np.sqrt(np.mean(diffy**2)))
                summary["diffx"] = diffx
                summary["diffy"] = diffy
            else:
                summary["rms_dx"] = None
                summary["rms_dy"] = None

            if "xs_um" in grp:
                summary["xs_um"] = grp["xs_um"][:]
            if "ys_um" in grp:
                summary["ys_um"] = grp["ys_um"][:]

            return summary

    except Exception:
        return None


def get_available_modes(h5_path: str) -> list[str]:
    """Get list of available calibration modes in H5 file.

    Parameters
    ----------
    h5_path : str
        Path to the pollen calibration H5 file.

    Returns
    -------
    list[str]
        List of available mode names ('auto', 'manual').
    """
    from pathlib import Path
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return []

    try:
        with h5py.File(h5_path, "r") as f:
            modes = []
            if "auto" in f:
                modes.append("auto")
            if "manual" in f:
                modes.append("manual")
            return modes
    except Exception:
        return []
