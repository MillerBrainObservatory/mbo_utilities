"""
Pollen calibration for multi-beam two-photon systems.

This script calibrates beamlet positions using pollen grain samples,
matching the behavior of the original MATLAB pollen_calibration.m script.

Uses the modern mbo_utilities API for reading ScanImage data.
"""
import warnings
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate
from scipy.optimize import curve_fit

from mbo_utilities.arrays import open_scanimage
from mbo_utilities.metadata import get_param
from mbo_utilities.metadata.scanimage import (
    is_lbm_stack,
    get_lbm_ai_sources,
    get_z_step_size,
)

from imgui_bundle import (
    imgui,
    hello_imgui,
    imgui_ctx,
    icons_fontawesome_6 as fa,
)
from imgui_bundle import portable_file_dialogs as pfd

# dark theme colors (matching FileDialog)
COL_BG = imgui.ImVec4(0.11, 0.11, 0.12, 1.0)
COL_BG_CARD = imgui.ImVec4(0.16, 0.16, 0.17, 1.0)
COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)
COL_ACCENT_HOVER = imgui.ImVec4(0.25, 0.55, 0.90, 1.0)
COL_ACCENT_ACTIVE = imgui.ImVec4(0.15, 0.45, 0.80, 1.0)
COL_TEXT = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
COL_TEXT_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)
COL_BORDER = imgui.ImVec4(0.35, 0.35, 0.37, 0.7)
COL_SECONDARY = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
COL_SECONDARY_HOVER = imgui.ImVec4(0.42, 0.42, 0.44, 1.0)
COL_SECONDARY_ACTIVE = imgui.ImVec4(0.28, 0.28, 0.30, 1.0)


warnings.simplefilter(action="ignore")


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
    }
)

# Default beam order for 30-channel system (matches MATLAB)
DEFAULT_ORDER_30 = [0, 4, 5, 6, 7, 8, 1, 9, 10, 11, 12, 13, 14, 15, 2, 16, 17, 18, 19, 20, 21, 3, 22, 23, 24, 25, 26, 27, 28, 29]


def get_cavity_indices(metadata: dict, nc: int) -> dict:
    """
    Get cavity A and cavity B channel indices from LBM metadata.

    For LBM acquisitions:
    - Cavity A = channels mapped to AI0 (first PMT)
    - Cavity B = channels mapped to AI1 (second PMT)

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key with ScanImage data.
    nc : int
        Total number of channels.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'cavity_a': list of 0-indexed channel indices for cavity A
        - 'cavity_b': list of 0-indexed channel indices for cavity B
        - 'is_lbm': True if this is an LBM stack
        - 'num_cavities': number of cavities detected (1 or 2)
    """
    result = {
        "cavity_a": [],
        "cavity_b": [],
        "is_lbm": False,
        "num_cavities": 1,
    }

    # Check if this is an LBM stack
    if not is_lbm_stack(metadata):
        # Not LBM - fall back to simple half/half split
        half = nc // 2
        result["cavity_a"] = list(range(half))
        result["cavity_b"] = list(range(half, nc))
        return result

    result["is_lbm"] = True

    # Get AI sources from virtualChannelSettings
    ai_sources = get_lbm_ai_sources(metadata)

    if not ai_sources:
        # No AI source info available - fall back to half/half
        half = nc // 2
        result["cavity_a"] = list(range(half))
        result["cavity_b"] = list(range(half, nc))
        return result

    # Sort AI sources by name (AI0 comes before AI1)
    sorted_sources = sorted(ai_sources.keys())

    if len(sorted_sources) >= 1:
        # AI0 = Cavity A
        cavity_a_channels = ai_sources.get(sorted_sources[0], [])
        # Convert to 0-indexed if needed (virtualChannelSettings uses 1-indexed)
        result["cavity_a"] = sorted([ch - 1 if ch > 0 else ch for ch in cavity_a_channels])

    if len(sorted_sources) >= 2:
        # AI1 = Cavity B
        cavity_b_channels = ai_sources.get(sorted_sources[1], [])
        result["cavity_b"] = sorted([ch - 1 if ch > 0 else ch for ch in cavity_b_channels])
        result["num_cavities"] = 2
    else:
        result["num_cavities"] = 1

    return result


def _push_button_style(primary=True):
    """Push button style colors."""
    if primary:
        imgui.push_style_color(imgui.Col_.button, COL_ACCENT)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_ACCENT_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_ACCENT_ACTIVE)
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
    else:
        imgui.push_style_color(imgui.Col_.button, COL_SECONDARY)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_SECONDARY_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_SECONDARY_ACTIVE)
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)


def _pop_button_style():
    """Pop button style colors."""
    imgui.pop_style_var(2)
    imgui.pop_style_color(4)


def _icon_button(icon: str, label: str, size: imgui.ImVec2, tooltip: str = "") -> bool:
    """Draw a styled icon button with MBO theme."""
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.18, 0.18, 0.20, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.22, 0.22, 0.25, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.15, 0.15, 0.17, 1.0))
    imgui.push_style_color(imgui.Col_.text, COL_ACCENT)
    imgui.push_style_color(imgui.Col_.border, COL_ACCENT)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.5)

    button_text = f"{icon}  {label}"
    clicked = imgui.button(button_text, size)

    if tooltip and imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)

    imgui.pop_style_var(2)
    imgui.pop_style_color(5)

    return clicked


class PollenDialog:
    def __init__(self):
        from mbo_utilities.file_io import get_mbo_dirs
        from mbo_utilities.preferences import get_default_open_dir

        self.selected_path = None
        self._open_multi = None
        self._default_dir = str(get_default_open_dir())

        # assets/settings from MBO utilities
        self._assets_path = Path(get_mbo_dirs()["assets"])
        self._settings_path = Path(get_mbo_dirs()["settings"])

    def _center_text(self, text: str, color: imgui.ImVec4 = None):
        """Draw centered text."""
        text_w = imgui.calc_text_size(text).x
        avail_w = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - text_w) * 0.5)
        if color:
            imgui.text_colored(color, text)
        else:
            imgui.text(text)

    def _center_widget(self, widget_w: float):
        """Center the next widget."""
        avail_w = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - widget_w) * 0.5)

    def render(self):
        # global style (matching FileDialog)
        imgui.push_style_color(imgui.Col_.window_bg, COL_BG)
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
        imgui.push_style_color(imgui.Col_.border, COL_BORDER)
        imgui.push_style_color(imgui.Col_.separator, imgui.ImVec4(0.35, 0.35, 0.37, 0.6))
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.22, 0.22, 0.23, 1.0))
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.28, 0.28, 0.29, 1.0))
        imgui.push_style_color(imgui.Col_.check_mark, COL_ACCENT)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(1.0, 0.8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, hello_imgui.em_to_vec2(0.6, 0.4))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, hello_imgui.em_to_vec2(0.6, 0.4))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)

        with imgui_ctx.begin_child("##main", size=imgui.ImVec2(0, 0), window_flags=imgui.WindowFlags_.no_scrollbar):
            imgui.push_id("pollen_fd")

            # header
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            self._center_text("Miller Brain Observatory", COL_ACCENT)
            self._center_text("Pollen Calibration", COL_TEXT_DIM)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            # description card
            avail_w = imgui.get_content_region_avail().x
            card_w = avail_w - hello_imgui.em_size(1)
            self._center_widget(card_w)

            imgui.push_style_color(imgui.Col_.child_bg, COL_BG_CARD)
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 6.0)
            imgui.push_style_var(imgui.StyleVar_.cell_padding, hello_imgui.em_to_vec2(0.4, 0.2))

            child_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
            window_flags = imgui.WindowFlags_.no_scrollbar

            with imgui_ctx.begin_child("##info", size=imgui.ImVec2(card_w, 0), child_flags=child_flags, window_flags=window_flags):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.indent(hello_imgui.em_size(0.6))

                imgui.text_wrapped(
                    "Calibrates beamlet Z and XY offsets using a piezo "
                    "z-stack of pollen grains. Click the same bead in each "
                    "beamlet image to measure focal plane positions."
                )

                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.text_colored(COL_TEXT_DIM, "Outputs: .h5 calibration + .png figures")

                imgui.unindent(hello_imgui.em_size(0.6))
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            imgui.pop_style_var(2)
            imgui.pop_style_color()

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.5))

            # action button
            btn_w = min(avail_w - hello_imgui.em_size(2), hello_imgui.em_size(16))
            btn_h = hello_imgui.em_size(1.8)

            self._center_widget(btn_w)
            if _icon_button(
                fa.ICON_FA_FILE_IMAGE,
                "Open Pollen TIFF",
                imgui.ImVec2(btn_w, btn_h),
                "Select a pollen calibration TIFF file"
            ):
                self._open_multi = pfd.open_file(
                    "Select pollen TIFF",
                    self._default_dir,
                    ["TIFF Files", "*.tif *.tiff", "All Files", "*"],
                    pfd.opt.none
                )

            # handle file selection
            if self._open_multi and self._open_multi.ready():
                result = self._open_multi.result()
                if result:
                    if isinstance(result, (list, tuple)):
                        self.selected_path = result[0]
                    else:
                        self.selected_path = result
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._open_multi = None

            # quit button (centered)
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            qsz = imgui.ImVec2(hello_imgui.em_size(6), hello_imgui.em_size(1.5))
            self._center_widget(qsz.x)
            _push_button_style(primary=False)
            if imgui.button(f"{fa.ICON_FA_XMARK}  Quit", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True
            _pop_button_style()

            imgui.pop_id()

        imgui.pop_style_var(4)
        imgui.pop_style_color(8)


def pollen_calibration_mbo(filepath, order=None, zoom=None, fov_um=None, dz_override=None):
    """
    Run pollen calibration matching MATLAB pollen_calibration.m behavior.

    Uses mbo_utilities.arrays.open_scanimage() for proper ScanImage data loading
    with automatic stack type detection (LBM, piezo, single-plane).

    Parameters
    ----------
    filepath : str or Path
        Path to the pollen TIFF file.
    order : list, optional
        Beamlet order sequence. If None, uses default for nc=30 or sequential.
    zoom : float, optional
        Zoom factor for FOV calculation. If None, reads from metadata.
    fov_um : float, optional
        Field of view in microns at zoom=1. If None, uses 600.0 (standard).
    dz_override : float, optional
        Z-step size in microns. If provided, overrides metadata value.
    """
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    # Load data using modern API - auto-detects stack type
    # Use dims="ZCYX" for pollen calibration:
    #   Z = piezo z-positions (scanning through depth)
    #   C = channels/beamlets (each beamlet has different focal plane)
    arr = open_scanimage(filepath, dims="ZCYX")
    metadata = arr.metadata


    # Get dimensions from array - shape is (Z, C, Y, X) for pollen calibration
    nz = arr.shape[0]  # Z dimension = number of piezo z-positions
    nc = arr.num_channels  # C dimension = number of beamlets/channels
    ny = arr.shape[2]  # Y dimension
    nx = arr.shape[3]  # X dimension

    # Get z-step from metadata or override
    if dz_override is not None:
        z_step_um = dz_override
    else:
        z_step_um = get_z_step_size(metadata)
        if z_step_um is None:
            z_step_um = get_param(metadata, "dz", default=1.0)
            if z_step_um == 1.0:
                pass
        else:
            pass

    # Get zoom factor - from CLI, metadata, or default
    if zoom is None:
        zoom = get_param(metadata, "zoom_factor", default=1.0)

    # FOV at zoom=1 (standard for MIMMS systems)
    if fov_um is None:
        fov_um = 600.0

    # Calculate pixel size in microns: dx = fov / zoom / pixels
    dx = fov_um / zoom / nx
    dy = fov_um / zoom / ny

    # Set up beam order
    if order is None:
        if nc == 30:
            # MATLAB order: [1 5:9 2 10:17 3 18:23 4 24:30] converted to 0-indexed
            order = DEFAULT_ORDER_30
        else:
            order = list(range(nc))

    # Validate order
    if len(order) != nc:
        order = list(range(nc))

    # Get cavity indices from LBM metadata
    # For LBM: Cavity A = AI0 channels, Cavity B = AI1 channels
    cavity_info = get_cavity_indices(metadata, nc)

    if cavity_info["is_lbm"]:
        if cavity_info["cavity_b"]:
            pass
    else:
        pass

    # Load full volume into memory
    # Array is (Z, C, Y, X) where Z=piezo positions, C=beamlets
    # This matches our internal (nz, nc, ny, nx) convention
    vol = np.asarray(arr[:]).astype(np.float32)  # Load all data
    vol -= vol.mean()

    # 1. Scan offset correction
    vol, _scan_corrections = correct_scan_phase(vol, filepath, z_step_um, metadata)

    # 2. Plot beamlet grid (Figure 701 in MATLAB)
    plot_beamlet_grid(vol, order, filepath)

    # 3. User marked pollen selection
    xs, ys, Iz, III, _zoi_per_channel = user_pollen_selection(vol, order, filepath)

    # 4. Power vs Z analysis
    ZZ, zoi, pp = analyze_power_vs_z(Iz, filepath, z_step_um, order, nc)

    # 5. Analyze Z positions with linear fit (using cavity info)
    analyze_z_positions(ZZ, zoi, order, filepath, cavity_info)

    # 6. Exponential decay with separate cavity fits
    fit_exp_decay(ZZ, zoi, order, filepath, pp, cavity_info, z_step_um, nz)

    # 7. Z spacing plot
    plot_z_spacing(ZZ, zoi, order, filepath)

    # 8. XY calibration with proper units
    calibrate_xy(xs, ys, III, filepath, dx, dy, nx, ny, cavity_info)



def correct_scan_phase(vol, filepath, z_step_um, metadata):
    """Detect and correct scan phase offsets along Y-axis."""
    scan_corrections = []
    nz, nc, ny, nx = vol.shape

    for c in range(nc):
        # Take z-projection like MATLAB Iinit(:,:,c)
        Iproj = vol[:, c, :, :].max(axis=0)  # (ny, nx)
        offset = return_scan_offset(Iproj)
        scan_corrections.append(offset)

    for c in range(nc):
        # Apply to each z-slice
        for z in range(nz):
            vol[z, c, :, :] = fix_scan_phase(vol[z, c, :, :], scan_corrections[c])

    # Save scan corrections with metadata
    h5_path = filepath.with_name(filepath.stem + "_pollen.h5")
    with h5py.File(h5_path, "a") as f:
        if "scan_corrections" in f:
            del f["scan_corrections"]
        f.create_dataset("scan_corrections", data=np.array(scan_corrections))

        # Save metadata as file attributes for H5Array compatibility
        f.attrs["num_planes"] = nc
        f.attrs["roi_width_px"] = nx
        f.attrs["roi_height_px"] = ny
        f.attrs["z_step_um"] = z_step_um
        f.attrs["source_file"] = filepath.name
        f.attrs["pollen_calibration_version"] = "2.0"

        # Save additional metadata if available
        frame_rate = get_param(metadata, "fs")
        if frame_rate is not None:
            f.attrs["frame_rate"] = frame_rate

        dx = get_param(metadata, "dx")
        dy = get_param(metadata, "dy")
        if dx is not None and dy is not None:
            f.attrs["pixel_resolution"] = f"({dx}, {dy})"

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


def plot_beamlet_grid(vol, order, filepath):
    """Plot 5x6 grid of max-projected beamlet images (Figure 701 in MATLAB)."""
    _nz, nc, ny, nx = vol.shape
    Imax = vol.max(axis=0)  # (nc, ny, nx)

    # Determine grid size
    n_cols = 6
    n_rows = int(np.ceil(nc / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()

    for idx in range(nc):
        ax = axes[idx]
        channel = order[idx]
        img = Imax[channel, :, :].T  # Transpose to match MATLAB's imagesc behavior
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        ax.set_xlim([0, ny])
        ax.set_ylim([0, nx])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Beam {idx+1}", fontsize=8)

    # Hide unused subplots
    for idx in range(nc, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Max-Projected Beamlet Images (Scan Corrected)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_beamlet_grid.png"), dpi=150)
    plt.close()


def user_pollen_selection(vol, order, filepath, num=10):
    """
    Interactive pollen bead selection matching MATLAB behavior.

    vol : ndarray, shape (nz, nc, ny, nx)
    order : list of channel indices in beam order
    """
    nz, nc, ny, nx = vol.shape
    xs, ys, Iz, III = [], [], [], []
    zoi_per_channel = []


    amt = max(1, 10)  # Smoothing window

    for idx in range(nc):
        channel = order[idx]
        img = vol[:, channel, :, :].max(axis=0)  # (ny, nx) - max projection over z

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            img,
            cmap="gray",
            vmin=np.percentile(img, 1),
            vmax=np.percentile(img, 99),
            origin="upper",
        )
        ax.set_title(f"Beamlet {idx + 1}/{nc} (Channel {channel})\nClick on pollen bead", fontsize=12)
        ax.set_xlim([0, nx])
        ax.set_ylim([ny, 0])  # Flip to match MATLAB
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        pts = plt.ginput(1, timeout=0)
        plt.close(fig)

        if not pts:
            continue

        x, y = pts[0]
        ix, iy = round(x), round(y)

        xs.append(x)
        ys.append(y)

        # Patch around point across z (matching MATLAB indexing)
        y0, y1 = max(0, iy - num), min(ny, iy + num + 1)
        x0, x1 = max(0, ix - num), min(nx, ix + num + 1)

        # Extract patch and compute trace
        # MATLAB: Iz(order(kk),:) = max(max(movmean(movmean(vol(...),3,1),3,2)))
        patch = vol[:, channel, y0:y1, x0:x1]  # (nz, roi_y, roi_x)

        # Apply moving mean smoothing like MATLAB (size 3 along each spatial dim)
        smoothed_patch = uniform_filter1d(patch, size=3, axis=1, mode="nearest")
        smoothed_patch = uniform_filter1d(smoothed_patch, size=3, axis=2, mode="nearest")

        # Max over spatial dimensions to get trace
        trace = smoothed_patch.max(axis=(1, 2))  # (nz,)
        Iz.append(trace)

        # Find best z using smoothed trace
        smoothed_trace = uniform_filter1d(trace, size=amt, mode="nearest")
        zoi = int(np.argmax(smoothed_trace))
        zoi_per_channel.append(zoi)

        # Store crop at best z
        crop = vol[zoi, channel, y0:y1, x0:x1]  # 2D crop at best z
        III.append(crop)

    # Plot selected patches (Figure 101 in MATLAB)
    if III:
        plot_selected_patches(III, filepath, nc)

    # Convert to arrays
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

    return np.array(xs), np.array(ys), Iz, III, zoi_per_channel


def plot_selected_patches(III, filepath, nc):
    """Plot 5x6 grid of selected pollen patches at best z (Figure 101 in MATLAB)."""
    n_patches = len(III)
    n_cols = 6
    n_rows = int(np.ceil(n_patches / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, crop in enumerate(III):
        ax = axes[idx]
        num = crop.shape[0] // 2
        extent = [-num, num, -num, num]
        ax.imshow(crop, cmap="gray", extent=extent,
                  vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99))
        ax.set_xlim([-num, num])
        ax.set_ylim([-num, num])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Beam {idx+1}", fontsize=8)

    # Hide unused subplots
    for idx in range(n_patches, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Selected Pollen Patches at Best Z", fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_selected_patches.png"), dpi=150)
    plt.close()


def analyze_power_vs_z(Iz, filepath, DZ, order, nc):
    """
    Analyze power vs Z depth matching MATLAB Figure 99.

    Returns ZZ (z positions), zoi (indices of max), pp (peak powers).
    """
    nz = Iz.shape[1]
    ZZ = np.flip(np.arange(nz) * DZ)  # Z positions in microns (flipped like MATLAB)

    amt = max(1, round(10.0 / DZ))
    smoothed = uniform_filter1d(Iz, size=amt, axis=1, mode="nearest")

    # Find peak position and value for each beamlet
    zoi = smoothed.argmax(axis=1)
    pp = smoothed.max(axis=1)

    # Create figure matching MATLAB Figure 99
    _fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all curves
    for i in range(len(order)):
        ax.plot(ZZ, np.sqrt(smoothed[i, :]), alpha=0.7)

    # Plot peak markers
    ax.plot(ZZ[zoi], np.sqrt(pp), "k.", markersize=10)

    # Add beam number labels at peaks
    # Use a small offset based on data range (3% of range)
    sqrt_pp = np.sqrt(pp)
    label_offset = 0.03 * (sqrt_pp.max() - sqrt_pp.min())
    for i in range(len(order)):
        ax.text(
            ZZ[zoi[i]],
            np.sqrt(pp[i]) + label_offset,
            str(i + 1),
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("Piezo Z (µm)", fontweight="bold")
    ax.set_ylabel("2p signal (a.u.)", fontweight="bold")
    ax.set_title("Power vs. Z-depth", fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_power_vs_z.png"), dpi=150)
    plt.close()

    return ZZ, zoi, pp


def analyze_z_positions(ZZ, zoi, order, filepath, cavity_info):
    """
    Analyze Z positions vs beam number with Cavity A/B split and linear fit.
    Matches MATLAB z_vs_N figure.

    Parameters
    ----------
    cavity_info : dict
        Dictionary with 'cavity_a' and 'cavity_b' channel indices.
    """
    n_beams = len(order)

    Z0 = ZZ[zoi[0]]  # Reference from first beam
    z_rel = ZZ[zoi] - Z0  # Relative Z positions

    _fig, ax = plt.subplots(figsize=(7, 5))

    # Get cavity indices from metadata-derived info
    cavity_a_channels = set(cavity_info["cavity_a"])
    cavity_b_channels = set(cavity_info["cavity_b"])

    # Map beam index to cavity based on which channel it corresponds to
    cavity_a_beams = []
    cavity_b_beams = []

    for beam_idx in range(n_beams):
        channel = order[beam_idx]
        if channel in cavity_a_channels:
            cavity_a_beams.append(beam_idx)
        elif channel in cavity_b_channels:
            cavity_b_beams.append(beam_idx)
        else:
            # Channel not in either cavity - assign to A by default
            cavity_a_beams.append(beam_idx)

    # Cavity A - blue circles
    if cavity_a_beams:
        ax.plot(
            [i+1 for i in cavity_a_beams],
            [z_rel[i] for i in cavity_a_beams],
            "bo", markersize=6, label="Cavity A"
        )

    # Cavity B - green squares
    if cavity_b_beams:
        ax.plot(
            [i+1 for i in cavity_b_beams],
            [z_rel[i] for i in cavity_b_beams],
            "gs", markersize=6, label="Cavity B", color=[0, 0.5, 0]
        )

    # Linear fit on all data
    beam_nums = np.arange(1, n_beams + 1)
    try:
        coeffs = np.polyfit(beam_nums, z_rel, 1)
        poly = np.poly1d(coeffs)

        # Calculate R²
        y_pred = poly(beam_nums)
        ss_res = np.sum((z_rel - y_pred) ** 2)
        ss_tot = np.sum((z_rel - np.mean(z_rel)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Plot fit line
        x_fit = np.linspace(0, n_beams + 1, 101)
        ax.plot(x_fit, poly(x_fit), "k-",
                label=f"Linear fit (r² = {r_squared:.3f})\ny = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
    except Exception:
        pass

    ax.set_xlabel("Beam number", fontweight="bold")
    ax.set_ylabel("Z position (µm)", fontweight="bold")
    ax.set_title("Z Position vs. Beam Number", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_z_vs_N.png"), dpi=150)
    plt.close()


def fit_exp_decay(ZZ, zoi, order, filepath, pp, cavity_info, DZ, nz):
    """
    Fit exponential decay with separate fits for Cavity A and Cavity B.
    Matches MATLAB Figure 1 and semilogy figure.

    Parameters
    ----------
    cavity_info : dict
        Dictionary with 'cavity_a' and 'cavity_b' channel indices.
    """
    def exp_func(z, a, b):
        return a * np.exp(b * z)

    n_beams = len(order)

    # Get cavity indices from metadata-derived info
    cavity_a_channels = set(cavity_info["cavity_a"])
    cavity_b_channels = set(cavity_info["cavity_b"])

    # Map beam index to cavity based on which channel it corresponds to
    cavity_a_beams = []
    cavity_b_beams = []

    for beam_idx in range(n_beams):
        channel = order[beam_idx]
        if channel in cavity_a_channels:
            cavity_a_beams.append(beam_idx)
        elif channel in cavity_b_channels:
            cavity_b_beams.append(beam_idx)
        else:
            cavity_a_beams.append(beam_idx)

    # Get z and power for each cavity
    z_all = ZZ[zoi]
    p_all = np.sqrt(pp)

    z1 = np.array([z_all[i] for i in cavity_a_beams]) if cavity_a_beams else np.array([])
    p1 = np.array([p_all[i] for i in cavity_a_beams]) if cavity_a_beams else np.array([])

    z2 = np.array([z_all[i] for i in cavity_b_beams]) if cavity_b_beams else np.array([])
    p2 = np.array([p_all[i] for i in cavity_b_beams]) if cavity_b_beams else np.array([])

    # === Linear scale plot ===
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data points
    if len(z1) > 0:
        ax.plot(z1, p1, "bo", markersize=6, label="Data Cavity A")
    if len(z2) > 0:
        ax.plot(z2, p2, "s", color=[0, 0.5, 0], markersize=6, label="Data Cavity B")

    z_fit_range = DZ * np.linspace(0, nz - 1, 1001)

    # Fit Cavity A
    popt1 = None
    ls1 = None
    if len(z1) > 2:
        try:
            popt1, _ = curve_fit(exp_func, z1, p1, p0=(p1.max(), -0.01), maxfev=5000)
            ax.plot(z_fit_range, exp_func(z_fit_range, *popt1), "r-",
                    label=f"Fit C1 (lₛ = {abs(1/popt1[1]):.0f} µm)")
            ls1 = abs(1/popt1[1])
        except Exception:
            pass

    # Fit Cavity B
    popt2 = None
    ls2 = None
    if len(z2) > 2:
        try:
            popt2, _ = curve_fit(exp_func, z2, p2, p0=(p2.max(), -0.01), maxfev=5000)
            ax.plot(z_fit_range, exp_func(z_fit_range, *popt2), "k-",
                    label=f"Fit C2 (lₛ = {abs(1/popt2[1]):.0f} µm)")
            ls2 = abs(1/popt2[1])
        except Exception:
            pass

    # Fit both cavities combined
    z_combined = np.concatenate([z1, z2]) if len(z2) > 0 else z1
    p_combined = np.concatenate([p1, p2]) if len(p2) > 0 else p1

    popt3 = None
    ls3 = None
    if len(z_combined) > 2:
        try:
            popt3, _ = curve_fit(exp_func, z_combined, p_combined, p0=(p_combined.max(), -0.01), maxfev=5000)
            ax.plot(z_fit_range, exp_func(z_fit_range, *popt3), "m-",
                    label=f"Fit both (lₛ = {abs(1/popt3[1]):.0f} µm)")
            ls3 = abs(1/popt3[1])
        except Exception:
            pass

    ax.set_xlabel("Z (µm)", fontweight="bold")
    ax.set_ylabel("Power (a.u.)", fontweight="bold")
    ax.set_title("Power vs Depth (Linear Scale)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_power_linear.png"), dpi=150)
    plt.close()

    # === Log scale plot ===
    _fig, ax = plt.subplots(figsize=(8, 6))

    if len(z1) > 0:
        ax.semilogy(z1, p1, "bo", markersize=6, label="Data Cavity A")
    if len(z2) > 0:
        ax.semilogy(z2, p2, "s", color=[0, 0.5, 0], markersize=6, label="Data Cavity B")

    # Plot fits on log scale
    if popt1 is not None and ls1 is not None:
        ax.semilogy(z_fit_range, exp_func(z_fit_range, *popt1), "r-",
                    label=f"Fit C1 (lₛ = {ls1:.0f} µm)")
    if popt2 is not None and ls2 is not None:
        ax.semilogy(z_fit_range, exp_func(z_fit_range, *popt2), "k-",
                    label=f"Fit C2 (lₛ = {ls2:.0f} µm)")
    if popt3 is not None and ls3 is not None:
        ax.semilogy(z_fit_range, exp_func(z_fit_range, *popt3), "m-",
                    label=f"Fit both (lₛ = {ls3:.0f} µm)")

    ax.set_xlabel("Z (µm)", fontweight="bold")
    ax.set_ylabel("Power (a.u.)", fontweight="bold")
    ax.set_title("Power vs Depth (Log Scale)", fontweight="bold")
    ax.set_xlim([0, DZ * nz])
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_power_log.png"), dpi=150)
    plt.close()


def plot_z_spacing(ZZ, zoi, order, filepath):
    """Plot Z spacing between consecutive beams (diff plot from MATLAB)."""
    z_positions = ZZ[zoi]
    z_diff = np.diff(z_positions)

    _fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(z_diff) + 1), z_diff, "k.", markersize=14)
    ax.set_xlabel("Beam pair", fontweight="bold")
    ax.set_ylabel("ΔZ (µm)", fontweight="bold")
    ax.set_title("Z Spacing Between Consecutive Beams", fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_z_spacing.png"), dpi=150)
    plt.close()


def calibrate_xy(xs, ys, III, filepath, dx, dy, nx, ny, cavity_info):
    """
    XY calibration with centroid refinement and proper units (microns).
    Matches MATLAB XY calibration section.

    Parameters
    ----------
    cavity_info : dict
        Dictionary with 'cavity_a' and 'cavity_b' channel indices.
    """
    n_patches = III.shape[2]
    num = III.shape[0] // 2  # Half-size of patch

    # Create coordinate grids for centroid calculation
    xx = np.arange(-num, num + 1)
    yy = np.arange(-num, num + 1)
    XX, YY = np.meshgrid(xx, yy)

    offx = np.zeros(n_patches)
    offy = np.zeros(n_patches)

    # Calculate centroid offsets for each patch (like MATLAB)
    for zz in range(n_patches):
        IOI = III[:, :, zz].copy()
        # Binarize
        IOI[IOI > 0] = 1
        IOI[IOI <= 0] = 0

        total = np.trapezoid(np.trapezoid(IOI, axis=0))
        if total > 0:
            offx[zz] = round(np.trapezoid(np.trapezoid(XX * IOI, axis=0)) / total)
            offy[zz] = round(np.trapezoid(np.trapezoid(YY * IOI, axis=0)) / total)

    # Refine positions with centroid offsets
    xs_refined = xs + offx
    ys_refined = ys + offy

    # Convert to physical coordinates (microns from center)
    vx = (np.arange(-nx//2, nx//2 + 1)) * dx
    vy = (np.arange(-ny//2, ny//2 + 1)) * dy

    # Map pixel positions to microns
    xs_um = np.zeros(n_patches)
    ys_um = np.zeros(n_patches)

    for i in range(n_patches):
        xi = round(xs_refined[i])
        yi = round(ys_refined[i])
        xi = np.clip(xi, 0, len(vx) - 1)
        yi = np.clip(yi, 0, len(vy) - 1)
        xs_um[i] = vx[xi]
        ys_um[i] = vy[yi]

    # Plot XY offsets in microns - show ALL selected beads
    _fig, ax = plt.subplots(figsize=(6, 6))

    # Plot all patches as a single series
    ax.plot(xs_um, ys_um, "bo", markersize=8)

    # Add beam number labels next to each point
    for i in range(n_patches):
        ax.annotate(str(i + 1), (xs_um[i], ys_um[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    # Store cavity info for reference but don't split the plot
    n_cavity_a = len(cavity_info["cavity_a"])
    len(cavity_info["cavity_b"])

    ax.set_xlabel("X (µm)", fontweight="bold")
    ax.set_ylabel("Y (µm)", fontweight="bold")
    ax.set_title("XY Offsets", fontweight="bold")
    ax.axis("equal")
    # Auto-scale with padding instead of fixed limits
    margin = 20
    ax.set_xlim([xs_um.min() - margin, xs_um.max() + margin])
    ax.set_ylim([ys_um.min() - margin, ys_um.max() + margin])
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath.with_name("pollen_calibration_xy_offsets.png"), dpi=150)
    plt.close()

    # Save calibration data
    # Calculate relative offsets like MATLAB
    diffx = xs_um.copy()
    diffy = ys_um.copy()

    # Use cavity A size for reference offset calculation
    if n_patches >= n_cavity_a > 0:
        diffx = diffx - max(diffx[0] if n_patches > 0 else 0,
                           diffx[n_cavity_a-1] if n_patches >= n_cavity_a else 0)
        diffy = diffy - min(diffy[0] if n_patches > 0 else 0,
                           diffy[n_cavity_a-1] if n_patches >= n_cavity_a else 0)

    h5_path = filepath.with_name(filepath.stem + "_pollen.h5")
    with h5py.File(h5_path, "a") as f:
        for key in ["diffx", "diffy", "xs_um", "ys_um", "centroid_offx", "centroid_offy",
                    "cavity_a_channels", "cavity_b_channels"]:
            if key in f:
                del f[key]

        f.create_dataset("diffx", data=diffx)
        f.create_dataset("diffy", data=diffy)
        f.create_dataset("xs_um", data=xs_um)
        f.create_dataset("ys_um", data=ys_um)
        f.create_dataset("centroid_offx", data=offx)
        f.create_dataset("centroid_offy", data=offy)

        # Save cavity info
        f.create_dataset("cavity_a_channels", data=np.array(cavity_info["cavity_a"]))
        f.create_dataset("cavity_b_channels", data=np.array(cavity_info["cavity_b"]))
        f.attrs["is_lbm"] = cavity_info["is_lbm"]
        f.attrs["num_cavities"] = cavity_info["num_cavities"]



def select_pollen_file() -> str | None:
    from imgui_bundle import immapp, hello_imgui
    from mbo_utilities.gui._setup import get_default_ini_path

    dlg = PollenDialog()

    def _render():
        dlg.render()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Pollen Calibration"
    params.app_window_params.window_geometry.size = (420, 380)
    params.ini_filename = get_default_ini_path("pollen_calibration")
    params.callbacks.show_gui = _render

    addons = immapp.AddOnsParams()
    addons.with_markdown = False  # not needed anymore
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)

    return dlg.selected_path if dlg.selected_path else None


@click.command()
@click.option(
    "--in",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    help="Input file or directory containing pollen data",
)
@click.option(
    "--zoom",
    type=float,
    default=None,
    help="Zoom factor for FOV calculation. If not provided, reads from metadata.",
)
@click.option(
    "--fov",
    type=float,
    default=None,
    help="Field of view in microns at zoom=1 (default: 600.0)",
)
@click.option(
    "--dz",
    type=float,
    default=None,
    help="Z-step size in microns (piezo step). Overrides metadata value.",
)
def main(input_path, zoom, fov, dz):
    """Run pollen calibration with optional input/output paths."""
    if input_path is None:
        data_in = select_pollen_file()
        if not data_in:
            click.echo("No file selected, exiting.")
            return
        input_path = data_in

    pollen_calibration_mbo(input_path, zoom=zoom, fov_um=fov, dz_override=dz)


if __name__ == "__main__":
    main()
