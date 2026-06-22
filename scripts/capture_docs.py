"""Capture documentation screenshots of the real Miller Brain Studio GUI.

Every panel is captured from the actual running GUI, never a reconstruction.

- Data-viewer panels (data view, save-as dialog, save options, metadata editor,
  suite2p Run tab) render into an *offscreen* fastplotlib canvas; the composited
  frame (scene + imgui overlay + popups) is read back pixel-perfect with no
  display, focus, or screen-grab needed.
- The two standalone imgui windows (file dialog, metadata inspector) use
  hello_imgui's framebuffer screenshot and require a desktop session.

Usage:
  uv run scripts/capture_docs.py [DATA_PATH]

DATA_PATH defaults to E:/demo/mk355/raw. Data-driven captures are skipped if it
is missing; the file dialog still captures.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

OUTPUT_DIR = Path("docs/_images/gui/readme")
DEFAULT_DATA = Path(r"E:/demo/mk355/raw")

PADDING = 30
SHADOW_BLUR = 12
SHADOW_OFFSET = (0, 8)
SHADOW_OPACITY = 0.25


def style_image(img: Image.Image, output_path: Path):
    """Add transparent padding and a soft drop shadow, then save as PNG."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    shadow_margin = SHADOW_BLUR * 2
    final_img = Image.new("RGBA", (img.width + PADDING * 2, img.height + PADDING * 2), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (img.width + shadow_margin, img.height + shadow_margin), (0, 0, 0, 0))
    shadow_draw = Image.new("RGBA", (img.width, img.height), (0, 0, 0, int(255 * SHADOW_OPACITY)))
    shadow.paste(shadow_draw, (shadow_margin // 2, shadow_margin // 2))
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))

    final_img.paste(
        shadow,
        (PADDING + SHADOW_OFFSET[0] - shadow_margin // 2, PADDING + SHADOW_OFFSET[1] - shadow_margin // 2),
        shadow,
    )
    final_img.paste(img, (PADDING, PADDING))
    final_img.save(output_path, "PNG")
    print(f"  -> {output_path}")


def _resolve_data_path(argv) -> Path | None:
    if len(argv) > 1:
        p = Path(argv[1])
        if p.exists():
            return p
        print(f"given DATA_PATH not found: {p}")
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA
    return None


# Offscreen viewer captures -------------------------------------------------
# An offscreen canvas composites scene + imgui overlay + popups into a frame we
# read back directly, so these need no display and are byte-for-byte the GUI.

def _build_viewer(data_in: Path, size: tuple[int, int]):
    from mbo_utilities.reader import imread
    from mbo_utilities.arrays import normalize_roi
    from mbo_utilities.gui.run_gui import _create_image_widget

    arr = imread(data_in, roi=normalize_roi(None))
    iw = _create_image_widget(
        arr,
        widget=True,
        figure_kwargs_override={"canvas": "offscreen", "size": size},
    )
    gui = next((g for g in iw.figure.guis.values() if g is not None), None)
    return iw, gui


def _draw(iw, n: int):
    """Pump n frames; return the last composited readback."""
    frame = None
    for _ in range(n):
        frame = iw.figure.canvas.draw()
    return frame


def _save(frame, name: str):
    arr = np.asarray(frame)[..., :3].copy()
    style_image(Image.fromarray(arr), OUTPUT_DIR / name)


def _shut(iw):
    try:
        iw.figure.canvas.close()
    except Exception:
        pass


def capture_data_view(data_in: Path):
    iw, _ = _build_viewer(data_in, (900, 600))
    _save(_draw(iw, 10), "02_step_data_view.png")
    _shut(iw)


def capture_metadata_editor(data_in: Path):
    iw, gui = _build_viewer(data_in, (900, 650))
    _draw(iw, 5)
    gui._show_metadata_popup = True
    _save(_draw(iw, 8), "04_configurable_metadata.png")
    _shut(iw)


def _capture_save_options(data_in: Path, name: str, ext_idx: int | None = None,
                          size: tuple[int, int] = (1000, 760)):
    """Open Save As -> Options and snapshot it. ext_idx selects the output
    format (0 .tiff, 1 .zarr, 2 .bin, 3 .h5, 4 .mp4), which changes the
    format-specific options shown."""
    iw, gui = _build_viewer(data_in, size)
    _draw(iw, 5)
    if ext_idx is not None:
        gui._ext_idx = ext_idx
    gui._saveas_popup_open = True
    _draw(iw, 5)
    gui._saveas_options_open = True
    _save(_draw(iw, 8), name)
    _shut(iw)


def capture_save_options(data_in: Path):
    _capture_save_options(data_in, "05_save_options.png")


def capture_save_options_zarr(data_in: Path):
    _capture_save_options(data_in, "08_save_options_zarr.png", ext_idx=1, size=(1000, 820))


def capture_save_options_mp4(data_in: Path):
    _capture_save_options(data_in, "09_save_options_mp4.png", ext_idx=4, size=(1000, 880))


def capture_suite2p_settings(data_in: Path):
    iw, gui = _build_viewer(data_in, (640, 820))
    _draw(iw, 5)
    gui._force_run_tab = True
    _save(_draw(iw, 8), "06_suite2p_settings.png")
    _shut(iw)


def _capture_popup(data_in: Path, flag: str, name: str, size: tuple[int, int]):
    """Set a one-shot popup flag on the side widget, then snapshot it."""
    iw, gui = _build_viewer(data_in, size)
    _draw(iw, 5)
    setattr(gui, flag, True)
    _save(_draw(iw, 8), name)
    _shut(iw)


def capture_keybinds(data_in: Path):
    _capture_popup(data_in, "_show_keybinds_popup", "10_keybinds.png", (900, 700))


def capture_process_console(data_in: Path):
    _capture_popup(data_in, "_show_process_console", "11_process_console.png", (1000, 650))


def capture_options(data_in: Path):
    _capture_popup(data_in, "_show_options_popup", "12_options.png", (900, 700))


def _set_s2p_demo_params(gui):
    """Set a few Suite2p params to non-default values so the 'modified from
    default' orange tint is visible in the captured settings panel."""
    s = getattr(gui, "s2p", None)
    if s is not None:
        s.tau = 0.7
        s.diameter_x = 6.0
        s.diameter_y = 6.0


def capture_suite2p_parameters(data_in: Path):
    iw, gui = _build_viewer(data_in, (1180, 880))
    # the offscreen imgui backend never registers the lazily-added bold font
    # the settings popup uses; fall back to the default font for the capture.
    gui._bold_font = None
    _set_s2p_demo_params(gui)
    _draw(iw, 5)
    gui._force_run_tab = True
    _draw(iw, 6)
    gui._force_pipe_settings = True
    _save(_draw(iw, 10), "07_suite2p_parameters.png")
    _shut(iw)


def capture_suite2p_legend(data_in: Path):
    iw, gui = _build_viewer(data_in, (1180, 880))
    gui._bold_font = None
    _set_s2p_demo_params(gui)
    _draw(iw, 5)
    gui._force_run_tab = True
    _draw(iw, 6)
    gui._force_pipe_settings = True
    _draw(iw, 6)
    gui._force_pipe_legend = True
    frame = _draw(iw, 8)
    # the Legend popup renders at the top-left of the settings modal; crop to
    # it so the small popup is legible as a standalone figure.
    arr = np.asarray(frame)[8:300, 2:412, :3].copy()
    style_image(Image.fromarray(arr), OUTPUT_DIR / "13_suite2p_legend.png")
    _shut(iw)


def capture_save_as_dialog(data_in: Path):
    iw, gui = _build_viewer(data_in, (1000, 720))
    _draw(iw, 5)
    gui._saveas_popup_open = True
    _save(_draw(iw, 8), "04_save_as_dialog.png")
    _shut(iw)


# Standalone imgui-window captures ------------------------------------------
# These run a real hello_imgui app and screenshot its framebuffer; they need a
# desktop session.

def capture_file_dialog():
    from mbo_utilities.gui.run_gui import run_gui
    from imgui_bundle import hello_imgui

    start = time.time()

    def pre_frame():
        if hello_imgui.get_runner_params().app_shall_exit:
            return
        if time.time() - start > 12.0:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Miller Brain Studio - Data Selection"
    params.app_window_params.window_geometry.size = (340, 620)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False
    params.fps_idling.enable_idling = False
    params.callbacks.pre_new_frame = pre_frame

    run_gui(select_only=True, runner_params=params)

    shot = hello_imgui.final_app_window_screenshot()
    if shot is not None and shot.size > 0:
        style_image(Image.fromarray(shot), OUTPUT_DIR / "01_step_file_dialog.png")
    else:
        print("  file dialog: empty screenshot buffer")


def capture_metadata_viewer(data_in: Path):
    from mbo_utilities.reader import imread
    from mbo_utilities.gui._metadata import draw_metadata_inspector
    from mbo_utilities.gui._setup import get_default_ini_path
    from imgui_bundle import hello_imgui, immapp

    arr = imread(data_in)
    metadata = arr.metadata
    if not metadata:
        print("  metadata viewer: no metadata, skipping")
        return

    start = time.time()

    def pre_frame():
        if hello_imgui.get_runner_params().app_shall_exit:
            return
        if time.time() - start > 3.0:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Miller Brain Studio - Metadata"
    params.app_window_params.window_geometry.size = (500, 700)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False
    params.ini_filename = get_default_ini_path("metadata_viewer")
    params.fps_idling.enable_idling = False
    params.callbacks.show_gui = lambda: draw_metadata_inspector(metadata, arr)
    params.callbacks.pre_new_frame = pre_frame

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    immapp.run(params, addons)

    shot = hello_imgui.final_app_window_screenshot()
    if shot is not None and shot.size > 0:
        style_image(Image.fromarray(shot), OUTPUT_DIR / "03_metadata_viewer.png")
    else:
        print("  metadata viewer: empty screenshot buffer")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_in = _resolve_data_path(sys.argv)

    # standalone imgui windows (need a desktop session)
    print("file dialog (01)...")
    try:
        capture_file_dialog()
    except Exception as e:
        print(f"  file dialog capture failed: {e}")

    if data_in is None:
        print(f"No data at {DEFAULT_DATA}; skipping data-driven captures.")
        sys.exit(0)

    print(f"data: {data_in}")
    captures = [
        ("data view (02)", capture_data_view),
        ("metadata viewer (03)", capture_metadata_viewer),
        ("metadata editor (04)", capture_metadata_editor),
        ("save as dialog (04b)", capture_save_as_dialog),
        ("save options (05)", capture_save_options),
        ("suite2p settings (06)", capture_suite2p_settings),
        ("suite2p parameters (07)", capture_suite2p_parameters),
        ("save options zarr (08)", capture_save_options_zarr),
        ("save options mp4 (09)", capture_save_options_mp4),
        ("keybinds (10)", capture_keybinds),
        ("process console (11)", capture_process_console),
        ("options (12)", capture_options),
        ("suite2p legend (13)", capture_suite2p_legend),
    ]
    for label, fn in captures:
        print(f"{label}...")
        try:
            fn(data_in)
        except Exception as e:
            print(f"  {label} capture failed: {e}")
