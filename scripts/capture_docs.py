"""
Script to capture and style screenshots for documentation.
Usage: uv run scripts/capture_docs.py
"""
import time
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import numpy as np

from mbo_utilities.gui.run_gui import run_gui
from imgui_bundle import hello_imgui

# Constants
OUTPUT_DIR = Path("docs/_images/gui/readme")
# Don't resize - keep original dimensions, just add shadow
PADDING = 30
SHADOW_BLUR = 12
SHADOW_OFFSET = (0, 8)
SHADOW_OPACITY = 0.25

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def style_image(img: Image.Image, output_path: Path):
    """Apply transparent padding and shadow to image and save."""
    print(f"Styling and saving -> {output_path}")

    # Convert to RGBA if needed
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Keep original size - no resizing

    # Add Shadow
    shadow_margin = SHADOW_BLUR * 2
    bg_width = img.width + (PADDING * 2)
    bg_height = img.height + (PADDING * 2)

    final_img = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (img.width + shadow_margin, img.height + shadow_margin), (0, 0, 0, 0))
    shadow_color = (0, 0, 0, int(255 * SHADOW_OPACITY))
    shadow_draw = Image.new("RGBA", (img.width, img.height), shadow_color)
    shadow.paste(shadow_draw, (shadow_margin//2, shadow_margin//2))
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))

    shadow_x = PADDING + SHADOW_OFFSET[0] - (shadow_margin//2)
    shadow_y = PADDING + SHADOW_OFFSET[1] - (shadow_margin//2)
    final_img.paste(shadow, (shadow_x, shadow_y), shadow)

    # 3. Paste Original Image
    final_img.paste(img, (PADDING, PADDING))

    # Save
    final_img.save(output_path, "PNG")

def capture_file_dialog():
    """Capture the file selection dialog."""
    print("Capturing File Dialog (01)...")

    start_time = time.time()
    def post_draw():
        if hello_imgui.get_runner_params().app_shall_exit:
            return

        # Wait 8 seconds to ensure installation checks complete and UI stabilizes
        if time.time() - start_time > 8.0:
            # Request exit - screenshot is taken at exit
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Utilities – Data Selection"
    params.app_window_params.window_geometry.size = (340, 620)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False

    # Critical flags for capture
    params.fps_idling.enable_idling = False

    params.callbacks.pre_new_frame = post_draw

    run_gui(select_only=True, runner_params=params)

    # Retrieve screenshot
    screenshot = hello_imgui.final_app_window_screenshot()

    if screenshot is not None and screenshot.size > 0:
        img = Image.fromarray(screenshot)
        style_image(img, OUTPUT_DIR / "01_step_file_dialog.png")
    else:
        print("Failed to capture screenshot (empty buffer)")


def capture_data_view(data_path: Path):
    """Capture the data view with loaded data."""
    print(f"Capturing Data View (02) for {data_path}...")

    from mbo_utilities.reader import imread
    from mbo_utilities.gui.run_gui import _create_image_widget
    from mbo_utilities.arrays import normalize_roi
    import fastplotlib as fpl

    # Load data
    try:
        data_array = imread(data_path, roi=normalize_roi(None))
    except Exception as e:
        print(f"Failed to load data from {data_path}: {e}")
        return

    # Create widget
    iw = _create_image_widget(data_array, widget=True)

    start_time = time.time()
    state = {"captured": False}

    def snapshot_callback():
        # Resize window to appropriate size for screenshot
        if not hasattr(snapshot_callback, "resized"):
            try:
                canvas = iw.figure.canvas
                if "QRenderCanvas" in str(type(canvas)):
                    win = canvas.window()
                    # Landscape mode - reasonable size for data viewer
                    win.resize(900, 600)
                snapshot_callback.resized = True
            except:
                pass

        if state["captured"]:
            iw.close()
            try:
                fpl.loop.close()
            except:
                pass
            return

        # Wait for 5 seconds to load/render
        if time.time() - start_time > 5.0:
            print("Taking snapshot via screen grab...")
            try:
                canvas = iw.figure.canvas

                # Check for PySide6/Qt canvas
                if "QRenderCanvas" in str(type(canvas)):
                    from PySide6.QtWidgets import QApplication
                    from PySide6.QtGui import QGuiApplication

                    window = canvas.window()
                    window.raise_()
                    window.activateWindow()
                    QApplication.instance().processEvents()

                    screen = window.screen()
                    if screen is None:
                        screen = QGuiApplication.primaryScreen()

                    pixmap = screen.grabWindow(window.winId())
                    qimg = pixmap.toImage()

                    # Convert QImage to PIL Image (Windows-compatible)
                    qimg = qimg.convertToFormat(qimg.Format.Format_RGBA8888)
                    width, height = qimg.width(), qimg.height()
                    byte_count = qimg.sizeInBytes()
                    ptr = qimg.bits()
                    # On Windows, use constBits or convert via numpy
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()
                    pil_img = Image.fromarray(arr, "RGBA")
                    style_image(pil_img, OUTPUT_DIR / "02_step_data_view.png")
                    state["captured"] = True

                    QApplication.instance().quit()
                    return

                # Fallback for non-Qt
                snapshot = iw.figure.renderer.snapshot()
                if snapshot is not None and snapshot.size > 0:
                    img = Image.fromarray(snapshot)
                    style_image(img, OUTPUT_DIR / "02_step_data_view.png")
                    state["captured"] = True
                else:
                    print("Snapshot was empty/None")
                    state["captured"] = True

                iw.close()
                try:
                    fpl.loop.close()
                except:
                    pass

            except Exception as e:
                print(f"Snapshot failed: {e}")
                state["captured"] = True # Exit anyway
                try:
                    from PySide6.QtWidgets import QApplication
                    QApplication.instance().quit()
                except:
                    pass

    iw.figure.add_animations(snapshot_callback)
    fpl.loop.run()


def capture_metadata_viewer(data_path: Path):
    """Capture the metadata viewer window."""
    print(f"Capturing Metadata Viewer (03) for {data_path}...")

    from mbo_utilities.reader import imread
    from mbo_utilities.gui._widgets import draw_metadata_inspector
    from mbo_utilities.gui._setup import get_default_ini_path
    from imgui_bundle import immapp

    # Load data to get metadata
    try:
        data_array = imread(data_path)
        metadata = data_array.metadata
        if not metadata:
            print("No metadata found, skipping metadata viewer capture")
            return
    except Exception as e:
        print(f"Failed to load data from {data_path}: {e}")
        return

    start_time = time.time()

    def gui_callback():
        draw_metadata_inspector(metadata)

    def post_draw():
        if hello_imgui.get_runner_params().app_shall_exit:
            return
        # Wait 3 seconds for UI to stabilize
        if time.time() - start_time > 3.0:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Metadata Viewer"
    params.app_window_params.window_geometry.size = (500, 700)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False
    params.fps_idling.enable_idling = False
    params.callbacks.show_gui = gui_callback
    params.callbacks.pre_new_frame = post_draw

    addons = immapp.AddOnsParams()
    addons.with_markdown = True

    immapp.run(params, addons)

    # Retrieve screenshot
    screenshot = hello_imgui.final_app_window_screenshot()

    if screenshot is not None and screenshot.size > 0:
        img = Image.fromarray(screenshot)
        style_image(img, OUTPUT_DIR / "03_metadata_viewer.png")
    else:
        print("Failed to capture metadata viewer screenshot (empty buffer)")


def capture_configurable_metadata(data_path: Path):
    """Capture a standalone window showing configurable metadata with explanation."""
    print(f"Capturing Configurable Metadata (04) for {data_path}...")

    from mbo_utilities.reader import imread
    from mbo_utilities.gui.feature_registry import get_feature
    from imgui_bundle import imgui, immapp

    # Load data to get required metadata fields
    try:
        data_array = imread(data_path)
    except Exception as e:
        print(f"Failed to load data from {data_path}: {e}")
        return

    # Get required metadata fields if available
    required_fields = []
    if hasattr(data_array, 'get_required_metadata'):
        required_fields = data_array.get_required_metadata()

    # Demo custom metadata entries
    custom_metadata = {
        "experiment_id": "exp_001",
        "subject": "mouse_42",
    }

    feature = get_feature("configurable_metadata")
    start_time = time.time()

    def gui_callback():
        imgui.set_next_window_pos(imgui.ImVec2(20, 20), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(460, 0), imgui.Cond_.once)

        if imgui.begin("Configurable Metadata", None, imgui.WindowFlags_.always_auto_resize):
            feature.draw_func(
                required_fields=required_fields,
                custom_metadata=custom_metadata,
                show_header=True,
                show_footer=True,
            )
        imgui.end()

    def post_draw():
        if hello_imgui.get_runner_params().app_shall_exit:
            return
        if time.time() - start_time > 2.5:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO - Configurable Metadata"
    params.app_window_params.window_geometry.size = (500, 520)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False
    params.fps_idling.enable_idling = False
    params.callbacks.show_gui = gui_callback
    params.callbacks.pre_new_frame = post_draw

    addons = immapp.AddOnsParams()
    addons.with_markdown = True

    immapp.run(params, addons)

    # Retrieve screenshot
    screenshot = hello_imgui.final_app_window_screenshot()

    if screenshot is not None and screenshot.size > 0:
        img = Image.fromarray(screenshot)
        style_image(img, OUTPUT_DIR / "04_configurable_metadata.png")
    else:
        print("Failed to capture configurable metadata screenshot (empty buffer)")


def capture_save_options(data_path: Path):
    """Capture a standalone window showing save-as options with explanations."""
    print(f"Capturing Save Options (05) for {data_path}...")

    from mbo_utilities.gui.feature_registry import get_feature
    from imgui_bundle import imgui, immapp

    feature = get_feature("save_options")
    start_time = time.time()
    state = {}  # Will use defaults from draw function

    def gui_callback():
        nonlocal state
        imgui.set_next_window_pos(imgui.ImVec2(20, 20), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(480, 0), imgui.Cond_.once)

        if imgui.begin("Save Options", None, imgui.WindowFlags_.always_auto_resize):
            state = feature.draw_func(state=state, show_header=True, show_footer=True)
        imgui.end()

    def post_draw():
        if hello_imgui.get_runner_params().app_shall_exit:
            return
        if time.time() - start_time > 2.5:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO - Save Options"
    params.app_window_params.window_geometry.size = (520, 520)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False
    params.fps_idling.enable_idling = False
    params.callbacks.show_gui = gui_callback
    params.callbacks.pre_new_frame = post_draw

    addons = immapp.AddOnsParams()
    addons.with_markdown = True

    immapp.run(params, addons)

    screenshot = hello_imgui.final_app_window_screenshot()

    if screenshot is not None and screenshot.size > 0:
        img = Image.fromarray(screenshot)
        style_image(img, OUTPUT_DIR / "05_save_options.png")
    else:
        print("Failed to capture save options screenshot (empty buffer)")


def capture_suite2p_settings():
    """Capture a standalone window showing suite2p pipeline settings with explanations."""
    print("Capturing Suite2p Settings (06)...")

    from imgui_bundle import imgui, immapp
    from mbo_utilities.gui.feature_registry import get_feature
    from mbo_utilities.gui.pipeline_widgets import Suite2pSettings

    feature = get_feature("suite2p_settings")
    start_time = time.time()
    settings = Suite2pSettings()

    def gui_callback():
        nonlocal settings
        imgui.set_next_window_pos(imgui.ImVec2(20, 20), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(500, 0), imgui.Cond_.once)

        if imgui.begin("Suite2p Settings", None, imgui.WindowFlags_.always_auto_resize):
            settings = feature.draw_func(settings=settings, show_header=True, show_footer=True)
        imgui.end()

    def post_draw():
        if hello_imgui.get_runner_params().app_shall_exit:
            return
        if time.time() - start_time > 2.5:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO - Suite2p Settings"
    params.app_window_params.window_geometry.size = (540, 620)
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False
    params.fps_idling.enable_idling = False
    params.callbacks.show_gui = gui_callback
    params.callbacks.pre_new_frame = post_draw

    addons = immapp.AddOnsParams()
    addons.with_markdown = True

    immapp.run(params, addons)

    screenshot = hello_imgui.final_app_window_screenshot()

    if screenshot is not None and screenshot.size > 0:
        img = Image.fromarray(screenshot)
        style_image(img, OUTPUT_DIR / "06_suite2p_settings.png")
    else:
        print("Failed to capture suite2p settings screenshot (empty buffer)")


def capture_save_as_dialog(data_path: Path):
    """Capture the save_as popup dialog from the data viewer."""
    print(f"Capturing Save As Dialog (07) for {data_path}...")

    from mbo_utilities.reader import imread
    from mbo_utilities.gui.run_gui import _create_image_widget
    from mbo_utilities.arrays import normalize_roi
    import fastplotlib as fpl

    # Load data
    try:
        data_array = imread(data_path, roi=normalize_roi(None))
    except Exception as e:
        print(f"Failed to load data from {data_path}: {e}")
        return

    # Create widget
    iw = _create_image_widget(data_array, widget=True)

    start_time = time.time()
    state = {"popup_opened": False, "captured": False, "open_attempts": 0}

    def snapshot_callback():
        # Resize window first
        if not hasattr(snapshot_callback, "resized"):
            try:
                canvas = iw.figure.canvas
                if "QRenderCanvas" in str(type(canvas)):
                    win = canvas.window()
                    win.resize(900, 650)
                snapshot_callback.resized = True
            except:
                pass

        if state["captured"]:
            iw.close()
            try:
                fpl.loop.close()
            except:
                pass
            return

        # After 2 seconds, start trying to open the save_as popup
        # Keep trying every frame until it sticks (imgui popup needs multiple frames)
        elapsed = time.time() - start_time
        if elapsed > 2.0 and elapsed < 4.0:
            try:
                # Access the PreviewDataWidget via figure.guis (it's a dict keyed by edge location)
                guis_dict = iw.figure.guis if hasattr(iw.figure, 'guis') else iw.figure._guis
                for edge, gui in guis_dict.items():
                    if gui is not None and hasattr(gui, '_saveas_popup_open'):
                        # Set the flag every frame to ensure popup opens
                        gui._saveas_popup_open = True
                        if state["open_attempts"] == 0:
                            print(f"Found gui at '{edge}': {type(gui).__name__}, setting _saveas_popup_open")
                        state["open_attempts"] += 1
                        state["popup_opened"] = True
                        break
                if not state["popup_opened"] and state["open_attempts"] == 0:
                    print(f"No gui with _saveas_popup_open found. guis: {guis_dict}")
            except Exception as e:
                if state["open_attempts"] == 0:
                    print(f"Failed to open save popup: {e}")

        # After 4.5 seconds, take screenshot (give popup time to render)
        if time.time() - start_time > 4.5 and not state["captured"]:
            print("Taking snapshot with save_as dialog...")
            try:
                canvas = iw.figure.canvas

                if "QRenderCanvas" in str(type(canvas)):
                    from PySide6.QtWidgets import QApplication
                    from PySide6.QtGui import QGuiApplication

                    window = canvas.window()
                    window.raise_()
                    window.activateWindow()
                    QApplication.instance().processEvents()

                    screen = window.screen()
                    if screen is None:
                        screen = QGuiApplication.primaryScreen()

                    pixmap = screen.grabWindow(window.winId())
                    qimg = pixmap.toImage()

                    qimg = qimg.convertToFormat(qimg.Format.Format_RGBA8888)
                    width, height = qimg.width(), qimg.height()
                    ptr = qimg.bits()
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()
                    pil_img = Image.fromarray(arr, "RGBA")
                    style_image(pil_img, OUTPUT_DIR / "07_save_as_dialog.png")
                    state["captured"] = True

                    QApplication.instance().quit()
                    return

            except Exception as e:
                print(f"Snapshot failed: {e}")
                state["captured"] = True
                try:
                    from PySide6.QtWidgets import QApplication
                    QApplication.instance().quit()
                except:
                    pass

    iw.figure.add_animations(snapshot_callback)
    fpl.loop.run()


if __name__ == "__main__":
    ensure_dirs()

    data_path = Path(r"C:\Users\flynn\mbo\data\raw")

    # Capture 1: File Dialog
    try:
        capture_file_dialog()
    except Exception as e:
        print(f"File dialog capture failed: {e}")

    # Capture 2: Data View (if path exists)
    if data_path.exists():
        try:
            capture_data_view(data_path)
        except Exception as e:
            print(f"Data view capture failed: {e}")
    else:
        print(f"Skipping data view capture: {data_path} not found")

    # Capture 3: Metadata Viewer
    if data_path.exists():
        try:
            capture_metadata_viewer(data_path)
        except Exception as e:
            print(f"Metadata viewer capture failed: {e}")
    else:
        print(f"Skipping metadata viewer capture: {data_path} not found")

    # Capture 4: Configurable Metadata
    if data_path.exists():
        try:
            capture_configurable_metadata(data_path)
        except Exception as e:
            print(f"Configurable metadata capture failed: {e}")
    else:
        print(f"Skipping configurable metadata capture: {data_path} not found")

    # Capture 5: Save Options
    if data_path.exists():
        try:
            capture_save_options(data_path)
        except Exception as e:
            print(f"Save options capture failed: {e}")
    else:
        print(f"Skipping save options capture: {data_path} not found")

    # Capture 6: Suite2p Settings
    try:
        capture_suite2p_settings()
    except Exception as e:
        print(f"Suite2p settings capture failed: {e}")

    # Capture 7: Save As Dialog (full popup)
    if data_path.exists():
        try:
            capture_save_as_dialog(data_path)
        except Exception as e:
            print(f"Save as dialog capture failed: {e}")
    else:
        print(f"Skipping save as dialog capture: {data_path} not found")
