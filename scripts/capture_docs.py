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
    # params.app_window_params.ask_for_screenshot_at_exit = True # Not available in this binding
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

if __name__ == "__main__":
    ensure_dirs()
    
    # Capture 1: File Dialog
    try:
        capture_file_dialog()
    except Exception as e:
        print(f"File dialog capture failed: {e}")

    # Capture 2: Data View (if path exists)
    data_path = Path(r"C:\Users\flynn\mbo\data\raw")
    if data_path.exists():
        capture_data_view(data_path)
    else:
        print(f"Skipping data view capture: {data_path} not found")
