"""
GUI automation and demo recording for MBO utilities.

This module provides tools to programmatically interact with the MBO GUI
for creating demo videos with real mouse cursor movement.

Example usage:
    # Run demo with automatic screen recording
    uv run python -m mbo_utilities.graphics.demo_automation "D:/data.zarr" --record

    # Run demo without recording
    uv run python -m mbo_utilities.graphics.demo_automation "D:/data.zarr"
"""

from pathlib import Path
from typing import Optional, Union, Callable
import time
import subprocess
import sys
import threading


class ScreenRecorder:
    """Screen recorder using ffmpeg with Windows-compatible encoding."""

    def __init__(self, output_path: str = "demo.mp4", fps: int = 30):
        self.output_path = output_path
        self.fps = fps
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        """Start screen recording."""
        if sys.platform == "win32":
            # Windows: use gdigrab with H.264 baseline profile for compatibility
            cmd = [
                "ffmpeg", "-y",
                "-f", "gdigrab",
                "-framerate", str(self.fps),
                "-i", "desktop",
                "-c:v", "libx264",
                "-profile:v", "baseline",  # Maximum compatibility
                "-level", "3.0",
                "-pix_fmt", "yuv420p",  # Required for Windows Photos
                "-preset", "medium",
                "-crf", "20",
                "-movflags", "+faststart",  # Web/streaming friendly
                self.output_path
            ]
        elif sys.platform == "darwin":
            # macOS: use avfoundation
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation",
                "-framerate", str(self.fps),
                "-i", "1:",
                "-c:v", "libx264",
                "-profile:v", "baseline",
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "20",
                "-movflags", "+faststart",
                self.output_path
            ]
        else:
            # Linux: use x11grab
            cmd = [
                "ffmpeg", "-y",
                "-f", "x11grab",
                "-framerate", str(self.fps),
                "-i", ":0.0",
                "-c:v", "libx264",
                "-profile:v", "baseline",
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "20",
                "-movflags", "+faststart",
                self.output_path
            ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Screen recording started: {self.output_path}")
        except FileNotFoundError:
            print("WARNING: ffmpeg not found. Install ffmpeg for screen recording.")
            self.process = None

    def stop(self):
        """Stop screen recording."""
        if self.process:
            # Send 'q' to ffmpeg to quit gracefully
            try:
                self.process.stdin.write(b'q')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
            print(f"Screen recording saved: {self.output_path}")
            self.process = None


class MouseAutomation:
    """
    Mouse automation using pyautogui for real cursor movement.

    This creates visible mouse movements that appear natural in recordings.
    """

    def __init__(self):
        try:
            import pyautogui
            self.pyautogui = pyautogui
            # Safety settings
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            pyautogui.PAUSE = 0.02  # Small pause between actions
        except ImportError:
            raise ImportError("pyautogui required: uv add pyautogui --group dev")

    def move_to(self, x: int, y: int, duration: float = 0.5):
        """Move mouse to absolute position with smooth animation."""
        self.pyautogui.moveTo(x, y, duration=duration, tween=self.pyautogui.easeInOutQuad)

    def drag_to(self, x: int, y: int, duration: float = 0.5, button: str = 'left'):
        """Drag from current position to target."""
        self.pyautogui.drag(
            x - self.pyautogui.position()[0],
            y - self.pyautogui.position()[1],
            duration=duration,
            button=button,
            tween=self.pyautogui.easeInOutQuad
        )

    def drag_slider(self, start_x: int, start_y: int, end_x: int, duration: float = 2.5):
        """Drag a horizontal slider from start to end position."""
        self.move_to(start_x, start_y, duration=0.3)
        time.sleep(0.1)
        self.pyautogui.mouseDown()
        time.sleep(0.05)
        self.pyautogui.moveTo(end_x, start_y, duration=duration, tween=self.pyautogui.linear)
        self.pyautogui.mouseUp()

    def click(self, x: int = None, y: int = None):
        """Click at position (or current position if not specified)."""
        if x is not None and y is not None:
            self.move_to(x, y, duration=0.2)
        self.pyautogui.click()

    def get_position(self):
        """Get current mouse position."""
        return self.pyautogui.position()


class DemoController:
    """
    Controller for demo automation with real mouse movement.

    Uses pyautogui to move the actual mouse cursor, creating
    natural-looking demonstrations.
    """

    def __init__(self, window_rect: tuple = None):
        """
        Initialize demo controller.

        Parameters
        ----------
        window_rect : tuple, optional
            (x, y, width, height) of the target window.
            If None, will try to find it automatically.
        """
        self.mouse = MouseAutomation()
        self.window_rect = window_rect
        self.steps: list[tuple[float, Callable, str]] = []
        self.start_time: float = 0
        self.step_idx: int = 0
        self.running: bool = False
        self.on_complete: Optional[Callable] = None
        self._stop_event = threading.Event()

    def set_window_rect(self, x: int, y: int, width: int, height: int):
        """Set the window rectangle for coordinate calculations."""
        self.window_rect = (x, y, width, height)

    def add_step(self, delay: float, action: Callable, description: str = ""):
        """Add a step to execute after delay seconds from start."""
        self.steps.append((delay, action, description))
        return self

    def start(self):
        """Start the demo sequence in a background thread."""
        self._stop_event.clear()
        thread = threading.Thread(target=self._run_sequence, daemon=True)
        thread.start()

    def stop(self):
        """Stop the demo sequence."""
        self._stop_event.set()
        self.running = False

    def _run_sequence(self):
        """Run the demo sequence (called in background thread)."""
        self.start_time = time.time()
        self.step_idx = 0
        self.running = True

        print(f"\n{'='*50}")
        print("DEMO STARTING")
        print(f"{'='*50}\n")

        while self.step_idx < len(self.steps) and not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            target_time, action, description = self.steps[self.step_idx]

            if elapsed >= target_time:
                if description:
                    print(f"[{elapsed:.1f}s] {description}")
                try:
                    action()
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                self.step_idx += 1
            else:
                time.sleep(0.016)  # ~60fps check rate

        self.running = False
        print(f"\n{'='*50}")
        print("DEMO COMPLETE")
        print(f"{'='*50}\n")

        if self.on_complete:
            self.on_complete()


def find_window_by_title(title_substring: str) -> tuple:
    """
    Find window position by title substring.

    Returns (x, y, width, height) or None if not found.
    """
    try:
        import pygetwindow as gw
        windows = gw.getWindowsWithTitle(title_substring)
        if windows:
            win = windows[0]
            return (win.left, win.top, win.width, win.height)
    except Exception:
        pass
    return None


def create_mouse_demo(
    window_rect: tuple,
    slider_duration: float = 2.5,
    close_after: bool = False,
) -> DemoController:
    """
    Create a demo with real mouse movements.

    Parameters
    ----------
    window_rect : tuple
        (x, y, width, height) of the target window
    slider_duration : float
        Time in seconds to drag each slider across
    close_after : bool
        Whether to close the window after demo

    Returns
    -------
    DemoController
        Controller ready to start
    """
    demo = DemoController(window_rect)
    mouse = demo.mouse

    win_x, win_y, win_w, win_h = window_rect

    # Calculate approximate widget positions
    # The GUI panel is on the right side, typically 350px wide
    panel_width = 350
    panel_x = win_x + win_w - panel_width

    # Image area is on the left
    image_center_x = win_x + (win_w - panel_width) // 2
    image_center_y = win_y + win_h // 2

    # Slider approximate positions (within the panel)
    # These are rough estimates - may need adjustment
    slider_left = panel_x + 60
    slider_right = panel_x + panel_width - 20

    # Time slider is usually near the bottom of sliders section
    time_slider_y = win_y + 120
    z_slider_y = win_y + 160

    # Checkbox area (lower in the panel)
    checkbox_x = panel_x + 30
    checkbox_start_y = win_y + 250

    t = 0.0

    # Initial pause
    demo.add_step(t, lambda: None, "Initializing demo...")
    t += 1.5

    # Move to image center first
    demo.add_step(t, lambda: mouse.move_to(image_center_x, image_center_y, 0.5),
                  "Moving to image view")
    t += 1.0

    # Drag the time slider
    demo.add_step(t, lambda: None, "Dragging time slider...")
    t += 0.3
    demo.add_step(t, lambda: mouse.drag_slider(slider_left, time_slider_y, slider_right, slider_duration),
                  f"  Time slider: left to right ({slider_duration}s)")
    t += slider_duration + 0.5

    # Drag back
    demo.add_step(t, lambda: mouse.drag_slider(slider_right, time_slider_y, slider_left, slider_duration),
                  f"  Time slider: right to left ({slider_duration}s)")
    t += slider_duration + 0.5

    # Drag the z slider (if 4D data)
    demo.add_step(t, lambda: None, "Dragging z-plane slider...")
    t += 0.3
    demo.add_step(t, lambda: mouse.drag_slider(slider_left, z_slider_y, slider_right, slider_duration),
                  f"  Z slider: left to right ({slider_duration}s)")
    t += slider_duration + 0.5

    demo.add_step(t, lambda: mouse.drag_slider(slider_right, z_slider_y, slider_left, slider_duration),
                  f"  Z slider: right to left ({slider_duration}s)")
    t += slider_duration + 0.5

    # Click checkboxes
    demo.add_step(t, lambda: None, "Clicking checkboxes...")
    t += 0.3

    # Fix Phase checkbox (first checkbox)
    demo.add_step(t, lambda: mouse.click(checkbox_x, checkbox_start_y),
                  "  Toggle Fix Phase")
    t += 1.0

    # Mean Subtraction checkbox (second checkbox)
    demo.add_step(t, lambda: mouse.click(checkbox_x, checkbox_start_y + 25),
                  "  Toggle Mean Subtraction")
    t += 1.5

    # Toggle back
    demo.add_step(t, lambda: mouse.click(checkbox_x, checkbox_start_y + 25),
                  "  Toggle Mean Subtraction off")
    t += 1.0

    demo.add_step(t, lambda: mouse.click(checkbox_x, checkbox_start_y),
                  "  Toggle Fix Phase off")
    t += 1.0

    # Final slider sweep
    demo.add_step(t, lambda: None, "Final time sweep...")
    t += 0.3
    demo.add_step(t, lambda: mouse.drag_slider(slider_left, time_slider_y, slider_right, slider_duration * 0.6),
                  "  Quick time sweep")
    t += slider_duration * 0.6 + 0.5

    # Move mouse away
    demo.add_step(t, lambda: mouse.move_to(image_center_x, image_center_y, 0.3),
                  "Demo finished!")
    t += 1.0

    if close_after:
        def close_window():
            try:
                import pygetwindow as gw
                windows = gw.getWindowsWithTitle("fastplotlib")
                if windows:
                    windows[0].close()
            except Exception:
                pass
        demo.add_step(t, close_window, "Closing window")

    return demo


def run_demo(
    data_path: Union[str, Path],
    record: bool = False,
    output_video: str = "mbo_demo.mp4",
    close_after: bool = False,
    slider_duration: float = 2.5,
):
    """
    Run the GUI with mouse automation demo.

    Parameters
    ----------
    data_path : str or Path
        Path to data file/folder to open
    record : bool
        Whether to record the screen to a video file
    output_video : str
        Output video filename (requires ffmpeg)
    close_after : bool
        Whether to close the window after demo completes
    slider_duration : float
        Time in seconds to drag each slider across (default 2.5)
    """
    import os

    # Set up Qt backend
    if sys.platform != "emscripten":
        try:
            import importlib.util
            if importlib.util.find_spec("PySide6") is not None:
                os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
                import PySide6  # noqa
        except ImportError:
            pass

    from mbo_utilities.lazy_array import imread
    from mbo_utilities.graphics.imgui import PreviewDataWidget
    import fastplotlib as fpl
    import numpy as np

    # Load data
    print(f"Loading data from: {data_path}")
    data_array = imread(data_path)
    print(f"Data shape: {data_array.shape}, dtype: {data_array.dtype}")

    # Create ImageWidget
    ndim = data_array.ndim
    if ndim == 4:
        slider_dim_names = ("t", "z")
        window_funcs = (np.mean, None)
        window_sizes = (1, None)
    elif ndim == 3:
        slider_dim_names = ("t",)
        window_funcs = (np.mean,)
        window_sizes = (1,)
    else:
        slider_dim_names = None
        window_funcs = None
        window_sizes = None

    iw = fpl.ImageWidget(
        data=data_array,
        slider_dim_names=slider_dim_names,
        window_funcs=window_funcs,
        window_sizes=window_sizes,
        histogram_widget=True,
        figure_kwargs={"size": (1200, 800)},
        graphic_kwargs={"vmin": -100, "vmax": 4000},
    )

    # Create PreviewDataWidget
    gui = PreviewDataWidget(
        iw=iw,
        fpath=str(data_path),
        size=350,
    )

    # Add GUI and show
    iw.figure.add_gui(gui)
    iw.show()

    # Wait for window to appear and get its position
    def start_automation():
        time.sleep(2.0)  # Wait for window to fully open

        # Find the window
        window_rect = find_window_by_title("fastplotlib")
        if not window_rect:
            # Fallback: assume window is at a default position
            print("WARNING: Could not find window, using default position")
            window_rect = (100, 100, 1200, 800)
        else:
            print(f"Found window at: {window_rect}")

        # Set up recorder
        recorder = None
        if record:
            recorder = ScreenRecorder(output_video)
            recorder.start()
            time.sleep(0.5)  # Let recording stabilize

        # Create and run demo
        demo = create_mouse_demo(
            window_rect=window_rect,
            slider_duration=slider_duration,
            close_after=close_after,
        )

        def on_complete():
            if recorder:
                time.sleep(0.5)
                recorder.stop()
            if close_after:
                time.sleep(0.5)
                fpl.loop.stop()

        demo.on_complete = on_complete
        demo.start()

    # Start automation in background
    threading.Thread(target=start_automation, daemon=True).start()

    # Run the event loop
    fpl.loop.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MBO GUI Demo Automation with Mouse Movement")
    parser.add_argument("data_path", help="Path to data file or folder")
    parser.add_argument("--record", action="store_true", help="Record screen to video")
    parser.add_argument("--output", "-o", default="mbo_demo.mp4", help="Output video filename")
    parser.add_argument("--close", action="store_true", help="Close window after demo")
    parser.add_argument("--slider-time", type=float, default=2.5,
                        help="Duration in seconds to drag each slider (default: 2.5)")

    args = parser.parse_args()

    run_demo(
        data_path=args.data_path,
        record=args.record,
        output_video=args.output,
        close_after=args.close,
        slider_duration=args.slider_time,
    )
