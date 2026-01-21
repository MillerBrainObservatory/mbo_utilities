#!/usr/bin/env python
"""
capture gui screenshots for documentation.

fully automatic - launches mbo gui, loads data, captures screenshots at each state.

usage:
    uv run scripts/capture_gui_screenshots.py --data /path/to/test.tiff

requires dev deps: uv add --dev pyautogui pygetwindow pillow
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "_images" / "gui" / "auto"


def get_window(title_contains: str = "MBO"):
    """find mbo window."""
    try:
        import pygetwindow as gw
        windows = gw.getWindowsWithTitle(title_contains)
        if windows:
            return windows[0]
    except Exception as e:
        print(f"error finding window: {e}")
    return None


def wait_for_window(title: str = "MBO", timeout: float = 30.0):
    """wait for window to appear and return it."""
    start = time.time()
    while time.time() - start < timeout:
        win = get_window(title)
        if win:
            return win
        time.sleep(0.3)
    return None


def capture_window(win, name: str, output_dir: Path = OUTPUT_DIR):
    """capture screenshot of window only."""
    import pyautogui

    if win is None:
        print(f"  skip: {name} (no window)")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # activate and bring to front
    try:
        win.activate()
        time.sleep(0.2)
    except Exception:
        pass

    # capture window region
    screenshot = pyautogui.screenshot(region=(win.left, win.top, win.width, win.height))

    filepath = output_dir / f"{name}.png"
    screenshot.save(filepath)
    print(f"  saved: {name}.png")
    return filepath


def send_key(key: str, delay: float = 0.3):
    """send a keypress."""
    import pyautogui
    pyautogui.press(key)
    time.sleep(delay)


def click_relative(win, x_offset: int, y_offset: int, delay: float = 0.3):
    """click at position relative to window top-left."""
    import pyautogui
    if win:
        pyautogui.click(win.left + x_offset, win.top + y_offset)
        time.sleep(delay)


def run_capture_sequence(data_path: str, output_dir: Path):
    """run full automated capture sequence."""
    print(f"data: {data_path}")
    print(f"output: {output_dir}")
    print()

    # launch gui with data
    cmd = [sys.executable, "-m", "mbo_utilities.cli", data_path]
    print(f"launching gui...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # wait for window to appear
        win = wait_for_window("MBO", timeout=30)
        if not win:
            print("error: window not found")
            proc.terminate()
            return

        # wait for data to load and render (watch for zstats to start)
        print("waiting for data to load...")
        time.sleep(5)  # initial load time

        # refresh window handle (size may have changed)
        win = get_window("MBO")

        # 01: main data view
        print("\ncapturing screenshots:")
        capture_window(win, "01_main_view", output_dir)

        # 02: save-as dialog
        time.sleep(0.5)
        send_key('s', delay=0.8)
        win = get_window("MBO")
        capture_window(win, "02_save_as_dialog", output_dir)

        # close save-as
        send_key('escape', delay=0.5)

        # 03: click Process tab (adjust x offset based on your tab layout)
        # tabs are typically near top of window, spaced ~80px apart
        win = get_window("MBO")
        if win:
            # try clicking "Process" tab - adjust these offsets for your layout
            # View=~60, Process=~140, Metadata=~220
            click_relative(win, 140, 35, delay=0.8)
            win = get_window("MBO")
            capture_window(win, "03_process_tab", output_dir)

        # 04: click Metadata tab
        win = get_window("MBO")
        if win:
            click_relative(win, 220, 35, delay=0.8)
            win = get_window("MBO")
            capture_window(win, "04_metadata_tab", output_dir)

        # 05: click back to View tab
        win = get_window("MBO")
        if win:
            click_relative(win, 60, 35, delay=0.8)
            win = get_window("MBO")
            capture_window(win, "05_view_tab", output_dir)

        print(f"\ndone! screenshots in: {output_dir}")

        # close gui
        proc.terminate()
        proc.wait(timeout=5)

    except KeyboardInterrupt:
        print("\ninterrupted")
        proc.terminate()
    except Exception as e:
        print(f"error: {e}")
        proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="capture mbo gui screenshots")
    parser.add_argument("--data", "-d", required=True, help="data file/folder to open")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--wait", "-w",
        type=float,
        default=5.0,
        help="seconds to wait for data to load (default: 5)"
    )
    args = parser.parse_args()

    # check deps
    try:
        import pyautogui
        import pygetwindow
        from PIL import Image
    except ImportError as e:
        print(f"missing: {e}")
        print("install: uv add --dev pyautogui pygetwindow pillow")
        sys.exit(1)

    if not Path(args.data).exists():
        print(f"error: data path not found: {args.data}")
        sys.exit(1)

    run_capture_sequence(args.data, args.output)


if __name__ == "__main__":
    main()
