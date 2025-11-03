"""Test utilities for graphics module testing."""

import os
from pathlib import Path
from typing import Callable

import numpy as np


def get_screenshot_dir() -> Path:
    """Get the screenshots directory."""
    return Path(__file__).parent / "screenshots"


def ensure_screenshot_dir() -> Path:
    """Ensure the screenshots directory exists."""
    screenshot_dir = get_screenshot_dir()
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    return screenshot_dir


def should_regenerate_screenshots() -> bool:
    """Check if we should regenerate screenshots."""
    return os.environ.get("REGENERATE_SCREENSHOTS", "0") == "1"


def get_wgpu_backend():
    """Get the WGPU backend being used."""
    try:
        import wgpu

        return wgpu.gpu.request_adapter_sync().request_device_sync()._backend_type
    except Exception:
        return "unknown"


def create_test_data_single_zplane(shape=(100, 512, 512)) -> np.ndarray:
    """
    Create synthetic test data for single z-plane.

    Parameters
    ----------
    shape : tuple
        Shape of the data (T, Y, X)

    Returns
    -------
    np.ndarray
        Synthetic test data
    """
    t, y, x = shape
    data = np.zeros(shape, dtype=np.int16)

    # Create some gaussian blobs to simulate cells
    rng = np.random.RandomState(42)
    n_cells = 10

    for i in range(n_cells):
        cy = rng.randint(y // 4, 3 * y // 4)
        cx = rng.randint(x // 4, 3 * x // 4)
        sigma = rng.randint(3, 8)

        yy, xx = np.ogrid[:y, :x]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= sigma**2

        # Add temporal dynamics
        baseline = rng.randint(500, 1000)
        amplitude = rng.randint(200, 500)
        for t_idx in range(t):
            signal = baseline + amplitude * np.sin(2 * np.pi * t_idx / 20 + i)
            data[t_idx][mask] += signal

    # Add noise
    noise = rng.normal(0, 50, shape).astype(np.int16)
    data = data + noise

    return data.clip(0, 4095)


def create_test_data_multi_zplane(shape=(100, 5, 512, 512)) -> np.ndarray:
    """
    Create synthetic test data for multi z-plane.

    Parameters
    ----------
    shape : tuple
        Shape of the data (T, Z, Y, X)

    Returns
    -------
    np.ndarray
        Synthetic test data
    """
    t, z, y, x = shape
    data = np.zeros(shape, dtype=np.int16)

    rng = np.random.RandomState(42)
    n_cells = 10

    for i in range(n_cells):
        cy = rng.randint(y // 4, 3 * y // 4)
        cx = rng.randint(x // 4, 3 * x // 4)
        cz = rng.randint(0, z)
        sigma = rng.randint(3, 8)

        yy, xx = np.ogrid[:y, :x]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= sigma**2

        # Add z-plane variation
        for z_idx in range(z):
            z_intensity = np.exp(-((z_idx - cz) ** 2) / (2 * 1.5**2))

            baseline = rng.randint(500, 1000)
            amplitude = rng.randint(200, 500)

            for t_idx in range(t):
                signal = (
                    baseline + amplitude * np.sin(2 * np.pi * t_idx / 20 + i)
                ) * z_intensity
                data[t_idx, z_idx][mask] += signal

    # Add noise
    noise = rng.normal(0, 50, shape).astype(np.int16)
    data = data + noise

    return data.clip(0, 4095)


def run_gui_with_screenshot(
    data: np.ndarray,
    screenshot_name: str,
    gui_action: Callable | None = None,
    frames: int = 5,
) -> None:
    """
    Run GUI and capture screenshot.

    Parameters
    ----------
    data : np.ndarray
        Data to display
    screenshot_name : str
        Name for the screenshot file
    gui_action : Callable, optional
        Function to call for GUI interactions before screenshot
    frames : int
        Number of frames to render before taking screenshot
    """
    import fastplotlib as fpl

    from mbo_utilities.graphics.imgui import PreviewDataWidget

    screenshot_dir = ensure_screenshot_dir()
    screenshot_path = screenshot_dir / screenshot_name

    # Create ImageWidget
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={"size": (800, 600)},
        graphic_kwargs={"vmin": 0, "vmax": 4000},
        window_funcs={"t": (np.mean, 0)},
    )

    # Add PreviewDataWidget
    gui = PreviewDataWidget(
        iw=iw,
        fpath=None,
        size=300,
        threading_enabled=False,  # Disable threading for deterministic tests
    )
    iw.figure.add_gui(gui)

    # Run GUI action if provided
    if gui_action is not None:
        gui_action(gui, iw)

    # Render a few frames to ensure GUI is ready
    for _ in range(frames):
        iw.canvas.update()
        iw.canvas.draw()

    # Capture screenshot
    img = iw.canvas.snapshot()

    # Save screenshot
    if should_regenerate_screenshots() or not screenshot_path.exists():
        import imageio

        imageio.imwrite(screenshot_path, img)

    # Close the widget
    iw.close()
