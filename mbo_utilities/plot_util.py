from __future__ import annotations

from pathlib import Path

import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from icecream import ic
from matplotlib import cm

from mbo_utilities.util import norm_minmax


def save_phase_images_png(
    before: np.ndarray,
    data_chunk: np.ndarray,
    save_path: str | Path,
    chan_id: int,
):
    after = data_chunk
    mid = len(before) // 2
    projection = before.std(axis=0)  # or max, or mean

    patch_size = 64
    max_val = -np.inf
    best_x = best_y = 0

    for y in range(0, projection.shape[0] - patch_size + 1, 8):
        for x in range(0, projection.shape[1] - patch_size + 1, 8):
            val = projection[y : y + patch_size, x : x + patch_size].sum()
            if val > max_val:
                max_val = val
                best_y, best_x = y, x

    ys = slice(best_y, best_y + patch_size)
    xs = slice(best_x, best_x + patch_size)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(before[mid, ys, xs], cmap="gray")
    axs[0].set_title("Before")
    axs[1].imshow(after[mid, ys, xs], cmap="gray")
    axs[1].set_title("After")
    fig.tight_layout()
    fig.savefig(save_path / f"chunk_{chan_id:03d}.png")
    plt.close(fig)


def update_colocalization(shift_x=None, shift_y=None, image_a=None, image_b=None):
    from scipy.ndimage import shift

    image_b_shifted = shift(image_b, shift=(shift_y, shift_x), mode="nearest")
    image_a = image_a / np.max(image_a)
    image_b_shifted = image_b_shifted / np.max(image_b_shifted)
    shape = image_a.shape
    colocalization = np.zeros((*shape, 3))
    colocalization[..., 1] = image_a
    colocalization[..., 0] = image_b_shifted
    mask = (image_a > 0.3) & (image_b_shifted > 0.3)
    colocalization[..., 2] = np.where(mask, np.minimum(image_a, image_b_shifted), 0)
    return colocalization


def plot_colocalization_hist(max_proj1, max_proj2_shifted, bins=100):
    x = max_proj1.flatten()
    y = max_proj2_shifted.flatten()
    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=bins, cmap="inferno", density=True)
    plt.colorbar(label="Density")
    plt.xlabel("Max Projection 1 (Green)")
    plt.ylabel("Max Projection 2 (Red)")
    plt.title("2D Histogram of Colocalization")
    plt.show()


def save_png(fname, data):
    """
    Saves a given image array as a PNG file using Matplotlib.

    Parameters
    ----------
    fname : str or Path
        The file name (or full path) where the PNG image will be saved.
    data : array-like
        The image data to be visualized and saved. Can be any 2D or 3D array that Matplotlib can display.

    Examples
    --------
    >>> import mbo_utilities as mbo
    >>> import tifffile
    >>> data = tifffile.memmap("path/to/plane_0.tiff")
    >>> frame = data[0, ...]
    >>> mbo.save_png("plane_0_frame_1.png", frame)
    """
    # TODO: move this to a separate module that imports matplotlib

    plt.imshow(data)
    plt.axis("tight")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    ic(f"Saved data to {fname}")


def save_mp4(
    fname: str | Path | np.ndarray,
    images,
    framerate=60,
    speedup=1,
    chunk_size=100,
    cmap="gray",
    win=7,
    vcodec="libx264",
    normalize=True,
):
    """
    Save a video from a 3D array or TIFF stack to `.mp4`.

    Parameters
    ----------
    fname : str
        Output video file name.
    images : numpy.ndarray or str
        Input 3D array (T x H x W) or a file path to a TIFF stack.
    framerate : int, optional
        Original framerate of the video, by default 60.
    speedup : int, optional
        Factor to increase the playback speed, by default 1 (no speedup).
    chunk_size : int, optional
        Number of frames to process and write in a single chunk, by default 100.
    cmap : str, optional
        Colormap to apply to the video frames, by default "gray".
        Must be a valid Matplotlib colormap name.
    win : int, optional
        Temporal averaging window size. If `win > 1`, frames are averaged over
        the specified window using convolution. By default, 7.
    vcodec : str, optional
        Video codec to use, by default 'libx264'.
    normalize : bool, optional
        Flag to min-max normalize the video frames, by default True.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist when `images` is provided as a file path.
    ValueError
        If `images` is not a valid 3D NumPy array or a file path to a TIFF stack.

    Notes
    -----
    - The input array `images` must have the shape (T, H, W), where T is the number of frames,
      H is the height, and W is the width.
    - The `win` parameter performs temporal smoothing by averaging over adjacent frames.

    Examples
    --------
    Save a video from a 3D NumPy array with a gray colormap and 2x speedup:

    >>> import numpy as np
    >>> images = np.random.rand(100, 600, 576) * 255
    >>> save_mp4('output.mp4', images, framerate=17, cmap='gray', speedup=2)

    Save a video with temporal averaging applied over a 5-frame window at 4x speed:

    >>> save_mp4('output_smoothed.mp4', images, framerate=30, speedup=4, cmap='gray', win=5)

    Save a video from a TIFF stack:

    >>> save_mp4('output.mp4', 'path/to/stack.tiff', framerate=60, cmap='gray')
    """
    if not isinstance(fname, (str, Path)):
        raise TypeError(f"Expected fname to be str or Path, got {type(fname)}")
    if isinstance(images, (str, Path)):
        ic(f"Loading TIFF stack from {images}")
        if Path(images).is_file():
            try:
                images = tifffile.memmap(images)
            except MemoryError:
                images = tifffile.imread(images)
        else:
            raise FileNotFoundError(
                f"Images given as a string or path, but not a valid file: {images}"
            )
    elif not isinstance(images, np.ndarray):
        raise ValueError(
            f"Expected images to be a numpy array or a file path, got {type(images)}"
        )

    T, height, width = images.shape
    colormap = cm.get_cmap(cmap)

    if normalize:
        ic("Normalizing mp4 images to [0, 1]")
        images = norm_minmax(images)

    if win and win > 1:
        ic(f"Applying temporal averaging with window size {win}")
        kernel = np.ones(win) / win
        images = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=images
        )

    ic(f"Saving {T} frames to {fname}")
    output_framerate = int(framerate * speedup)
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            framerate=output_framerate,
        )
        .output(str(fname), pix_fmt="yuv420p", vcodec=vcodec, r=output_framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = images[start:end]
        colored_chunk = (colormap(chunk)[:, :, :, :3] * 255).astype(np.uint8)
        for frame in colored_chunk:
            process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()
    ic(f"Video saved to {fname}")
