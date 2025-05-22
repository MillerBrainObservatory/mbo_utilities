from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
            val = projection[
                  y: y + patch_size, x: x + patch_size
                  ].sum()
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
