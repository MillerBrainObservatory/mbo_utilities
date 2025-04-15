import numpy as np
import matplotlib.pyplot as plt


def update_colocalization(shift_x=None, shift_y=None, image_a=None, image_b=None):
    from scipy.ndimage import shift
    image_b_shifted = shift(image_b, shift=(shift_y, shift_x), mode='nearest')
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
    plt.hist2d(x, y, bins=bins, cmap='inferno', density=True)
    plt.colorbar(label='Density')
    plt.xlabel('Max Projection 1 (Green)')
    plt.ylabel('Max Projection 2 (Red)')
    plt.title('2D Histogram of Colocalization')
    plt.show()