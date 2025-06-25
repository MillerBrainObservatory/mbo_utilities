# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "mbo_utilities",
#     "fastplotlib",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "dev" }
import tifffile
from pathlib import Path
import numpy as np

import fastplotlib as fpl
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities.file_io import MBO_SUPPORTED_FTYPES

if __name__ == "__main__":
    path = Path(r"D://demo//test")
    path.mkdir(exist_ok=True)
    data = imread(r"D://W2_DATA//kbarber//2025_03_01//mk301//green")
    # data = imread(r"D://demo//roi2//plane11.h5")
    data.roi = 2
    # for ftype in MBO_SUPPORTED_FTYPES:
    # for ftype in [".bin"]:
    for ftype in [".tif", ".h5", ".bin"]:
            imwrite(
            data,
            path,
            ext=ftype,
            overwrite=True,
            planes=[10],
        )
    files = [x for x in Path(path).glob("*")]
    check = imread(files[-1])
    fpl.ImageWidget(check,
                    names=[f.name for f in files],
                    histogram_widget=True,
                    figure_kwargs={"size": (800, 1000),},
                    graphic_kwargs={"vmin": -300, "vmax": 3000},
                    window_funcs={"t": (np.mean, 0)},
                   ).show()
    fpl.loop.run()

    # data = scan[:20, 11, :, :]
    # title = f"200 frames, {data.shape[0]} planes, plane {zplane}"
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(data.mean(axis=0)[200:280, 300:380], cmap="gray", vmin=-300, vmax=2500)
    # ax[1].imshow(data.mean(axis=0), cmap="gray", vmin=-300, vmax=2500)
    # plt.title(title)
    # plt.savefig("/tmp/01/both.png")
    # print(data.shape)