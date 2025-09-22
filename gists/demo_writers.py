# /// script
# requires-python = ">=3.12.7, <3.12.10"
# dependencies = [
#     "mbo_utilities",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "dev" }

from pathlib import Path
from mbo_utilities.lazy_array import imread, imwrite

raw = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green")
data = imread(raw)
data.roi = 2
data.fix_phase = True
imwrite(
    data,
    raw.parent.joinpath("planes"),
    ext=".h5",
    overwrite=True,
    planes=[4, 10]
)

# files = [x for x in Path(path).glob("*")]
# check = imread(files[0])
# fpl.ImageWidget(check,
#                 names=[f.name for f in files],
#                 histogram_widget=True,
#                 figure_kwargs={"size": (800, 1000),},
#                 graphic_kwargs={"vmin": -300, "vmax": 3000},
#                 window_funcs={"t": (np.mean, 0)},
#                ).show()
# fpl.loop.run()

# data = scan[:20, 11, :, :]
# title = f"200 frames, {data.shape[0]} planes, plane {zplane}"
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(data.mean(axis=0)[200:280, 300:380], cmap="gray", vmin=-300, vmax=2500)
# ax[1].imshow(data.mean(axis=0), cmap="gray", vmin=-300, vmax=2500)
# plt.title(title)
# plt.savefig("/tmp/01/both.png")
# print(data.shape)