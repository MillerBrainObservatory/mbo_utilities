# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "mbo_utilities",
#     "fastplotlib",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "master" }

import mbo_utilities as mbo
from pathlib import Path
import tifffile

import mbo_utilities.plot_util

save_path = Path().home().joinpath("dev")

files = mbo.get_files(save_path, "tif", 4)
data = tifffile.imread(files[0])

fps = 17
duration_s = 5

# For each call:
mbo_utilities.plot_util.save_mp4(
    save_path.joinpath("default.mp4"),
    data[: duration_s * fps],  # no speedup
    framerate=fps,
)

mbo_utilities.plot_util.save_mp4(
    save_path.joinpath("speedup_2x.mp4"),
    data[: duration_s * fps * 2],
    framerate=fps,
    speedup=2,
)

mbo_utilities.plot_util.save_mp4(
    save_path.joinpath("17_frame_window.mp4"),
    data[: duration_s * fps],
    framerate=fps,
    win=17,
)

mbo_utilities.plot_util.save_mp4(
    save_path.joinpath("windowed_speedup_6x.mp4"),
    data[: duration_s * fps * 6],
    framerate=fps,
    speedup=6,
    win=17,
)

x = 2
