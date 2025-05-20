# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "mbo_utilities",
#     "lbm_suite2p_python",
#     "fastplotlib",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "master" }
# lbm_suite2p_python = { git = "https://github.com/MillerBrainObservatory/lbm_suite2p_python", branch = "master" }

import mbo_utilities as mbo
from mbo_utilities.file_io import mbo_home
import fastplotlib as fpl

from imgui_bundle import portable_file_dialogs as pfd

# def select_folder():
#     dialog = pfd.select_folder("Select a folder")
#     return dialog


mbo.run_gui("/home/flynn/lbm_data/raw", gui=True)
print("done")
# path = select_folder()
# print(path.result())

#
# raw_scan = mbo.read_scan(r"/home/flynn/lbm_data/raw")
# data = raw_scan
#
# # main_window = LBMMainWindow()
# iw = fpl.ImageWidget(
#     data=data,
#     histogram_widget=True,
#     figure_kwargs={"size": (data.shape[-1], data.shape[-2])}
# )
#
# # start the widget playing in a loop
# iw._image_widget_sliders._loop = True  # noqa
# iw.show()  # need to display before playing
#
# fpl.loop.run()