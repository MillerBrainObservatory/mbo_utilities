"""
Example Data Loader
===================

Use portable file dialogs to open a file.

"""

# test_example = false
# sphinx_gallery_pygfx_docs = 'screenshot'

from pathlib import Path
import mbo_utilities as mbo
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui, portable_file_dialogs as pfd

# Initial data path
data_path = r"D:/raw_scanimage_tiffs"
data = mbo.imread(data_path)[:2000, 0, :, :]
print(f"Initial data shape: {data.shape}")

# Create ImageWidget
iw = fpl.ImageWidget(
    [data],
    names=["My Custom Filetype"],
    slider_dim_names=("t", "z"),
)

iw.show()

fpl.loop.run()
