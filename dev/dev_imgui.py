from pathlib import Path
import mbo_utilities as mbo
from mbo_utilities.widgets import select_files
import fastplotlib as fpl

# data_path_a = select_files()
data_path_a = r"\\rbo-w1\W1_E_USER_DATA\kbarber\11_04_2025_green_full\plane_1"
tiffs = [x for x in Path(data_path_a).glob("*.tif")]

data = mbo.imread(tiffs)

iw = fpl.ImageWidget(
    [data],
    names=["No Plane Alignment"],
)

iw.show()
fpl.loop.run()
