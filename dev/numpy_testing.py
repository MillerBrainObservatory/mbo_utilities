from pathlib import Path
import fastplotlib as fpl
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities.array_types import NumpyArray, MboRawArray
import fastplotlib as fpl
import time

if __name__ == "__main__":
    raw_data_path = Path(r"D:\demo\raw")
    lazy_array: MboRawArray = imread(raw_data_path)
    data = lazy_array[:, 0:5, :, :]

    np_lazy = NumpyArray(data, metadata=lazy_array.metadata)
    iw = fpl.ImageWidget(
        data=np_lazy,
        histogram_widget=True,
        figure_kwargs={"size": (800, 1000)},
        graphic_kwargs={"vmin": -400, "vmax": 4000},
    )
    iw.show()
    fpl.loop.run()
    x = 2