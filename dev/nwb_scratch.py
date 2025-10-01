import mbo_utilities as mbo
import zarr
from pathlib import Path
from pynwb import read_nwb
import fastplotlib as fpl
import numpy as np

# my_nwb = read_nwb(r"D:\W2_DATA\kbarber\07_27_2025\mk355\0727_mk355_testing.zarr")
# data = my_nwb.acquisition["TwoPhotonSeries"].data
# chunks = data.chunks
# reshaped = np.moveaxis(data, -1, 1)

uri_rechunked = r"s3://dandiarchive/zarr/a1f93cbf-fb83-4a02-a02b-e7b92d59fea0/"
uri_original = r"s3://dandiarchive/zarr/2f0c6b0f-dc77-4f99-8291-40182bb7c33d/"

z = zarr.open(
    store=uri_rechunked,
    storage_options=dict(anon=True),
    mode="r",
)
z_old = zarr.open(
    store=uri_original,
    storage_options=dict(anon=True),
    mode="r",
)

full_file = Path(
    r"D:\W2_DATA\kbarber\07_27_2025\mk355\green\mk355_7_27_2025_180mw_right_m2_go_to_2x-mROI-880x1100um_220x550px_2um-px_14p00Hz_00001_00001_00001.tif"
)

metadata = mbo.get_metadata(
    full_file,
    verbose=True,
)

data_mbo = mbo.imread(full_file)
# data_rechunked = z["acquisition"]["TwoPhotonSeries"].data
data = z_old["acquisition"]["TwoPhotonSeries"].data

iw = fpl.ImageWidget(data_mbo)
iw.show()
fpl.loop.run()

iw = fpl.ImageWidget(data)
iw.show()
fpl.loop.run()
