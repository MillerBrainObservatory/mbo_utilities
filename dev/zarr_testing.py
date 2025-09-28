from pathlib import Path
import shutil
import numpy as np
import time
import mbo_utilities as mbo
import zarr
import tifffile
from zarr.codecs import BloscCodec
import fastplotlib as fpl
# from mbo_utilities.util import align_zplanes

# arr = mbo.imread(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_planar.zarr\plane01_stitched.zarr")
# fpl.ImageWidget(arr).show()
# fpl.loop.run()
#
# arr = mbo.imread(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_planar.zarr")
# fpl.ImageWidget(arr).show()
# fpl.loop.run()

if __name__ == "__main__":

    import zarr
    # import fastplotlib as fpl
    # x = mbo.imread(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
    # fpl.ImageWidget(x).show()
    # fpl.loop.run()
    #
    # z = zarr.open(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_planar\plane01_stitched.zarr", mode='r')

    data_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
    files = list(data_path.glob("*.tif*"))

    compressors = BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    x = mbo.imread(data_path)
    x.phasecorr_method = "mean"

    out_file = data_path.parent / "zarr"
    out_file.mkdir(exist_ok=True)

    x = mbo.imwrite(x, out_file / "data_planar_aligned", preprocess=True, ext=".zarr")


# group = zarr.create_group(store=out_file)
# array = group.create_array(
#     name="volume",
#     shape=x.shape,
#     dtype=x.dtype,
#     chunks=(x.shape[0], 1, x.shape[-2], x.shape[-1]),
#     compressor=compressors
# )
#
# out_file = data_path.parent / "data_planar_compressed.zarr"
# txt_log = out_file.parent / "planar_log_compressed.txt"
# with open(txt_log, "w") as f:
#     f.write(f"Original files:\n")
#     for file in files:
#         f.write(f"{file}\n")
#     f.write(f"\nShape: {x.shape}\n")
#     f.write(f"Dtype: {x.dtype}\n")
#     f.write(f"Spatial dimensions: {spatial_dims}\n")
#     f.write(f"Number of planes: {nz}\n")
#     for i in range(x.shape[1]):
#         start = time.time()
#         data = x[:, i]
#         end = time.time()
#         f.write(f"Time to read plane {i}: {end - start:.2f}\n")
#         start = time.time()
#         array[:, i, ...] = x[:, i]
#         end = time.time()
#         f.write(f"Time to write plane {i}: {end - start:.2f}\n")
#
# txt_log = out_file.parent / "volume_compressed_log.txt"
# with open(txt_log, "w") as f:
#     f.write(f"Original files:\n")
#     for file in files:
#         f.write(f"{file}\n")
#     f.write(f"\nShape: {x.shape}\n")
#     f.write(f"Dtype: {x.dtype}\n")
#     f.write(f"Spatial dimensions: {spatial_dims}\n")
#     f.write(f"Number of planes: {nz}\n")
#
#     group = zarr.create_group(store=out_file)
#
#     for i in range(x.shape[1]):
#         array = group.create_array(
#             name=f"plane{i + 1}",
#             shape=(x.shape[0], x.shape[-2], x.shape[-1]),
#             dtype=x.dtype,
#             chunks=(x.shape[0], x.shape[-2], x.shape[-1]),
#             compressor=compressors
#         )
#         start = time.time()
#         data = x[:, i]
#         end = time.time()
#         f.write(f"Time to read plane {i}: {end - start:.2f}\n")
#         start = time.time()
#         array[:] = data
#         end = time.time()
#         f.write(f"Time to write plane {i}: {end - start:.2f}\n")
