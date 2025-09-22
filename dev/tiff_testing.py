from pathlib import Path
import time
import mbo_utilities as mbo

data_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
files = list(data_path.glob("*.tif*"))

first_file = files[0]
file_size_gb = first_file.stat().st_size / 1e9
print(f"First file: {first_file.name}, Size: {file_size_gb:.2f} GB")

x = mbo.imread(first_file)
x.phasecorr_method = "frame"

outpath = data_path.parent / "tiff_test_output_mean_roll"

start = time.time()
mbo.imwrite(x, outpath, planes=[2], ext=".tif")
end = time.time()
print(f"Time to write with mean phase correction: {end - start:.2f}")

# start = time.time()
# mbo.imwrite(x, outpath, planes=[2], ext=".bin")
# end = time.time()
# print(f"Time to write binary: {end - start:.2f}")
#
# start = time.time()
# mbo.imwrite(x, outpath, planes=[2], ext=".h5")
# end = time.time()
# print(f"Time to write h5: {end - start:.2f}")

x = 2
# mbo.imwrite(x, temp_outfile)