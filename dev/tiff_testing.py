from pathlib import Path
import time
import mbo_utilities as mbo

data_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
files = list(data_path.glob("*.tif*"))

first_file = files[0]
file_size_gb = first_file.stat().st_size / 1e9
print(f"First file: {first_file.name}, Size: {file_size_gb:.2f} GB")

outpath = data_path.parent / "tiff_test_output"
x = mbo.imread(first_file)
mbo.imwrite(x, outpath, planes=[1, 2], ext=".bin")
x = 2
# mbo.imwrite(x, temp_outfile)