"""
To run this code:

uv pip install mbo_utilities lbm_suite2p_python
"""
from pathlib import Path
import mbo_utilities as mbo
import lbm_suite2p_python as lsp

# raw_data_path = Path("../raw_data").expanduser().resolve()
# output_data_path = Path("../extracted").expanduser().resolve()

raw_data_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green").expanduser().resolve()
output_data_path = Path(r"E:\W2_DATA\lbm\2025-07-27")
# if not len(list(output_data_path.glob("*.tif*"))):
#     output_data_path = Path.cwd() / output_data_path
#     data = mbo.imread(raw_data_path)
#     mbo.imwrite(data, output_data_path, roi=None)

files = list(output_data_path.glob("*.tif*"))

ops = lsp.default_ops()
ops["roidetect"] = True
ops["keep_movie_raw"] = False
ops["tau"] = 0.7
ops["batch_size"] = 300
ops["nimg_init"] = 300
ops["block_size"] = [64, 64]

results_path = Path(output_data_path).parent / "results"
results_path.mkdir(exist_ok=True)
lsp.run_volume(
    input_files=files,
    ops=ops,
    save_path=results_path,
    keep_reg=True,
    keep_raw=True,
    dff_window_size=500,
    dff_percentile=20,
)