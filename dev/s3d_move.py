from pathlib import Path
import numpy as np
import time
import warnings
import mbo_utilities as mbo

warnings.simplefilter(action='ignore')

# repo_root = Path(__file__).resolve().parents[2] / "suite3d"
# sys.path.insert(0, str(repo_root))

if __name__ == "__main__":
    fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green")
    arr = mbo.imread(fpath)
    arr.preprocess()