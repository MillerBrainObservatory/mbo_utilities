import numpy as np
import pytest
import tifffile
import h5py
import mbo_utilities as mbo

# class DummyScan:
#     def __init__(self, data):
#         self.data = data
#     @property
#     def num_planes(self):
#         return self.data.shape[0]
#     def __getitem__(self, idx):
#         return self.data[idx]
#
# @pytest.fixture
# def scan(pathnames):
#     """A Mock Scanimage Scan"""
#     return mbo.read_scan(pathnames)
#
# @pytest.mark.parametrize("ext", [".tiff", ".h5", ".bin"])
# @pytest.mark.parametrize("planes", [None, [0], [0,1]])
# def test_save(tmp_path, scan, ext, planes):
#     out = tmp_path / "out"
#     mbo.save_as(scan, out, planes=planes, ext=ext, overwrite=True)
#     if ext == ".tiff":
#         files = sorted(out.glob(f"plane_*{ext}"))
#         arrs = [tifffile.memmap(f) for f in files]
#     elif ext == ".h5":
#         files = sorted(out.glob(f"plane_*{ext}"))
#         arrs = [h5py.File(f, "r")["data"][()] for f in files]
#     else:
#         files = sorted(out.glob(f"plane_*{ext}"))
#         arrs = [np.memmap(f, dtype=np.int16, mode="r",
#                           shape=scan.data.shape[1:]) for f in files]
#     saved = np.stack(arrs, 0)
#     orig = scan.data if planes is None else scan.data[planes]
#     assert np.array_equal(saved, orig)
