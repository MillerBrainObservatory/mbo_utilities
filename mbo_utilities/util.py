# backwards-compatibility shim — functions moved to their proper modules
from mbo_utilities.file_io import load_npy
from mbo_utilities.arrays._base import get_dtype
from mbo_utilities.arrays.features._slicing import listify_index, index_length
