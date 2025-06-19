import zarr
from fsspec.implementations.reference import ReferenceFileSystem
from zarr.storage import FsspecStore

def imshow_lazy_array(array, **kwargs):
    try:
        import fastplotlib as fpl
    except ImportError:
        raise ImportError("fastplotlib is required for image display.")

    if hasattr(array.data, "reference"):
        store = FsspecStore(ReferenceFileSystem(str(array.data.reference)))
        z_arr = zarr.open(store, mode="r")
        iw = fpl.ImageWidget(z_arr, **kwargs)
        iw.show()
        return iw
    else:
        raise ValueError("No reference file found. Please call save_fsspec() first.")
