import time
from uuid import uuid4

from neuroconv.tools.nwb_helpers import get_default_backend_configuration
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from pynwb.file import Subject, NWBFile
from neuroconv.datainterfaces.ophys.tiff.tiffdatainterface import GeneralTiffImagingInterface
from neuroconv.tools import configure_and_write_nwbfile
from mbo_utilities.array_types import NWBArray

def reader():
    fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\0727_mk355_testing.zarr")
    nwb_array = NWBArray(fpath)
    x = 4


def run_conversion():
    """
    Run the conversion process for the TIFF imaging data.
    """

    session_start_time = datetime(2025, 5, 5, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))

    full_file = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green\mk355_7_27_2025_180mw_right_m2_go_to_2x-mROI-880x1100um_220x550px_2um-px_14p00Hz_00001_00001_00001.tif")
    file_path = full_file.parent
    files = mbo.get_files(file_path)

    interface = GeneralTiffImagingInterface(
        file_paths=files,
        sampling_frequency=14.0,
        num_planes=14,
    )
    metadata = interface.get_metadata()
    metadata["NWBFile"].update(session_start_time=session_start_time)
    nwbfile = NWBFile(
        session_description="Mouse doing a visual discrimination task",
        identifier=str(uuid4()),
        session_id="m2-left-0727-testing",
        session_start_time=session_start_time,
        experimenter=[
            "Barber, Kevin",
        ],
        lab="Alipasha Vaziri",
        institution="Rockefeller University",
        keywords=["behavior", "vision", "imaging", "mouse", "two-photon", "light-beads-microscopy"],
    )

    subject = Subject(
        subject_id="mk355",
        species="Mus musculus",
        description="White mouse",
        sex="M",
        age="P2Y",
    )
    nwbfile.subject = subject
    interface.add_to_nwbfile(nwbfile=nwbfile, metadata=metadata)
    backend_configuration = get_default_backend_configuration(
        nwbfile=nwbfile, backend="zarr"
    )
    backend_configuration.number_of_jobs = -1
    # backend_configuration.dataset_configurations["acquisition/TwoPhotonSeries/data"].chunk_shape = (14, 224, 279, 14)

    nwb_outpath = file_path.parent.joinpath("m355_0727_full-dataset.zarr")
    start = time.time()
    configure_and_write_nwbfile(nwbfile, nwbfile_path=nwb_outpath, backend_configuration=backend_configuration,)
    end = time.time()
    print(f"NWBFile writing took {end - start:.2f} seconds")

    interface.run_conversion(
        nwbfile_path=nwb_outpath,
        nwbfile=nwbfile,
        metadata=metadata,
        overwrite=True,
        backend_configuration=backend_configuration,
    )

if __name__ == "__main__":
    import mbo_utilities as mbo
    import zarr
    import numpy as np

    reader()
    # run_conversion()
    # my_nwb = read_nwb(r"D:\W2_DATA\kbarber\07_27_2025\mk355\0727_mk355_testing.zarr")
    # data = my_nwb.acquisition["TwoPhotonSeries"].data
    # chunks = data.chunks
    # reshaped = np.moveaxis(data, -1, 1)

    # iw = fpl.ImageWidget(reshaped)
    # iw.show()
    # fpl.loop.run()
    uri_rechunked = r"s3://dandiarchive/zarr/a1f93cbf-fb83-4a02-a02b-e7b92d59fea0/"
    uri_original = r"s3://dandiarchive/zarr/2f0c6b0f-dc77-4f99-8291-40182bb7c33d/"

    z = zarr.open(
        store=uri_rechunked,
        storage_options = dict(anon=True),
        mode="r",
    )
    z_old = zarr.open(
        store=uri_original,
        storage_options = dict(anon=True),
        mode="r",
    )

    full_file = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green\mk355_7_27_2025_180mw_right_m2_go_to_2x-mROI-880x1100um_220x550px_2um-px_14p00Hz_00001_00001_00001.tif")
    data_mbo = mbo.imread(full_file)
    data_rechunked = z["acquisition"]["TwoPhotonSeries"].data
    data = z_old["acquisition"]["TwoPhotonSeries"].data
    chunks = data.chunks

    chunk_size_MB = np.prod(data.chunks) * data.dtype.itemsize / 1024 ** 2
    full_size_MB = np.prod(data.shape) * data.dtype.itemsize / 1024 ** 2

    ats = z.attrs
    x = 4
    # metadata = mbo.get_metadata(
    #     r"D:\W2_DATA\kbarber\07_27_2025\mk355\green\mk355_7_27_2025_180mw_right_m2_go_to_2x-mROI-880x1100um_220x550px_2um-px_14p00Hz_00001_00001_00001.tif",
    #     verbose=True,
    # )

    start = time.time()
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")
    print("Conversion completed successfully.")
