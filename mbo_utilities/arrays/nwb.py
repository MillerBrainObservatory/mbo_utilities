"""
NWB (Neurodata Without Borders) array reader.

This module provides NWBArray for reading two-photon imaging data from NWB files.
"""

from __future__ import annotations

from pathlib import Path

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.nwb")

# register nwb pipeline info
_NWB_INFO = PipelineInfo(
    name="nwb",
    description="NWB (Neurodata Without Borders) files",
    input_patterns=[
        "**/*.nwb",
    ],
    output_patterns=[
        "**/*.nwb",
    ],
    input_extensions=["nwb"],
    output_extensions=["nwb"],
    marker_files=[],
    category="reader",
)
register_pipeline(_NWB_INFO)


class NWBArray:
    """
    Lazy array reader for NWB (Neurodata Without Borders) files.

    Reads TwoPhotonSeries acquisition data from NWB files.

    Parameters
    ----------
    path : Path or str
        Path to NWB file.

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : np.dtype
        Data type.
    ndim : int
        Number of dimensions.

    Examples
    --------
    >>> arr = NWBArray("experiment.nwb")
    >>> arr.shape
    (10000, 512, 512)
    >>> frame = arr[0]  # Get first frame
    """

    def __init__(self, path: Path | str):
        try:
            from pynwb import read_nwb
        except ImportError:
            raise ImportError(
                "pynwb is not installed. Install with `pip install pynwb`."
            )
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No NWB file found at {self.path}")

        self.filenames = [self.path]

        nwbfile = read_nwb(path)
        self.data = nwbfile.acquisition["TwoPhotonSeries"].data
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.ndim = self.data.ndim
        self._metadata = {}

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def min(self) -> float:
        return float(self.data[0].min())

    @property
    def max(self) -> float:
        return float(self.data[0].max())

    @property
    def metadata(self) -> dict:
        """Return metadata as dict. Always returns dict, never None."""
        return self._metadata if self._metadata is not None else {}

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        """Write NWBArray to disk in various formats."""
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            **kwargs,
        )
