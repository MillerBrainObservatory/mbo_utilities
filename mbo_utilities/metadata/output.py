"""
output metadata computation for subsetted data.

handles automatic adjustment of metadata when writing subsets:
- z-step size scales with plane step
- frame rate validity depends on contiguity
- format-specific builders for ImageJ, OME, napari
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mbo_utilities.metadata.params import get_param, get_voxel_size
from mbo_utilities.metadata.base import VoxelSize


@dataclass
class OutputMetadata:
    """
    computes adjusted metadata for output based on selection.

    handles the transformation of source metadata to output metadata
    when subsets of data are being written (e.g., every Nth z-plane,
    specific frame ranges, etc.)

    parameters
    ----------
    source : dict
        source metadata dictionary
    frame_indices : list[int] | None
        0-based frame indices being written (None = all frames)
    plane_indices : list[int] | None
        0-based plane indices being written (None = all planes)
    source_num_frames : int | None
        total frames in source (for contiguity check)
    source_num_planes : int | None
        total planes in source (for step factor)

    examples
    --------
    >>> meta = {"dz": 5.0, "fs": 30.0}
    >>> out = OutputMetadata(meta, plane_indices=[0, 2, 4, 6])
    >>> out.dz  # every 2nd plane -> dz doubles
    10.0

    >>> out = OutputMetadata(meta, frame_indices=[0, 5, 10])
    >>> out.fs  # non-contiguous -> no valid fs
    None
    """

    source: dict
    frame_indices: list[int] | None = None
    plane_indices: list[int] | None = None
    source_num_frames: int | None = None
    source_num_planes: int | None = None

    # computed fields
    _is_contiguous: bool = field(default=True, init=False)
    _frame_step: int = field(default=1, init=False)
    _z_step_factor: int = field(default=1, init=False)

    def __post_init__(self):
        """compute derived values after init."""
        self._compute_contiguity()
        self._compute_z_step_factor()

    def _compute_contiguity(self):
        """determine if frame selection is contiguous with uniform step."""
        if self.frame_indices is None or len(self.frame_indices) <= 1:
            self._is_contiguous = True
            self._frame_step = 1
            return

        # check for uniform spacing
        steps = [
            self.frame_indices[i + 1] - self.frame_indices[i]
            for i in range(len(self.frame_indices) - 1)
        ]

        if not steps:
            self._is_contiguous = True
            self._frame_step = 1
            return

        unique_steps = set(steps)
        if len(unique_steps) == 1:
            self._frame_step = steps[0]
            self._is_contiguous = True
        else:
            # non-uniform spacing
            self._is_contiguous = False
            self._frame_step = 1

    def _compute_z_step_factor(self):
        """compute z-step multiplication factor from plane selection."""
        if self.plane_indices is None or len(self.plane_indices) <= 1:
            self._z_step_factor = 1
            return

        # check for uniform spacing
        steps = [
            self.plane_indices[i + 1] - self.plane_indices[i]
            for i in range(len(self.plane_indices) - 1)
        ]

        if not steps:
            self._z_step_factor = 1
            return

        unique_steps = set(steps)
        if len(unique_steps) == 1:
            self._z_step_factor = steps[0]
        else:
            # non-uniform z selection - use first step, log warning
            import logging
            logging.getLogger("mbo_utilities").warning(
                f"Non-uniform z-plane spacing detected (steps: {steps[:5]}...). "
                f"Using first step ({steps[0]}) for dz calculation."
            )
            self._z_step_factor = steps[0]

    @property
    def is_contiguous(self) -> bool:
        """whether frame selection is contiguous with uniform step."""
        return self._is_contiguous

    @property
    def frame_step(self) -> int:
        """step between selected frames (1 if contiguous)."""
        return self._frame_step

    @property
    def z_step_factor(self) -> int:
        """multiplication factor for z-step (step between selected planes)."""
        return self._z_step_factor

    @property
    def num_frames(self) -> int | None:
        """number of frames in output."""
        if self.frame_indices is not None:
            return len(self.frame_indices)
        return self.source_num_frames

    @property
    def num_planes(self) -> int | None:
        """number of planes in output."""
        if self.plane_indices is not None:
            return len(self.plane_indices)
        return self.source_num_planes

    @property
    def dz(self) -> float | None:
        """adjusted z-step for output planes."""
        source_dz = get_param(self.source, "dz")
        if source_dz is None:
            return None
        return source_dz * self._z_step_factor

    @property
    def dx(self) -> float:
        """pixel size in x (unchanged from source)."""
        return get_param(self.source, "dx", default=1.0)

    @property
    def dy(self) -> float:
        """pixel size in y (unchanged from source)."""
        return get_param(self.source, "dy", default=1.0)

    @property
    def voxel_size(self) -> VoxelSize:
        """adjusted voxel size for output."""
        return VoxelSize(dx=self.dx, dy=self.dy, dz=self.dz)

    @property
    def fs(self) -> float | None:
        """
        frame rate - only valid for contiguous frames.

        if frames are non-contiguous, returns None since
        fs has no physical meaning.
        """
        if not self._is_contiguous:
            return None
        source_fs = get_param(self.source, "fs")
        if source_fs is None:
            return None
        # adjust for frame step (e.g., every 2nd frame = half the rate)
        return source_fs / self._frame_step

    @property
    def finterval(self) -> float | None:
        """frame interval in seconds (1/fs)."""
        fs = self.fs
        if fs is None or fs <= 0:
            return None
        return 1.0 / fs

    @property
    def total_duration(self) -> float | None:
        """total duration in seconds (only for contiguous frames)."""
        if not self._is_contiguous:
            return None
        fs = self.fs
        n_frames = self.num_frames
        if fs is None or n_frames is None:
            return None
        return n_frames / fs

    def to_imagej(self, shape: tuple) -> tuple[dict, tuple]:
        """
        build ImageJ-compatible metadata dict and resolution tuple.

        parameters
        ----------
        shape : tuple
            output array shape (T, Z, Y, X) or (T, Y, X) or (Y, X)

        returns
        -------
        tuple[dict, tuple]
            (imagej_metadata, resolution) ready for tifffile
        """
        vs = self.voxel_size

        ij_meta = {
            "unit": "um",
            "loop": False,
        }

        # imagej hyperstack dimensions: frames (T), slices (Z), channels (C)
        # tifffile infers axes from shape + counts, explicit axes causes validation errors
        ndim = len(shape)
        if ndim == 4:
            # TZYX input
            n_frames = shape[0]
            n_slices = shape[1]
            ij_meta["images"] = n_frames * n_slices  # total pages
            ij_meta["frames"] = n_frames
            ij_meta["slices"] = n_slices
            ij_meta["channels"] = 1
            ij_meta["hyperstack"] = True
        elif ndim == 3:
            # TYX input
            ij_meta["images"] = shape[0]
            ij_meta["frames"] = shape[0]
            ij_meta["slices"] = 1
            ij_meta["channels"] = 1
        else:
            # YX input
            ij_meta["images"] = 1
            ij_meta["frames"] = 1
            ij_meta["slices"] = 1
            ij_meta["channels"] = 1

        # z-spacing (adjusted for plane step)
        if vs.dz is not None:
            ij_meta["spacing"] = vs.dz

        # frame interval (only if contiguous)
        if self._is_contiguous and self.finterval is not None:
            ij_meta["finterval"] = self.finterval

        # resolution is pixels per um (inverse of um/pixel)
        res_x = 1.0 / vs.dx if vs.dx and vs.dx > 0 else 1.0
        res_y = 1.0 / vs.dy if vs.dy and vs.dy > 0 else 1.0

        return ij_meta, (res_x, res_y)

    def to_ome_ngff(self, dims: tuple[str, ...] = ("T", "Z", "Y", "X")) -> dict:
        """
        build OME-NGFF v0.5 compliant metadata.

        parameters
        ----------
        dims : tuple[str, ...]
            dimension labels for the output array

        returns
        -------
        dict
            OME-NGFF v0.5 multiscales metadata
        """
        from mbo_utilities.arrays.features._dim_tags import (
            dims_to_ome_axes,
            normalize_dims,
        )

        vs = self.voxel_size
        dims = normalize_dims(dims)
        axes = dims_to_ome_axes(dims)

        # build scale values matching dimension order
        scales = []
        for dim in dims:
            if dim == "T":
                # time scale is finterval if contiguous, else 1.0
                if self._is_contiguous and self.finterval is not None:
                    scales.append(self.finterval)
                else:
                    scales.append(1.0)
            elif dim == "Z":
                scales.append(vs.dz if vs.dz is not None else 1.0)
            elif dim == "Y":
                scales.append(vs.dy)
            elif dim == "X":
                scales.append(vs.dx)
            else:
                scales.append(1.0)  # C, V, B, etc.

        return {
            "axes": axes,
            "coordinateTransformations": [{"type": "scale", "scale": scales}],
        }

    def to_napari_scale(self, dims: tuple[str, ...] = ("T", "Z", "Y", "X")) -> tuple:
        """
        build napari-compatible scale tuple.

        parameters
        ----------
        dims : tuple[str, ...]
            dimension labels for the output array

        returns
        -------
        tuple
            scale values in same order as dims
        """
        vs = self.voxel_size
        scale = []

        for dim in dims:
            dim_upper = dim.upper()
            if dim_upper == "T":
                if self._is_contiguous and self.finterval is not None:
                    scale.append(self.finterval)
                else:
                    scale.append(1.0)
            elif dim_upper == "Z":
                scale.append(vs.dz if vs.dz is not None else 1.0)
            elif dim_upper == "Y":
                scale.append(vs.dy)
            elif dim_upper == "X":
                scale.append(vs.dx)
            elif dim_upper == "C":
                scale.append(1.0)

        return tuple(scale)

    def to_dict(self, include_aliases: bool = True) -> dict:
        """
        export as flat metadata dict with all aliases.

        parameters
        ----------
        include_aliases : bool
            if True, includes all standard aliases (OME, ImageJ, legacy)

        returns
        -------
        dict
            metadata dictionary with adjusted values
        """
        result = dict(self.source)  # copy source

        # update with computed voxel size
        vs = self.voxel_size
        result.update(vs.to_dict(include_aliases=include_aliases))

        # frame rate / timing
        if self._is_contiguous:
            if self.fs is not None:
                result["fs"] = self.fs
                result["frame_rate"] = self.fs
            if self.finterval is not None:
                result["finterval"] = self.finterval
            result["is_contiguous"] = True
        else:
            # non-contiguous frames - fs/finterval not meaningful for output
            result["is_contiguous"] = False
            # clear fs-related keys so downstream writers don't use invalid values
            result["fs"] = None
            result["frame_rate"] = None
            result["finterval"] = None
            # keep source fs for reference with different key
            source_fs = get_param(self.source, "fs")
            if source_fs is not None:
                result["source_fs"] = source_fs

        # update dimension counts with all aliases
        if self.num_frames is not None:
            result["num_timepoints"] = self.num_frames
            result["nframes"] = self.num_frames
            result["num_frames"] = self.num_frames
            result["n_frames"] = self.num_frames
            result["timepoints"] = self.num_frames
            result["T"] = self.num_frames
            result["nt"] = self.num_frames
        if self.num_planes is not None:
            result["num_zplanes"] = self.num_planes
            result["nplanes"] = self.num_planes
            result["num_planes"] = self.num_planes
            result["n_planes"] = self.num_planes
            result["zplanes"] = self.num_planes
            result["Z"] = self.num_planes
            result["nz"] = self.num_planes
            result["slices"] = self.num_planes  # imagej alias
            result["num_channels"] = self.num_planes  # lbm: z-planes stored as channels

        # record selection info
        if self._z_step_factor > 1:
            result["z_step_factor"] = self._z_step_factor
        if self._frame_step > 1:
            result["frame_step"] = self._frame_step

        return result

    def __repr__(self) -> str:
        parts = [f"OutputMetadata("]
        if self.dz is not None:
            parts.append(f"dz={self.dz:.2f}um")
            if self._z_step_factor > 1:
                parts.append(f" (x{self._z_step_factor})")
        if self.fs is not None:
            parts.append(f", fs={self.fs:.2f}Hz")
        elif not self._is_contiguous:
            parts.append(", fs=N/A (non-contiguous)")
        parts.append(")")
        return "".join(parts)
