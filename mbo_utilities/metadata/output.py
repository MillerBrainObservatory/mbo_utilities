"""
output metadata computation for subsetted data.

handles automatic adjustment of metadata when writing subsets:
- all dimensions are reactive (Lx, Ly, num_zplanes, num_timepoints)
- z-step size scales with plane step
- frame rate validity depends on contiguity
- format-specific builders for ImageJ, OME, napari
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from mbo_utilities.metadata.params import get_param, get_voxel_size
from mbo_utilities.metadata.base import VoxelSize

if TYPE_CHECKING:
    from mbo_utilities.arrays.features._dim_spec import DimensionSpecs


@dataclass
class OutputMetadata:
    """
    computes adjusted metadata for output based on selection.

    handles the transformation of source metadata to output metadata
    when subsets of data are being written (e.g., every Nth z-plane,
    specific frame ranges, spatial cropping, etc.)

    parameters
    ----------
    source : dict
        source metadata dictionary
    source_shape : tuple[int, ...]
        shape of the source array
    source_dims : tuple[str, ...] | None
        dimension labels for source array (e.g., ("T", "Z", "Y", "X"))
    selections : dict[str, list[int]] | None
        mapping of dim name -> 0-based indices selected
        e.g., {"T": [0,1,2], "Z": [0,2,4], "Y": range(100,400), "X": range(50,450)}

    examples
    --------
    >>> meta = {"dz": 5.0, "fs": 30.0, "dx": 0.5, "dy": 0.5}
    >>> out = OutputMetadata(meta, (100, 28, 512, 512), ("T", "Z", "Y", "X"),
    ...                      selections={"Z": [0, 2, 4, 6]})
    >>> out.dz  # every 2nd plane -> dz doubles
    10.0
    >>> out.num_zplanes
    4
    >>> out.Ly  # unchanged (no Y selection)
    512
    """

    source: dict
    source_shape: tuple[int, ...] = field(default_factory=tuple)
    source_dims: tuple[str, ...] | None = None
    selections: dict[str, list[int]] | None = None

    # legacy params for backwards compatibility
    frame_indices: list[int] | None = None
    plane_indices: list[int] | None = None
    source_num_frames: int | None = None
    source_num_planes: int | None = None

    # computed fields
    _is_contiguous: bool = field(default=True, init=False)
    _frame_step: int = field(default=1, init=False)
    _z_step_factor: int = field(default=1, init=False)
    _output_shape: tuple[int, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self):
        """compute derived values after init."""
        # migrate legacy params to selections dict
        if self.selections is None:
            self.selections = {}

        if self.frame_indices is not None and "T" not in self.selections:
            self.selections["T"] = list(self.frame_indices)
        if self.plane_indices is not None and "Z" not in self.selections:
            self.selections["Z"] = list(self.plane_indices)

        # infer source dims if not provided
        if self.source_dims is None:
            ndim = len(self.source_shape) if self.source_shape else 4
            from mbo_utilities.arrays.features._dim_labels import DEFAULT_DIMS
            self.source_dims = DEFAULT_DIMS.get(ndim, ("T", "Z", "Y", "X"))

        # compute derived values
        self._compute_contiguity()
        self._compute_z_step_factor()
        self._compute_output_shape()

    def _compute_contiguity(self):
        """determine if frame selection is contiguous with uniform step."""
        t_sel = self.selections.get("T") if self.selections else None
        if t_sel is None:
            t_sel = self.frame_indices

        if t_sel is None or len(t_sel) <= 1:
            self._is_contiguous = True
            self._frame_step = 1
            return

        steps = [t_sel[i + 1] - t_sel[i] for i in range(len(t_sel) - 1)]

        if not steps:
            self._is_contiguous = True
            self._frame_step = 1
            return

        unique_steps = set(steps)
        if len(unique_steps) == 1:
            self._frame_step = steps[0]
            self._is_contiguous = True
        else:
            self._is_contiguous = False
            self._frame_step = 1

    def _compute_z_step_factor(self):
        """compute z-step multiplication factor from plane selection."""
        z_sel = self.selections.get("Z") if self.selections else None
        if z_sel is None:
            z_sel = self.plane_indices

        if z_sel is None or len(z_sel) <= 1:
            self._z_step_factor = 1
            return

        steps = [z_sel[i + 1] - z_sel[i] for i in range(len(z_sel) - 1)]

        if not steps:
            self._z_step_factor = 1
            return

        unique_steps = set(steps)
        if len(unique_steps) == 1:
            self._z_step_factor = steps[0]
        else:
            import logging
            logging.getLogger("mbo_utilities").warning(
                f"Non-uniform z-plane spacing detected (steps: {steps[:5]}...). "
                f"Using first step ({steps[0]}) for dz calculation."
            )
            self._z_step_factor = steps[0]

    def _compute_output_shape(self):
        """compute output shape based on selections."""
        if not self.source_shape or not self.source_dims:
            self._output_shape = self.source_shape
            return

        output = []
        for i, dim in enumerate(self.source_dims):
            dim_upper = dim.upper()
            if self.selections and dim_upper in self.selections:
                output.append(len(self.selections[dim_upper]))
            elif i < len(self.source_shape):
                output.append(self.source_shape[i])
            else:
                output.append(1)
        self._output_shape = tuple(output)

    def _get_step_factor(self, dim: str) -> int:
        """get step factor for a dimension (1 if not uniformly spaced)."""
        sel = self.selections.get(dim.upper()) if self.selections else None
        if sel is None or len(sel) <= 1:
            return 1

        steps = [sel[i + 1] - sel[i] for i in range(len(sel) - 1)]
        unique_steps = set(steps)
        if len(unique_steps) == 1:
            return steps[0]
        return 1

    # shape and dimension properties

    @property
    def output_shape(self) -> tuple[int, ...]:
        """shape of output array after selections."""
        return self._output_shape

    @property
    def Lx(self) -> int:
        """output width (X dimension size)."""
        if not self.source_dims:
            return self.source_shape[-1] if self.source_shape else 0
        try:
            idx = list(self.source_dims).index("X")
            return self._output_shape[idx] if idx < len(self._output_shape) else 0
        except ValueError:
            return self.source_shape[-1] if self.source_shape else 0

    @property
    def Ly(self) -> int:
        """output height (Y dimension size)."""
        if not self.source_dims:
            return self.source_shape[-2] if len(self.source_shape) >= 2 else 0
        try:
            idx = list(self.source_dims).index("Y")
            return self._output_shape[idx] if idx < len(self._output_shape) else 0
        except ValueError:
            return self.source_shape[-2] if len(self.source_shape) >= 2 else 0

    @property
    def num_timepoints(self) -> int:
        """number of timepoints in output."""
        if not self.source_dims:
            return self.num_frames or 1
        try:
            idx = list(self.source_dims).index("T")
            return self._output_shape[idx] if idx < len(self._output_shape) else 1
        except ValueError:
            return 1

    @property
    def num_zplanes(self) -> int:
        """number of z-planes in output."""
        if not self.source_dims:
            return self.num_planes or 1
        try:
            idx = list(self.source_dims).index("Z")
            return self._output_shape[idx] if idx < len(self._output_shape) else 1
        except ValueError:
            return 1

    @property
    def num_color_channels(self) -> int:
        """number of color channels in output."""
        if not self.source_dims:
            return get_param(self.source, "num_color_channels", default=1) or 1
        try:
            idx = list(self.source_dims).index("C")
            return self._output_shape[idx] if idx < len(self._output_shape) else 1
        except ValueError:
            return get_param(self.source, "num_color_channels", default=1) or 1

    # legacy compatibility properties

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
        """number of frames in output (legacy, use num_timepoints)."""
        t_sel = self.selections.get("T") if self.selections else None
        if t_sel is None:
            t_sel = self.frame_indices
        if t_sel is not None:
            return len(t_sel)
        return self.source_num_frames

    @property
    def num_planes(self) -> int | None:
        """number of planes in output (legacy, use num_zplanes)."""
        z_sel = self.selections.get("Z") if self.selections else None
        if z_sel is None:
            z_sel = self.plane_indices
        if z_sel is not None:
            return len(z_sel)
        return self.source_num_planes

    # scale properties

    @property
    def dz(self) -> float | None:
        """adjusted z-step for output planes."""
        source_dz = get_param(self.source, "dz")
        if source_dz is None:
            return None
        return source_dz * self._z_step_factor

    @property
    def dx(self) -> float:
        """pixel size in x (adjusted if X selection has step > 1)."""
        source_dx = get_param(self.source, "dx", default=1.0) or 1.0
        x_step = self._get_step_factor("X")
        return source_dx * x_step

    @property
    def dy(self) -> float:
        """pixel size in y (adjusted if Y selection has step > 1)."""
        source_dy = get_param(self.source, "dy", default=1.0) or 1.0
        y_step = self._get_step_factor("Y")
        return source_dy * y_step

    @property
    def voxel_size(self) -> VoxelSize:
        """adjusted voxel size for output."""
        return VoxelSize(dx=self.dx, dy=self.dy, dz=self.dz)

    @property
    def fs(self) -> float | None:
        """frame rate - only valid for contiguous frames."""
        if not self._is_contiguous:
            return None
        source_fs = get_param(self.source, "fs")
        if source_fs is None:
            return None
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

    @classmethod
    def from_dimension_specs(
        cls,
        source_specs: "DimensionSpecs",
        selections: dict[str, list[int]] | None = None,
        source_metadata: dict | None = None,
    ) -> "OutputMetadata":
        """
        create OutputMetadata from DimensionSpecs.

        parameters
        ----------
        source_specs : DimensionSpecs
            dimension specs from source array
        selections : dict | None
            mapping of dim name -> 0-based indices selected
        source_metadata : dict | None
            additional source metadata

        returns
        -------
        OutputMetadata
        """
        source = source_metadata.copy() if source_metadata else {}

        # add scale values from specs
        for spec in source_specs:
            if spec.name == "X":
                source.setdefault("dx", spec.scale)
            elif spec.name == "Y":
                source.setdefault("dy", spec.scale)
            elif spec.name == "Z":
                source.setdefault("dz", spec.scale)
            elif spec.name == "T" and spec.scale > 0:
                source.setdefault("fs", 1.0 / spec.scale)

        return cls(
            source=source,
            source_shape=source_specs.shape,
            source_dims=source_specs.dims,
            selections=selections,
        )

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

        ndim = len(shape)
        if ndim == 5:
            # TZCYX order
            n_frames = shape[0]
            n_slices = shape[1]
            n_channels = shape[2]
            ij_meta["images"] = n_frames * n_slices * n_channels
            ij_meta["frames"] = n_frames
            ij_meta["slices"] = n_slices
            ij_meta["channels"] = n_channels
            ij_meta["hyperstack"] = True
        elif ndim == 4:
            n_frames = shape[0]
            n_slices = shape[1]
            ij_meta["images"] = n_frames * n_slices
            ij_meta["frames"] = n_frames
            ij_meta["slices"] = n_slices
            ij_meta["channels"] = 1
            ij_meta["hyperstack"] = True
        elif ndim == 3:
            ij_meta["images"] = shape[0]
            ij_meta["frames"] = shape[0]
            ij_meta["slices"] = 1
            ij_meta["channels"] = 1
        else:
            ij_meta["images"] = 1
            ij_meta["frames"] = 1
            ij_meta["slices"] = 1
            ij_meta["channels"] = 1

        if vs.dz is not None:
            ij_meta["spacing"] = vs.dz

        if self._is_contiguous and self.finterval is not None:
            ij_meta["finterval"] = self.finterval

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

        scales = []
        for dim in dims:
            if dim == "T":
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
                scales.append(1.0)

        return {
            "axes": axes,
            "coordinateTransformations": [{"type": "scale", "scale": scales}],
        }

    def to_napari_scale(self, dims: tuple[str, ...] = ("T", "Z", "Y", "X")) -> tuple:
        """build napari-compatible scale tuple."""
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
        export as flat metadata dict with all reactive values.

        all dimension sizes and scales are adjusted based on selections.

        parameters
        ----------
        include_aliases : bool
            if True, includes all standard aliases (OME, ImageJ, legacy)

        returns
        -------
        dict
            metadata dictionary with adjusted values
        """
        result = dict(self.source)

        # update shape
        result["shape"] = self.output_shape

        # update spatial dimensions (always reactive)
        result["Lx"] = self.Lx
        result["Ly"] = self.Ly

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
            result["is_contiguous"] = False
            result["fs"] = None
            result["frame_rate"] = None
            result["finterval"] = None
            source_fs = get_param(self.source, "fs")
            if source_fs is not None:
                result["source_fs"] = source_fs

        # update dimension counts with all aliases
        result["num_timepoints"] = self.num_timepoints
        result["nframes"] = self.num_timepoints
        result["num_frames"] = self.num_timepoints
        result["n_frames"] = self.num_timepoints
        result["timepoints"] = self.num_timepoints
        result["T"] = self.num_timepoints
        result["nt"] = self.num_timepoints

        result["num_zplanes"] = self.num_zplanes
        result["nplanes"] = self.num_zplanes
        result["num_planes"] = self.num_zplanes
        result["n_planes"] = self.num_zplanes
        result["zplanes"] = self.num_zplanes
        result["Z"] = self.num_zplanes
        result["nz"] = self.num_zplanes
        result["slices"] = self.num_zplanes
        result["num_channels"] = self.num_zplanes  # lbm: z-planes as channels
        result["num_color_channels"] = self.num_color_channels

        # record selection info
        if self._z_step_factor > 1:
            result["z_step_factor"] = self._z_step_factor
        if self._frame_step > 1:
            result["frame_step"] = self._frame_step

        return result

    def __repr__(self) -> str:
        parts = ["OutputMetadata("]
        parts.append(f"shape={self.output_shape}")
        if self.dz is not None:
            parts.append(f", dz={self.dz:.2f}um")
            if self._z_step_factor > 1:
                parts.append(f" (x{self._z_step_factor})")
        if self.fs is not None:
            parts.append(f", fs={self.fs:.2f}Hz")
        elif not self._is_contiguous:
            parts.append(", fs=N/A (non-contiguous)")
        parts.append(")")
        return "".join(parts)
