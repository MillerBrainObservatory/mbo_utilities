"""
dimension tags for filename generation.

extensible system for labeling output files with dimension ranges.
tags follow array dims order and use format: {label}{start}[-{stop}[-{step}]]

examples:
    tp00001-10000_zplane01-14_stack.tif  (TZYX)
    zplane01-14_stack.tif                (ZYX, single timepoint)
    tp00001-10000_stack.tif              (TYX, single plane)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# spatial dims that don't appear in filenames
SPATIAL_DIMS = {"Y", "X"}


@dataclass(frozen=True)
class TagDefinition:
    """definition for a dimension tag type."""

    label: str        # filename prefix: "tp", "zplane", "cam"
    description: str  # human readable
    zero_pad: int = 2 # pad with how many digits could this value be


# built-in tag definitions
TAG_REGISTRY: dict[str, TagDefinition] = {
    "T": TagDefinition("tp", "timepoint", zero_pad=5),
    "Z": TagDefinition("zplane", "z-plane", zero_pad=2),
    "C": TagDefinition("ch", "channel", zero_pad=2),
    "V": TagDefinition("view", "view", zero_pad=2),
    "R": TagDefinition("roi", "region", zero_pad=2),
    # future extensions
    "B": TagDefinition("beamlet", "beamlet", zero_pad=2),
    "A": TagDefinition("cam", "camera", zero_pad=1),
}


@dataclass
class DimensionTag:
    """single dimension tag instance with range information."""

    definition: TagDefinition
    start: int        # 1-based start index
    stop: int | None  # 1-based stop index (None = single value)
    step: int = 1     # step size (default 1)

    def to_string(self) -> str:
        """format as filename component.

        examples:
            zplane01       (single value)
            zplane01-14    (range, step=1)
            zplane01-14-2  (range with step)
        """
        pad = self.definition.zero_pad
        label = self.definition.label

        if self.stop is None or self.start == self.stop:
            # single value
            return f"{label}{self.start:0{pad}d}"

        if self.step == 1:
            # range without step
            return f"{label}{self.start:0{pad}d}-{self.stop:0{pad}d}"

        # range with step
        return f"{label}{self.start:0{pad}d}-{self.stop:0{pad}d}-{self.step}"

    def to_slice(self) -> slice:
        """convert to 0-based slice for array indexing."""
        start_0 = self.start - 1
        stop_0 = self.stop if self.stop is not None else self.start
        return slice(start_0, stop_0, self.step if self.step != 1 else None)

    @property
    def size(self) -> int:
        """number of elements in this range."""
        if self.stop is None:
            return 1
        return len(range(self.start, self.stop + 1, self.step))

    @classmethod
    def from_dim_size(
        cls,
        definition: TagDefinition,
        size: int,
        selection: slice | list | Sequence[int] | None = None,
    ) -> "DimensionTag":
        """create from dimension size and optional selection.

        parameters
        ----------
        definition : TagDefinition
            tag type definition
        size : int
            total size of dimension
        selection : slice | list | Sequence[int] | None
            optional selection (indices are 1-based)

        returns
        -------
        DimensionTag
            tag with appropriate range
        """
        if selection is None:
            # full range
            if size == 1:
                return cls(definition, start=1, stop=None, step=1)
            return cls(definition, start=1, stop=size, step=1)

        if isinstance(selection, slice):
            # convert slice to range
            start = (selection.start or 0) + 1  # convert to 1-based
            stop = selection.stop if selection.stop else size
            step = selection.step or 1
            if start == stop or (stop - start) <= step:
                return cls(definition, start=start, stop=None, step=1)
            return cls(definition, start=start, stop=stop, step=step)

        # list/sequence of indices (assumed 1-based)
        indices = list(selection)
        if len(indices) == 1:
            return cls(definition, start=indices[0], stop=None, step=1)

        # check if contiguous with consistent step
        if len(indices) >= 2:
            step = indices[1] - indices[0]
            is_uniform = all(
                indices[i + 1] - indices[i] == step
                for i in range(len(indices) - 1)
            )
            if is_uniform and step > 0:
                return cls(definition, start=indices[0], stop=indices[-1], step=step)

        # non-uniform selection - use start-stop with step showing non-uniform
        return cls(definition, start=indices[0], stop=indices[-1], step=1)

    @classmethod
    def from_string(cls, s: str) -> "DimensionTag":
        """parse from filename component.

        examples:
            "zplane01"      -> DimensionTag(start=1, stop=None)
            "zplane01-14"   -> DimensionTag(start=1, stop=14)
            "zplane01-14-2" -> DimensionTag(start=1, stop=14, step=2)
        """
        import re

        # find matching tag definition
        definition = None
        for dim_char, tag_def in TAG_REGISTRY.items():
            if s.startswith(tag_def.label):
                definition = tag_def
                s = s[len(tag_def.label):]
                break

        if definition is None:
            raise ValueError(f"unknown tag prefix in '{s}'")

        # parse numbers
        parts = s.split("-")
        if len(parts) == 1:
            return cls(definition, start=int(parts[0]), stop=None, step=1)
        elif len(parts) == 2:
            return cls(definition, start=int(parts[0]), stop=int(parts[1]), step=1)
        elif len(parts) == 3:
            return cls(definition, start=int(parts[0]), stop=int(parts[1]), step=int(parts[2]))
        else:
            raise ValueError(f"invalid tag format: '{s}'")


class OutputFilename:
    """builds output filename from dimension tags."""

    def __init__(self, tags: list[DimensionTag], suffix: str = "stack"):
        self.tags = tags
        self.suffix = suffix

    def build(self, ext: str = ".tif") -> str:
        """build filename string.

        returns
        -------
        str
            filename like "tp00001-10000_zplane01-14_stack.tif"
        """
        if not ext.startswith("."):
            ext = "." + ext

        parts = [tag.to_string() for tag in self.tags]
        parts.append(self.suffix)
        return "_".join(parts) + ext

    @classmethod
    def from_array(
        cls,
        arr,
        planes: int | list | Sequence[int] | None = None,
        frames: int | list | Sequence[int] | None = None,
        suffix: str = "stack",
    ) -> "OutputFilename":
        """create from array and optional selections.

        parameters
        ----------
        arr : array-like
            array with shape and dims
        planes : int | list | None
            z-plane selection (1-based indices)
        frames : int | list | None
            timepoint selection (1-based indices)
        suffix : str
            filename suffix (default "stack")

        returns
        -------
        OutputFilename
            builder with tags derived from array dims
        """
        from mbo_utilities.arrays.features._dim_labels import get_dims

        dims = get_dims(arr)
        shape = arr.shape
        tags = []

        for i, dim_char in enumerate(dims):
            if dim_char in SPATIAL_DIMS:
                continue

            if dim_char not in TAG_REGISTRY:
                continue

            definition = TAG_REGISTRY[dim_char]
            dim_size = shape[i]

            # apply selection if matching dimension
            selection = None
            if dim_char == "Z" and planes is not None:
                selection = [planes] if isinstance(planes, int) else planes
            elif dim_char == "T" and frames is not None:
                selection = [frames] if isinstance(frames, int) else frames

            tag = DimensionTag.from_dim_size(definition, dim_size, selection)
            tags.append(tag)

        return cls(tags, suffix=suffix)

    def __repr__(self) -> str:
        return f"OutputFilename({self.build()!r})"
