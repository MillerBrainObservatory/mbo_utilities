"""
Pipeline registry for tracking inputs, outputs, and file patterns.

Each array type and pipeline declares what files it reads/writes
"""

from dataclasses import dataclass, field
from collections.abc import Callable


@dataclass
class PipelineInfo:
    """describes a pipeline's file patterns and metadata."""

    # unique identifier
    name: str

    # human-readable description
    description: str = ""

    # glob patterns for input files this pipeline reads
    # e.g. ["**/*.tif", "**/*.tiff"] for raw scanimage
    input_patterns: list[str] = field(default_factory=list)

    # glob patterns for output files this pipeline produces
    # e.g. ["**/ops.npy", "**/data.bin"] for suite2p
    output_patterns: list[str] = field(default_factory=list)

    # file extensions this pipeline reads (without dot)
    # e.g. ["tif", "tiff", "bin"]
    input_extensions: list[str] = field(default_factory=list)

    # file extensions this pipeline writes (without dot)
    # e.g. ["bin", "npy", "zarr"]
    output_extensions: list[str] = field(default_factory=list)

    # marker files that identify this pipeline's output directory
    # e.g. ["ops.npy"] for suite2p - if this exists, it's a suite2p dir
    marker_files: list[str] = field(default_factory=list)

    # optional validator function: (path) -> bool
    # returns True if path is valid input/output for this pipeline
    validator: Callable | None = None

    # pipeline category for grouping in UI
    # e.g. "reader", "writer", "processor", "segmentation"
    category: str = "unknown"


# global registry
_PIPELINES: dict[str, PipelineInfo] = {}


def register_pipeline(info: PipelineInfo) -> None:
    """Register a pipeline's info in the global registry."""
    _PIPELINES[info.name] = info


def get_pipeline_info(name: str) -> PipelineInfo | None:
    """Get info for a specific pipeline by name."""
    return _PIPELINES.get(name)


def get_all_pipelines() -> dict[str, PipelineInfo]:
    """Get all registered pipelines."""
    return dict(_PIPELINES)


def get_all_input_patterns() -> dict[str, list[str]]:
    """Get input patterns grouped by pipeline name."""
    return {name: info.input_patterns for name, info in _PIPELINES.items()}


def get_all_output_patterns() -> dict[str, list[str]]:
    """Get output patterns grouped by pipeline name."""
    return {name: info.output_patterns for name, info in _PIPELINES.items()}


def get_all_marker_files() -> dict[str, list[str]]:
    """Get marker files grouped by pipeline name."""
    return {name: info.marker_files for name, info in _PIPELINES.items()}


def get_pipelines_by_category(category: str) -> dict[str, PipelineInfo]:
    """Get all pipelines in a category."""
    return {
        name: info for name, info in _PIPELINES.items()
        if info.category == category
    }


def get_readable_extensions() -> set[str]:
    """Get all file extensions that can be read by any pipeline."""
    extensions = set()
    for info in _PIPELINES.values():
        extensions.update(info.input_extensions)
    return extensions


def get_writable_extensions() -> set[str]:
    """Get all file extensions that can be written by any pipeline."""
    extensions = set()
    for info in _PIPELINES.values():
        extensions.update(info.output_extensions)
    return extensions


# convenience decorator for registering pipelines
def pipeline(
    name: str,
    description: str = "",
    input_patterns: list[str] | None = None,
    output_patterns: list[str] | None = None,
    input_extensions: list[str] | None = None,
    output_extensions: list[str] | None = None,
    marker_files: list[str] | None = None,
    category: str = "unknown",
):
    """Decorator to register a class as a pipeline with file patterns."""
    def decorator(cls):
        info = PipelineInfo(
            name=name,
            description=description,
            input_patterns=input_patterns or [],
            output_patterns=output_patterns or [],
            input_extensions=input_extensions or [],
            output_extensions=output_extensions or [],
            marker_files=marker_files or [],
            category=category,
        )
        register_pipeline(info)
        # attach info to class for introspection
        cls._pipeline_info = info
        return cls
    return decorator
