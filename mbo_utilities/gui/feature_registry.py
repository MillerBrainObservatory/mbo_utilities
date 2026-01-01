"""
GUI Feature Registry for self-documenting features.

This module provides a registry system that:
1. Defines GUI features with metadata (name, description, image path)
2. Associates reusable draw functions with each feature
3. Enables auto-generation of documentation from feature definitions
4. Powers screenshot capture for documentation

Usage
-----
Register a feature:

    @register_feature(
        name="suite2p_settings",
        title="Suite2p Settings",
        description="Configure Suite2p pipeline parameters for calcium imaging analysis.",
        category="pipeline",
    )
    def draw_suite2p_feature(settings=None, **kwargs):
        from mbo_utilities.gui.pipeline_widgets import Suite2pSettings, draw_suite2p_settings_panel
        settings = settings or Suite2pSettings()
        return draw_suite2p_settings_panel(settings, show_header=True, show_footer=True)

Capture all feature screenshots:

    from mbo_utilities.gui.feature_registry import capture_all_features
    capture_all_features(output_dir="docs/_images/gui/features")

Generate documentation:

    from mbo_utilities.gui.feature_registry import generate_feature_docs
    generate_feature_docs(output_path="docs/gui_features.md")
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from pathlib import Path


@dataclass
class GUIFeature:
    """
    Metadata and rendering function for a GUI feature.

    Attributes
    ----------
    name : str
        Unique identifier for the feature (e.g., "suite2p_settings").
    title : str
        Human-readable title (e.g., "Suite2p Settings").
    description : str
        Short description for documentation (1-2 sentences).
    category : str
        Feature category: "pipeline", "viewer", "dialog", "widget".
    draw_func : Callable
        Function that renders the feature UI. Should accept **kwargs.
    long_description : str
        Extended description with usage details (for docs).
    tip : str
        Usage tip displayed in footer (optional).
    image_filename : str
        Output filename for screenshot (auto-generated from name if not set).
    window_size : tuple[int, int]
        Default window size for standalone rendering.
    requires_data : bool
        Whether the feature requires loaded data to render.
    """

    name: str
    title: str
    description: str
    category: str
    draw_func: Callable
    long_description: str = ""
    tip: str = ""
    image_filename: str = ""
    window_size: tuple[int, int] = (500, 500)
    requires_data: bool = False

    def __post_init__(self):
        if not self.image_filename:
            self.image_filename = f"{self.name}.png"
        if not self.long_description:
            self.long_description = self.description


# Global feature registry
_FEATURE_REGISTRY: dict[str, GUIFeature] = {}


def register_feature(
    name: str,
    title: str,
    description: str,
    category: str = "widget",
    long_description: str = "",
    tip: str = "",
    image_filename: str = "",
    window_size: tuple[int, int] = (500, 500),
    requires_data: bool = False,
):
    """
    Decorator to register a GUI feature drawing function.

    Parameters
    ----------
    name : str
        Unique identifier for the feature.
    title : str
        Human-readable title.
    description : str
        Short description (1-2 sentences).
    category : str
        Category: "pipeline", "viewer", "dialog", "widget".
    long_description : str
        Extended description for documentation.
    tip : str
        Usage tip for footer.
    image_filename : str
        Screenshot filename (defaults to {name}.png).
    window_size : tuple[int, int]
        Window size for standalone rendering.
    requires_data : bool
        Whether feature requires loaded data.

    Returns
    -------
    Callable
        Decorator function.

    Example
    -------
    @register_feature(
        name="suite2p_settings",
        title="Suite2p Settings",
        description="Configure Suite2p pipeline parameters.",
        category="pipeline",
    )
    def draw_suite2p_feature(**kwargs):
        ...
    """

    def decorator(func: Callable) -> Callable:
        feature = GUIFeature(
            name=name,
            title=title,
            description=description,
            category=category,
            draw_func=func,
            long_description=long_description or func.__doc__ or description,
            tip=tip,
            image_filename=image_filename or f"{name}.png",
            window_size=window_size,
            requires_data=requires_data,
        )
        _FEATURE_REGISTRY[name] = feature
        return func

    return decorator


def get_feature(name: str) -> Optional[GUIFeature]:
    """Get a registered feature by name."""
    return _FEATURE_REGISTRY.get(name)


def get_all_features() -> dict[str, GUIFeature]:
    """Get all registered features."""
    return _FEATURE_REGISTRY.copy()


def get_features_by_category(category: str) -> list[GUIFeature]:
    """Get all features in a category."""
    return [f for f in _FEATURE_REGISTRY.values() if f.category == category]


@register_feature(
    name="suite2p_settings",
    title="Suite2p Settings",
    description="Configure Suite2p pipeline parameters for calcium imaging analysis.",
    category="pipeline",
    long_description=(
        "Suite2p is a popular pipeline for processing calcium imaging data. "
        "This panel lets you configure registration, ROI detection, and signal "
        "extraction parameters. Defaults are optimized for LBM (Light Beads "
        "Microscopy) datasets."
    ),
    tip=(
        "For LBM data, tau=1.3 and diameter=4 are good starting points. "
        "Increase threshold_scaling if detecting too many false ROIs."
    ),
    window_size=(540, 620),
)
def draw_suite2p_feature(settings=None, **kwargs):
    """
    Draw Suite2p settings panel.

    Parameters
    ----------
    settings : Suite2pSettings, optional
        Settings instance. Creates default if not provided.
    **kwargs
        Additional arguments passed to draw_suite2p_settings_panel.

    Returns
    -------
    Suite2pSettings
        The (potentially modified) settings.
    """
    from mbo_utilities.gui.pipeline_widgets import (
        Suite2pSettings,
        draw_suite2p_settings_panel,
    )

    settings = settings or Suite2pSettings()
    return draw_suite2p_settings_panel(
        settings,
        show_header=kwargs.get("show_header", True),
        show_footer=kwargs.get("show_footer", True),
        **{k: v for k, v in kwargs.items() if k not in ("show_header", "show_footer")},
    )


@register_feature(
    name="save_options",
    title="Save Options",
    description="Configure processing and output options when saving data.",
    category="dialog",
    long_description=(
        "When exporting data to a new format, these options control how "
        "data is transformed. Includes processing options (phase correction, "
        "z-registration) and format-specific settings (Zarr sharding, OME metadata)."
    ),
    tip=(
        "Zarr with sharding is recommended for large datasets. "
        "OME-Zarr enables compatibility with viewers like napari."
    ),
    window_size=(520, 520),
)
def draw_save_options_feature(state=None, **kwargs):
    """
    Draw save options panel.

    Parameters
    ----------
    state : dict, optional
        State dictionary with option values.
    **kwargs
        show_header, show_footer control header/footer display.

    Returns
    -------
    dict
        The (potentially modified) state.
    """
    from imgui_bundle import imgui, hello_imgui

    state = state or {
        "overwrite": False,
        "register_z": False,
        "fix_phase": True,
        "subpixel": False,
        "debug": False,
        "chunk_mb": 100,
        "zarr_sharded": True,
        "zarr_ome": True,
        "zarr_level": 1,
    }

    show_header = kwargs.get("show_header", True)
    show_footer = kwargs.get("show_footer", True)

    if show_header:
        imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
        imgui.text_colored(
            imgui.ImVec4(0.7, 0.85, 1.0, 1.0),
            "Configure processing and output options when saving data to a new format. "
            "These options control how data is transformed during export.",
        )
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    # Processing Options section
    imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "Processing Options:")
    imgui.dummy(imgui.ImVec2(0, 4))

    _, state["overwrite"] = imgui.checkbox("Overwrite", state["overwrite"])
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Replace existing files")

    _, state["register_z"] = imgui.checkbox(
        "Register Z-Planes Axially", state["register_z"]
    )
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Align z-planes (suite3d)")

    _, state["fix_phase"] = imgui.checkbox("Fix Scan Phase", state["fix_phase"])
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Correct bidirectional offset")

    _, state["subpixel"] = imgui.checkbox(
        "Subpixel Phase Correction", state["subpixel"]
    )
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "FFT-based (slower)")

    _, state["debug"] = imgui.checkbox("Debug", state["debug"])
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Verbose output")

    imgui.spacing()

    imgui.text("Chunk Size (MB)")
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Memory per read/write")
    imgui.set_next_item_width(hello_imgui.em_size(12))
    _, state["chunk_mb"] = imgui.drag_int("##chunk", state["chunk_mb"], 1, 1, 1024)

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # Zarr Options section
    imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "Zarr Options:")
    imgui.dummy(imgui.ImVec2(0, 4))

    _, state["zarr_sharded"] = imgui.checkbox("Sharded", state["zarr_sharded"])
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Group chunks into shards")

    _, state["zarr_ome"] = imgui.checkbox("OME-Zarr", state["zarr_ome"])
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Write OME-NGFF metadata")

    imgui.text("Compression Level")
    imgui.same_line(hello_imgui.em_size(22))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "0=none, 9=max")
    imgui.set_next_item_width(hello_imgui.em_size(12))
    _, state["zarr_level"] = imgui.slider_int("##level", state["zarr_level"], 0, 9)

    if show_footer:
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
            "Tip: Zarr with sharding is recommended for large datasets. "
            "OME-Zarr enables compatibility with viewers like napari.",
        )
        imgui.pop_text_wrap_pos()

    return state


@register_feature(
    name="configurable_metadata",
    title="Configurable Metadata",
    description="Configure required and custom metadata fields when saving data.",
    category="dialog",
    long_description=(
        "When saving data, you can configure metadata that will be embedded "
        "in the output file. Required fields (like z-step size) must be set "
        "before saving. Custom fields let you add experiment-specific annotations."
    ),
    tip=(
        "Required fields like 'dz' (z-step size) are needed for proper 3D "
        "reconstruction. Custom fields let you add any experiment-specific annotations."
    ),
    window_size=(500, 520),
    requires_data=True,
)
def draw_configurable_metadata_feature(
    required_fields=None, custom_metadata=None, **kwargs
):
    """
    Draw configurable metadata panel.

    Parameters
    ----------
    required_fields : list[dict], optional
        List of required metadata field definitions.
    custom_metadata : dict, optional
        Dictionary of custom metadata key-value pairs.
    **kwargs
        show_header, show_footer control header/footer display.

    Returns
    -------
    tuple[list, dict]
        The required_fields and custom_metadata (potentially modified).
    """
    from imgui_bundle import imgui, hello_imgui

    required_fields = required_fields or []
    custom_metadata = custom_metadata or {
        "experiment_id": "exp_001",
        "subject": "mouse_42",
    }

    show_header = kwargs.get("show_header", True)
    show_footer = kwargs.get("show_footer", True)

    if show_header:
        imgui.push_text_wrap_pos(imgui.get_font_size() * 28.0)
        imgui.text_colored(
            imgui.ImVec4(0.7, 0.85, 1.0, 1.0),
            "When saving data, you can configure metadata that will be "
            "embedded in the output file. Required fields must be set "
            "before saving; custom fields are optional.",
        )
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    # Required metadata section
    if required_fields:
        imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "Required:")
        imgui.dummy(imgui.ImVec2(0, 4))

        for field in required_fields:
            canonical = field.get("canonical", "unknown")
            label = field.get("label", canonical)
            unit = field.get("unit", "")
            desc = field.get("description", "")
            value = field.get("value")

            is_set = value is not None
            color = (
                imgui.ImVec4(0.4, 0.9, 0.4, 1.0)
                if is_set
                else imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
            )

            imgui.text_colored(color, f"  {label}")
            imgui.same_line(hello_imgui.em_size(10))

            if is_set:
                imgui.text_colored(color, f"{value} {unit}")
            else:
                imgui.text_colored(color, "required")

            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 20.0)
                imgui.text_unformatted(desc)
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()

            imgui.same_line(hello_imgui.em_size(22))
            imgui.set_next_item_width(hello_imgui.em_size(6))
            imgui.input_text(f"##{canonical}_input", "")
            imgui.same_line()
            imgui.small_button(f"Set##{canonical}")

        imgui.spacing()
    else:
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
            "No required metadata for this data type.",
        )
        imgui.spacing()

    imgui.separator()
    imgui.spacing()

    # Custom metadata section
    imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "Custom:")
    imgui.dummy(imgui.ImVec2(0, 4))

    for key, value in custom_metadata.items():
        imgui.text(f"  {key}:")
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), str(value))
        imgui.same_line()
        imgui.small_button(f"X##{key}")

    imgui.spacing()
    imgui.set_next_item_width(hello_imgui.em_size(10))
    imgui.input_text("##custom_key", "new_field")
    imgui.same_line()
    imgui.text("=")
    imgui.same_line()
    imgui.set_next_item_width(hello_imgui.em_size(12))
    imgui.input_text("##custom_value", "value")
    imgui.same_line()
    imgui.button("Add")

    if show_footer:
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 28.0)
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
            "Tip: Required fields like 'dz' (z-step size) are needed for "
            "proper 3D reconstruction. Custom fields let you add any "
            "experiment-specific annotations.",
        )
        imgui.pop_text_wrap_pos()

    return required_fields, custom_metadata


def capture_feature(
    feature: GUIFeature,
    output_dir: Path,
    **kwargs,
) -> Optional[Path]:
    """
    Capture a screenshot of a single feature.

    Parameters
    ----------
    feature : GUIFeature
        The feature to capture.
    output_dir : Path
        Directory for output images.
    **kwargs
        Additional arguments passed to the feature's draw function.

    Returns
    -------
    Path or None
        Path to saved image, or None if capture failed.
    """
    import time
    from PIL import Image
    from imgui_bundle import imgui, immapp, hello_imgui

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / feature.image_filename

    print(f"Capturing {feature.title} -> {output_path}")

    start_time = time.time()

    def gui_callback():
        imgui.set_next_window_pos(imgui.ImVec2(20, 20), imgui.Cond_.once)
        imgui.set_next_window_size(
            imgui.ImVec2(feature.window_size[0] - 40, 0), imgui.Cond_.once
        )

        if imgui.begin(feature.title, None, imgui.WindowFlags_.always_auto_resize):
            feature.draw_func(**kwargs)
        imgui.end()

    def post_draw():
        if hello_imgui.get_runner_params().app_shall_exit:
            return
        if time.time() - start_time > 2.5:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = f"MBO - {feature.title}"
    params.app_window_params.window_geometry.size = feature.window_size
    params.app_window_params.window_geometry.size_auto = False
    params.app_window_params.resizable = False
    params.fps_idling.enable_idling = False
    params.callbacks.show_gui = gui_callback
    params.callbacks.pre_new_frame = post_draw

    addons = immapp.AddOnsParams()
    addons.with_markdown = True

    immapp.run(params, addons)

    screenshot = hello_imgui.final_app_window_screenshot()

    if screenshot is not None and screenshot.size > 0:
        img = Image.fromarray(screenshot)
        # Apply styling (shadow, padding) if desired
        img.save(output_path, "PNG")
        return output_path
    else:
        print(f"Failed to capture {feature.title}")
        return None


def capture_all_features(
    output_dir: str | Path = "docs/_images/gui/features",
    categories: Optional[list[str]] = None,
    skip_requires_data: bool = True,
) -> dict[str, Path]:
    """
    Capture screenshots of all registered features.

    Parameters
    ----------
    output_dir : str or Path
        Directory for output images.
    categories : list[str], optional
        Only capture features in these categories. None = all.
    skip_requires_data : bool
        Skip features that require loaded data.

    Returns
    -------
    dict[str, Path]
        Map of feature name to output path.
    """
    output_dir = Path(output_dir)
    results = {}

    for name, feature in _FEATURE_REGISTRY.items():
        if categories and feature.category not in categories:
            continue
        if skip_requires_data and feature.requires_data:
            print(f"Skipping {feature.title} (requires data)")
            continue

        path = capture_feature(feature, output_dir)
        if path:
            results[name] = path

    return results


# =============================================================================
# Documentation generation
# =============================================================================


def generate_feature_docs(
    output_path: str | Path = "docs/gui_features.md",
    image_base_path: str = "/_images/gui/features",
) -> str:
    """
    Generate markdown documentation from registered features.

    Parameters
    ----------
    output_path : str or Path
        Output file path for generated markdown.
    image_base_path : str
        Base path for images in the generated docs.

    Returns
    -------
    str
        The generated markdown content.
    """
    output_path = Path(output_path)

    lines = [
        "# GUI Features",
        "",
        "This page documents the configurable features available in the MBO GUI.",
        "",
    ]

    # Group by category
    categories = {}
    for feature in _FEATURE_REGISTRY.values():
        if feature.category not in categories:
            categories[feature.category] = []
        categories[feature.category].append(feature)

    category_titles = {
        "pipeline": "Processing Pipelines",
        "dialog": "Dialogs",
        "viewer": "Viewer Features",
        "widget": "Widgets",
    }

    for category, features in sorted(categories.items()):
        title = category_titles.get(category, category.title())
        lines.append(f"## {title}")
        lines.append("")

        for feature in features:
            lines.append(f"### {feature.title}")
            lines.append("")
            lines.append(feature.long_description)
            lines.append("")
            lines.append(f"```{{image}} {image_base_path}/{feature.image_filename}")
            lines.append(":width: 80%")
            lines.append(":align: center")
            lines.append("```")
            lines.append("")
            if feature.tip:
                lines.append(f":::{tip}")
                lines.append(feature.tip)
                lines.append(":::")
                lines.append("")

    content = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    return content
