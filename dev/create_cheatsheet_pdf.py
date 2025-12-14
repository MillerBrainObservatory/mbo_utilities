"""Generate mbo_utilities dark mode cheat sheet as PDF."""

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import platform

# Dark theme colors
DARK_BG = HexColor("#1a1a2e")
ACCENT_YELLOW = HexColor("#ffd700")
ACCENT_CYAN = HexColor("#00d4ff")
ACCENT_GREEN = HexColor("#4ec94e")
WHITE = HexColor("#ffffff")
LIGHT_GRAY = HexColor("#cccccc")
MED_GRAY = HexColor("#888888")
CODE_BG = HexColor("#2d2d3a")

# Try to register a monospace font
try:
    if platform.system() == "Windows":
        pdfmetrics.registerFont(TTFont("Consolas", "C:/Windows/Fonts/consola.ttf"))
        pdfmetrics.registerFont(TTFont("ConsolasBd", "C:/Windows/Fonts/consolab.ttf"))
        MONO_FONT = "Consolas"
        MONO_BOLD = "ConsolasBd"
    else:
        MONO_FONT = "Courier"
        MONO_BOLD = "Courier-Bold"
except:
    MONO_FONT = "Courier"
    MONO_BOLD = "Courier-Bold"


def draw_dark_background(c, width, height):
    """Fill page with dark background."""
    c.setFillColor(DARK_BG)
    c.rect(0, 0, width, height, fill=True, stroke=False)


def draw_title(c, text, x, y, size=28, color=ACCENT_YELLOW, bold=True):
    """Draw a title."""
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, text)


def draw_text(c, text, x, y, size=10, color=WHITE, font="Helvetica"):
    """Draw regular text."""
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, text)


def draw_code(c, text, x, y, size=9, color=WHITE):
    """Draw monospace code text."""
    c.setFont(MONO_FONT, size)
    c.setFillColor(color)
    c.drawString(x, y, text)


def draw_bullet(c, text, x, y, size=10, color=WHITE, bullet_color=ACCENT_CYAN):
    """Draw a bullet point."""
    c.setFont("Helvetica", size)
    c.setFillColor(bullet_color)
    c.drawString(x, y, "•")
    c.setFillColor(color)
    c.drawString(x + 12, y, text)


def draw_section_header(c, text, x, y, size=14, color=ACCENT_GREEN):
    """Draw a section header."""
    c.setFont("Helvetica-Bold", size)
    c.setFillColor(color)
    c.drawString(x, y, text)


def draw_code_block(c, lines, x, y, line_height=12, size=8):
    """Draw a code block with multiple lines."""
    for i, line in enumerate(lines):
        if line.startswith("#"):
            draw_code(c, line, x, y - i * line_height, size, ACCENT_GREEN)
        else:
            draw_code(c, line, x, y - i * line_height, size, WHITE)
    return y - len(lines) * line_height


def create_page1(c, width, height):
    """Page 1: Title + Python API Core I/O."""
    draw_dark_background(c, width, height)

    # Title section
    draw_title(c, "mbo_utilities", 0.5 * inch, height - 0.6 * inch, size=32)
    draw_text(c, "Cheat Sheet - Python API, CLI & GUI", 0.5 * inch, height - 0.95 * inch,
              size=14, color=LIGHT_GRAY)

    # Divider line
    c.setStrokeColor(ACCENT_YELLOW)
    c.setLineWidth(2)
    c.line(0.5 * inch, height - 1.1 * inch, width - 0.5 * inch, height - 1.1 * inch)

    # Left column: Core I/O
    col1_x = 0.5 * inch
    y = height - 1.5 * inch

    draw_section_header(c, "Python API: Core I/O", col1_x, y, size=16)
    y -= 0.35 * inch

    functions = [
        ("imread(path)", "Lazy-load any supported format"),
        ("imwrite(arr, path, ext)", "Stream-write to disk"),
        ("get_metadata(path)", "Extract file metadata dict"),
        ("get_voxel_size(path)", "Get physical dimensions (µm)"),
        ("get_files(path, **kw)", "Discover files with filtering"),
    ]

    for func, desc in functions:
        draw_code(c, func, col1_x, y, size=10, color=ACCENT_CYAN)
        y -= 14
        draw_text(c, desc, col1_x + 15, y, size=9, color=LIGHT_GRAY)
        y -= 18

    # Code examples
    y -= 0.15 * inch
    draw_section_header(c, "Examples", col1_x, y, size=12)
    y -= 0.25 * inch

    code1 = [
        "# Load data",
        "from mbo_utilities import imread",
        "arr = imread('/path/to/data.tiff')",
        "arr = imread('/path/to/raw/')  # ScanImage",
    ]
    y = draw_code_block(c, code1, col1_x, y, size=8)
    y -= 0.2 * inch

    code2 = [
        "# Write data",
        "from mbo_utilities import imwrite",
        "imwrite(arr, 'out.zarr', ext='.zarr')",
        "imwrite(arr, 'out/', planes=[0,1,2])",
    ]
    y = draw_code_block(c, code2, col1_x, y, size=8)
    y -= 0.2 * inch

    code3 = [
        "# Get metadata",
        "from mbo_utilities import get_metadata",
        "meta = get_metadata('/path/to/data')",
        "print(meta['nframes'], meta['shape'])",
    ]
    draw_code_block(c, code3, col1_x, y, size=8)

    # Right column: Utilities
    col2_x = 4.2 * inch
    y = height - 1.5 * inch

    draw_section_header(c, "Utilities & Visualization", col2_x, y, size=16)
    y -= 0.35 * inch

    utils = [
        ("save_mp4(fname, images, **kw)", "Export video from 3D array"),
        ("save_png(fname, data)", "Save image via matplotlib"),
        ("norm_minmax(images)", "Normalize to 0-1 range"),
        ("smooth_data(data, window)", "Temporal smoothing"),
        ("subsample_array(arr, factor)", "Downsample array"),
        ("files_to_dask(files)", "Build Dask array from files"),
        ("expand_paths(paths)", "Expand wildcards/lists"),
    ]

    for func, desc in utils:
        draw_code(c, func, col2_x, y, size=9, color=ACCENT_CYAN)
        y -= 14
        draw_text(c, desc, col2_x + 15, y, size=9, color=LIGHT_GRAY)
        y -= 16

    # Video example
    y -= 0.15 * inch
    draw_section_header(c, "Examples", col2_x, y, size=12)
    y -= 0.25 * inch

    code4 = [
        "# Video export",
        "from mbo_utilities import save_mp4",
        "save_mp4('movie.mp4', arr[:500],",
        "         framerate=30, temporal_avg=5)",
    ]
    y = draw_code_block(c, code4, col2_x, y, size=8)
    y -= 0.2 * inch

    code5 = [
        "# Dask arrays",
        "from mbo_utilities import files_to_dask",
        "darr = files_to_dask(tiff_files,",
        "                     chunk_t=250)",
    ]
    draw_code_block(c, code5, col2_x, y, size=8)


def create_page2(c, width, height):
    """Page 2: CLI Commands."""
    draw_dark_background(c, width, height)

    draw_title(c, "CLI Commands", 0.5 * inch, height - 0.6 * inch, size=24)

    # Commands table
    y = height - 1.1 * inch

    commands = [
        ("mbo view [PATH]", "Launch GUI viewer", "--roi 0,1  --widget  --metadata"),
        ("mbo convert IN OUT", "Convert formats", "-e .zarr  -p 0,1,2  --fix-phase  --register-z"),
        ("mbo info PATH", "Show file metadata", "--metadata"),
        ("mbo scanphase [PATH]", "Analyze scan phase", "-o output/  --format png  --show"),
        ("mbo formats", "List supported formats", ""),
        ("mbo --download-notebook", "Get user guide", "[PATH]"),
    ]

    for cmd, desc, opts in commands:
        draw_code(c, cmd, 0.5 * inch, y, size=11, color=ACCENT_CYAN)
        draw_text(c, desc, 2.8 * inch, y, size=10, color=WHITE)
        if opts:
            draw_code(c, opts, 5.0 * inch, y, size=9, color=LIGHT_GRAY)
        y -= 0.32 * inch

    # Quick examples
    y -= 0.2 * inch
    draw_section_header(c, "Quick Examples", 0.5 * inch, y, size=14)
    y -= 0.3 * inch

    examples = [
        "mbo /path/to/data.tiff                    # View TIFF in GUI",
        "mbo convert raw/ out.zarr -e .zarr        # Convert to Zarr",
        "mbo convert data.tiff out/ --fix-phase    # Fix bidirectional scan",
        "mbo view data/ --roi 0,1                  # View specific ROIs",
    ]

    for ex in examples:
        draw_code(c, ex, 0.6 * inch, y, size=9)
        y -= 0.22 * inch

    # Supported formats section
    y -= 0.3 * inch
    draw_title(c, "Supported Formats", 0.5 * inch, y, size=20)
    y -= 0.4 * inch

    # Input formats - left
    draw_section_header(c, "Input", 0.5 * inch, y, size=14)
    y_input = y - 0.25 * inch

    inputs = [
        (".tif, .tiff", "BigTIFF, OME-TIFF, ScanImage"),
        (".zarr", "Zarr v3, OME-NGFF"),
        (".h5, .hdf5", "HDF5 datasets"),
        (".bin", "Suite2p binary + ops.npy"),
        (".npy", "NumPy arrays"),
        (".nwb", "Neurodata Without Borders"),
        ("In-memory", "NumPy/Dask arrays"),
    ]

    for ext, desc in inputs:
        draw_code(c, ext, 0.6 * inch, y_input, size=10, color=ACCENT_CYAN)
        draw_text(c, desc, 1.7 * inch, y_input, size=9, color=LIGHT_GRAY)
        y_input -= 0.19 * inch

    # Output formats - right
    draw_section_header(c, "Output", 4.2 * inch, y, size=14)
    y_output = y - 0.25 * inch

    outputs = [
        (".tiff", "BigTIFF (streaming write)"),
        (".zarr", "Zarr v3 with OME metadata"),
        (".h5", "HDF5 with chunking"),
        (".bin", "Suite2p binary format"),
        (".npy", "NumPy array"),
        (".mp4", "Video export"),
    ]

    for ext, desc in outputs:
        draw_code(c, ext, 4.3 * inch, y_output, size=10, color=ACCENT_CYAN)
        draw_text(c, desc, 5.0 * inch, y_output, size=9, color=LIGHT_GRAY)
        y_output -= 0.19 * inch

    # Lazy array types
    y = min(y_input, y_output) - 0.2 * inch
    draw_section_header(c, "Lazy Array Types (returned by imread)", 0.5 * inch, y, size=12)
    y -= 0.2 * inch

    arrays = [
        "MboRawArray - Raw ScanImage multi-ROI with phase correction",
        "TiffArray / MBOTiffArray - Standard/Dask-backed TIFF (auto-detects volumes)",
        "Suite2pArray - Suite2p output (auto-detects volumes)",
        "ZarrArray - OME-Zarr / Zarr v3 stores",
        "H5Array - HDF5 datasets",
        "NumpyArray - .npy files or in-memory arrays",
        "BinArray - Direct binary file manipulation",
        "NWBArray - Neurodata Without Borders",
        "IsoviewArray - Lightsheet multi-view data",
    ]

    for arr in arrays:
        draw_bullet(c, arr, 0.6 * inch, y, size=8, color=LIGHT_GRAY)
        y -= 0.16 * inch


def create_page3(c, width, height):
    """Page 3: GUI Features."""
    draw_dark_background(c, width, height)

    draw_title(c, "GUI Features", 0.5 * inch, height - 0.6 * inch, size=24)

    # Preview & Visualization
    y = height - 1.0 * inch
    draw_section_header(c, "Preview & Visualization", 0.5 * inch, y, size=16, color=ACCENT_YELLOW)
    y -= 0.3 * inch

    preview_features = [
        "Image Viewer - FastPlotLib 2D/3D rendering with WGPU",
        "Frame Navigation - Time slider with playback controls",
        "Z-Plane Slider - Navigate through imaging planes",
        "Window Functions - mean, max, std, mean-subtracted",
        "Scan-Phase Correction - Fix bidirectional artifacts",
        "Contrast Controls - V-Min/V-Max adjustment",
        "Summary Stats - Per-plane mean, std, SNR tables",
    ]

    for feat in preview_features:
        draw_bullet(c, feat, 0.6 * inch, y, size=10)
        y -= 0.2 * inch

    # Processing & Export
    y -= 0.15 * inch
    draw_section_header(c, "Processing & Export", 0.5 * inch, y, size=16, color=ACCENT_YELLOW)
    y -= 0.3 * inch

    process_features = [
        "Spatial Crop - Select ROI region for processing",
        "Suite2p Pipeline - Integrated registration & cell detection",
        "Registration Settings - Rigid/non-rigid, 1P mode options",
        "Save As Dialog - Export to .tiff/.zarr/.h5/.bin",
        "Multi-ROI Support - Process ROIs separately or combined",
        "Suite3D Registration - Axial z-plane alignment",
        "Phase Correction - Automatic bidirectional scan fix",
    ]

    for feat in process_features:
        draw_bullet(c, feat, 0.6 * inch, y, size=10)
        y -= 0.2 * inch

    # ROI Diagnostics - right column
    col2_x = 4.5 * inch
    y = height - 1.0 * inch
    draw_section_header(c, "ROI Diagnostics (Suite2p Results)", col2_x, y, size=16, color=ACCENT_YELLOW)
    y -= 0.3 * inch

    diag_features = [
        "dF/F Traces - Adjustable baseline (median/percentile)",
        "Quality Metrics - SNR, skewness, activity histograms",
        "Filter Sliders - Interactive threshold adjustment",
        "Auto-save - Syncs iscell.npy to disk on change",
        "File Watching - Detects external modifications",
        "Suite2p Sync - Bi-directional with Suite2p GUI",
        "ROI Statistics - Detailed per-ROI information",
    ]

    for feat in diag_features:
        draw_bullet(c, feat, col2_x + 0.1 * inch, y, size=10)
        y -= 0.2 * inch

    # Launch commands
    y -= 0.3 * inch
    draw_section_header(c, "Launching the GUI", col2_x, y, size=14)
    y -= 0.25 * inch

    launch_code = [
        "# From command line",
        "mbo view /path/to/data",
        "mbo /path/to/data.tiff",
        "",
        "# From Python",
        "from mbo_utilities import run_gui",
        "run_gui('/path/to/data')",
        "run_gui()  # opens file dialog",
    ]

    for line in launch_code:
        if line.startswith("#"):
            draw_code(c, line, col2_x + 0.1 * inch, y, size=9, color=ACCENT_GREEN)
        elif line:
            draw_code(c, line, col2_x + 0.1 * inch, y, size=9, color=WHITE)
        y -= 0.16 * inch

    # Image placeholders note
    y = 0.6 * inch
    c.setStrokeColor(MED_GRAY)
    c.setLineWidth(1)
    c.setDash(3, 3)
    c.rect(0.5 * inch, 0.4 * inch, 3 * inch, 0.4 * inch, fill=False, stroke=True)
    c.setDash()
    draw_text(c, "See docs/_images/GUI_Slide1.png, GUI_Slide2.png",
              0.6 * inch, 0.52 * inch, size=9, color=MED_GRAY)


def create_cheatsheet_pdf():
    """Create the complete dark mode PDF cheat sheet."""
    out_path = Path(__file__).parent.parent / "cheat_sheet.pdf"

    width, height = landscape(letter)
    c = canvas.Canvas(str(out_path), pagesize=landscape(letter))

    # Page 1: Title + API
    create_page1(c, width, height)
    c.showPage()

    # Page 2: CLI + Formats
    create_page2(c, width, height)
    c.showPage()

    # Page 3: GUI Features
    create_page3(c, width, height)
    c.showPage()

    c.save()
    print(f"Created: {out_path}")
    return out_path


if __name__ == "__main__":
    create_cheatsheet_pdf()
