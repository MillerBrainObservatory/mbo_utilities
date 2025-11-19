"""
MBO Utilities - GPU/CPU Renderer Diagnostic and Setup Tool

This script diagnoses your GPU/rendering setup and automatically selects the best
available adapter for fastplotlib/wgpu rendering. It will work on:
- Systems with dedicated GPUs (Nvidia, AMD)
- Systems with integrated GPUs (Intel, AMD APU)
- Systems without GPU access (uses CPU software rendering via LavaPipe)

Usage:
    # Run diagnostics only
    uv run https://gist.githubusercontent.com/FlynnOConnell/431c1de03d22d082afaa42e2735a1a3e/raw/force_cpu_renderer.py --diagnose

    # Run with data file
    uv run https://gist.githubusercontent.com/FlynnOConnell/431c1de03d22d082afaa42e2735a1a3e/raw/force_cpu_renderer.py path/to/data.tif

    # Run file selector GUI
    uv run https://gist.githubusercontent.com/FlynnOConnell/431c1de03d22d082afaa42e2735a1a3e/raw/force_cpu_renderer.py
"""

import sys
import os
import subprocess
import pprint
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Print a formatted section header."""
    print(f"\n█ {text}")
    print("-" * 70)


def check_system_packages():
    """Check if required system packages are installed on Linux."""
    if sys.platform != "linux":
        return True, []

    print_section("Checking System Packages (Linux)")

    required_packages = {
        "mesa-vulkan-drivers": "Provides Vulkan drivers including LavaPipe (CPU renderer)",
        "libvulkan1": "Vulkan runtime library",
        "libegl1-mesa-dev": "EGL library for rendering context",
        "libgl1-mesa-dri": "Mesa DRI drivers",
    }

    optional_packages = {
        "libjpeg-turbo8-dev": "For optimal jupyter-rfb performance",
        "libturbojpeg0-dev": "For optimal jupyter-rfb performance",
    }

    missing_required = []
    missing_optional = []

    for pkg, desc in required_packages.items():
        result = subprocess.run(
            ["dpkg", "-l", pkg],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  ✗ {pkg}: NOT INSTALLED - {desc}")
            missing_required.append(pkg)
        else:
            print(f"  ✓ {pkg}: installed - {desc}")

    for pkg, desc in optional_packages.items():
        result = subprocess.run(
            ["dpkg", "-l", pkg],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  ⚠ {pkg}: not installed (optional) - {desc}")
            missing_optional.append(pkg)
        else:
            print(f"  ✓ {pkg}: installed - {desc}")

    if missing_required:
        print("\n⚠ Missing required packages!")
        print("\nTo install missing packages, run:")
        print(f"  sudo apt install {' '.join(missing_required)}")

    if missing_optional:
        print("\nTo install optional packages for better performance, run:")
        print(f"  sudo apt install {' '.join(missing_optional)}")

    return len(missing_required) == 0, missing_required


def check_environment_variables():
    """Check relevant environment variables."""
    print_section("Environment Variables")

    env_vars = {
        "DISPLAY": "X11 display server (required for on-screen windows)",
        "WAYLAND_DISPLAY": "Wayland display server",
        "WGPU_BACKEND_TYPE": "Override backend selection (Vulkan/Metal/D3D12/OpenGL)",
        "WGPU_LIB_PATH": "Custom path to wgpu-native library",
        "VK_ICD_FILENAMES": "Vulkan ICD (driver) to use",
        "LIBGL_ALWAYS_SOFTWARE": "Force software rendering",
        "MBO_DEBUG": "Enable debug logging in mbo_utilities",
    }

    any_set = False
    display_available = False

    for var, desc in env_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  {var}={value}")
            print(f"    → {desc}")
            any_set = True
            if var in ("DISPLAY", "WAYLAND_DISPLAY"):
                display_available = True

    if not any_set:
        print("  No relevant environment variables set")

    # Check for display access on Linux
    if sys.platform == "linux":
        if not display_available:
            print("\n  ⚠ WARNING: No DISPLAY environment variable set!")
            print("    This is a headless server - on-screen GUI windows will NOT work!")
            print("    Offscreen rendering (tested above) works fine.")
            print("\n    Solutions for on-screen GUIs:")
            print("    1. Use SSH with X11 forwarding: ssh -X user@host")
            print("    2. Set up virtual display:")
            print("       Xvfb :99 -screen 0 1024x768x24 &")
            print("       export DISPLAY=:99")
            print("    3. Use Jupyter notebook (no display needed)")
            print("    4. Use VNC or other remote desktop")

    return display_available


def enumerate_and_describe_adapters():
    """Enumerate all available GPU/CPU adapters and describe them."""
    print_section("Available Rendering Adapters")

    try:
        import wgpu
    except ImportError:
        print("  ✗ wgpu not installed. Install with: pip install wgpu")
        return None

    try:
        adapters = wgpu.gpu.enumerate_adapters_sync()
    except Exception as e:
        print(f"  ✗ Failed to enumerate adapters: {e}")
        return None

    if not adapters:
        print("  ✗ No adapters found!")
        return None

    print(f"\n  Found {len(adapters)} adapter(s):\n")

    adapter_list = []
    for i, adapter in enumerate(adapters):
        info = adapter.info
        adapter_type = info.get("adapter_type", "Unknown")
        backend = info.get("backend_type", "Unknown")
        device = info.get("device", "Unknown")
        vendor = info.get("vendor", "Unknown")
        description = info.get("description", "")

        # Store the index in the adapter info for subprocess testing
        info['_index'] = i

        print(f"  [{i}] {device}")
        print(f"      Vendor: {vendor}")
        print(f"      Type: {adapter_type}")
        print(f"      Backend: {backend}")
        if description:
            print(f"      Driver: {description}")

        # Explain what this adapter is
        if adapter_type == "DiscreteGPU":
            print(f"      ℹ This is a dedicated GPU card (best performance)")
        elif adapter_type == "IntegratedGPU":
            print(f"      ℹ This is an integrated GPU (good performance, lower power)")
        elif adapter_type == "CPU":
            print(f"      ℹ This is a software renderer running on CPU (limited performance)")
        elif "llvmpipe" in device.lower() or "lavapipe" in device.lower():
            print(f"      ℹ This is LavaPipe - CPU-based Vulkan software renderer")
        elif "swiftshader" in device.lower():
            print(f"      ℹ This is SwiftShader - CPU-based software renderer")

        print()
        adapter_list.append(adapter)

    return adapter_list


def test_adapter(adapter):
    """Test if an adapter actually works by creating a device, buffer, and rendering surface."""
    print_section(f"Testing Adapter: {adapter.info.get('device', 'Unknown')}")

    try:
        import wgpu

        # Try to create a device
        device = adapter.request_device_sync()
        print(f"  ✓ Device created successfully")

        # Try to create a small buffer (10 MB)
        buffer = device.create_buffer(
            size=10 * 2**20,
            usage=wgpu.BufferUsage.COPY_DST
        )
        print(f"  ✓ Buffer allocation successful (10 MB)")

        # CRITICAL: Test if we can create an ON-SCREEN window (this is what fails for GUI)
        # This causes a Rust panic, so we need to run in subprocess to catch it
        print(f"  Testing on-screen window creation...")

        test_code = f"""
import sys
try:
    import wgpu
    from rendercanvas.auto import RenderCanvas

    # Select the adapter by index
    adapter = wgpu.gpu.enumerate_adapters_sync()[{adapter.info.get('_index', 0)}]
    device = adapter.request_device_sync()

    # Create on-screen window
    canvas = RenderCanvas(size=(640, 480), title="GPU Test")

    # Configure surface - THIS IS WHERE IT FAILS
    context = canvas.get_wgpu_context()
    canvas_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=canvas_format)

    # Try to get texture - also can fail with queue family error
    current_texture = context.get_current_texture()
    texture_view = current_texture.create_view()

    canvas.close()
    print("SUCCESS")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy()
        )

        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print(f"  ✓ On-screen window test PASSED!")
            print(f"  ✓ Surface configuration successful")
            print(f"  ✓ Can acquire texture for on-screen rendering")
        else:
            print(f"  ✗ On-screen window test FAILED!")

            # Analyze the error
            error_output = result.stderr + result.stdout
            if "queue family" in error_output.lower():
                print(f"  ⚠ Surface doesn't support adapter's queue family")
                print(f"  ⚠ This means DRI2/GPU access permissions issue")
            if "DRI2" in error_output or "authenticate" in error_output:
                print(f"  ⚠ DRI2 authentication failed - GPU can't access display")

            print(f"\n  Suggested fixes:")
            print(f"    1. Install: sudo apt install libegl1-mesa-dev xserver-xorg-core")
            print(f"    2. Check GPU device permissions: ls -la /dev/dri/")
            print(f"    3. Add user to groups: sudo usermod -a -G video,render $USER")
            print(f"    4. Then logout and login again")
            print(f"\n  ⚠ This adapter CANNOT create on-screen windows!")
            print(f"  ⚠ Your GUI (uv run mbo) will NOT work with this adapter!")
            return False

        # Try to get device limits (may not be available on all wgpu versions)
        try:
            limits = device.limits
            if hasattr(limits, '__getitem__'):
                # Dict-like access
                max_buf = limits.get('max_buffer_size', None)
                max_tex = limits.get('max_texture_dimension_2d', None)
            else:
                # Attribute access
                max_buf = getattr(limits, 'max_buffer_size', None)
                max_tex = getattr(limits, 'max_texture_dimension_2d', None)

            if max_buf:
                print(f"  ✓ Max buffer size: {max_buf / 2**30:.2f} GB")
            if max_tex:
                print(f"  ✓ Max texture dimension 2D: {max_tex} px")
        except Exception:
            # Limits not critical for basic functionality
            pass

        return True

    except Exception as e:
        print(f"  ✗ Adapter test failed: {e}")
        return False


def select_best_adapter(adapters, prefer_cpu=False):
    """Select the best available adapter based on type and working status."""
    print_section("Selecting Best Adapter")

    if not adapters:
        print("  ✗ No adapters available!")
        return None

    # Priority order: DiscreteGPU > IntegratedGPU > CPU/Software
    priority = {
        "DiscreteGPU": 3,
        "IntegratedGPU": 2,
        "CPU": 1,
        "Unknown": 0,
    }

    if prefer_cpu:
        print("  ℹ CPU rendering requested, looking for software renderers...")
        # When CPU is preferred, reverse the priority
        candidates = [a for a in adapters
                     if a.info.get("adapter_type") == "CPU" or
                     "llvmpipe" in a.info.get("device", "").lower() or
                     "lavapipe" in a.info.get("device", "").lower() or
                     "swiftshader" in a.info.get("device", "").lower()]
    else:
        # Sort by priority, test each until we find one that works
        candidates = sorted(
            adapters,
            key=lambda a: priority.get(a.info.get("adapter_type", "Unknown"), 0),
            reverse=True
        )

    for adapter in candidates:
        info = adapter.info
        device = info.get("device", "Unknown")
        adapter_type = info.get("adapter_type", "Unknown")

        print(f"\n  Testing: {device} ({adapter_type})")

        if test_adapter(adapter):
            print(f"\n  ✓ Selected: {device}")
            print(f"    Type: {adapter_type}")
            print(f"    Backend: {info.get('backend_type', 'Unknown')}")
            return adapter

    print("\n  ✗ No working adapter found!")
    return None


def print_installation_help(missing_packages=None):
    """Print helpful installation instructions."""
    print_header("Installation Help")

    if sys.platform == "linux":
        print("\nFor Linux systems without GPU or with rendering issues:\n")

        if missing_packages:
            print("1. Install missing system packages:")
            print(f"   sudo apt install {' '.join(missing_packages)}")
            print()

        print("2. Common issue: 'DRI2: failed to authenticate' or 'Surface does not support queue family'")
        print("   This means the GPU can't create rendering surfaces for GUI windows.")
        print("   Solutions:")
        print("   a) Install missing EGL/DRI libraries:")
        print("      sudo apt install libegl1-mesa-dev libgl1-mesa-dri mesa-vulkan-drivers")
        print()
        print("   b) For headless servers (no X11), you need xserver-xorg-core:")
        print("      sudo apt install xserver-xorg-core")
        print()
        print("   c) Check your user has access to GPU devices:")
        print("      ls -la /dev/dri/")
        print("      # You should see card0, card1, renderD128, etc.")
        print("      # If permission denied, add yourself to video/render groups:")
        print("      sudo usermod -a -G video,render $USER")
        print("      # Then log out and back in")
        print()

        print("3. For CPU software rendering (LavaPipe) as fallback:")
        print("   sudo apt install mesa-vulkan-drivers libvulkan1")
        print()

        print("4. If you have a dedicated GPU, ensure vendor drivers are installed:")
        print("   - Nvidia: Check with 'nvidia-smi', update drivers if needed")
        print("   - AMD: sudo apt install mesa-vulkan-drivers firmware-amd-graphics")
        print()

        print("5. Environment variables to force software rendering (if GPU fails):")
        print("   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
        print("   export LIBGL_ALWAYS_SOFTWARE=1")
        print("   export __GLX_VENDOR_LIBRARY_NAME=mesa")

    elif sys.platform == "darwin":
        print("\nFor macOS:")
        print("  - macOS 10.13+ required for Metal/Vulkan support")
        print("  - Metal should be pre-installed")
        print("  - Update to latest macOS if having issues")

    elif sys.platform == "win32":
        print("\nFor Windows:")
        print("  - Windows 10+ recommended")
        print("  - Update GPU drivers from manufacturer:")
        print("    - Nvidia: https://www.nvidia.com/drivers")
        print("    - AMD: https://www.amd.com/support")
        print("  - Vulkan should be installed by default on Windows 11")


def run_diagnostics():
    """Run complete diagnostic suite."""
    print_header("MBO Utilities - GPU/Rendering Diagnostics")

    print(f"\nPlatform: {sys.platform}")
    print(f"Python: {sys.version}")

    # Check system packages (Linux only)
    packages_ok, missing = check_system_packages()

    # Check environment variables
    check_environment_variables()

    # Enumerate adapters
    adapters = enumerate_and_describe_adapters()

    if adapters:
        # Select best adapter
        best = select_best_adapter(adapters)

        if best:
            print_section("Recommendation")
            info = best.info
            adapter_type = info.get("adapter_type", "Unknown")

            if adapter_type == "DiscreteGPU":
                print("  ✓ You have a working dedicated GPU - excellent!")
                print("  ✓ fastplotlib should work with great performance")
            elif adapter_type == "IntegratedGPU":
                print("  ✓ You have a working integrated GPU - good!")
                print("  ✓ fastplotlib should work well for most use cases")
            elif adapter_type == "CPU" or "llvmpipe" in info.get("device", "").lower():
                print("  ✓ CPU software rendering available via LavaPipe")
                print("  ⚠ Performance will be limited - consider using a GPU if available")

            return True
        else:
            print_section("Issues Detected")
            print("  ✗ No working adapters found!")
            print_installation_help(missing if not packages_ok else None)
            return False
    else:
        print_section("Issues Detected")
        print("  ✗ Could not enumerate adapters!")
        print_installation_help(missing if not packages_ok else None)
        return False


def setup_and_run_gui(data_path=None):
    """Setup rendering and run mbo_utilities GUI."""
    print_header("Setting Up Rendering for MBO Utilities")

    # Import wgpu and fastplotlib
    try:
        import wgpu
        import fastplotlib as fpl
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("\n  Install with: pip install wgpu fastplotlib")
        return False

    # Enumerate adapters
    try:
        adapters = wgpu.gpu.enumerate_adapters_sync()
        print(f"  Found {len(adapters)} adapter(s)")
    except Exception as e:
        print(f"  ✗ Failed to enumerate adapters: {e}")
        return False

    # Convert to fastplotlib format
    try:
        fpl_adapters = fpl.enumerate_adapters()
    except Exception as e:
        print(f"  ✗ Failed to enumerate fastplotlib adapters: {e}")
        return False

    # Select adapter - prefer GPU, fall back to CPU
    selected = None

    # First try GPU adapters
    for i, adapter in enumerate(adapters):
        adapter_type = adapter.info.get("adapter_type", "Unknown")
        device = adapter.info.get("device", "Unknown")

        if adapter_type in ("DiscreteGPU", "IntegratedGPU"):
            print(f"  ✓ Selecting GPU: {device}")
            try:
                fpl.select_adapter(fpl_adapters[i])
                selected = fpl_adapters[i]
                break
            except Exception as e:
                print(f"  ✗ Failed to select {device}: {e}")

    # If no GPU worked, try CPU
    if not selected:
        for i, adapter in enumerate(adapters):
            adapter_type = adapter.info.get("adapter_type", "Unknown")
            device = adapter.info.get("device", "Unknown")

            if (adapter_type == "CPU" or
                "llvmpipe" in device.lower() or
                "lavapipe" in device.lower()):
                print(f"  ⚠ No GPU available, using CPU renderer: {device}")
                print(f"  ℹ Performance will be limited")
                try:
                    fpl.select_adapter(fpl_adapters[i])
                    selected = fpl_adapters[i]
                    break
                except Exception as e:
                    print(f"  ✗ Failed to select {device}: {e}")

    if not selected:
        print("  ✗ Could not select any working adapter!")
        return False

    # Now run the GUI
    print_header("Starting MBO Utilities GUI")

    try:
        from mbo_utilities.graphics import run_gui

        if data_path:
            print(f"  Loading data: {data_path}")
            run_gui(data_in=data_path)
        else:
            print(f"  Opening file selector...")
            run_gui()

    except Exception as e:
        print(f"  ✗ Failed to start GUI: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Main entry point."""
    args = sys.argv[1:]

    # Check for diagnostic mode
    if "--diagnose" in args or "--diagnostic" in args or "-d" in args:
        success = run_diagnostics()
        sys.exit(0 if success else 1)

    # Check for help
    if "--help" in args or "-h" in args:
        print(__doc__)
        print("\nOptions:")
        print("  --diagnose, -d    Run diagnostics only (don't start GUI)")
        print("  --help, -h        Show this help message")
        print("\nExamples:")
        print("  # Run diagnostics")
        print("  python force_cpu_renderer.py --diagnose")
        print()
        print("  # Open file selector")
        print("  python force_cpu_renderer.py")
        print()
        print("  # Load specific file")
        print("  python force_cpu_renderer.py path/to/data.tif")
        sys.exit(0)

    # Get data path if provided
    data_path = args[0] if args else None

    # Setup and run
    success = setup_and_run_gui(data_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
