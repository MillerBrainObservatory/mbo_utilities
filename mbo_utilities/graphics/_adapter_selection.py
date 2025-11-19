"""
GPU/CPU adapter selection for fastplotlib rendering.

This module MUST be imported before any fastplotlib imports to ensure
the correct adapter is selected before any GPU initialization happens.
"""

import subprocess
import sys
import os

_ADAPTER_SELECTED = False


def ensure_working_adapter():
    """
    Select a working GPU/CPU adapter before any rendering operations.

    Tests each available adapter (GPU and CPU) in priority order to find
    one that can create on-screen windows. Falls back to llvmpipe (CPU
    software renderer) if GPUs fail due to permission or driver issues.

    This function is idempotent - it only runs once per process.
    """
    global _ADAPTER_SELECTED

    if _ADAPTER_SELECTED:
        return

    try:
        import wgpu
        import fastplotlib as fpl

        adapters = wgpu.gpu.enumerate_adapters_sync()
        fpl_adapters = fpl.enumerate_adapters()

        # Try adapters in priority order: DiscreteGPU > IntegratedGPU > CPU
        priority = {"DiscreteGPU": 3, "IntegratedGPU": 2, "CPU": 1, "Unknown": 0}

        sorted_adapters = sorted(
            enumerate(adapters),
            key=lambda x: priority.get(x[1].info.get("adapter_type", "Unknown"), 0),
            reverse=True
        )

        for i, adapter in sorted_adapters:
            adapter_type = adapter.info.get("adapter_type", "Unknown")
            device_name = adapter.info.get("device", "Unknown")

            # Quick test: can this adapter create an on-screen window?
            # We run this in a subprocess because wgpu Rust panics can't be caught in Python
            test_code = f"""
import sys
try:
    import wgpu
    from rendercanvas.auto import RenderCanvas
    adapter = wgpu.gpu.enumerate_adapters_sync()[{i}]
    device = adapter.request_device_sync()
    canvas = RenderCanvas(size=(640, 480), title="GPU Test")
    context = canvas.get_wgpu_context()
    canvas_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=canvas_format)
    current_texture = context.get_current_texture()
    canvas.close()
    sys.exit(0)
except:
    sys.exit(1)
"""
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                timeout=5,
                env=os.environ.copy()
            )

            if result.returncode == 0:
                # This adapter works!
                print(f"✓ Selected rendering adapter: {device_name} ({adapter_type})")
                fpl.select_adapter(fpl_adapters[i])
                _ADAPTER_SELECTED = True
                return

        # If we get here, no adapter worked
        print("✗ Warning: No working GPU adapter found, using default")
        _ADAPTER_SELECTED = True

    except Exception as e:
        print(f"⚠ Warning: Adapter selection failed: {e}")
        print("  Continuing with default adapter...")
        _ADAPTER_SELECTED = True


# Auto-select adapter when this module is imported
ensure_working_adapter()
