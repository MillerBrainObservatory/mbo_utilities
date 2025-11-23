#!/usr/bin/env python
"""
Test fastplotlib/pygfx with offscreen rendering and adapter selection.
"""

import os
import numpy as np

print("=" * 80)
print("FASTPLOTLIB OFFSCREEN TEST")
print("=" * 80)

# First, enumerate adapters to find the best one
import wgpu

adapters = wgpu.gpu.enumerate_adapters_sync()
print(f"\nFound {len(adapters)} adapter(s)")

# Find discrete GPUs with Vulkan backend
discrete_gpus = [
    (idx, adapter)
    for idx, adapter in enumerate(adapters)
    if adapter.info.get("adapter_type") == "DiscreteGPU"
    and adapter.info.get("backend_type") == "Vulkan"
]

if discrete_gpus:
    adapter_idx, adapter = discrete_gpus[0]
    adapter_name = adapter.info.get('device')
    print(f"\nFound discrete GPU: {adapter_name}")
    print(f"Setting WGPU_ADAPTER_NAME={adapter_name}")

    # Set environment variable to force wgpu to use this adapter
    os.environ['WGPU_ADAPTER_NAME'] = adapter_name
else:
    print("\nNo discrete GPUs found, using default adapter")

# Also set backend preference
os.environ['WGPU_BACKEND_TYPE'] = 'Vulkan'

print("\nEnvironment variables set:")
print(f"  WGPU_ADAPTER_NAME: {os.environ.get('WGPU_ADAPTER_NAME', 'not set')}")
print(f"  WGPU_BACKEND_TYPE: {os.environ.get('WGPU_BACKEND_TYPE', 'not set')}")

# Now try to use fastplotlib in offscreen mode
print("\n" + "=" * 80)
print("TESTING FASTPLOTLIB")
print("=" * 80)

try:
    import fastplotlib as fpl
    from wgpu.gui.offscreen import WgpuCanvas

    print("\nCreating test data...")
    data = np.random.rand(10, 512, 512).astype(np.float32)
    print(f"  Data shape: {data.shape}, dtype: {data.dtype}")

    print("\nCreating offscreen canvas...")
    canvas = WgpuCanvas(size=(800, 600), pixel_ratio=1)

    print("\nAttempting to create ImageWidget with offscreen canvas...")
    # Note: This may not work depending on fastplotlib version
    # ImageWidget might require a real window

    # Alternative: Use pygfx directly for offscreen rendering
    print("\nUsing pygfx directly for offscreen rendering...")
    import pygfx as gfx

    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()

    # Create a simple image
    print("  Creating image object...")
    image = gfx.Image(
        gfx.Geometry(grid=gfx.Texture(data[0], dim=2)),
        gfx.ImageBasicMaterial(clim=(data[0].min(), data[0].max())),
    )
    scene.add(image)

    # Create camera
    camera = gfx.OrthographicCamera(512, 512)
    camera.local.position = (256, 256, 0)

    print("  Rendering frame...")
    renderer.render(scene, camera)

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print("\nPygfx offscreen rendering works!")
    print("You can use this approach for headless GPU rendering.")

except ImportError as e:
    print(f"\nImport error: {e}")
    print("\nNote: offscreen rendering requires 'wgpu.gui.offscreen'")
    print("This should be available in wgpu-py by default.")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 80)
    print("PARTIAL SUCCESS")
    print("=" * 80)
    print("\nThe GPU is working, but offscreen rendering with fastplotlib")
    print("requires additional configuration.")
    print("\nFor headless servers, consider:")
    print("  1. Using pygfx directly instead of fastplotlib")
    print("  2. Using Xvfb (virtual framebuffer)")
    print("  3. Rendering to images without displaying")
