#!/usr/bin/env python
"""Test if we can use NVIDIA GPU with a different rendering path."""
import os
os.environ['DISPLAY'] = ':17'
os.environ['WGPU_BACKEND_TYPE'] = 'Vulkan'

import wgpu
import numpy as np

# Get NVIDIA adapter
adapters = wgpu.gpu.enumerate_adapters_sync()
nvidia_adapters = [a for a in adapters if 'NVIDIA' in str(a.info.get('device')) and a.info.get('backend_type') == 'Vulkan']

if not nvidia_adapters:
    print("No NVIDIA Vulkan adapters found!")
    exit(1)

adapter = nvidia_adapters[0]
print(f"Using: {adapter.info.get('device')}")

# Create device
device = adapter.request_device_sync()
print("Device created!")

# Try using pygfx directly with explicit device
print("\nTrying pygfx with explicit device...")
try:
    import pygfx as gfx
    from rendercanvas.glfw import GlfwRenderCanvas

    canvas = GlfwRenderCanvas(size=(800, 600), title="NVIDIA Test")
    renderer = gfx.WgpuRenderer(canvas, device=device)

    scene = gfx.Scene()

    # Create simple image
    data = np.random.rand(512, 512).astype(np.float32)
    image = gfx.Image(
        gfx.Geometry(grid=gfx.Texture(data, dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 1)),
    )
    scene.add(image)

    camera = gfx.OrthographicCamera(512, 512)
    camera.local.position = (256, 256, 0)

    print("Rendering...")
    renderer.render(scene, camera)

    print("\nSUCCESS! Window should be visible.")
    print("Close the window to continue...")

    canvas.request_draw(lambda: renderer.render(scene, camera))

    # Run event loop
    from rendercanvas import loop
    loop.run()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
