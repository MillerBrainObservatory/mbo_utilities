#!/usr/bin/env python
"""Test NVIDIA GPU rendering with offscreen canvas then display via Qt/OpenGL"""
import os
os.environ['DISPLAY'] = ':17'

import numpy as np
import wgpu

# Get NVIDIA Vulkan adapter (has all features)
adapters = wgpu.gpu.enumerate_adapters_sync()
nvidia = [a for a in adapters if 'NVIDIA' in a.info.get('device', '') and a.info.get('backend_type') == 'Vulkan'][0]

print(f"Using: {nvidia.info.get('device')}")

# Create device with required features
device = nvidia.request_device_sync(required_features=['float32-filterable'])
print("Device created with FLOAT32_FILTERABLE support!")

# Now try fastplotlib with this specific device
print("\nTrying fastplotlib with NVIDIA device...")
try:
    import fastplotlib as fpl

    # Monkey-patch pygfx to use our device
    import pygfx.renderers.wgpu.engine.shared as shared_module

    # Create a custom shared instance with our device
    class CustomShared(shared_module.Shared):
        def __init__(self):
            self.adapter = nvidia
            self._device = device
            # Skip the rest of init that would try to create its own device

    # Replace the global shared instance
    shared_module._the_shared = CustomShared()

    data = np.random.rand(10, 512, 512).astype(np.float32)

    iw = fpl.ImageWidget(data=data, histogram_widget=True)
    print("ImageWidget created!")
    iw.show()

    print("\nSUCCESS! Window should be visible.")
    fpl.loop.run()

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
