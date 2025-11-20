#!/usr/bin/env python
import os
os.environ['WGPU_BACKEND_TYPE'] = 'Vulkan'
os.environ['RENDERCANVAS_BACKEND'] = 'qt'

import numpy as np
import wgpu
import fastplotlib as fpl

adapters = wgpu.gpu.enumerate_adapters_sync()
print(f"Found {len(adapters)} adapters\n")

for idx, adapter in enumerate(adapters):
    info = adapter.info
    print(f"[{idx}] {info.get('device')} ({info.get('backend_type')})")
    try:
        device = adapter.request_device_sync()
        limits = device.limits
        print(f"    max-texture-2d: {limits.get('max-texture-dimension-2d')}")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\nTesting ImageWidget...")
data = np.random.rand(10, 512, 512).astype(np.float32)
iw = fpl.ImageWidget(data=data, histogram_widget=True)
iw.show()
fpl.loop.run()
