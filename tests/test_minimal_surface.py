#!/usr/bin/env python
"""Minimal test to find where surface creation fails"""
import os
os.environ['DISPLAY'] = ':17'

print("Step 1: Import wgpu")
import wgpu

print("\nStep 2: Get adapter")
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print(f"  Got: {adapter.info.get('device')}")

print("\nStep 3: Create device")
device = adapter.request_device_sync()
print("  Device created!")

print("\nStep 4: Try to create Qt canvas")
try:
    from qtpy import QtWidgets
    import sys
    from rendercanvas.qt import QRenderCanvas

    app = QtWidgets.QApplication(sys.argv)
    print("  Qt app created!")

    canvas = QRenderCanvas(size=(800, 600), title="Test")
    print("  Qt canvas created!")

    print("\nStep 5: Try to get wgpu surface from canvas")
    surface = canvas.get_surface()
    print(f"  Surface: {surface}")

    print("\nStep 6: Configure surface")
    # This is where it probably fails
    from wgpu import TextureFormat
    config = {
        "device": device,
        "format": TextureFormat.bgra8unorm,
        "usage": wgpu.TextureUsage.RENDER_ATTACHMENT,
        "width": 800,
        "height": 600,
    }
    surface.configure(**config)
    print("  SUCCESS! Surface configured!")

except Exception as e:
    print(f"\nFAILED at current step!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
