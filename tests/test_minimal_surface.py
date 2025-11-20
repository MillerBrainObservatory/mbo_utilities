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

    print("\nStep 5: Get wgpu context from canvas")
    present_context = canvas.get_wgpu_context()
    print(f"  Context: {present_context}")

    print("\nStep 6: Configure context")
    render_texture_format = present_context.get_preferred_format(adapter)
    present_context.configure(device=device, format=render_texture_format)
    print("  SUCCESS! Context configured!")

except Exception as e:
    print(f"\nFAILED at current step!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
