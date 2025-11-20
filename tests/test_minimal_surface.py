import os
import wgpu

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print(f"  Request adapter sync: {adapter.info.get('device')}")

device = adapter.request_device_sync()
print("  Device created successfully: \n  {device}")

print("\nCreate Qt canvas")
try:
    from qtpy import QtWidgets
    import sys
    from rendercanvas.qt import QRenderCanvas

    print("Creating app")
    app = QtWidgets.QApplication(sys.argv)
    print("  Qt app created")

    print("Creating canvas")
    canvas = QRenderCanvas(size=(800, 600), title="Test")
    print("  Qt canvas created")

    print("Retrieve wgpu context")
    present_context = canvas.get_wgpu_context()
    print(f"  Context: {present_context}")

    print("\nConfigure context")
    render_texture_format = present_context.get_preferred_format(adapter)
    present_context.configure(device=device, format=render_texture_format)
    print("  Context configured")

except Exception as e:
    print(f"\nFailed with error: {e}")
    import traceback

    traceback.print_exc()
