#!/usr/bin/env python
import numpy as np
import wgpu
import pprint

print("=" * 80)
print("ENUMERATING WGPU ADAPTERS")
print("=" * 80)

# Enumerate all available adapters
adapters = wgpu.gpu.enumerate_adapters_sync()

print(f"\nFound {len(adapters)} adapter(s):\n")

for idx, adapter in enumerate(adapters):
    print(f"\n{'=' * 80}")
    print(f"ADAPTER {idx}")
    print(f"{'=' * 80}")

    # Get adapter info
    info = adapter.info

    print("\nAdapter Info:")
    pprint.pprint(info, indent=2, width=100)

    # Get a more readable summary
    print(f"\nSummary:")
    print(f"  Device: {info.get('device', 'N/A')}")
    print(f"  Vendor: {info.get('vendor', 'N/A')}")
    print(f"  Type: {info.get('adapter_type', 'N/A')}")
    print(f"  Backend: {info.get('backend_type', 'N/A')}")
    print(f"  Description: {info.get('description', 'N/A')}")
    print(f"  Vendor ID: {info.get('vendor_id', 'N/A')}")
    print(f"  Device ID: {info.get('device_id', 'N/A')}")

    # Try to get device and show limits
    try:
        device = adapter.request_device_sync()
        limits = device.limits
        print(f"\nDevice Limits (subset):")
        print(f"  max-texture-dimension-2d: {limits.get('max-texture-dimension-2d', 'N/A')}")
        print(f"  max-buffer-size: {limits.get('max-buffer-size', 'N/A')}")
        print(f"  max-bindings-per-bind-group: {limits.get('max-bindings-per-bind-group', 'N/A')}")
    except Exception as e:
        print(f"\n  WARNING: Could not request device: {e}")

print("\n" + "=" * 80)
print("ADAPTER ENUMERATION COMPLETE")
print("=" * 80)

# Now try to use fastplotlib with default adapter
print("\n\nAttempting to create ImageWidget with default adapter...")
try:
    import fastplotlib as fpl

    data = np.random.rand(10, 512, 512).astype(np.float32)
    np.save("test_data.npy", data)

    data = np.load("test_data.npy")
    print(f"Data shape: {data.shape}, dtype: {data.dtype}")

    iw = fpl.ImageWidget(data=data, histogram_widget=True)
    iw.show()
    fpl.loop.run()
except Exception as e:
    print(f"\nERROR creating ImageWidget: {e}")
    import traceback
    traceback.print_exc()
