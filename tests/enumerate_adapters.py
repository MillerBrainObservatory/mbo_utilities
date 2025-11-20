#!/usr/bin/env python
"""
Simple script to enumerate and display information about all available WGPU adapters.
This helps diagnose GPU/rendering issues without attempting to create any surfaces.
"""

import wgpu
import pprint

print("=" * 80)
print("WGPU ADAPTER ENUMERATION")
print("=" * 80)

try:
    adapters = wgpu.gpu.enumerate_adapters_sync()
    print(f"\nFound {len(adapters)} adapter(s)\n")

    for idx, adapter in enumerate(adapters):
        print(f"\n{'=' * 80}")
        print(f"ADAPTER {idx}")
        print(f"{'=' * 80}")

        # Get adapter info
        try:
            info = adapter.request_adapter_info()
            print("\nFull Adapter Info:")
            pprint.pprint(info, indent=2, width=100)

            print(f"\nQuick Summary:")
            print(f"  Device      : {info.get('device', 'N/A')}")
            print(f"  Vendor      : {info.get('vendor', 'N/A')}")
            print(f"  Type        : {info.get('adapter_type', 'N/A')}")
            print(f"  Backend     : {info.get('backend_type', 'N/A')}")
            print(f"  Description : {info.get('description', 'N/A')}")
            print(f"  Vendor ID   : {info.get('vendor_id', 'N/A')}")
            print(f"  Device ID   : {info.get('device_id', 'N/A')}")

        except Exception as e:
            print(f"\n  ERROR getting adapter info: {e}")
            import traceback
            traceback.print_exc()

        # Try to request device and show limits
        try:
            print("\nRequesting device...")
            device = adapter.request_device()
            print("  Device requested successfully!")

            limits = device.limits
            print(f"\nDevice Limits:")
            print(f"  max_texture_dimension_1d      : {limits.get('max_texture_dimension_1d', 'N/A')}")
            print(f"  max_texture_dimension_2d      : {limits.get('max_texture_dimension_2d', 'N/A')}")
            print(f"  max_texture_dimension_3d      : {limits.get('max_texture_dimension_3d', 'N/A')}")
            print(f"  max_buffer_size               : {limits.get('max_buffer_size', 'N/A')}")
            print(f"  max_bindings_per_bind_group   : {limits.get('max_bindings_per_bind_group', 'N/A')}")
            print(f"  max_vertex_buffers            : {limits.get('max_vertex_buffers', 'N/A')}")

            # Try to get features
            try:
                features = device.features
                print(f"\nDevice Features ({len(features)} total):")
                for feature in sorted(features):
                    print(f"  - {feature}")
            except Exception as e:
                print(f"\n  WARNING: Could not get device features: {e}")

        except Exception as e:
            print(f"\n  ERROR requesting device: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ENUMERATION COMPLETE")
    print("=" * 80)

    # Identify discrete GPUs
    discrete_gpus = [
        (idx, a.request_adapter_info())
        for idx, a in enumerate(adapters)
        if a.request_adapter_info().get("adapter_type") == "DiscreteGPU"
    ]

    if discrete_gpus:
        print(f"\n\nFound {len(discrete_gpus)} discrete GPU(s):")
        for idx, info in discrete_gpus:
            print(f"  [{idx}] {info.get('device', 'Unknown')} ({info.get('backend_type', 'Unknown')})")
    else:
        print("\n\nNo discrete GPUs found.")

    # Suggest which adapter to use
    vulkan_discrete = [
        (idx, info)
        for idx, info in discrete_gpus
        if info.get("backend_type") == "Vulkan"
    ]

    if vulkan_discrete:
        print(f"\nRecommendation: Use adapter {vulkan_discrete[0][0]} (Discrete GPU with Vulkan backend)")
    elif discrete_gpus:
        print(f"\nRecommendation: Use adapter {discrete_gpus[0][0]} (Discrete GPU)")

except Exception as e:
    print(f"\nFATAL ERROR during enumeration: {e}")
    import traceback
    traceback.print_exc()
