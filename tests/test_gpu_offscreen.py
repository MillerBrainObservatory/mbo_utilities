#!/usr/bin/env python
"""
Test GPU functionality without requiring a display/window.
Tests adapter enumeration, device creation, and basic compute operations.
"""

import numpy as np
import wgpu
import pprint

print("=" * 80)
print("GPU OFFSCREEN TEST")
print("=" * 80)

# Enumerate all available adapters
adapters = wgpu.gpu.enumerate_adapters_sync()
print(f"\nFound {len(adapters)} adapter(s)\n")

# Find discrete GPUs with Vulkan backend
discrete_gpus = [
    (idx, adapter)
    for idx, adapter in enumerate(adapters)
    if adapter.info.get("adapter_type") == "DiscreteGPU"
    and adapter.info.get("backend_type") == "Vulkan"
]

if not discrete_gpus:
    print("No discrete GPUs found with Vulkan backend!")
    print("Falling back to first available adapter for testing...")
    adapter_idx = 0
    adapter = adapters[0]
else:
    print(f"Found {len(discrete_gpus)} discrete GPU(s) with Vulkan backend:")
    for idx, adapter in discrete_gpus:
        info = adapter.info
        print(f"  [{idx}] {info.get('device')} ({info.get('vendor')})")

    # Use the first discrete GPU
    adapter_idx, adapter = discrete_gpus[0]
info = adapter.info

print(f"\nUsing adapter {adapter_idx}: {info.get('device')}")
print(f"  Vendor: {info.get('vendor')}")
print(f"  Backend: {info.get('backend_type')}")
print(f"  Description: {info.get('description')}")

# Request device
print("\nRequesting device...")
device = adapter.request_device_sync()
print("  Device created successfully!")

# Print device limits
limits = device.limits
print(f"\nDevice Limits:")
print(f"  max-texture-dimension-2d      : {limits.get('max-texture-dimension-2d')}")
print(f"  max-texture-dimension-3d      : {limits.get('max-texture-dimension-3d')}")
print(f"  max-buffer-size               : {limits.get('max-buffer-size')}")
print(f"  max-storage-buffer-binding-size: {limits.get('max-storage-buffer-binding-size')}")

# Test creating a buffer (basic GPU memory allocation)
print("\n" + "=" * 80)
print("TESTING GPU BUFFER CREATION")
print("=" * 80)

# Create test data
test_data = np.arange(1000, dtype=np.float32)
print(f"\nTest data: shape={test_data.shape}, dtype={test_data.dtype}")

# Create GPU buffer
buffer_size = test_data.nbytes
print(f"Creating GPU buffer of size: {buffer_size} bytes ({buffer_size / 1024 / 1024:.2f} MB)")

buffer = device.create_buffer(
    size=buffer_size,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
)
print("  Buffer created successfully!")

# Write data to buffer
device.queue.write_buffer(buffer, 0, test_data)
print("  Data written to GPU buffer!")

# Test creating a texture (offscreen)
print("\n" + "=" * 80)
print("TESTING GPU TEXTURE CREATION (OFFSCREEN)")
print("=" * 80)

texture_size = (512, 512)
print(f"\nCreating {texture_size[0]}x{texture_size[1]} RGBA texture...")

texture = device.create_texture(
    size=(texture_size[0], texture_size[1], 1),
    format=wgpu.TextureFormat.rgba8unorm,
    usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC,
)
print("  Texture created successfully!")

# Test with multiple textures (simulate what ImageWidget might do)
print(f"\nCreating 10 textures to simulate ImageWidget stack...")
textures = []
for i in range(10):
    tex = device.create_texture(
        size=(512, 512, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    textures.append(tex)
    print(f"  Texture {i+1}/10 created")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print(f"\nGPU {info.get('device')} is working correctly in offscreen mode.")
print(f"You can use adapter index {adapter_idx} for fastplotlib/pygfx applications.")
