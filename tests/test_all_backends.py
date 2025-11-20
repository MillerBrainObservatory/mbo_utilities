#!/usr/bin/env python
import subprocess
import sys

backends_wgpu = ['Vulkan', 'OpenGL', 'OpenGLES']
backends_canvas = ['glfw', 'qt', 'wx', 'offscreen']

test_code = """
import os
os.environ['WGPU_BACKEND_TYPE'] = '{wgpu}'
os.environ['RENDERCANVAS_BACKEND'] = '{canvas}'
import numpy as np
import fastplotlib as fpl
data = np.random.rand(5, 64, 64).astype(np.float32)
iw = fpl.ImageWidget(data=data, histogram_widget=False)
"""

print("Testing all backend combinations...\n")

for wgpu_backend in backends_wgpu:
    for canvas_backend in backends_canvas:
        code = test_code.format(wgpu=wgpu_backend, canvas=canvas_backend)
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print(f"✓ {wgpu_backend:10} + {canvas_backend:10}")
        else:
            error = result.stderr.split('\n')[-2] if result.stderr else 'unknown'
            error = error[:60]
            print(f"✗ {wgpu_backend:10} + {canvas_backend:10} {error}")
