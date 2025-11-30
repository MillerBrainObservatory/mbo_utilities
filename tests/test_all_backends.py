#!/usr/bin/env python
import subprocess
import sys

backends = [
    ('Vulkan', 'glfw'),
    ('Vulkan', 'qt'),
    ('Vulkan', 'offscreen'),
    ('OpenGL', 'glfw'),
    ('OpenGL', 'qt'),
]

test_code = """
import os
os.environ['WGPU_BACKEND_TYPE'] = '{wgpu}'
os.environ['RENDERCANVAS_BACKEND'] = '{canvas}'
import numpy as np
import fastplotlib as fpl
data = np.random.rand(5, 64, 64).astype(np.float32)
iw = fpl.ImageWidget(data=data, histogram_widget=False)
"""

for wgpu, canvas in backends:
    code = test_code.format(wgpu=wgpu, canvas=canvas)
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=5
    )

    combo = f"{wgpu:8} + {canvas:10}"

    if result.returncode == 0:
        print(f"✓ {combo}")
    else:
        errors = [l for l in result.stderr.split('\n')
                  if l.strip() and 'Error' in l or 'Caused by:' in l or 'panic' in l]
        error_msg = errors[-1][:70] if errors else 'unknown'
        print(f"✗ {combo} → {error_msg}")
