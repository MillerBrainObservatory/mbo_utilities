#!/usr/bin/env python
import os
# Use Qt canvas (works over remote desktop) with Vulkan (has features)
os.environ['WGPU_BACKEND_TYPE'] = 'Vulkan'
os.environ['RENDERCANVAS_BACKEND'] = 'qt'

import numpy as np
import fastplotlib as fpl

data = np.random.rand(10, 512, 512).astype(np.float32)
np.save("test_data.npy", data)

data = np.load("test_data.npy")
iw = fpl.ImageWidget(data=data, histogram_widget=True)
iw.show()
fpl.loop.run()
