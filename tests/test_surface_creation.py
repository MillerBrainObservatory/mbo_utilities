#!/usr/bin/env python
import os
# Force OpenGL backend like napari uses (works over remote desktop)
os.environ['WGPU_BACKEND_TYPE'] = 'OpenGL'
os.environ['RENDERCANVAS_BACKEND'] = 'glfw'

import numpy as np
import fastplotlib as fpl

data = np.random.rand(10, 512, 512).astype(np.float32)
np.save("test_data.npy", data)

data = np.load("test_data.npy")
iw = fpl.ImageWidget(data=data, histogram_widget=True)
iw.show()
fpl.loop.run()
