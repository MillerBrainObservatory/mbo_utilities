"""fastplotlib config for every mbo figure, applied on import: no subplot
toolbar, a thin frame, no axes, and the image filling its subplot (auto_scale
fits the camera to the image's bounding box, so zoom 1 is the exact fill).
"""

from fastplotlib import Figure
from fastplotlib.layouts import Subplot
from fastplotlib.utils import global_config

FRAME_SPACING = {
    "x0": 1,
    "sides": 2,
    "title_flanks": 2,
    "resize_handle_space": 0,
    "bottom": 2,
}
TITLE_FONT_SIZE = 12

FPL_CONFIG = {
    Subplot: {
        "init": {
            "toolbar": False,
            "frame_kwargs": {
                "spacing": FRAME_SPACING,
                "title_kwargs": {"font_size": TITLE_FONT_SIZE},
            },
        },
        "auto_scale": {"zoom": 1.0},
    },
    Figure: {"show": {"axes_visible": False}},
}

for _cls, _methods in FPL_CONFIG.items():
    for _method, _options in _methods.items():
        global_config.update(getattr(_cls.config, _method), **_options)
