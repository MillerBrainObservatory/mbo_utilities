#!/usr/bin/env python
"""
Wrapper to run mbo with CPU fallback if GPUs don't work.

Sets WGPU_ADAPTER_NAME to force llvmpipe if needed.
"""
import os
import sys
import subprocess

# Force llvmpipe (CPU renderer) via environment variable
# This is the correct way to select adapter before ANY wgpu initialization
os.environ["WGPU_ADAPTER_NAME"] = "llvmpipe"

# Now run mbo with remaining arguments
from mbo_utilities.graphics import run_gui
sys.exit(run_gui())
