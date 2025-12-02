"""
Backwards compatibility shim for lazy_array module.

This module re-exports imread and imwrite from their new locations
for backwards compatibility with existing code.

New code should import directly from:
- mbo_utilities.reader for imread
- mbo_utilities.writer for imwrite
"""

from mbo_utilities.reader import imread
from mbo_utilities.writer import imwrite

__all__ = ["imread", "imwrite"]
