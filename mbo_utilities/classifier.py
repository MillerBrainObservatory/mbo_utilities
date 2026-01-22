"""
command-line entry point for integrated classifier.

usage:
    python -m mbo_utilities.classifier /path/to/plane0
    python -m mbo_utilities.classifier /path/to/stat.npy
"""

import sys
from pathlib import Path


def main():
    """launch integrated classifier from command line."""
    from mbo_utilities.gui.widgets.integrated_classifier import IntegratedClassifierWindow
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    plane_dir = None
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        # if user passed a file, use its parent directory
        if path.is_file():
            plane_dir = path.parent
        else:
            plane_dir = path

    window = IntegratedClassifierWindow(plane_dir)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
