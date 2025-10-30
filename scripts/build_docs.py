#!/usr/bin/env python
"""Build documentation by cleaning and rebuilding with Sphinx."""
import subprocess
import sys
from pathlib import Path


def main():
    """Clean and build the documentation."""
    docs_dir = Path(__file__).parent.parent / "docs"

    print("Cleaning documentation build...")
    result = subprocess.run(
        ["make", "clean"],
        cwd=docs_dir,
        capture_output=False,
    )

    if result.returncode != 0:
        print("Failed to clean docs")
        sys.exit(1)

    print("\nBuilding documentation...")
    result = subprocess.run(
        ["make", "html"],
        cwd=docs_dir,
        capture_output=False,
    )

    if result.returncode != 0:
        print("Failed to build docs")
        sys.exit(1)

    print("\n[SUCCESS] Documentation built successfully!")
    print(f"Open: {docs_dir / '_build' / 'html' / 'index.html'}")


if __name__ == "__main__":
    main()
