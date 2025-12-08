# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mbo-utilities[gui]",
# ]
# ///
"""
Launch the MBO Utilities GUI for viewing imaging data.

Usage:
    uv run https://gist.github.com/your-username/gist-id/raw/mbo.py
    uv run mbo.py
    uv run mbo.py /path/to/data
"""

from mbo_utilities.cli import main

if __name__ == "__main__":
    main()
