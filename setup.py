#!/usr/bin/env python3

import setuptools
import versioneer
from pathlib import Path

install_deps = [
    "tifffile",
    "scikit-image",
    "fastplotlib @ git+https://github.com/fastplotlib/fastplotlib.git@main",
    "pygfx @ git+https://github.com/pygfx/pygfx.git@main",
    "ffmpeg-python",
    "PyWavelets",
    "qtpy",
    "pyqt5",
    "dask",
    "zarr",
    "tqdm",
    "fabric",
]

with open(Path(__file__).parent / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mbo_utilities",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Various utilities for the Miller Brain Observatory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Flynn OConnell",
    author_email="FlynnOConnell@gmail.com",
    license="",
    url="https://github.com/millerbrainobservatory/mbo_utilities",
    keywords="Conda Microscopy ScanImage multiROI Tiff Slurm",
    install_requires=install_deps,
    packages=setuptools.find_packages(exclude=["data", "data.*"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
    ],
    entry_points={
        "console_scripts": [
            "mbo = mbo_utilities.__main__:main",
            "assemble = mbo_utilities.assembly:main",
        ]
    },
)

