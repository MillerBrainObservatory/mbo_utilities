#!/usr/bin/env python3

import setuptools
import versioneer
from pathlib import Path

install_deps = [
    "tifffile",
    "numpy",
    "scikit-image",
    "fastplotlib[notebook]",
    "scanreader @ git+https://github.com/atlab/scanreader.git@master#egg=scanreader",
    "PyWavelets",
    "zarr",
    "dask",
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
            "mbo_utilities = mbo_utilities.main:main",
        ]
    },
)

