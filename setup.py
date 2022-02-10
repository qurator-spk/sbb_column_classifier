#!/usr/bin/env python3
from io import open
from setuptools import setup


with open("requirements.txt") as fp:
    install_requires = fp.read()

setup(
    name="sbb_column_classifier",
    packages=["sbb_column_classifier"],
    install_requires=install_requires,
    entry_points={"console_scripts": ["sbb_column_classifier=sbb_column_classifier.sbb_column_classifier:main"]},
)
