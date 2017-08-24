#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="rl",
    version="0.0.1",
    author="Evan Gui",
    author_email="evan176.gui@gmail.com",
    description=(""),
    keywords="reinforcement learning",
    packages=["rl"],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)
