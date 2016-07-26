#!/usr/local/bin/python
# -*- coding: utf-8 -*-

#  Copyright (c) 2016 Tjeerd Fokkens, Andreas Fognini, Val Zwiller
#  Licensed MIT: http://opensource.org/licenses/MIT

import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Tomography",
    version = "0.0.1",
    author = "Tjeerd Fokkens, Andreas Fognini, Val Zwiller",
    author_email = "a.w.fognini@tudelft.nl",
    description = ("Tomography installer. "),
    license = "MIT",
    keywords = "tomography density matrix quantum information entanglement",
    py_modules =["Tomography"],
    setup_requires=[],
    install_requires=['numpy','scipy>=0.17'],
    long_description=read('README.txt'),
    #test_suite="tests",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
)
