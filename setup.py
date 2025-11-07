#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:23:09 2020

@author: Vicky

Python Package Setup - PyTorch Version
"""
import io
from os import path
from setuptools import setup
from setuptools import find_packages

with io.open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

setup(
    name='torch-pde',
    version='0.651dev',
    description="Deep learning library for solving partial differential equations using PyTorch",
    author="Vignesh Gopakumar",
    author_email="vignesh7g@gmail.com",
    url = "https://github.com/gitvicky/torch-pde",
    packages = find_packages(),
    license='MIT',
    long_description = long_desc,
    long_description_content_type = "text/markdown",
    install_requires = ['numpy>=1.18.5',
                        'matplotlib>=3.2.1',
                        'scipy>=1.4.1',
                        'sympy>=1.6',
                        'torch>=1.9.0',
                        'pydoe>=0.3.8',
                        'cloudpickle>=1.4.1',
    ],
)
