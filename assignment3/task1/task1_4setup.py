# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("task1_4.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)