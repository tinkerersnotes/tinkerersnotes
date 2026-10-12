from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Seam carver Cython demo app',
    ext_modules=cythonize("seam_carver.pyx"),
)
