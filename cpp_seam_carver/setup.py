from setuptools import setup
from Cython.Build import cythonize

# C++ language, sources and flags are set by "# distutils:" directives in the .pyx
setup(
    name='Seam carver Cython + C++ demo app',
    ext_modules=cythonize("cpp_seam_carver.pyx", annotate=True),
)
