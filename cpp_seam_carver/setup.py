from setuptools import setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# C++ language, sources and include dirs are set by "# distutils:" directives in the .pyx.
# Compile flags are set here instead: they depend on the compiler, and a directive can't.
GCC_COMPILE_ARGS = ['-O3', '-std=c++17', '-fvisibility=hidden']  # GCC and Clang
# MSVC would only warn about each GCC flag, and needs no visibility flag:
# a DLL exports only __declspec(dllexport) symbols, and calls inside it are never interposed.
# /wd4551: Cython's own "(void) helper;" lines in the generated C++, which MSVC flags.
MSVC_COMPILE_ARGS = ['/std:c++17', '/wd4551']


class BuildExt(build_ext):
    """Picks the compile flags for the compiler build_ext chose, which setup() can't know yet."""

    def build_extensions(self):
        msvc = self.compiler.compiler_type == 'msvc'
        for extension in self.extensions:
            extension.extra_compile_args = MSVC_COMPILE_ARGS if msvc else GCC_COMPILE_ARGS
        super().build_extensions()


setup(
    name='Seam carver Cython + C++ demo app',
    ext_modules=cythonize("cpp_seam_carver.pyx", annotate=True),
    cmdclass={'build_ext': BuildExt},
)
