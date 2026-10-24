from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# The C++ core is shared by all native modules, only the Python binding differs
cpp_core_dir = Path('..', 'seam_carver_core')

# GCC and Clang. Partially initialized PyTypeObject / PyModuleDef structs are the C API norm.
# Hidden visibility lets the compiler inline the core's helpers (seam_carver_optimizations.md, step 2).
GCC_COMPILE_ARGS = ['-O3', '-std=c++17', '-Wall', '-Wextra', '-Wno-missing-field-initializers',
                    '-fvisibility=hidden']
# MSVC rejects -Wextra as an invalid /W level, and needs no visibility flag:
# a DLL exports only __declspec(dllexport) symbols, and calls inside it are never interposed.
MSVC_COMPILE_ARGS = ['/std:c++17']


class BuildExt(build_ext):
    """Picks the compile flags for the compiler build_ext chose, which setup() can't know yet."""

    def build_extensions(self):
        msvc = self.compiler.compiler_type == 'msvc'
        for extension in self.extensions:
            extension.extra_compile_args = MSVC_COMPILE_ARGS if msvc else GCC_COMPILE_ARGS
        super().build_extensions()


setup(
    name='Seam carver Python C API demo app',
    ext_modules=[
        Extension(
            'c_seam_carver',
            sources=['c_seam_carver_module.cpp', str(cpp_core_dir / 'seam_carver_impl.cpp')],
            include_dirs=[str(cpp_core_dir)],
            language='c++',
        ),
    ],
    cmdclass={'build_ext': BuildExt},
)
