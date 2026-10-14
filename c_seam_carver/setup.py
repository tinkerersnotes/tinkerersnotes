from pathlib import Path

from setuptools import Extension, setup

# The C++ core is shared by all native modules, only the Python binding differs
cpp_core_dir = Path('..', 'seam_carver_core')

setup(
    name='Seam carver Python C API demo app',
    ext_modules=[
        Extension(
            'c_seam_carver',
            sources=['c_seam_carver_module.cpp', str(cpp_core_dir / 'seam_carver_impl.cpp')],
            include_dirs=[str(cpp_core_dir)],
            language='c++',
            # Partially initialized PyTypeObject / PyModuleDef structs are the C API norm
            extra_compile_args=['-O3', '-std=c++17', '-Wall', '-Wextra', '-Wno-missing-field-initializers',
                                '-fvisibility=hidden'],
        ),
    ],
)
