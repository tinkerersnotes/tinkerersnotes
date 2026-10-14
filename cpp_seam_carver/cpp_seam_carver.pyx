# distutils: language = c++
# distutils: sources = ../seam_carver_core/seam_carver_impl.cpp
# distutils: include_dirs = ../seam_carver_core
# distutils: extra_compile_args = -O3 -std=c++17 -fvisibility=hidden
# cython: language_level=3

from libc.stdint cimport uint8_t
from libcpp cimport bool as cpp_bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from PIL import Image

cdef extern from "seam_carver_impl.hpp" namespace "seam_carving":
    cdef cppclass SeamCarver:
        SeamCarver(const uint8_t* rgb, int width, int height) except +
        void shrink_horizontally(int pixels_to_shrink, cpp_bool incremental_cost_updates) except + nogil
        int width()
        int height()
        vector[uint8_t] shrunk_rgb() except +


cdef class CppSeamCarver:
    cdef unique_ptr[SeamCarver] carver
    # Set while the GIL is released around native code, so other threads
    # can't use the carver at the same time.
    cdef bint busy

    # Construction happens exactly once in __cinit__, like in immutable built-in
    # types, so calling __init__ again can't replace the carver.
    def __cinit__(self, image):
        rgb_image = image if image.mode == 'RGB' else image.convert('RGB')
        width, height = rgb_image.size
        cdef bytes rgb = rgb_image.tobytes()
        cdef const unsigned char* rgb_data = rgb
        # The C++ side copies the pixels into its mesh, rgb may be freed afterwards
        self.carver.reset(new SeamCarver(rgb_data, width, height))

    # Intentionally a no-op. It lets subclasses call super().__init__(image),
    # which object.__init__ would reject once a subclass defines __init__.
    def __init__(self, image):
        pass

    cdef SeamCarver* checked_carver(self) except NULL:
        if self.busy:
            raise RuntimeError('CppSeamCarver is in use by another thread')
        return self.carver.get()

    @property
    def width(self):
        return self.checked_carver().width()

    @property
    def height(self):
        return self.checked_carver().height()

    def shrink_horizontally(self, int pixels_to_shrink, bint incremental_cost_updates=True):
        cdef SeamCarver* carver = self.checked_carver()
        self.busy = True
        try:
            with nogil:
                carver.shrink_horizontally(pixels_to_shrink, incremental_cost_updates)
        finally:
            self.busy = False

    def get_shrunk_image(self):
        cdef SeamCarver* carver = self.checked_carver()
        cdef vector[uint8_t] rgb = carver.shrunk_rgb()
        cdef bytes rgb_bytes = (<const char*> rgb.data())[:rgb.size()]
        return Image.frombytes('RGB', (carver.width(), carver.height()), rgb_bytes)
