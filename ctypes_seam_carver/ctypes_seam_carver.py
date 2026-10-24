"""Seam carver calling a C-style dynamic library through ctypes.

Same public interface as cpp_seam_carver.CppSeamCarver and
c_seam_carver.CSeamCarver. Build the library first with ./build.sh (build.ps1 on Windows).
"""
import contextlib
import ctypes
import operator
import sys
import threading
import weakref
from pathlib import Path

from PIL import Image

__all__ = ['CtypesSeamCarver']

_LIBRARY_SUFFIX = {'darwin': 'dylib', 'win32': 'dll'}.get(sys.platform, 'so')
_LIBRARY_PATH = Path(__file__).with_name(f'libseam_carver.{_LIBRARY_SUFFIX}')

# sc_status values from seam_carver_c_api.h, mapped like Cython's "except +"
_SC_OK = 0
_STATUS_EXCEPTIONS = {1: ValueError, 2: IndexError, 3: MemoryError, 4: RuntimeError}

_C_INT_MIN = -2**31
_C_INT_MAX = 2**31 - 1


class _SeamCarverStruct(ctypes.Structure):
    """Opaque sc_seam_carver, only ever used through a pointer."""


_Handle = ctypes.POINTER(_SeamCarverStruct)


def _check_status(status, func, args):
    if status != _SC_OK:
        # sc_last_error is thread local, and errcheck runs on the calling thread
        message = _lib.sc_last_error().decode(errors='replace')
        raise _STATUS_EXCEPTIONS.get(status, RuntimeError)(message)
    return status


def _load_library():
    try:
        lib = ctypes.CDLL(str(_LIBRARY_PATH))
    except OSError as e:
        raise ImportError(f'cannot load {_LIBRARY_PATH.name}, run build.sh (build.ps1 on Windows) first: {e}') from e

    # Without argtypes / restype ctypes assumes int, which truncates 64-bit pointers
    lib.sc_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(_Handle)]
    lib.sc_destroy.argtypes = [_Handle]
    lib.sc_destroy.restype = None
    lib.sc_shrink_horizontally.argtypes = [_Handle, ctypes.c_int, ctypes.c_int]
    lib.sc_width.argtypes = [_Handle]
    lib.sc_width.restype = ctypes.c_int
    lib.sc_height.argtypes = [_Handle]
    lib.sc_height.restype = ctypes.c_int
    lib.sc_copy_shrunk_rgb.argtypes = [_Handle, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
    lib.sc_last_error.argtypes = []
    lib.sc_last_error.restype = ctypes.c_char_p
    for function in (lib.sc_create, lib.sc_shrink_horizontally, lib.sc_copy_shrunk_rgb):
        function.restype = ctypes.c_int
        function.errcheck = _check_status
    return lib


_lib = _load_library()


class CtypesSeamCarver:
    # Construction happens exactly once in __new__, like in immutable built-in
    # types, so calling __init__ again can't replace the native object.
    def __new__(cls, image):
        self = super().__new__(cls)
        rgb_image = image if image.mode == 'RGB' else image.convert('RGB')
        width, height = rgb_image.size
        rgb = rgb_image.tobytes()
        handle = _Handle()
        # The C++ side copies the pixels into its mesh, rgb may be freed afterwards
        _lib.sc_create(rgb, len(rgb), width, height, ctypes.byref(handle))
        self._handle = handle
        # Destroys the native object once self is garbage, or at interpreter exit
        self._finalizer = weakref.finalize(self, _lib.sc_destroy, handle)
        # ctypes releases the GIL during every foreign call, so other threads
        # must not use the native object at the same time.
        self._lock = threading.Lock()
        return self

    # Intentionally a no-op. It lets subclasses call super().__init__(image),
    # which object.__init__ would reject once a subclass defines __init__.
    def __init__(self, image):
        pass

    @contextlib.contextmanager
    def _native(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('CtypesSeamCarver is in use by another thread')
        try:
            yield self._handle
        finally:
            self._lock.release()

    @property
    def width(self):
        with self._native() as handle:
            return _lib.sc_width(handle)

    @property
    def height(self):
        with self._native() as handle:
            return _lib.sc_height(handle)

    def shrink_horizontally(self, pixels_to_shrink, incremental_cost_updates=True):
        # ctypes silently truncates ints that don't fit into a C int
        pixels_to_shrink = operator.index(pixels_to_shrink)
        if not _C_INT_MIN <= pixels_to_shrink <= _C_INT_MAX:
            raise OverflowError('Python int too large to convert to C int')
        with self._native() as handle:
            _lib.sc_shrink_horizontally(handle, pixels_to_shrink, bool(incremental_cost_updates))

    def get_shrunk_image(self):
        with self._native() as handle:
            width = _lib.sc_width(handle)
            height = _lib.sc_height(handle)
            rgb = ctypes.create_string_buffer(width * height * 3)
            _lib.sc_copy_shrunk_rgb(handle, rgb, len(rgb))
        return Image.frombytes('RGB', (width, height), rgb.raw)
