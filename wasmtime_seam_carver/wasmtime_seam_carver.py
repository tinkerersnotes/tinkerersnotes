"""Seam carver running the C++ core as WebAssembly, in the Wasmtime runtime.

Same public interface as ctypes_seam_carver.CtypesSeamCarver, and the same C API
(../ctypes_seam_carver/seam_carver_c_api.h), compiled by Emscripten into seam_carver.wasm
instead of a native library. Build it first with ./build.sh (build.ps1 on Windows), which
installs the Emscripten it needs on first use.

Each carver runs in its own Wasmtime store: a sandbox with its own linear memory, where the
module's malloc holds the mesh. Python only reaches into that memory to copy the pixels in and
out. The store, and with it everything the carver allocated, goes away with the carver.
"""
import contextlib
import ctypes
import operator
import threading
from pathlib import Path

import wasmtime
from PIL import Image

__all__ = ['WasmtimeSeamCarver']

_MODULE_PATH = Path(__file__).with_name('seam_carver.wasm')

# sc_status values from seam_carver_c_api.h, mapped like Cython's "except +"
_SC_OK = 0
_STATUS_EXCEPTIONS = {1: ValueError, 2: IndexError, 3: MemoryError, 4: RuntimeError}

# WebAssembly i32 parameters; build.sh caps the memory at 2 GB, so addresses and sizes fit too
_I32_MIN = -2**31
_I32_MAX = 2**31 - 1


def _load_module():
    """Compiles seam_carver.wasm once per process, with its imports resolved."""
    if not _MODULE_PATH.exists():
        raise ImportError(f'{_MODULE_PATH.name} is missing, run build.sh (build.ps1 on Windows) first')
    config = wasmtime.Config()
    config.wasm_exceptions = True  # the C API catches the core's C++ exceptions inside the module
    engine = wasmtime.Engine(config)
    module = wasmtime.Module.from_file(engine, str(_MODULE_PATH))
    linker = wasmtime.Linker(engine)
    # Emscripten's standalone runtime writes messages through WASI, for instance before an abort
    linker.define_wasi()
    # And reports memory growth, which matters only to callers that cache views of the memory
    linker.define_func('env', 'emscripten_notify_memory_growth',
                       wasmtime.FuncType([wasmtime.ValType.i32()], []), lambda index: None)
    return engine, linker.instantiate_pre(module)


_engine, _instance_pre = _load_module()


class _Module:
    """One instance of seam_carver.wasm in a store of its own, and its exports."""

    def __init__(self):
        self.store = wasmtime.Store(_engine)
        wasi = wasmtime.WasiConfig()
        wasi.inherit_stderr()
        self.store.set_wasi(wasi)
        exports = _instance_pre.instantiate(self.store).exports(self.store)
        exports['_initialize'](self.store)  # a library module's static constructors
        self.memory = exports['memory']
        self.functions = {name: exports[name] for name in (
            'sc_create', 'sc_shrink_horizontally', 'sc_width', 'sc_height', 'sc_copy_shrunk_rgb',
            'sc_last_error', 'malloc', 'free')}

    def call(self, name, *args):
        return self.functions[name](self.store, *args)

    def check(self, status):
        if status != _SC_OK:
            message = self.c_string(self.call('sc_last_error'))
            raise _STATUS_EXCEPTIONS.get(status, RuntimeError)(message)

    def malloc(self, size):
        address = self.call('malloc', size)
        if address == 0:
            raise MemoryError(f'cannot allocate {size} bytes in WebAssembly memory')
        return address

    def _host_address(self, address, size):
        # Where the module's address lies in this process. Valid until the module's memory grows,
        # which only happens during a call into it.
        if not 0 <= address <= address + size <= self.memory.data_len(self.store):
            raise IndexError('outside the WebAssembly memory')
        return ctypes.addressof(self.memory.data_ptr(self.store).contents) + address

    def write(self, address, data):
        ctypes.memmove(self._host_address(address, len(data)), data, len(data))

    def read(self, address, size):
        return ctypes.string_at(self._host_address(address, size), size)

    def c_string(self, address, limit=4096):
        size = min(limit, self.memory.data_len(self.store) - address)
        return self.read(address, size).split(b'\0', 1)[0].decode(errors='replace')


class WasmtimeSeamCarver:
    # Construction happens exactly once in __new__, like in immutable built-in
    # types, so calling __init__ again can't replace the WebAssembly object.
    def __new__(cls, image):
        self = super().__new__(cls)
        rgb_image = image if image.mode == 'RGB' else image.convert('RGB')
        width, height = rgb_image.size
        rgb = rgb_image.tobytes()
        if len(rgb) > _I32_MAX:
            raise MemoryError("the image doesn't fit into WebAssembly memory")
        module = _Module()
        rgb_address = module.malloc(max(1, len(rgb)))
        handle_address = module.malloc(4)
        try:
            module.write(rgb_address, rgb)
            # The C++ side copies the pixels into its mesh
            module.check(module.call('sc_create', rgb_address, len(rgb), width, height, handle_address))
            self._handle = int.from_bytes(module.read(handle_address, 4), 'little')
        finally:
            module.call('free', rgb_address)
            module.call('free', handle_address)
        self._module = module
        # A store must not be used by two threads at once
        self._lock = threading.Lock()
        return self

    # Intentionally a no-op. It lets subclasses call super().__init__(image),
    # which object.__init__ would reject once a subclass defines __init__.
    def __init__(self, image):
        pass

    @contextlib.contextmanager
    def _native(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('WasmtimeSeamCarver is in use by another thread')
        try:
            yield self._module
        finally:
            self._lock.release()

    @property
    def width(self):
        with self._native() as module:
            return module.call('sc_width', self._handle)

    @property
    def height(self):
        with self._native() as module:
            return module.call('sc_height', self._handle)

    def shrink_horizontally(self, pixels_to_shrink, incremental_cost_updates=True):
        # A WebAssembly i32, like the C API's int
        pixels_to_shrink = operator.index(pixels_to_shrink)
        if not _I32_MIN <= pixels_to_shrink <= _I32_MAX:
            raise OverflowError('Python int too large to convert to C int')
        with self._native() as module:
            module.check(module.call('sc_shrink_horizontally', self._handle, pixels_to_shrink,
                                     int(bool(incremental_cost_updates))))

    def get_shrunk_image(self):
        with self._native() as module:
            width = module.call('sc_width', self._handle)
            height = module.call('sc_height', self._handle)
            size = width * height * 3
            address = module.malloc(max(1, size))
            try:
                module.check(module.call('sc_copy_shrunk_rgb', self._handle, address, size))
                rgb = module.read(address, size)
            finally:
                module.call('free', address)
        return Image.frombytes('RGB', (width, height), rgb)
