#!/bin/sh
# Builds seam_carver.wasm, the WebAssembly module wasmtime_seam_carver.py runs with Wasmtime: the
# C++ core and the C API of ../ctypes_seam_carver, compiled by Emscripten into a standalone module
# that needs no JavaScript. The C++ core is shared by all modules, only the binding differs.
#
# Needs Emscripten 6.0.3, which ../wasm_seam_carver/emscripten.sh puts on PATH: the emcc already
# there if it is that version, otherwise the repo's own copy in ../.emsdk/, installed on first use
# (about 1 GB, a few minutes). Needs git and python3.
set -e
cd "$(dirname "$0")"
. ../wasm_seam_carver/emscripten.sh

# -sSTANDALONE_WASM --no-entry: a library module ("reactor") for any WebAssembly runtime, importing
#   only a few WASI functions (fd_write and friends, for messages written before an abort)
# -fwasm-exceptions -sWASM_LEGACY_EXCEPTIONS=0: the C API turns C++ exceptions into status codes,
#   with WebAssembly's exception handling in its standardized form, the one Wasmtime implements
# -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=2GB: the mesh takes 64 bytes per pixel; below 2 GB every
#   address fits the signed 32-bit integers WebAssembly functions pass
em++ -O3 -std=c++17 -Wall -Wextra -fwasm-exceptions -sWASM_LEGACY_EXCEPTIONS=0 \
    -I../seam_carver_core -I../ctypes_seam_carver \
    ../ctypes_seam_carver/seam_carver_c_api.cpp ../seam_carver_core/seam_carver_impl.cpp \
    -sSTANDALONE_WASM --no-entry \
    -sEXPORTED_FUNCTIONS=_sc_create,_sc_destroy,_sc_shrink_horizontally,_sc_width,_sc_height,_sc_copy_shrunk_rgb,_sc_last_error,_malloc,_free \
    -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=2GB \
    -o seam_carver.wasm
echo "Built $(pwd)/seam_carver.wasm with $(emcc --version | head -n 1 | sed 's/^emcc ([^)]*) //')"
