#!/bin/sh
# Builds seam_carver.html: a standalone page running the C++ core from ../seam_carver_core,
# compiled to WebAssembly with Emscripten, its JavaScript port, a multithreaded C++ engine and
# two WebGPU engines, in JavaScript and in C++. The page opens straight from disk, no web server
# needed; only the multithreaded engine needs one (see isolation_service_worker.js).
#
# Needs Emscripten 6.0.3, which emscripten.sh puts on PATH: the emcc already there if it is that
# version, otherwise the repo's own copy in .emsdk/, installed on the first build (about 1 GB, a
# few minutes). The Pages workflow builds the page the same way. Needs git and python3.
set -e
cd "$(dirname "$0")"
. ./emscripten.sh
mkdir -p build

# -fwasm-exceptions: the core's C++ exceptions as native WebAssembly exceptions, free until thrown
# -sEXPORTED_RUNTIME_METHODS=getExceptionMessage: the page shows a C++ exception's type and message
# -sSINGLE_FILE: the .wasm embedded into the .js, since browsers refuse to fetch files from file:// pages
# -sALLOW_MEMORY_GROWTH: the mesh takes 64 bytes per pixel, 78 MB for 1280x958 already
# -sENVIRONMENT=web,node: the same module runs in the page and in Node.js tests
em++ -O3 -std=c++17 -Wall -Wextra -fwasm-exceptions -I../seam_carver_core \
    seam_carver_wasm.cpp ../seam_carver_core/seam_carver_impl.cpp \
    -lembind -fwasm-exceptions -sEXPORTED_RUNTIME_METHODS=getExceptionMessage \
    -sMODULARIZE -sEXPORT_NAME=createSeamCarverModule -sENVIRONMENT=web,node \
    -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=4GB -sSINGLE_FILE \
    -o build/seam_carver_wasm.js

# The multithreaded engine, ../parallel_seam_carver/tiled_seam_carver.hpp, with the same flags plus:
# -pthread: shared memory and std::thread on Web Workers; the page needs cross-origin isolation
# -sPTHREAD_POOL_SIZE: workers started with the module, Module.poolSize from the page (one fewer
#   than the most threads a carver may use, since the calling thread works too)
# -sPTHREAD_POOL_SIZE_STRICT=2: a thread beyond the pool fails instead of deadlocking the page
# -sENVIRONMENT=...,worker: the workers run this module's script too
# -sINCOMING_MODULE_JS_API: the options the page passes in; mainScriptUrlOrBlob, the Blob URL its
#   workers start from, is no longer accepted by default (Emscripten 6), and without it workers
#   would start from the calling worker's own script and never load the module
# -sMAXIMUM_MEMORY=2GB: shared memory reserves its maximum up front, keep it modest
# -Wno-pthreads-mem-growth: growing shared memory slows JavaScript access to it, which only
#   copies the picture in and out here
em++ -O3 -std=c++20 -Wall -Wextra -pthread -fwasm-exceptions -I../parallel_seam_carver \
    tiled_seam_carver_wasm.cpp \
    -lembind -fwasm-exceptions -sEXPORTED_RUNTIME_METHODS=getExceptionMessage \
    -sMODULARIZE -sEXPORT_NAME=createTiledSeamCarverModule -sENVIRONMENT=web,worker,node \
    '-sPTHREAD_POOL_SIZE=Module.poolSize??Math.min(navigator.hardwareConcurrency,8)-1' -sPTHREAD_POOL_SIZE_STRICT=2 \
    -sINCOMING_MODULE_JS_API=mainScriptUrlOrBlob,print,printErr \
    -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=2GB -Wno-pthreads-mem-growth -sSINGLE_FILE \
    -o build/tiled_seam_carver_wasm.js

# The WebGPU engine in C++ (webgpu_seam_carver_cpp.cpp) uses webgpu.h from Emdawnwebgpu, Dawn's
# implementation for Emscripten, through the port built into Emscripten: --use-port=emdawnwebgpu
# downloads the Dawn release this Emscripten pins, once, into its cache. Its WGSL comes from
# webgpu_seam_carver.js, the only copy, as a raw string literal.
python3 - <<'EOF'
import re
from pathlib import Path

js = Path('webgpu_seam_carver.js').read_text()
wgsl = re.search(r'const WEBGPU_SEAM_CARVER_WGSL = /\* wgsl \*/ `(.*?)`;', js, re.S).group(1)
assert '${' not in wgsl and '\\' not in wgsl and ')wgsl"' not in wgsl, 'WGSL must stay a plain template literal'
Path('build/seam_carver_wgsl.inc').write_text(f'R"wgsl({wgsl})wgsl"\n')
EOF

# The same flags as the C++ core, plus the Emdawnwebgpu port at compiling and linking.
# -sENVIRONMENT=web: WebGPU is a browser API
em++ -O3 -std=c++20 -Wall -Wextra -fwasm-exceptions -Ibuild --use-port=emdawnwebgpu \
    webgpu_seam_carver_cpp.cpp \
    -lembind -fwasm-exceptions -sEXPORTED_RUNTIME_METHODS=getExceptionMessage \
    -sMODULARIZE -sEXPORT_NAME=createWebGPUCppModule -sENVIRONMENT=web \
    -sALLOW_MEMORY_GROWTH -sSINGLE_FILE \
    -o build/webgpu_seam_carver_cpp.js

# Inline the modules, the JavaScript engines and the sample picture into the page template
python3 - <<'EOF'
import base64
from pathlib import Path

def script(path):
    # A literal "</script" anywhere in the code would end the inline <script> element early
    return Path(path).read_text().replace('</script', '<\\/script')

sample = base64.b64encode(Path('../tundra_1280x958.jpeg').read_bytes()).decode()
page = Path('seam_carver.template.html').read_text()
for placeholder, value in (('@@SEAM_CARVER_WASM_JS@@', script('build/seam_carver_wasm.js')),
                           ('@@TILED_SEAM_CARVER_WASM_JS@@', script('build/tiled_seam_carver_wasm.js')),
                           ('@@TILED_SEAM_CARVER_WORKER_JS@@', script('tiled_seam_carver_worker.js')),
                           ('@@SEAM_CARVER_JS@@', script('seam_carver.js')),  # JavaScript port of the core
                           ('@@WEBGPU_SEAM_CARVER_JS@@', script('webgpu_seam_carver.js')),
                           ('@@WEBGPU_SEAM_CARVER_CPP_JS@@', script('build/webgpu_seam_carver_cpp.js')),
                           ('@@SAMPLE_IMAGE_DATA_URL@@', 'data:image/jpeg;base64,' + sample)):
    assert page.count(placeholder) == 1, placeholder
    page = page.replace(placeholder, value)
Path('seam_carver.html').write_text(page)
print(f'seam_carver.html: {len(page) // 1024} KiB')
EOF
