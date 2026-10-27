#!/bin/sh
# Builds seam_carver.html: a standalone page running the C++ core from ../seam_carver_core,
# compiled to WebAssembly with Emscripten (sudo apt install emscripten), its JavaScript port,
# a multithreaded C++ engine and a WebGPU engine. The page opens straight from disk, no web server
# needed; only the multithreaded engine needs one (see isolation_service_worker.js).
set -e
cd "$(dirname "$0")"
mkdir -p build

# -fwasm-exceptions: the core's C++ exceptions as native WebAssembly exceptions, free until thrown
# -sSINGLE_FILE: the .wasm embedded into the .js, since browsers refuse to fetch files from file:// pages
# -sALLOW_MEMORY_GROWTH: the mesh takes 64 bytes per pixel, 78 MB for 1280x958 already
# -sENVIRONMENT=web,node: the same module runs in the page and in Node.js tests
em++ -O3 -std=c++17 -Wall -Wextra -fwasm-exceptions -I../seam_carver_core \
    seam_carver_wasm.cpp ../seam_carver_core/seam_carver_impl.cpp \
    -lembind -fwasm-exceptions -sEXPORT_EXCEPTION_HANDLING_HELPERS \
    -sMODULARIZE -sEXPORT_NAME=createSeamCarverModule -sENVIRONMENT=web,node \
    -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=4GB -sSINGLE_FILE \
    -o build/seam_carver_wasm.js

# The multithreaded engine, ../parallel_seam_carver/tiled_seam_carver.hpp, with the same flags plus:
# -pthread: shared memory and std::thread on Web Workers; the page needs cross-origin isolation
# -sPTHREAD_POOL_SIZE: workers started with the module, Module.poolSize from the page (one fewer
#   than the most threads a carver may use, since the calling thread works too)
# -sPTHREAD_POOL_SIZE_STRICT=2: a thread beyond the pool fails instead of deadlocking the page
# -sENVIRONMENT=...,worker: the workers run this module's script too
# -sMAXIMUM_MEMORY=2GB: shared memory reserves its maximum up front, keep it modest
# -Wno-pthreads-mem-growth: growing shared memory slows JavaScript access to it, which only
#   copies the picture in and out here
em++ -O3 -std=c++20 -Wall -Wextra -pthread -fwasm-exceptions -I../parallel_seam_carver \
    tiled_seam_carver_wasm.cpp \
    -lembind -fwasm-exceptions -sEXPORT_EXCEPTION_HANDLING_HELPERS \
    -sMODULARIZE -sEXPORT_NAME=createTiledSeamCarverModule -sENVIRONMENT=web,worker,node \
    '-sPTHREAD_POOL_SIZE=Module.poolSize??Math.min(navigator.hardwareConcurrency,8)-1' -sPTHREAD_POOL_SIZE_STRICT=2 \
    -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=2GB -Wno-pthreads-mem-growth -sSINGLE_FILE \
    -o build/tiled_seam_carver_wasm.js

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
                           ('@@SAMPLE_IMAGE_DATA_URL@@', 'data:image/jpeg;base64,' + sample)):
    assert page.count(placeholder) == 1, placeholder
    page = page.replace(placeholder, value)
Path('seam_carver.html').write_text(page)
print(f'seam_carver.html: {len(page) // 1024} KiB')
EOF
