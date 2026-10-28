#!/bin/sh
# Builds seam_carver.html: a standalone page running the C++ core from ../seam_carver_core,
# compiled to WebAssembly with Emscripten (sudo apt install emscripten), its JavaScript port,
# a multithreaded C++ engine and two WebGPU engines, in JavaScript and in C++. The page opens
# straight from disk, no web server needed; only the multithreaded engine needs one (see
# isolation_service_worker.js). Downloads Dawn's Emdawnwebgpu package once, into build/.
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

# The WebGPU engine in C++ (webgpu_seam_carver_cpp.cpp) uses webgpu.h from Emdawnwebgpu, Dawn's
# implementation for Emscripten. This Emscripten predates the port built into newer releases
# (--use-port=emdawnwebgpu), so a pinned release package serves as a local port. Its WGSL comes
# from webgpu_seam_carver.js, the only copy, as a raw string literal.
EMDAWNWEBGPU=v20261027.184652
python3 - "$EMDAWNWEBGPU" <<'EOF'
import hashlib, io, re, sys, urllib.request, zipfile
from pathlib import Path

tag = sys.argv[1]
sha256 = 'e0144a36f5a9e23c5d949bfa25d124e9b9e99c7864f7ede758a734719276e46c'
target = Path(f'build/emdawnwebgpu-{tag}')
if not (target / 'emdawnwebgpu_pkg' / 'emdawnwebgpu.port.py').is_file():
    url = f'https://github.com/google/dawn/releases/download/{tag}/emdawnwebgpu_pkg-{tag}.zip'
    data = urllib.request.urlopen(url).read()
    assert hashlib.sha256(data).hexdigest() == sha256, f'{url} has an unexpected SHA-256'
    zipfile.ZipFile(io.BytesIO(data)).extractall(target)

js = Path('webgpu_seam_carver.js').read_text()
wgsl = re.search(r'const WEBGPU_SEAM_CARVER_WGSL = /\* wgsl \*/ `(.*?)`;', js, re.S).group(1)
assert '${' not in wgsl and '\\' not in wgsl and ')wgsl"' not in wgsl, 'WGSL must stay a plain template literal'
Path('build/seam_carver_wgsl.inc').write_text(f'R"wgsl({wgsl})wgsl"\n')
EOF

# The same flags as the C++ core, plus the Emdawnwebgpu port at compiling and linking.
# -sENVIRONMENT=web: WebGPU is a browser API
em++ -O3 -std=c++20 -Wall -Wextra -fwasm-exceptions -Ibuild \
    "--use-port=build/emdawnwebgpu-$EMDAWNWEBGPU/emdawnwebgpu_pkg/emdawnwebgpu.port.py" \
    webgpu_seam_carver_cpp.cpp \
    -lembind -fwasm-exceptions -sEXPORT_EXCEPTION_HANDLING_HELPERS \
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
