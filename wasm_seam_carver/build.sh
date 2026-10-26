#!/bin/sh
# Builds seam_carver.html: a standalone page running the C++ core from ../seam_carver_core,
# compiled to WebAssembly with Emscripten (sudo apt install emscripten). The page opens straight
# from disk, no web server needed.
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

# Inline the module, the JavaScript port and the sample picture into the page template
python3 - <<'EOF'
import base64
from pathlib import Path

module = Path('build/seam_carver_wasm.js').read_text()
# A literal "</script" anywhere in the module would end the inline <script> element early
module = module.replace('</script', '<\\/script')
sample = base64.b64encode(Path('../tundra_1280x958.jpeg').read_bytes()).decode()
page = Path('seam_carver.template.html').read_text()
port = Path('seam_carver.js').read_text().replace('</script', '<\\/script')  # JavaScript port of the core
for placeholder, value in (('@@SEAM_CARVER_WASM_JS@@', module),
                           ('@@SEAM_CARVER_JS@@', port),
                           ('@@SAMPLE_IMAGE_DATA_URL@@', 'data:image/jpeg;base64,' + sample)):
    assert page.count(placeholder) == 1, placeholder
    page = page.replace(placeholder, value)
Path('seam_carver.html').write_text(page)
print(f'seam_carver.html: {len(page) // 1024} KiB')
EOF
