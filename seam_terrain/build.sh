#!/bin/sh
# Builds seam_terrain.html: a standalone page showing a picture as a 3D terrain, each pixel as high
# as the cost of the cheapest seam from the top row down to it, rendered with WebGL2. The costs come
# from the JavaScript port of the seam carver, ../wasm_seam_carver/seam_carver.js, so unlike the
# seam carver page this needs no Emscripten, only python3. The page opens straight from disk.
set -e
cd "$(dirname "$0")"

python3 - <<'EOF'
import base64
from pathlib import Path

def script(path):
    # A literal "</script" anywhere in the code would end the inline <script> element early
    return Path(path).read_text().replace('</script', '<\\/script')

sample = base64.b64encode(Path('../tundra_1280x958.jpeg').read_bytes()).decode()
page = Path('seam_terrain.template.html').read_text()
for placeholder, value in (('@@SEAM_CARVER_JS@@', script('../wasm_seam_carver/seam_carver.js')),
                           ('@@SAMPLE_IMAGE_DATA_URL@@', 'data:image/jpeg;base64,' + sample)):
    assert page.count(placeholder) == 1, placeholder
    page = page.replace(placeholder, value)
Path('seam_terrain.html').write_text(page)
print(f'seam_terrain.html: {len(page) // 1024} KiB')
EOF
