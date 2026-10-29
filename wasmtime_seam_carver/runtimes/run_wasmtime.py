"""Times seam_carver.wasm in Wasmtime through WasmtimeSeamCarver, as the notebook does, on the
pixels make_page.py wrote to build/tundra.rgb. The same workloads and repeats as host.js.
Usage: python run_wasmtime.py (the repo's .venv-tn Python)"""
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

from PIL import Image

here = Path(__file__).parent
sys.path.insert(0, str(here.parent.parent))
from wasmtime_seam_carver.wasmtime_seam_carver import WasmtimeSeamCarver  # noqa: E402

WORKLOADS = [('600 seams, incremental', 600, True), ('50 seams, full recompute', 50, False)]
REPEATS = 5


def fnv1a(data):
    h = 0x811c9dc5
    for byte in data:
        h = ((h ^ byte) * 0x01000193) & 0xffffffff
    return f'{h:08x}'


image = Image.frombytes('RGB', (1280, 958), (here / 'build' / 'tundra.rgb').read_bytes())
print(f'Wasmtime {version("wasmtime")} (Python {sys.version.split()[0]})')
for name, seams, incremental in WORKLOADS:
    times = []
    for _ in range(REPEATS):
        carver = WasmtimeSeamCarver(image)
        start = time.perf_counter()
        carver.shrink_horizontally(seams, incremental)
        times.append((time.perf_counter() - start) * 1000)
    picture = fnv1a(carver.get_shrunk_image().tobytes())
    print(f'  {name:26s} median {statistics.median(times):5.0f} ms, best {min(times):5.0f} ms, picture {picture}')
