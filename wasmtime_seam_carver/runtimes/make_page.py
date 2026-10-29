"""Writes build/tundra.rgb, the sample picture's pixels as Pillow decodes them (the same for
every runtime), and build/compare_runtimes.html, a page that runs seam_carver.wasm on them in the
browser it is opened in. The page embeds everything, so it opens straight from disk.
Usage: python make_page.py (the repo's .venv-tn Python)"""
import base64
from pathlib import Path

from PIL import Image

here = Path(__file__).parent
build = here / 'build'
build.mkdir(exist_ok=True)
rgb = Image.open(here.parent.parent / 'tundra_1280x958.jpeg').convert('RGB').tobytes()
(build / 'tundra.rgb').write_bytes(rgb)
wasm = (here.parent / 'seam_carver.wasm').read_bytes()

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WebAssembly runtime comparison</title>
<style>
  :root {{ color-scheme: light dark; --bg: #fbfaf7; --fg: #1d1d1b; --muted: #5d5b55; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #17181a; --fg: #e8e6e1; --muted: #a29f97; }} }}
  body {{ margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }}
  main {{ max-width: 760px; margin: 0 auto; padding: 32px 16px; }}
  pre {{ font: 13px/1.5 ui-monospace, Menlo, monospace; white-space: pre-wrap; }}
  p {{ color: var(--muted); }}
</style>
</head>
<body>
<main>
<h1>seam_carver.wasm in this browser</h1>
<p>The module wasmtime_seam_carver.py runs, on the same pixels, with the same calls and timing as
run_wasmtime.py and run_node.mjs. Each run is a fresh instance; only shrink_horizontally is timed.</p>
<pre id="out">Running…</pre>
</main>
<script>{(here / 'host.js').read_text()}</script>
<script>
const decode = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
const wasm = decode('{base64.b64encode(wasm).decode()}');
const rgb = decode('{base64.b64encode(rgb).decode()}');
const out = document.getElementById('out');
const engine = /Version\\/([\\d.]+).*Safari/.exec(navigator.userAgent)?.[0] ?? navigator.userAgent;
const progress = [];
setTimeout(async () => {{
  try {{
    const results = await compare(wasm, rgb, 1280, 958, (line) => {{ progress.push(line); out.textContent = progress.join('\\n'); }});
    out.textContent = engine + '\\n' + results.map((r) =>
      `  ${{r.workload.padEnd(26)}} median ${{r.median.toFixed(0).padStart(5)}} ms, best ${{r.best.toFixed(0).padStart(5)}} ms, picture ${{r.hash}}`).join('\\n');
  }} catch (error) {{
    out.textContent = 'Failed: ' + error;
  }}
}}, 300);
</script>
</body>
</html>
'''
(build / 'compare_runtimes.html').write_text(page)
print(f'Wrote {build / "compare_runtimes.html"} ({len(page) // 1024} KiB) and {build / "tundra.rgb"}')
