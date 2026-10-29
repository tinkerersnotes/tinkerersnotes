#!/bin/sh
# Times the same seam_carver.wasm on the same pixels in three WebAssembly runtimes: Wasmtime from
# Python (the notebook's WasmtimeSeamCarver), Node (Chrome's V8) and a page in Safari (or the
# default browser elsewhere). Equal picture hashes mean all of them removed the same seams.
# Run ../../init.sh first: it builds the module and installs Emscripten, which brings a Node.
set -e
cd "$(dirname "$0")"
python=../../.venv-tn/bin/python
[ -f ../seam_carver.wasm ] || ../build.sh
"$python" make_page.py
echo
"$python" run_wasmtime.py
echo
# Node: the one on PATH, else the one Emscripten installed
node=$(command -v node || ls ../../.emsdk/node/*/bin/node 2>/dev/null | tail -n 1)
if [ -n "$node" ]; then
    "$node" run_node.mjs
else
    echo "No Node found: skipped"
fi
echo
echo "The browser's results appear on the page:"
case "$(uname -s)" in
    Darwin) open -a Safari build/compare_runtimes.html ;;
    *) xdg-open build/compare_runtimes.html 2>/dev/null || echo "  open build/compare_runtimes.html" ;;
esac
