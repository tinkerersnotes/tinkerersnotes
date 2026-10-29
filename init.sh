#!/usr/bin/env bash
# Sets up this repo, safe to rerun: a Python virtual environment with the pinned dependencies
# of pyproject.toml (constraints.txt), then builds the seam carver modules. One of them,
# wasmtime_seam_carver, is WebAssembly built by Emscripten: the first run installs Emscripten into
# .emsdk/ (wasm_seam_carver/emscripten.sh), about 1 GB and a few minutes.
#
# The WebAssembly page has a build of its own, with the same Emscripten: wasm_seam_carver/build.sh
#
# System requirements (Ubuntu packages; on macOS the Xcode command line tools):
#   Python >= 3.12 with venv support       python3 python3-venv
#   Python headers for extension modules   python3-dev
#   C/C++ compiler                         g++
#   git, for Emscripten's installer        git
set -euo pipefail
cd "$(dirname "$0")"
tnvenv="./.venv-tn"

python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' \
    || { echo "Python >= 3.12 is required, found $(python3 --version)" >&2; exit 1; }
command -v c++ >/dev/null \
    || { echo "No C++ compiler found, install one with: sudo apt install g++" >&2; exit 1; }
python3 -c 'import os, sysconfig; raise SystemExit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' \
    || { echo "Python headers are missing, install them with: sudo apt install python3-dev" >&2; exit 1; }
command -v git >/dev/null \
    || { echo "git is missing, install it with: sudo apt install git" >&2; exit 1; }

# Recreate the environment if its interpreter is gone, e.g. after a distro upgrade replaced Python
if ! "$tnvenv/bin/python" -c '' 2>/dev/null; then
    python3 -m venv --clear "$tnvenv"
fi
source "$tnvenv/bin/activate"
python -m pip install --upgrade 'pip>=25.1'  # dependency groups need pip >= 25.1
python -m pip install --group notebooks --group dev -c constraints.txt

# Commit notebooks without their outputs: sets up the nbstripout git filter for this clone
python configure_git.py

# Seam carver modules, imported by seam_carver.ipynb
for module in cy_seam_carver cpp_seam_carver c_seam_carver ctypes_seam_carver wasmtime_seam_carver; do
    (cd "$module" && ./build.sh)
done
