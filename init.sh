#!/usr/bin/env bash
# Sets up this repo, safe to rerun: a Python virtual environment with the pinned dependencies
# of pyproject.toml (constraints.txt), then builds the native seam carver modules.
#
# System requirements (Ubuntu packages):
#   Python >= 3.12 with venv support       python3 python3-venv
#   Python headers for extension modules   python3-dev
#   C/C++ compiler                         g++
set -euo pipefail
cd "$(dirname "$0")"
tnvenv="./.venv-tn"

python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' \
    || { echo "Python >= 3.12 is required, found $(python3 --version)" >&2; exit 1; }
command -v c++ >/dev/null \
    || { echo "No C++ compiler found, install one with: sudo apt install g++" >&2; exit 1; }
python3 -c 'import os, sysconfig; raise SystemExit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' \
    || { echo "Python headers are missing, install them with: sudo apt install python3-dev" >&2; exit 1; }

# Recreate the environment if its interpreter is gone, e.g. after a distro upgrade replaced Python
if ! "$tnvenv/bin/python" -c '' 2>/dev/null; then
    python3 -m venv --clear "$tnvenv"
fi
source "$tnvenv/bin/activate"
python -m pip install --upgrade 'pip>=25.1'  # dependency groups need pip >= 25.1
python -m pip install --group notebooks -c constraints.txt

# Native seam carver modules, imported by seam_carver.ipynb
for module in cy_seam_carver cpp_seam_carver c_seam_carver ctypes_seam_carver; do
    (cd "$module" && ./build.sh)
done
