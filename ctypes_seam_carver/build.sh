#!/bin/sh
# Builds the C API dynamic library loaded by ctypes_seam_carver.py.
# The C++ core is shared by all native modules, only the binding differs.
set -e
cd "$(dirname "$0")"
case "$(uname -s)" in
    Darwin) suffix=dylib ;;
    *) suffix=so ;;
esac
${CXX:-c++} -O3 -std=c++17 -Wall -Wextra -fPIC -fvisibility=hidden -shared \
    -I../seam_carver_core \
    seam_carver_c_api.cpp ../seam_carver_core/seam_carver_impl.cpp \
    -o libseam_carver.$suffix
