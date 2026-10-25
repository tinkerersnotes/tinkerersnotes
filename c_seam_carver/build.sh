#!/bin/sh
# --force rebuilds even when no source file changed: build_ext does not notice changes
# to setup.py, such as new compile flags, and would keep the old module
python setup.py build_ext --inplace --force
