#!/usr/bin/env bash
tnvenv="./.venv-tn"
if [ ! -d $tnvenv ]; then
  python3 -m venv $tnvenv
fi
source $tnvenv/bin/activate
pip install jupyterlab setuptools cython

