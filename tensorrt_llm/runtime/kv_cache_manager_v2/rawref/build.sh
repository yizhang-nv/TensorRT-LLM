#!/bin/bash

# Build the rawref C extension in-place

cd "$(dirname "$0")"
python setup.py build_ext --inplace
