#!/bin/bash
# Post-create step: install QDK Chemistry from the mounted source.
set -euo pipefail
source "$HOME/qdk_chemistry_venv/bin/activate"
export CMAKE_BUILD_PARALLEL_LEVEL=4

# Install python library
cd ./python
pip install -v .[all]
