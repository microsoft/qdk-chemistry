#!/bin/bash
# Post-create step: install QDK Chemistry from the mounted source.
set -euo pipefail
source "$HOME/qdk_chemistry_venv/bin/activate"
export CMAKE_BUILD_PARALLEL_LEVEL=4

# Build C++ and install to /usr/local
cmake -S cpp -B cpp/build -G Ninja
cmake --build cpp/build
sudo cmake --install cpp/build

# Install python library
cd ./python
pip install -v .[all]
