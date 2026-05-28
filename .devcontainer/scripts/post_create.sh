#!/bin/bash
# Post-create step: install QDK Chemistry from the mounted source.
set -euo pipefail
source "$HOME/qdk_chemistry_venv/bin/activate"

# Build C++ and install to a user-local prefix.
# Use the in-tree macis (external/macis) per INSTALL.md; prevents a
# full rebuild on reconfigure.
cmake -S cpp -B cpp/build -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
    -DCMAKE_DISABLE_FIND_PACKAGE_macis=ON
cmake --build cpp/build
cmake --install cpp/build

# Install python library
cd ./python
pip install -v .[all]
