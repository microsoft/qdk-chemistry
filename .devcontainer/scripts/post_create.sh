#!/bin/bash
# Post-create step: install QDK Chemistry from the mounted source.
set -euo pipefail
source "$HOME/qdk_chemistry_venv/bin/activate"

# Set a memory-/core-aware CMAKE_BUILD_PARALLEL_LEVEL (unless already set) so the
# initial build below does not oversubscribe CPU or OOM on constrained machines.
source /usr/local/share/qdk/parallelism.sh

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
