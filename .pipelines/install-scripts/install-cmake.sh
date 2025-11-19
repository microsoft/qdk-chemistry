#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
BUILD_TYPE=${2:-Release}

echo "Installing CMake 3.to ${INSTALL_PREFIX}..."

# Work from /ext directory
cd /ext

# Check if CMake source exists
if [ ! -d "cmake" ]; then
    echo "Error: CMake source not found in /ext/"
    echo "Available directories in /ext/:"
    ls -la .
    exit 1
fi

# Build CMake from source
echo "Building CMake from source..."
cd cmake

# Bootstrap CMake (CMake builds itself)
echo "Bootstrapping CMake..."
./bootstrap --prefix=${INSTALL_PREFIX} --parallel=$(nproc)

echo "Building CMake..."
make -j$(nproc)

echo "Installing CMake..."
make install

# Cleanup (return to /ext)
cd /ext

echo "CMake installation completed."

# Verify installation
if command -v cmake >/dev/null 2>&1; then
    cmake --version
else
    echo "Warning: cmake command not found in PATH"
fi
