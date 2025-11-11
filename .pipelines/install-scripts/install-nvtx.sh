#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
BUILD_TYPE=${2:-Release}

echo "Installing NVTX to ${INSTALL_PREFIX}..."

# Work from /ext directory
cd /ext

# Check if NVTX source exists
if [ ! -d "NVTX" ]; then
    echo "Error: NVTX source not found in /ext/"
    echo "Available directories in /ext/:"
    ls -la .
    exit 1
fi

# Install NVTX headers from /ext (header-only library)
mkdir -p ${INSTALL_PREFIX}/include
cp -a NVTX/c/include/* ${INSTALL_PREFIX}/include/

echo "NVTX installation completed."
