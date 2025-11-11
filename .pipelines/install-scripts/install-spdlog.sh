#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
BUILD_TYPE=${2:-Release}
MARCH=${3:-native}
BUILD_SHARED=${4:-ON}

echo "Installing spdlog to ${INSTALL_PREFIX}..."

# Work from /ext directory
cd /ext

# Check if spdlog already exists
if [ -d "${INSTALL_PREFIX}/spdlog" ]; then
    echo "spdlog exists, skip"
    exit 0
fi

# Check if spdlog source exists
if [ ! -d "spdlog" ]; then
    echo "Error: spdlog source not found in /ext/"
    echo "Available directories in /ext/:"
    ls -la .
    exit 1
fi

# Build spdlog
cmake -S spdlog -B build-spdlog -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DSPDLOG_BUILD_SHARED=${BUILD_SHARED} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_CXX_FLAGS="-march=${MARCH}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build build-spdlog
cmake --install build-spdlog

echo "spdlog installation completed."
