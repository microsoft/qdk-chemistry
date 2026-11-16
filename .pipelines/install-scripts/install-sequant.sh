#!/bin/bash
set -e

# install-sequant.sh — build and install SeQuant with its BTAS backend through cpp/cmake/sequant.
#
# The pins and patches live in cpp/cmake/sequant.cmake. BTAS's installed config looks for BLAS++ where it found
# it at build time, so install-blaspp.sh and install-lapackpp.sh MUST run first, into the same install_prefix.
#
# Usage: install-sequant.sh <install_prefix> [march] [build_type]
#   install_prefix - CMAKE_INSTALL_PREFIX; also where the already-installed BLAS++/LAPACK++ are found via
#                    CMAKE_PREFIX_PATH.
#   march          - -march= value for CMAKE_CXX_FLAGS (default: x86-64-v3).
#   build_type     - CMAKE_BUILD_TYPE (default: Release).

INSTALL_PREFIX=${1:-/usr/local}
MARCH=${2:-x86-64-v3}
BUILD_TYPE=${3:-Release}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(pwd)/sequant"

echo "Installing SeQuant to ${INSTALL_PREFIX}..."

# Clean up any leftover directory from a previous (possibly failed) run.
rm -rf "${WORKDIR}"

cmake -S "${SCRIPT_DIR}/../../cpp/cmake/sequant" -B "${WORKDIR}" -GNinja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
    -DCMAKE_CXX_FLAGS="-march=${MARCH}"

cmake --build "${WORKDIR}"
cmake --install "${WORKDIR}"

rm -rf "${WORKDIR}"
echo "SeQuant installation completed."
