#!/bin/bash
set -e

# install-lapackpp.sh — build and install LAPACK++ (icl-utk-edu/lapackpp) from source.
#
# LAPACK++ provides the C++ `lapack::` API used by MACIS/TAMM (e.g. dense eigensolves). It depends on BLAS++, so
# install-blaspp.sh MUST run first, into the same install_prefix.
#
# The commit hash is NOT defaulted here: it must always be resolved from external/macis/manifest/cgmanifest.json
# by the caller and passed in.
#
# Usage: install-lapackpp.sh <install_prefix> <commit> [march] [build_shared_libs]
#   install_prefix     - CMAKE_INSTALL_PREFIX; also where the already-installed BLAS++ (blasppConfig.cmake) is
#                        found via CMAKE_PREFIX_PATH.
#   commit              - lapackpp commit hash, resolved from external/macis/manifest/cgmanifest.json.
#   march               - -march= value for CMAKE_CXX_FLAGS (default: x86-64-v3).
#   build_shared_libs   - ON/OFF (default: OFF, matches install_cpp_dependencies.sh's static default).
#
# NOTE: LAPACK++ only accepts `-Dlapack=auto`/`generic` -- there's no vendor-specific value; vendor selection
# belongs entirely to BLAS++'s `-Dblas=` option (install-blaspp.sh). `auto` link-tests against the already-found
# BLAS++ target, and OpenBLAS/BLIS+LibFLAME both provide LAPACK in that same library, so no separate liblapack
# install is needed.

INSTALL_PREFIX=${1:-/usr/local}
COMMIT=${2:?commit hash is required (resolve it from external/macis/manifest/cgmanifest.json)}
MARCH=${3:-x86-64-v3}
BUILD_SHARED_LIBS=${4:-OFF}

LAPACKPP_REPO="https://github.com/icl-utk-edu/lapackpp.git"
WORKDIR="$(pwd)/lapackpp"

echo "Installing LAPACK++ (${COMMIT}) to ${INSTALL_PREFIX}..."

# Clean up any leftover directory from a previous (possibly failed) run.
rm -rf "${WORKDIR}"
git clone "${LAPACKPP_REPO}" "${WORKDIR}"
git -C "${WORKDIR}" checkout "${COMMIT}"

cmake -S "${WORKDIR}" -B "${WORKDIR}/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_FLAGS="-march=${MARCH}" \
    -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
    -Dlapack=auto \
    -Dgpu_backend=none \
    -Dbuild_tests=OFF \
    -Dcolor=no

cmake --build "${WORKDIR}/build"
cmake --install "${WORKDIR}/build"

rm -rf "${WORKDIR}"
echo "LAPACK++ installation completed."
