#!/bin/bash
set -e

# install-blaspp.sh — build and install BLAS++ (icl-utk-edu/blaspp) from source.
#
# BLAS++ provides the C++ `blas::` API used directly by TAMM/MACIS.
#
# The commit hash is NOT defaulted here: it must always be resolved from external/macis/manifest/cgmanifest.json
# by the caller and passed in.
#
# Usage: install-blaspp.sh <install_prefix> <commit> [blas_vendor] [march] [build_shared_libs]
#   install_prefix     - CMAKE_INSTALL_PREFIX (also searched for an existing BLAS install, e.g. OpenBLAS/BLIS).
#   commit              - blaspp commit hash, resolved from external/macis/manifest/cgmanifest.json.
#   blas_vendor         - BLAS++'s `-Dblas=` value: "openblas" (default) or "blis" (ADO's BLIS+LibFLAME stack).
#   march               - -march= value for CMAKE_CXX_FLAGS (default: x86-64-v3).
#   build_shared_libs   - ON/OFF (default: OFF, matches its callers' static default).

INSTALL_PREFIX=${1:-/usr/local}
COMMIT=${2:?commit hash is required (resolve it from external/macis/manifest/cgmanifest.json)}
BLAS_VENDOR=${3:-openblas}
MARCH=${4:-x86-64-v3}
BUILD_SHARED_LIBS=${5:-OFF}

BLASPP_REPO="https://github.com/icl-utk-edu/blaspp.git"
WORKDIR="$(pwd)/blaspp"

echo "Installing BLAS++ (${COMMIT}, vendor=${BLAS_VENDOR}) to ${INSTALL_PREFIX}..."

# Clean up any leftover directory from a previous (possibly failed) run.
rm -rf "${WORKDIR}"
git clone "${BLASPP_REPO}" "${WORKDIR}"
git -C "${WORKDIR}" checkout "${COMMIT}"

cmake -S "${WORKDIR}" -B "${WORKDIR}/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_FLAGS="-march=${MARCH}" \
    -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
    -Dblas="${BLAS_VENDOR}" \
    -Dblas_int=int32 \
    -Dblas_threaded=false \
    -Dblas_fortran=gfortran \
    -Dgpu_backend=none \
    -Dbuild_tests=OFF \
    -Dcolor=no

cmake --build "${WORKDIR}/build"
cmake --install "${WORKDIR}/build"

rm -rf "${WORKDIR}"
echo "BLAS++ installation completed."
