#!/bin/bash
set -e

# install-blaspp.sh — build and install BLAS++ (icl-utk-edu/blaspp) from source.
#
# BLAS++ provides the C++ `blas::` API used directly by TAMM/MACIS (e.g. MACIS's own blaspp dependency, and
# TAMM's cpu_blas.cpp / blockops_blas.hpp). It is not distributed via apt, so we build it from source. This is the
# single canonical BLAS++ build used by both the devcontainer (.devcontainer/scripts/install_cpp_dependencies.sh)
# and CI pipelines (.pipelines/install-scripts/install_cpp_dependencies.sh) -- do not duplicate this logic
# elsewhere; add a caller instead.
#
# The commit hash is NOT defaulted here: it must always be resolved from external/macis/manifest/cgmanifest.json
# by the caller and passed in, so the built version can never silently drift from the pinned manifest.
#
# Usage: install-blaspp.sh <install_prefix> <commit> [blas_vendor] [march] [build_shared_libs]
#   install_prefix     - CMAKE_INSTALL_PREFIX (also searched for an existing BLAS install, e.g. OpenBLAS/BLIS).
#   commit              - blaspp commit hash, resolved from external/macis/manifest/cgmanifest.json.
#   blas_vendor         - BLAS++'s `-Dblas=` value: "openblas" (default; qdk-chemistry's own CI/devcontainer
#                         vendor) or "blis" (for the ADO wheel pipeline's BLIS+LibFLAME stack). See BLAS++'s
#                         cmake/BLASFinder.cmake for the full set of supported values.
#   march               - -march= value for CMAKE_CXX_FLAGS (default: x86-64-v3).
#   build_shared_libs   - ON/OFF (default: OFF, matches install_cpp_dependencies.sh's static default).

INSTALL_PREFIX=${1:-/usr/local}
COMMIT=${2:?commit hash is required (resolve it from external/macis/manifest/cgmanifest.json)}
BLAS_VENDOR=${3:-openblas}
MARCH=${4:-x86-64-v3}
BUILD_SHARED_LIBS=${5:-OFF}

BLASPP_REPO="https://github.com/icl-utk-edu/blaspp.git"
WORKDIR="$(pwd)/blaspp"

echo "Installing BLAS++ (${COMMIT}, vendor=${BLAS_VENDOR}) to ${INSTALL_PREFIX}..."

# Clean up any leftover state from a previous (possibly failed) attempt on this self-hosted agent -- the
# workspace persists across builds and retries.
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
