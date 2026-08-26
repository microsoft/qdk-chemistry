#!/bin/bash
set -ex

# build-exachem-linux.sh — build ExaChem (+ its TAMM tensor backend) for the ADO python-wheels pipeline's Linux
# x86_64 test environment.
#
# Runs inside the same ubuntu:24.04 Docker container used to build/test the qdk-chemistry wheel itself (see
# build-pip-wheels.sh / test-pip-wheels.sh) -- NOT directly on the ADO host pool -- so the resulting ExaChem
# binary stays ABI-compatible with the test container regardless of what OS the host pool itself runs (the pool
# is expected to migrate from Ubuntu to Azure Linux at some point; this script doesn't need to change for that).
#
# Uses BLIS+LibFLAME as its BLAS/LAPACK vendor, unlike GHA's OpenBLAS -- see build-pip-wheels.sh's own comment
# for why, and install-exachem.sh's LINALG_VENDOR=BLIS handling for the LAPACK_PREFERENCE_LIST fix that combo
# needs.
#
# Only the final exachem_install directory is written under the bind-mounted repo checkout (so it survives to be
# published as a pipeline artifact / cached across pipeline runs -- see .pipelines/templates/build-exachem-linux.yml);
# cpp_deps_install and the ExaChem/TAMM build trees are written under /tmp (container-local, discarded on exit)
# since nothing downstream needs them -- BLAS++/LAPACK++/LibInt2/GauXC/EcpInt/spdlog/nlohmann_json are all built
# static (BUILD_SHARED_LIBS=OFF), so they end up statically linked into the ExaChem binary itself, not left as
# separate runtime artifacts to carry forward.
#
# Usage: build-exachem-linux.sh [march] [blis_version] [libflame_version]
# Must be run from the repo root (uses relative paths to cpp/manifest, external/macis/manifest, and the other
# .pipelines/install-scripts/* scripts it delegates to), and writes ./exachem_install into the current directory.

MARCH=${1:-x86-64-v3}
BLIS_VERSION=${2:-2.0}
LIBFLAME_VERSION=${3:-5.2.0}

export DEBIAN_FRONTEND=noninteractive
export CFLAGS="-fPIC -Os"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CFSClean3: redirect Ubuntu apt endpoints to the Azure-internal mirror (see build-pip-wheels.sh for the fuller
# version of this, including the non-x86 ubuntu-ports case -- not needed here, this leg is x86_64-only).
_cfs_apt_redirect() {
    sed -i \
        -e 's|https\?://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
        -e 's|https\?://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
        "$1"
}
[ -f /etc/apt/sources.list.d/ubuntu.sources ] && _cfs_apt_redirect /etc/apt/sources.list.d/ubuntu.sources
[ -f /etc/apt/sources.list ]                  && _cfs_apt_redirect /etc/apt/sources.list

echo "Installing apt dependencies..."
apt-get update -q
apt-get install -y -q \
    build-essential \
    ca-certificates \
    cmake \
    gfortran \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libhdf5-dev \
    libnuma-dev \
    ninja-build \
    openmpi-bin \
    libopenmpi-dev \
    pkg-config \
    python3 \
    unzip \
    wget
cmake --version

echo "Installing BLIS ${BLIS_VERSION}..."
bash "${SCRIPT_DIR}/install-blis.sh" /usr/local "${MARCH}" "${BLIS_VERSION}" "${CFLAGS}"

echo "Installing libflame ${LIBFLAME_VERSION}..."
bash "${SCRIPT_DIR}/install-libflame.sh" /usr/local "${MARCH}" "${LIBFLAME_VERSION}" "${CFLAGS}"

CPP_DEPS_PREFIX=/tmp/cpp_deps_install
echo "Installing qdk-chemistry C++ dependencies (BLIS vendor) into ${CPP_DEPS_PREFIX}..."
INSTALL_PREFIX="${CPP_DEPS_PREFIX}" MARCH="${MARCH}" \
    bash "${SCRIPT_DIR}/install-cpp-deps.sh" \
    "$(pwd)/cpp/manifest/qdk-chemistry/cgmanifest.json" \
    "$(pwd)/external/macis/manifest/cgmanifest.json" \
    blis

echo "Installing ExaChem/TAMM into $(pwd)/exachem_install..."
CPP_DEPS_PREFIX="${CPP_DEPS_PREFIX}" \
INSTALL_PREFIX="$(pwd)/exachem_install" \
BUILD_ROOT=/tmp/exachem_build \
MARCH="${MARCH}" \
LINALG_VENDOR=BLIS \
LINALG_PREFIX=/usr/local \
    bash "${SCRIPT_DIR}/exachem/install-exachem.sh"

echo "==> ExaChem build complete: $(pwd)/exachem_install"
