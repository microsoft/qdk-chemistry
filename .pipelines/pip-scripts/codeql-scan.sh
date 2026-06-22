#!/bin/bash
# =============================================================================
# Dedicated CodeQL C++ scan helper.
#
# The wheel build (build-pip-wheels.sh) does too much for CodeQL to wrap
# cleanly: dep install + configure-time toolchain probes + compile + wheel
# repair all in one go. This script runs only the cmake configure +
# compile from python/CMakeLists.txt — which add_subdirectory()s cpp/ and
# adds the pybind11 _core module, so a single cmake build covers both the
# C++ library and the Python bindings. Split into two phases so the heavy
# env setup + CMake configure run *before* CodeQL Init, and only the
# actual `cmake --build` is traced.
#
# Usage:
#   bash codeql-scan.sh deps    # outside CodeQL
#   bash codeql-scan.sh build   # inside CodeQL
#
# Only Linux x86_64 on Ubuntu 24.04 is supported (matches the new
# CodeQLScan stage in python-wheels.yaml).
# =============================================================================

set -ex

PHASE=${1:-deps}

MARCH=x86-64-v3
HDF5_VERSION=1.13.0
BLIS_VERSION=2.0
LIBFLAME_VERSION=5.2.0

BUILD_DIR=python/build
CFLAGS_COMMON="-march=${MARCH} -fPIC -Os -fvisibility=hidden -g"

export CFLAGS="-fPIC -Os"

if [ "$PHASE" = "deps" ]; then
    # Use sudo only for the commands that touch system locations (apt,
    # /usr/local). Everything else — downloads, builds in the workspace,
    # cmake configure — runs as the agent user so the produced artifacts
    # (cpp/build/) are writable by the subsequent `build` phase without
    # further escalation.
    SUDO="sudo"
    if [ "$(id -u)" = "0" ]; then
        SUDO=""
    fi
    APT="$SUDO env DEBIAN_FRONTEND=noninteractive apt-get"

    # Mirrors build-pip-wheels.sh: prevent stochastic libc-bin segfault.
    echo "Reinstalling libc-bin..."
    $SUDO rm -f /var/lib/dpkg/info/libc-bin.*
    $SUDO apt-get clean
    $APT update -q
    $APT install -y -q libc-bin

    echo "Installing apt dependencies..."
    $APT update -q
    $APT install -y -q \
        build-essential \
        cmake \
        curl \
        gcc g++ \
        git \
        libboost-all-dev \
        libbz2-dev \
        libeigen3-dev \
        libffi-dev \
        libfmt-dev \
        libgmock-dev \
        libgtest-dev \
        liblzma-dev \
        libncursesw5-dev \
        libpugixml-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        make \
        ninja-build \
        nlohmann-json3-dev \
        patchelf \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        tk-dev \
        unzip \
        wget \
        xz-utils \
        zlib1g-dev

    # The install-{blis,libflame,hdf5}.sh helpers do `make install` into
    # /usr/local without sudo internally, so wrap their invocation.
    echo "Downloading and installing BLIS..."
    $SUDO bash .pipelines/install-scripts/install-blis.sh /usr/local ${MARCH} ${BLIS_VERSION} "${CFLAGS}"

    echo "Downloading and installing libflame..."
    $SUDO bash .pipelines/install-scripts/install-libflame.sh /usr/local ${MARCH} ${LIBFLAME_VERSION} "${CFLAGS}"

    echo "Installing HDF5..."
    $SUDO bash .pipelines/install-scripts/install-hdf5.sh /usr/local Release ${PWD} "${CFLAGS}" OFF ${HDF5_VERSION}

    # Create a venv with the pinned build-time requirements
    echo "Setting up Python venv with build requirements..."
    VENV_DIR="${PWD}/.codeql-venv"
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --quiet --upgrade pip
    "${VENV_DIR}/bin/pip" install --quiet -r .pipelines/requirements.txt
    PYBIND11_CMAKEDIR="$("${VENV_DIR}/bin/python" -m pybind11 --cmakedir)"
    echo "Using pybind11 cmake dir: ${PYBIND11_CMAKEDIR}"

    # Configure CMake outside the CodeQL trace so configure-time toolchain
    # probes don't pollute the database. Flags mirror what scikit-build-core
    # injects in the production wheel build (build-pip-wheels.sh) so the
    # scanned compilation matches what ships.
    cmake -S python -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DQDK_UARCH=${MARCH} \
        -DBUILD_SHARED_LIBS=OFF \
        -DQDK_CHEMISTRY_ENABLE_MPI=OFF \
        -DQDK_ENABLE_OPENMP=OFF \
        -DQDK_CHEMISTRY_ENABLE_COVERAGE=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_C_FLAGS="${CFLAGS_COMMON}" \
        -DCMAKE_CXX_FLAGS="${CFLAGS_COMMON}" \
        -DPython_EXECUTABLE="${VENV_DIR}/bin/python" \
        -Dpybind11_DIR="${PYBIND11_CMAKEDIR}"

elif [ "$PHASE" = "build" ]; then
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
    cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

else
    echo "ERROR: unknown PHASE '$PHASE' (expected 'deps' or 'build')"
    exit 1
fi
