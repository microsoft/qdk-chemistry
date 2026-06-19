#!/bin/bash
# =============================================================================
# Dedicated CodeQL C++ scan helper.
#
# The wheel build (build-pip-wheels.sh) does too much for CodeQL to wrap
# cleanly: dep install + configure-time toolchain probes + compile + wheel
# repair all in one go. This script splits the C++ build so the pipeline can
# run the heavy environment setup + CMake configure *before* CodeQL Init,
# and only the actual `cmake --build` step is traced by CodeQL.
#
# Usage:
#   bash codeql-scan.sh deps  [PYTHON_VERSION]   # outside CodeQL
#   bash codeql-scan.sh build [PYTHON_VERSION]   # inside CodeQL
#
# Only Linux x86_64 on Ubuntu 24.04 is supported (matches the new
# CodeQLScan stage in python-wheels.yaml).
# =============================================================================

set -ex

PHASE=${1:-deps}
PYTHON_VERSION=${2:-3.13}

MARCH=x86-64-v3
CMAKE_VERSION=3.28.3
HDF5_VERSION=1.13.0
BLIS_VERSION=2.0
LIBFLAME_VERSION=5.2.0

BUILD_DIR=build/codeql-scan
CFLAGS_COMMON="-march=${MARCH} -fPIC -Os -fvisibility=hidden -g"

export CFLAGS="-fPIC -Os"

if [ "$PHASE" = "deps" ]; then
    # Use sudo only for the commands that touch system locations (apt,
    # /usr/local). Everything else — downloads, builds in the workspace,
    # conda bootstrap, cmake configure — runs as the agent user so the
    # produced artifacts (build/codeql-scan/, conda env) are writable by
    # the subsequent `build` phase without further escalation.
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
        pybind11-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-pybind11 \
        python3-venv \
        tk-dev \
        unzip \
        wget \
        xz-utils \
        zlib1g-dev

    # Upgrade cmake (Ubuntu 22/24 apt only ships up to 3.22/3.28).
    echo "Downloading and installing CMake ${CMAKE_VERSION}..."
    export CMAKE_CHECKSUM=72b7570e5c8593de6ac4ab433b73eab18c5fb328880460c86ce32608141ad5c1
    wget -q https://cmake.org/files/v3.28/cmake-${CMAKE_VERSION}.tar.gz -O cmake-${CMAKE_VERSION}.tar.gz
    echo "${CMAKE_CHECKSUM}  cmake-${CMAKE_VERSION}.tar.gz" | shasum -a 256 -c || exit 1
    tar -xzf cmake-${CMAKE_VERSION}.tar.gz
    rm cmake-${CMAKE_VERSION}.tar.gz
    cd cmake-${CMAKE_VERSION}
    ./bootstrap --parallel=$(nproc) --prefix=/usr/local
    make --silent -j$(nproc)
    $SUDO make install
    cd ..
    rm -r cmake-${CMAKE_VERSION}
    cmake --version

    # The install-{blis,libflame,hdf5}.sh helpers do `make install` into
    # /usr/local without sudo internally, so wrap their invocation.
    echo "Downloading and installing BLIS..."
    $SUDO bash .pipelines/install-scripts/install-blis.sh /usr/local ${MARCH} ${BLIS_VERSION} "${CFLAGS}"

    echo "Downloading and installing libflame..."
    $SUDO bash .pipelines/install-scripts/install-libflame.sh /usr/local ${MARCH} ${LIBFLAME_VERSION} "${CFLAGS}"

    echo "Installing HDF5..."
    $SUDO bash .pipelines/install-scripts/install-hdf5.sh /usr/local Release ${PWD} "${CFLAGS}" OFF ${HDF5_VERSION}

    # Bootstrap Anaconda's `conda` and create the build env.
    # shellcheck disable=SC1091
    . .pipelines/pip-scripts/bootstrap-conda.sh buildenv
    conda activate buildenv

    python3 --version
    python3 -m pip install --upgrade pip
    # cmake's find_package(pybind11) needs pybind11 importable in the env we
    # configure from, and numpy is pulled in by some headers transitively.
    python3 -m pip install pybind11 numpy

    # Configure CMake outside the CodeQL trace. Mirrors the cmake.define
    # flags scikit-build-core injects in build-pip-wheels.sh.
    PYBIND11_CMAKEDIR=$(python3 -m pybind11 --cmakedir)
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
        -DPython3_EXECUTABLE="$(which python3)" \
        -DCMAKE_PREFIX_PATH="${PYBIND11_CMAKEDIR}"

elif [ "$PHASE" = "build" ]; then
    # The second pipeline step starts a fresh shell, so re-activate conda.
    # shellcheck disable=SC1091
    . .pipelines/pip-scripts/bootstrap-conda.sh buildenv
    conda activate buildenv

    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
    cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

else
    echo "ERROR: unknown PHASE '$PHASE' (expected 'deps' or 'build')"
    exit 1
fi
