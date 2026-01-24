#!/bin/bash
set -e

MARCH=${1:-x86-64-v3}
PYTHON_VERSION=${2:-3.11}
BUILD_TYPE=${3:-Release}
BUILD_TESTING=${4:-OFF}
ENABLE_COVERAGE=${5:-OFF}
CMAKE_VERSION=${6:-3.28.3}
HDF5_VERSION=${7:-1.13.0}
BLIS_VERSION=${8:-2.0}
LIBFLAME_VERSION=${9:-5.2.0}
PYENV_VERSION=${10:-2.6.15}
MAC_BUILD=${11:-OFF}

export CFLAGS="-fPIC -Os"
if [ "$MAC_BUILD" == "OFF" ]; then # Build/install Linux dependencies
    export DEBIAN_FRONTEND=noninteractive
    # Try to prevent stochastic segfault from libc-bin
    echo "Reinstalling libc-bin..."
    rm /var/lib/dpkg/info/libc-bin.*
    apt-get clean
    apt-get update -q
    apt install -q libc-bin

    # Update and install dependencies
    echo "Installing apt dependencies..."
    apt-get update -q
    apt-get install -y -q \
        python3 python3-pip python3-dev \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
        libffi-dev liblzma-dev \
        libeigen3-dev \
        nlohmann-json3-dev \
        libboost-all-dev \
        libgtest-dev \
        libgmock-dev \
        libfmt-dev \
        ninja-build \
        gcc g++ \
        make \
        git \
        wget \
        curl \
        unzip \
        patchelf \
        build-essential \
        libpugixml-dev \
        python3-pybind11 pybind11-dev

    # Upgrade cmake as Ubuntu 22.04 only has up to v3.22 in apt
    echo "Downloading and installing CMake ${CMAKE_VERSION}..."
    export CMAKE_CHECKSUM=72b7570e5c8593de6ac4ab433b73eab18c5fb328880460c86ce32608141ad5c1
    wget -q https://cmake.org/files/v3.28/cmake-${CMAKE_VERSION}.tar.gz -O cmake-${CMAKE_VERSION}.tar.gz
    echo "${CMAKE_CHECKSUM}  cmake-${CMAKE_VERSION}.tar.gz" | shasum -a 256 -c || exit 1
    tar -xzf cmake-${CMAKE_VERSION}.tar.gz
    rm cmake-${CMAKE_VERSION}.tar.gz
    cd cmake-${CMAKE_VERSION}
    ./bootstrap --parallel=$(nproc) --prefix=/usr/local
    make --silent -j$(nproc)
    make install
    cd ..
    rm -r cmake-${CMAKE_VERSION}
    cmake --version

    # We use BLIS/libflame as the BLAS/LAPACK vendors to prevent symbol collisions
    # with qiskit's shared OpenBLAS
    echo "Downloading and installing BLIS..."
    bash .pipelines/install-scripts/install-blis.sh /usr/local ${MARCH} ${BLIS_VERSION} "${CFLAGS}"

    echo "Downloading and installing libflame..."
    bash .pipelines/install-scripts/install-libflame.sh /usr/local ${MARCH} ${LIBFLAME_VERSION} "${CFLAGS}"

    export PYENV_ROOT="/workspace/.pyenv"
elif [ "$MAC_BUILD" == "ON" ]; then
    brew update
    brew upgrade
    arch -arm64 brew install \
        ninja \
        eigen \
        wget \
        curl \
        cmake \
        gcc \
        boost \
        pybind11
    export CMAKE_PREFIX_PATH="/opt/homebrew"
    export PYENV_ROOT="$PWD/.pyenv"
fi

echo "Downloading HDF5 $HDF5_VERSION..."
export HDF5_CHECKSUM=1826e198df8dac679f0d3dc703aba02af4c614fd6b7ec936cf4a55e6aa0646ec
wget -q -nc --no-check-certificate https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.bz2
echo "${HDF5_CHECKSUM}  hdf5-${HDF5_VERSION}.tar.bz2" | shasum -a 256 -c || exit 1
tar -xjf hdf5-${HDF5_VERSION}.tar.bz2
rm hdf5-${HDF5_VERSION}.tar.bz2
mv hdf5-${HDF5_VERSION} hdf5
echo "HDF5 $HDF5_VERSION downloaded and extracted successfully"

echo "Installing HDF5..."
bash .pipelines/install-scripts/install-hdf5.sh /usr/local ${BUILD_TYPE} ${PWD} "${CFLAGS}" ${MAC_BUILD}

# Install pyenv to use non-system python3 versions
# pyenv is used in place of a venv to prevent any collisions with the system Python
# when building with a non-system Python version.
echo "Installing pyenv ${PYENV_VERSION}..."
export PYENV_CHECKSUM=95187d6ad9bc8310662b5b805a88506e5cbbe038f88890e5aabe3021711bf3c8
wget -q https://github.com/pyenv/pyenv/archive/refs/tags/v${PYENV_VERSION}.zip -O pyenv.zip
echo "${PYENV_CHECKSUM}  pyenv.zip" | shasum -a 256 -c || exit 1
unzip -q pyenv.zip
mv pyenv-${PYENV_VERSION} "$PYENV_ROOT"
rm pyenv.zip
"$PYENV_ROOT/bin/pyenv" install ${PYTHON_VERSION}
"$PYENV_ROOT/bin/pyenv" global ${PYTHON_VERSION}
export PATH="$PYENV_ROOT/versions/${PYTHON_VERSION}/bin:$PATH"
export PATH="$PYENV_ROOT/shims:$PATH"

python3 --version

# Update pip and install build tools
python3 -m pip install --upgrade pip
python3 -m pip install auditwheel build
python3 -m pip install "fonttools>=4.61.0" "urllib3>=2.6.0"

# Prepare README for PyPI
bash .pipelines/pip-scripts/prepare-readme.sh

# Install Python package
cd python

# Build wheel with all necessary CMake flags
if [ "$MAC_BUILD" == "OFF" ]; then
    export CMAKE_C_FLAGS="-march=${MARCH} -fPIC -Os -fvisibility=hidden"
    export CMAKE_CXX_FLAGS="-march=${MARCH} -fPIC -Os -fvisibility=hidden"
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
    python3 -m build --wheel \
        -C build-dir="build/{wheel_tag}" \
        -C cmake.define.QDK_UARCH=${MARCH} \
        -C cmake.define.BUILD_SHARED_LIBS=OFF \
        -C cmake.define.QDK_CHEMISTRY_ENABLE_MPI=OFF \
        -C cmake.define.QDK_ENABLE_OPENMP=OFF \
        -C cmake.define.QDK_CHEMISTRY_ENABLE_COVERAGE=${ENABLE_COVERAGE} \
        -C cmake.define.BUILD_TESTING=${BUILD_TESTING} \
        -C cmake.define.CMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
        -C cmake.define.CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"

    echo "Checking shared dependencies..."
    ldd build/cp*/_core.*.so

    # Repair wheel
    auditwheel repair dist/qdk_chemistry-*.whl -w repaired_wheelhouse/

    # Fix RPATH
    WHEEL_FILE=$(ls repaired_wheelhouse/qdk_chemistry-*.whl)
    FULL_WHEEL_PATH="$PWD/$WHEEL_FILE"
    TEMP_DIR=$(mktemp -d)
    python3 -m zipfile -e "$WHEEL_FILE" "$TEMP_DIR"

    find "$TEMP_DIR" -name '*.so*' -type f -not -path '*/qdk_chemistry.libs/*' | while read so_file; do
        echo "Fixing RPATH for main package: $so_file"
        patchelf --set-rpath '$ORIGIN/../../qdk_chemistry.libs' "$so_file" || true
    done

    find "$TEMP_DIR" -path '*/qdk_chemistry.libs/*' -name '*.so*' -type f | while read so_file; do
        echo "Fixing RPATH for bundled library: $so_file"
        patchelf --set-rpath '$ORIGIN' "$so_file" || true
    done

    rm "$WHEEL_FILE"
    (cd "$TEMP_DIR" && python3 -m zipfile -c "$FULL_WHEEL_PATH" .)
    rm -rf "$TEMP_DIR"

elif [ "$MAC_BUILD" == "ON" ]; then
    export CMAKE_C_FLAGS="-fPIC -Os -fvisibility=hidden -target arm64-apple-darwin"
    export CMAKE_CXX_FLAGS="-fPIC -Os -fvisibility=hidden -target arm64-apple-darwin"
    python3 -m build --wheel \
        -C build-dir="build/{wheel_tag}" \
        -C cmake.define.QDK_UARCH=native \
        -C cmake.define.BUILD_SHARED_LIBS=OFF \
        -C cmake.define.QDK_CHEMISTRY_ENABLE_MPI=OFF \
        -C cmake.define.QDK_ENABLE_OPENMP=OFF \
        -C cmake.define.QDK_CHEMISTRY_ENABLE_COVERAGE=${ENABLE_COVERAGE} \
        -C cmake.define.BUILD_TESTING=${BUILD_TESTING} \
        -C cmake.define.CMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
        -C cmake.define.CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
        -C cmake.define.CMAKE_PREFIX_PATH="/opt/homebrew"
    echo "Repairing wheel for macOS..."
    pip install delocate==0.13.0
    WHEEL_FILE=$(ls dist/qdk_chemistry-*.whl)
    delocate-wheel -w repaired_wheelhouse/ "$WHEEL_FILE"
    delocate-listdeps --all repaired_wheelhouse/qdk_chemistry*.whl

    echo "Checking shared dependencies..."
    otool -L build/cp*/_core.*.so
fi
