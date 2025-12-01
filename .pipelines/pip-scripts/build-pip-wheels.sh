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

export DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
apt-get update
apt-get install -y \
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
    libpugixml-dev

# Upgrade cmake as Ubuntu 22.04 only has up to v3.22 in apt
if [[ ${MARCH} == 'armv8-a' ]]; then
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.sh
    chmod +x cmake-${CMAKE_VERSION}-linux-aarch64.sh
    /bin/sh cmake-${CMAKE_VERSION}-linux-aarch64.sh --skip-license --prefix=/usr/local
    rm cmake-${CMAKE_VERSION}-linux-aarch64.sh
elif [[ ${MARCH} == 'x86-64-v3' ]]; then
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh
    /bin/sh cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh
fi
cmake --version

export CFLAGS="-fPIC -Os"
echo "Downloading and installing BLIS..."
bash .pipelines/install-scripts/install-blis.sh /usr/local ${MARCH} ${BLIS_VERSION} ${CFLAGS}

echo "Downloading and installing libflame..."
bash .pipelines/install-scripts/install-libflame.sh /usr/local ${MARCH} ${LIBFLAME_VERSION} ${CFLAGS}

echo "Downloading HDF5 $HDF5_VERSION..."
wget -q -nc --no-check-certificate https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.bz2
tar -xjf hdf5-${HDF5_VERSION}.tar.bz2
rm hdf5-${HDF5_VERSION}.tar.bz2
mv hdf5-${HDF5_VERSION} hdf5
echo "HDF5 $HDF5_VERSION downloaded and extracted successfully"

echo "Installing HDF5..."
bash .pipelines/install-scripts/install-hdf5.sh /usr/local ${BUILD_TYPE} ${PWD} ${CFLAGS}

# Install pyenv to use non-system python3 versions
export PYENV_ROOT="/workspace/.pyenv" && \
wget -q https://github.com/pyenv/pyenv/archive/refs/heads/master.zip -O pyenv.zip && \
unzip -q pyenv.zip && \
mv pyenv-master "$PYENV_ROOT" && \
rm pyenv.zip && \
"$PYENV_ROOT/bin/pyenv" install ${PYTHON_VERSION} && \
"$PYENV_ROOT/bin/pyenv" global ${PYTHON_VERSION} && \
export PATH="$PYENV_ROOT/versions/${PYTHON_VERSION}/bin:$PATH"
export PATH="$PYENV_ROOT/shims:$PATH"

python3 --version

# Update pip and install build tools
python3 -m pip install --upgrade pip
python3 -m pip install auditwheel build

# Install Python package
cd python

# Build wheel with all necessary CMake flags
export CMAKE_C_FLAGS="-march=${MARCH} -fPIC -Os -fvisibility=hidden"
export CMAKE_CXX_FLAGS="-march=${MARCH} -fPIC -Os -fvisibility=hidden"

python3 -m build --wheel \
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
auditwheel repair dist/qdk_chemistry-*.whl -w repaired_wheelhouse/ \
    --exclude libopen-rte.so.40 \
    --exclude libopen-pal.so.40 \
    --exclude libmpi.so.40

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
