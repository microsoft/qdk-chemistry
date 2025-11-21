#!/bin/bash
set -e

echo "Installing C++ dependencies for QDK Chemistry..."

# Configuration
BUILD_DIR="${BUILD_DIR:-/tmp/qdk_deps_build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Install spdlog
echo "=== Installing spdlog ==="
git clone --depth 1 --branch v1.15.3 https://github.com/gabime/spdlog.git spdlog
cd spdlog
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DCMAKE_CXX_FLAGS="-march=native -fPIC"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf spdlog

# Install blaspp
echo "=== Installing blaspp ==="
git clone https://github.com/icl-utk-edu/blaspp.git blaspp
cd blaspp
git checkout 13622021629f5fd27591bb7da60bae5b19561f01
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_TESTING=OFF
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf blaspp

# Install lapackpp
echo "=== Installing lapackpp ==="
git clone https://github.com/icl-utk-edu/lapackpp.git lapackpp
cd lapackpp
git checkout 5bc9c85201ace48213df5ac7d1ef026c9668dfbd
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DCMAKE_CXX_FLAGS="-DLAPACK_COMPLEX_CPP" \
         -DBUILD_TESTING=OFF
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf lapackpp

# Install libint2
echo "=== Installing libint2 ==="
wget -q https://github.com/evaleev/libint/releases/download/v2.9.0/libint-2.9.0-mpqc4.tgz
tar xzf libint-2.9.0-mpqc4.tgz
cd libint-2.9.0
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf libint-2.9.0 libint-2.9.0-mpqc4.tgz

# Install ecpint
echo "=== Installing ecpint ==="
git clone --depth 1 --branch v1.0.7 https://github.com/robashaw/libecpint ecpint
cd ecpint
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_TESTING=OFF \
         -DLIBECPINT_BUILD_TESTS=OFF \
         -DLIBECPINT_USE_PUGIXML=OFF
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf ecpint

# Install gauxc
echo "=== Installing gauxc ==="
git clone --depth 1 --branch v1.0 https://github.com/wavefunction91/gauxc.git gauxc
cd gauxc
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_TESTING=OFF \
         -DEXCHCXX_ENABLE_LIBXC=OFF \
         -DGAUXC_ENABLE_HDF5=OFF \
         -DGAUXC_ENABLE_MAGMA=OFF \
         -DGAUXC_ENABLE_CUTLASS=ON \
         -DGAUXC_ENABLE_CUDA=OFF \
         -DGAUXC_ENABLE_MPI=OFF
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf gauxc

# Cleanup
cd /
rm -rf "$BUILD_DIR"

echo "=== All dependencies installed successfully ==="
