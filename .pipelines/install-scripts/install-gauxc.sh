#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
BUILD_TYPE=${2:-Release}
CUDA_ARCH=${3:-90}
MARCH=${4:-native}
BUILD_SHARED=${5:-ON}
ENABLE_MPI=${6:-OFF}

echo "Installing GauXC to ${INSTALL_PREFIX}..."

# Work from /ext directory
cd /ext

# Check if GauXC source exists
if [ ! -d "GauXC" ]; then
    echo "Error: GauXC source not found in /ext/"
    echo "Available directories in /ext/:"
    ls -la .
    exit 1
fi

# Check if ExchCXX source exists
if [ ! -d "ExchCXX" ]; then
    EXCHCXX_SOURCE_ARGS="-DFETCHCONTENT_SOURCE_DIR_EXCHCXX=/ext/ExchCXX"
fi


# Determine CUDA settings
if [ "${CUDA_ARCH}" = "none" ]; then
    echo "Building GauXC without CUDA support..."
    CUDA_ENABLE=OFF
    CUDA_ARCH_FLAG=""
    CUTLASS_ENABLE=OFF
else
    echo "Building GauXC with CUDA support for architecture ${CUDA_ARCH}..."
    CUDA_ENABLE=ON
    CUDA_ARCH_FLAG="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
    CUTLASS_ENABLE=ON
fi

# Build GauXC with ExchCXX dependency
cmake -S GauXC -B build-gauxc -G Ninja \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  ${CUDA_ARCH_FLAG} \
  ${EXCHCXX_SOURCE_ARGS} \
  -DEXCHCXX_ENABLE_LIBXC=OFF \
  -DGAUXC_ENABLE_CUDA=${CUDA_ENABLE} \
  -DGAUXC_ENABLE_MPI=${ENABLE_MPI} \
  -DGAUXC_ENABLE_CUTLASS=${CUTLASS_ENABLE} \
  -DGAUXC_ENABLE_MAGMA=OFF \
  -DCMAKE_CXX_FLAGS="-march=${MARCH} -fPIC" -DCMAKE_Fortran_FLAGS="-march=${MARCH} -fPIC" \
  -DBUILD_SHARED_LIBS=${BUILD_SHARED} \
  -DBUILD_TESTING=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build build-gauxc
cmake --install build-gauxc

# Cleanup
rm -rf build-gauxc

echo "GauXC installation completed."
