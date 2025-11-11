#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
BUILD_TYPE=${2:-Release}
CUDA_ARCH=${3:-90}
MARCH=${4:-native}
BUILD_SHARED=${5:-ON}

echo "Installing Libint2 to ${INSTALL_PREFIX}..."

# Work from /ext directory
cd /ext

echo "Contents of /ext directory:"
ls -la

# Look for libint2 directory (version-independent name)
if [ -d "libint2" ]; then
    LIBINT_DIR="libint2"
    echo "Found Libint2 source directory: ${LIBINT_DIR}"
else
    echo "Error: Libint2 source not found in /ext/"
    echo "Expected to find directory named 'libint2'"
    echo "Available directories in /ext/:"
    ls -la .
    echo "Searching entire /ext for any libint directories:"
    find /ext -name "*libint*" -type d
    exit 1
fi

# Build Libint2 from /ext
cmake -S ${LIBINT_DIR} -B build-libint -G Ninja \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  -DBUILD_SHARED_LIBS=${BUILD_SHARED} \
  -DCMAKE_UNITY_BUILD=ON \
  -DCMAKE_CXX_FLAGS="-march=${MARCH}" -DCMAKE_Fortran_FLAGS="-march=${MARCH}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build build-libint
cmake --install build-libint

# Cleanup build directory only
rm -rf build-libint

echo "Libint2 installation completed."
