#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
BUILD_TYPE=${2:-Release}

echo "Installing HDF5 to ${INSTALL_PREFIX}..."

# Work from /ext directory
cd /ext

# Check if HDF5 is already installed
if [ -d "${INSTALL_PREFIX}/hdf5" ]; then
    echo "HDF5 exists, skip"
    exit 0
fi

# Check if HDF5 source exists
if [ ! -d "hdf5" ]; then
    echo "Error: HDF5 source not found in /ext/"
    echo "Available directories in /ext/:"
    ls -la .
    exit 1
fi

# Build and install HDF5 from extracted source
cd hdf5
echo "Configuring HDF5..."
./configure --prefix=${INSTALL_PREFIX} \
    --enable-cxx \
    --enable-fortran \
    --enable-shared \
    --disable-static

make -j${nproc}

echo "Installing HDF5..."
make install

# Cleanup (return to /ext but leave source for potential reuse)
cd /ext

echo "HDF5 1.13.0 installation completed."
