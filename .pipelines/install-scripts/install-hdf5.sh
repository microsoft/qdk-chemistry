#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
BUILD_TYPE=${2:-Release}
HDF5_PARENT_DIR=${3:-/ext}
CXX_FLAGS=${4:-"-fPIC -O3"}
MAC_BUILD=${5:-OFF}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
CGMANIFEST="$(cd "${SCRIPT_DIR}/../.." && pwd)/cpp/manifest/qdk-chemistry/cgmanifest.json"

HDF5_VERSION="$(get_version "${CGMANIFEST}" "HDF5")"
if [ -z "${HDF5_VERSION}" ]; then
    echo "ERROR: Could not find HDF5 version in ${CGMANIFEST}" >&2
    exit 1
fi

echo "Installing HDF5 to ${INSTALL_PREFIX}..."

# Work from HDF5 parent directory
cd ${HDF5_PARENT_DIR}

# Check if HDF5 is already installed
if [ -d "${INSTALL_PREFIX}/hdf5" ]; then
    echo "HDF5 exists, skip"
    exit 0
fi

# Download HDF5 source (URL + SHA-1 come from cgmanifest.json, the single source of truth for this pin).
# Clean up any leftover state from a previous (possibly failed) attempt on
# this self-hosted agent — the workspace persists across builds and retries.
echo "Downloading HDF5 ${HDF5_VERSION}..."
rm -rf hdf5 hdf5-${HDF5_VERSION} hdf5-${HDF5_VERSION}.tar.bz2
HDF5_TARBALL="$(download_and_verify "${CGMANIFEST}" "HDF5")" || exit 1
tar -xjf "${HDF5_TARBALL}"
rm "${HDF5_TARBALL}"
mv hdf5-${HDF5_VERSION} hdf5
echo "HDF5 ${HDF5_VERSION} downloaded and extracted successfully"

# Build and install HDF5 from extracted source
cd hdf5
echo "Configuring HDF5..."
CXXFLAGS=${CXX_FLAGS} ./configure --prefix=${INSTALL_PREFIX} \
    --enable-cxx \
    --enable-fortran=no \
    --enable-static \
    --enable-shared=no \
    --with-pic

if [ "$MAC_BUILD" == "ON" ]; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=$(nproc)
fi
make -j${JOBS}

echo "Installing HDF5..."
if [ "$MAC_BUILD" == "ON" ]; then
    sudo make install
elif [ "$MAC_BUILD" == "OFF" ]; then
    make install
fi

# Cleanup (return to HDF5 parent directory but leave source for potential reuse)
cd ${HDF5_PARENT_DIR}

echo "HDF5 ${HDF5_VERSION} installation completed."
