#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
BUILD_TYPE=${2:-Release}
HDF5_PARENT_DIR=${3:-/ext}
CXX_FLAGS=${4:-"-fPIC -O3"}
MAC_BUILD=${5:-OFF}
HDF5_VERSION=${6:-1.13.0}

# Checksum is pinned to the supported HDF5 source tarball. If you bump
# HDF5_VERSION here (or in the templates that call this script), you MUST
# also update HDF5_CHECKSUM below to the sha256 of the new tarball.
SUPPORTED_HDF5_VERSION=1.13.0
HDF5_CHECKSUM=1826e198df8dac679f0d3dc703aba02af4c614fd6b7ec936cf4a55e6aa0646ec

if [ "${HDF5_VERSION}" != "${SUPPORTED_HDF5_VERSION}" ]; then
    echo "ERROR: HDF5 ${HDF5_VERSION} is not supported by this install script." >&2
    echo "       The pinned checksum and URL release-directory only match ${SUPPORTED_HDF5_VERSION}." >&2
    echo "       Update HDF5_CHECKSUM and SUPPORTED_HDF5_VERSION together if you intend to bump." >&2
    exit 1
fi

# Derive the upstream HDF5 release-directory ("hdf5-<major>.<minor>") from
# the version so a future point-release bump within the same minor line only
# needs SUPPORTED_HDF5_VERSION + HDF5_CHECKSUM updated.
HDF5_RELEASE_DIR="hdf5-${HDF5_VERSION%.*}"

echo "Installing HDF5 to ${INSTALL_PREFIX}..."

# Work from HDF5 parent directory
cd ${HDF5_PARENT_DIR}

# Check if HDF5 is already installed
if [ -d "${INSTALL_PREFIX}/hdf5" ]; then
    echo "HDF5 exists, skip"
    exit 0
fi

# Download HDF5 source.
# Clean up any leftover state from a previous (possibly failed) attempt on
# this self-hosted agent — the workspace persists across builds and retries.
echo "Downloading HDF5 ${HDF5_VERSION}..."
rm -rf hdf5 hdf5-${HDF5_VERSION} hdf5-${HDF5_VERSION}.tar.bz2
wget -q "https://support.hdfgroup.org/ftp/HDF5/releases/${HDF5_RELEASE_DIR}/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.bz2"
if command -v sha256sum >/dev/null 2>&1; then
    echo "${HDF5_CHECKSUM}  hdf5-${HDF5_VERSION}.tar.bz2" | sha256sum -c || exit 1
else
    echo "${HDF5_CHECKSUM}  hdf5-${HDF5_VERSION}.tar.bz2" | shasum -a 256 -c || exit 1
fi
tar -xjf hdf5-${HDF5_VERSION}.tar.bz2
rm hdf5-${HDF5_VERSION}.tar.bz2
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
