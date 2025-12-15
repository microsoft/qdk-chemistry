#!/bin/bash
INSTALL_PREFIX=${1:-/usr/local}
MARCH=${2:-x86-64-v3}
BLIS_VERSION=${3:-2.0}
CFLAGS=${4:-"-fPIC -O3"}

# Download BLIS v2.0
echo "Downloading BLIS ${BLIS_VERSION}..."
export BLIS_CHECKSUM=40134f6570d5539609c6328252ad1530c010931bb96f4e249e08279fd978da7a
wget -q https://github.com/flame/blis/archive/refs/tags/${BLIS_VERSION}.zip -O blis.zip
echo "${BLIS_CHECKSUM}  blis.zip" | shasum -a 256 -c || exit 1
unzip -q blis.zip
rm blis.zip
mv blis-${BLIS_VERSION} blis

cd blis

# Select architectures to build BLIS for
if [[ ${MARCH} == 'armv8-a' ]]; then
    # Compile for generic architecture due to issues with block
    # size allocations for certain ARM instruction sets
    export BLIS_ARCH=generic
    CFLAGS=${CFLAGS} ./configure \
    --disable-shared \
    --enable-static \
    --enable-cblas \
    --prefix=${INSTALL_PREFIX} \
    $BLIS_ARCH
elif [[ ${MARCH} == 'x86-64-v3' ]]; then
    # Compile for intel64, amd64, and amd64_legacy architectures
    export BLIS_ARCH=x86_64
    CFLAGS=${CFLAGS} ./configure \
    --disable-shared \
    --enable-static \
    --enable-cblas \
    --prefix=${INSTALL_PREFIX} \
    $BLIS_ARCH
fi

make -j$(nproc)
make install

cd ..
