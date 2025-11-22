#!/bin/bash
INSTALL_PREFIX=${1:-/usr/local}
MARCH=${2:-x86-64-v3}
BLIS_VERSION=${3:-2.0}
CFLAGS=${4:-"-fPIC -O3"}

# Select architectures to build BLIS for
if [[ ${MARCH} == 'armv8-a' ]]; then
    # Compile for armsve, firestorm, thunderx2, cortexa57, cortexa53, and generic architectures
    export BLIS_ARCH=arm64
elif [[ ${MARCH} == 'x86-64-v3' ]]; then
    # Compile for intel64, amd64, and amd64_legacy architectures
    export BLIS_ARCH=x86_64
fi
# Download BLIS v2.0
echo "Downloading BLIS ${BLIS_VERSION}..."
wget -q https://github.com/flame/blis/archive/refs/tags/${BLIS_VERSION}.zip -O blis.zip
unzip -q blis.zip
rm blis.zip
mv blis-${BLIS_VERSION} blis

cd blis

CFLAGS=${CFLAGS} ./configure \
    --disable-shared \
    --enable-static \
    --prefix=${INSTALL_PREFIX} \
    $BLIS_ARCH

make -j$(nproc)
make install

cd ..
