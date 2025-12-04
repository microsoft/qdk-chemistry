#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
MARCH=${2:-x86-64-v3}
LIBFLAME_VERSION=${3:-5.2.0}
CFLAGS=${4:-"-fPIC -O3"}

# Select architectures to build BLIS for
if [[ ${MARCH} == 'armv8-a' ]]; then
    # Compile for armsve, firestorm, thunderx2, cortexa57, cortexa53, and generic architectures
    export LIBFLAME_ARCH=arm64
    export LIBFLAME_BUILD=aarch64-unknown-linux-gnu
elif [[ ${MARCH} == 'x86-64-v3' ]]; then
    # Compile for intel64, amd64, and amd64_legacy architectures
    export LIBFLAME_BUILD=x86_64-unknown-linux-gnu
    export LIBFLAME_ARCH=x86_64
fi

# Download libflame
echo "Downloading libflame ${LIBFLAME_VERSION}..."
export LIBFLAME_CHECKSUM=6ebdc4a68bd55fd4e555ef96c4bb259224b5c077
wget -q https://github.com/flame/libflame/archive/refs/tags/${LIBFLAME_VERSION}.zip -O libflame.zip
echo "${LIBFLAME_CHECKSUM}  libflame.zip" | shasum -c || exit 1
unzip -q libflame.zip
rm libflame.zip
mv libflame-${LIBFLAME_VERSION} libflame

# Configure and build libflame
cd libflame
ln -s /usr/bin/python3 /usr/bin/python
CFLAGS="${CFLAGS} -march=${MARCH}" ./configure \
    --build=$LIBFLAME_BUILD \
    --enable-static-build \
    --prefix=${INSTALL_PREFIX} \
    --enable-lapack2flame \
    --enable-legacy-lapack \
    --enable-default-m-blocksize=72 \
    --enable-default-k-blocksize=256 \
    --enable-default-n-blocksize=4080 \
    --enable-max-arg-list-hack \
    --target=$LIBFLAME_ARCH

make -j$(nproc)
make install

cd ..
