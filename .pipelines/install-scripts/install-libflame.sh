#!/bin/bash
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
wget -q https://github.com/flame/libflame/archive/refs/tags/${LIBFLAME_VERSION}.zip -O libflame.zip
unzip -q libflame.zip
rm libflame.zip
mv libflame-${LIBFLAME_VERSION} libflame

# Configure and build libflame
cd libflame
ln -s /usr/bin/python3 /usr/bin/python
CFLAGS=${CFLAGS} ./configure \
    --build=$LIBFLAME_BUILD \
    --enable-static-build \
    --prefix=${INSTALL_PREFIX} \
    --enable-lapack2flame \
    --enable-legacy-lapack \
    --enable-max-arg-list-hack \
    --target=$LIBFLAME_ARCH

make -j$(nproc)
make install

cd ..
