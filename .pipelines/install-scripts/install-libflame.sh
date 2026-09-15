#!/bin/bash
set -e

INSTALL_PREFIX=${1:-/usr/local}
MARCH=${2:-x86-64-v3}
CFLAGS=${3:-"-fPIC -O3"}

# Download libflame v5.2.0 (a stable release tag, not a floating branch -- checksum-verified below)
# Pinned to the version LIBFLAME_CHECKSUM verifies; bump both together.
LIBFLAME_VERSION=5.2.0
LIBFLAME_CHECKSUM=e120f559758c21392448f45301918f45760f5ab59d246e4d144079c664d5b64b

# Select architectures to build libflame for
if [[ ${MARCH} == 'armv8-a' ]]; then
    # Compile for armsve, firestorm, thunderx2, cortexa57, cortexa53, and generic architectures
    export LIBFLAME_ARCH=arm64
    export LIBFLAME_BUILD=aarch64-unknown-linux-gnu
elif [[ ${MARCH} == 'x86-64-v3' ]]; then
    # Compile for intel64, amd64, and amd64_legacy architectures
    export LIBFLAME_BUILD=x86_64-unknown-linux-gnu
    export LIBFLAME_ARCH=x86_64
fi

echo "Downloading libflame ${LIBFLAME_VERSION}..."
# Clean up any leftover state from a previous (possibly failed) attempt on
# this self-hosted agent — the workspace may persist across builds and retries.
rm -rf libflame libflame-${LIBFLAME_VERSION} libflame.zip
wget -q https://github.com/flame/libflame/archive/refs/tags/${LIBFLAME_VERSION}.zip -O libflame.zip
# sha256sum (coreutils) is always present on Linux, even minimal images that lack shasum (a Perl script not
# always pulled in); shasum is the fallback for macOS, which has no sha256sum by default.
if command -v sha256sum >/dev/null 2>&1; then
    echo "${LIBFLAME_CHECKSUM}  libflame.zip" | sha256sum -c || exit 1
else
    echo "${LIBFLAME_CHECKSUM}  libflame.zip" | shasum -a 256 -c || exit 1
fi
unzip -q libflame.zip
rm libflame.zip
mv libflame-${LIBFLAME_VERSION} libflame

# Configure and build libflame
cd libflame

export PYTHON=/usr/bin/python3
CFLAGS="${CFLAGS}" ./configure \
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
