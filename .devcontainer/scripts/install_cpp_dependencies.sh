#!/bin/bash
set -e

# Usage: install_cpp_dependencies.sh <cpp_cgmanifest_path> <macis_cgmanifest_path>
#
# Arguments:
#   cpp_cgmanifest_path   - Full path to cpp/manifest/qdk-chemistry/cgmanifest.json
#   macis_cgmanifest_path - Full path to external/macis/manifest/cgmanifest.json

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <cpp_cgmanifest_path> <macis_cgmanifest_path>"
    echo ""
    echo "Example:"
    echo "  $0 /repo/cpp/manifest/qdk-chemistry/cgmanifest.json /repo/external/macis/manifest/cgmanifest.json"
    exit 1
fi

CGMANIFEST="$1"
MACIS_CGMANIFEST="$2"

if [[ ! -f "$CGMANIFEST" ]]; then
    echo "Error: cgmanifest.json not found at $CGMANIFEST"
    exit 1
fi

if [[ ! -f "$MACIS_CGMANIFEST" ]]; then
    echo "Error: macis cgmanifest.json not found at $MACIS_CGMANIFEST"
    exit 1
fi

echo "Installing C++ dependencies for QDK Chemistry..."
echo "Using cgmanifest: $CGMANIFEST"
echo "Using macis cgmanifest: $MACIS_CGMANIFEST"

# Configuration
BUILD_DIR="/tmp/qdk_deps_build"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-OFF}"  # Default to static libraries

# Helper function to extract commit hash from cgmanifest by repository URL pattern
get_commit_hash() {
    local manifest="$1"
    local repo_pattern="$2"
    python3 -c "
import json
with open('$manifest') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'git' and '$repo_pattern' in comp['git'].get('repositoryUrl', ''):
        print(comp['git']['commitHash'].strip())
        break
"
}

# Helper function to extract tag from cgmanifest by repository URL pattern
get_tag() {
    local manifest="$1"
    local repo_pattern="$2"
    python3 -c "
import json
with open('$manifest') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'git' and '$repo_pattern' in comp['git'].get('repositoryUrl', ''):
        print(comp['git'].get('tag', ''))
        break
"
}

# Helper function to get download URL for "other" type components
get_download_url() {
    local manifest="$1"
    local name="$2"
    python3 -c "
import json
with open('$manifest') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'other' and comp['other'].get('name') == '$name':
        print(comp['other']['downloadUrl'])
        break
"
}

# Read versions from cpp cgmanifest
SPDLOG_COMMIT=$(get_commit_hash "$CGMANIFEST" "gabime/spdlog")
if [[ -z "$SPDLOG_COMMIT" ]]; then
    echo "Error: Could not find spdlog commit hash in $CGMANIFEST"
    exit 1
fi
SPDLOG_TAG=$(get_tag "$CGMANIFEST" "gabime/spdlog")
if [[ -z "$SPDLOG_TAG" ]]; then
    echo "Error: Could not find spdlog tag in $CGMANIFEST"
    exit 1
fi
LIBECPINT_COMMIT=$(get_commit_hash "$CGMANIFEST" "robashaw/libecpint")
if [[ -z "$LIBECPINT_COMMIT" ]]; then
    echo "Error: Could not find libecpint commit hash in $CGMANIFEST"
    exit 1
fi
LIBECPINT_TAG=$(get_tag "$CGMANIFEST" "robashaw/libecpint")
if [[ -z "$LIBECPINT_TAG" ]]; then
    echo "Error: Could not find libecpint tag in $CGMANIFEST"
    exit 1
fi
LIBINT_URL=$(get_download_url "$CGMANIFEST" "Libint")
if [[ -z "$LIBINT_URL" ]]; then
    echo "Error: Could not find Libint download URL in $CGMANIFEST"
    exit 1
fi
GAUXC_COMMIT=$(get_commit_hash "$CGMANIFEST" "wavefunction91/gauxc")
if [[ -z "$GAUXC_COMMIT" ]]; then
    echo "Error: Could not find gauxc commit hash in $CGMANIFEST"
    exit 1
fi

# Read versions from macis cgmanifest
BLASPP_COMMIT=$(get_commit_hash "$MACIS_CGMANIFEST" "icl-utk-edu/blaspp")
if [[ -z "$BLASPP_COMMIT" ]]; then
    echo "Error: Could not find blaspp commit hash in $MACIS_CGMANIFEST"
    exit 1
fi
LAPACKPP_COMMIT=$(get_commit_hash "$MACIS_CGMANIFEST" "icl-utk-edu/lapackpp")
if [[ -z "$LAPACKPP_COMMIT" ]]; then
    echo "Error: Could not find lapackpp commit hash in $MACIS_CGMANIFEST"
    exit 1
fi

echo "Using versions from cgmanifest.json:"
echo "  spdlog: ${SPDLOG_TAG:-$SPDLOG_COMMIT}"
echo "  blaspp: $BLASPP_COMMIT"
echo "  lapackpp: $LAPACKPP_COMMIT"
echo "  libecpint: ${LIBECPINT_TAG:-$LIBECPINT_COMMIT}"
echo "  libint: $LIBINT_URL"
echo "  gauxc: $GAUXC_COMMIT"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Install spdlog
echo "=== Installing spdlog ==="
git clone https://github.com/gabime/spdlog.git spdlog
cd spdlog
git checkout "$SPDLOG_COMMIT"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DCMAKE_CXX_FLAGS="-march=native -fPIC" \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf spdlog

# Install blaspp
echo "=== Installing blaspp ==="
git clone https://github.com/icl-utk-edu/blaspp.git blaspp
cd blaspp
git checkout "$BLASPP_COMMIT"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_TESTING=OFF \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf blaspp

# Install lapackpp
echo "=== Installing lapackpp ==="
git clone https://github.com/icl-utk-edu/lapackpp.git lapackpp
cd lapackpp
git checkout "$LAPACKPP_COMMIT"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_TESTING=OFF \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf lapackpp

# Install libint2
echo "=== Installing libint2 ==="
LIBINT_TARBALL=$(basename "$LIBINT_URL")
wget -q "$LIBINT_URL"
tar xzf "$LIBINT_TARBALL"
# The tarball libint-2.9.0-mpqc4.tgz extracts to libint-2.9.0, not libint-2.9.0-mpqc4
# Find the actual extracted directory (excluding macOS metadata files starting with ._)
LIBINT_DIR=$(ls -d libint-*/ 2>/dev/null | grep -v '^\._' | head -1 | tr -d '/')
if [[ -z "$LIBINT_DIR" || ! -d "$LIBINT_DIR" ]]; then
    echo "Error: Could not find libint directory after extraction"
    ls -la
    exit 1
fi
echo "Found libint directory: $LIBINT_DIR"
cd "$LIBINT_DIR"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf "$LIBINT_DIR" "$LIBINT_TARBALL"

# Install ecpint
echo "=== Installing ecpint ==="
git clone https://github.com/robashaw/libecpint ecpint
cd ecpint
git checkout "$LIBECPINT_COMMIT"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_TESTING=OFF \
         -DLIBECPINT_BUILD_TESTS=OFF \
         -DLIBECPINT_USE_PUGIXML=OFF \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf ecpint

# Install gauxc
echo "=== Installing gauxc ==="
git clone https://github.com/wavefunction91/gauxc.git gauxc
cd gauxc
git checkout "$GAUXC_COMMIT"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_TESTING=OFF \
         -DEXCHCXX_ENABLE_LIBXC=OFF \
         -DGAUXC_ENABLE_HDF5=OFF \
         -DGAUXC_ENABLE_MAGMA=OFF \
         -DGAUXC_ENABLE_CUTLASS=ON \
         -DGAUXC_ENABLE_CUDA=OFF \
         -DGAUXC_ENABLE_MPI=OFF \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf gauxc

# Cleanup
cd /
rm -rf "$BUILD_DIR"

echo "=== All dependencies installed successfully ==="
