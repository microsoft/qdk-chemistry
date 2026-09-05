#!/bin/bash
set -e

# install-cpp-deps.sh — build and install qdk-chemistry's C++ dependencies for CI pipelines.
#
# Builds from source: nlohmann_json, googletest, Catch2, spdlog, BLAS++, LAPACK++, LibInt2, ECPint, GauXC.
# The actual BLAS/LAPACK implementation (OpenBLAS via apt, or Apple Accelerate on macOS) is reused from the
# system, not built here.
#
# GoogleTest/Catch2 are installed here so qdk-chemistry's own find_package() calls succeed, avoiding a
# redundant per-job FetchContent rebuild. Eigen3/HDF5/Boost/OpenMP/MPI are installed separately by the
# pipeline (apt/brew), not by this script.
#
# Usage: install-cpp-deps.sh <cpp_cgmanifest_path> <macis_cgmanifest_path> [blas_vendor]
#
# Arguments:
#   cpp_cgmanifest_path   - Path to cpp/manifest/qdk-chemistry/cgmanifest.json
#   macis_cgmanifest_path - Path to external/macis/manifest/cgmanifest.json
#   blas_vendor           - BLAS++'s `-Dblas=` value (default: "auto"; see install-blaspp.sh).

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <cpp_cgmanifest_path> <macis_cgmanifest_path> [blas_vendor]"
    echo ""
    echo "Example:"
    echo "  $0 /repo/cpp/manifest/qdk-chemistry/cgmanifest.json /repo/external/macis/manifest/cgmanifest.json openblas"
    exit 1
fi

CGMANIFEST="$1"
MACIS_CGMANIFEST="$2"
BLAS_VENDOR="${3:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ ! -f "$CGMANIFEST" ]]; then
    echo "Error: cgmanifest.json not found at $CGMANIFEST"
    exit 1
fi

if [[ ! -f "$MACIS_CGMANIFEST" ]]; then
    echo "Error: macis cgmanifest.json not found at $MACIS_CGMANIFEST"
    exit 1
fi

echo "Installing C++ dependencies for QDK Chemistry (CI)..."
echo "Using cgmanifest: $CGMANIFEST"
echo "Using macis cgmanifest: $MACIS_CGMANIFEST"

# Configuration
BUILD_DIR="${BUILD_DIR:-/tmp/qdk_deps_build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-OFF}"  # Default to static
MARCH="${MARCH:-x86-64-v3}"  # matches the uarch qdk-chemistry's Linux CI runners build for
KEEP_BUILD_DIR="${KEEP_BUILD_DIR:-0}"
if command -v nproc >/dev/null 2>&1; then
    DEFAULT_JOBS=$(nproc) # Linux
else
    DEFAULT_JOBS=$(sysctl -n hw.logicalcpu) # macOS
fi
# Falls back to the pipeline's own memory-aware CMAKE_BUILD_PARALLEL_LEVEL (see build-and-test.yaml's "Set
# memory-aware build parallelism" step) when JOBS isn't set explicitly, so this script doesn't independently
# re-derive a job count that could exceed what the runner's memory actually supports.
JOBS="${JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-$DEFAULT_JOBS}}"
MAC_BUILD="OFF"
if [[ "$OSTYPE" == "darwin"* ]]; then
    MAC_BUILD="ON"
fi

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
NJSON_COMMIT=$(get_commit_hash "$CGMANIFEST" "nlohmann/json")
if [[ -z "$NJSON_COMMIT" ]]; then
    echo "Error: Could not find nlohmann_json commit hash in $CGMANIFEST"
    exit 1
fi
NJSON_TAG=$(get_tag "$CGMANIFEST" "nlohmann/json")
if [[ -z "$NJSON_TAG" ]]; then
    echo "Error: Could not find nlohmann_json tag in $CGMANIFEST"
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
GTEST_COMMIT=$(get_commit_hash "$CGMANIFEST" "google/googletest")
if [[ -z "$GTEST_COMMIT" ]]; then
    echo "Error: Could not find googletest commit hash in $CGMANIFEST"
    exit 1
fi
GTEST_TAG=$(get_tag "$CGMANIFEST" "google/googletest")
if [[ -z "$GTEST_TAG" ]]; then
    echo "Error: Could not find googletest tag in $CGMANIFEST"
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
CATCH2_COMMIT=$(get_commit_hash "$MACIS_CGMANIFEST" "catchorg/Catch2")
if [[ -z "$CATCH2_COMMIT" ]]; then
    echo "Error: Could not find Catch2 commit hash in $MACIS_CGMANIFEST"
    exit 1
fi
CATCH2_TAG=$(get_tag "$MACIS_CGMANIFEST" "catchorg/Catch2")
if [[ -z "$CATCH2_TAG" ]]; then
    echo "Error: Could not find Catch2 tag in $MACIS_CGMANIFEST"
    exit 1
fi

echo "Using versions from cgmanifest.json:"
echo "  spdlog: ${SPDLOG_TAG:-$SPDLOG_COMMIT}"
echo "  blaspp: $BLASPP_COMMIT (vendor: $BLAS_VENDOR)"
echo "  lapackpp: $LAPACKPP_COMMIT"
echo "  libecpint: ${LIBECPINT_TAG:-$LIBECPINT_COMMIT}"
echo "  libint: $LIBINT_URL"
echo "  gauxc: $GAUXC_COMMIT"
echo "  nlohmann_json: ${NJSON_TAG:-$NJSON_COMMIT}"
echo "  googletest: ${GTEST_TAG:-$GTEST_COMMIT}"
echo "  Catch2: ${CATCH2_TAG:-$CATCH2_COMMIT}"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Install nlohmann_json (header-only: no compilation, just installs headers + CMake config)
echo "=== Installing nlohmann_json ==="
shallow_checkout https://github.com/nlohmann/json.git "$NJSON_COMMIT" nlohmann_json
cd nlohmann_json
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DJSON_BuildTests=OFF
make install
cd "$BUILD_DIR"
rm -rf nlohmann_json

# Install googletest (avoids a redundant per-job FetchContent rebuild; see header).
echo "=== Installing googletest ==="
shallow_checkout https://github.com/google/googletest.git "$GTEST_COMMIT" googletest
cd googletest
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
         -DINSTALL_GTEST=ON
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf googletest

# Install Catch2 (same rationale as googletest above).
echo "=== Installing Catch2 ==="
shallow_checkout https://github.com/catchorg/Catch2.git "$CATCH2_COMMIT" catch2
cd catch2
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
         -DCATCH_BUILD_TESTING=OFF \
         -DCATCH_INSTALL_DOCS=OFF
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf catch2

# Install spdlog
echo "=== Installing spdlog ==="
shallow_checkout https://github.com/gabime/spdlog.git "$SPDLOG_COMMIT" spdlog
cd spdlog
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
         -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DCMAKE_CXX_FLAGS="-march=${MARCH} -fPIC" \
         -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf spdlog

# Install blaspp / lapackpp
echo "=== Installing blaspp ==="
bash "${SCRIPT_DIR}/install-blaspp.sh" "$INSTALL_PREFIX" "$BLASPP_COMMIT" "$BLAS_VENDOR" "$MARCH" "$BUILD_SHARED_LIBS" "$BUILD_TYPE"

echo "=== Installing lapackpp ==="
bash "${SCRIPT_DIR}/install-lapackpp.sh" "$INSTALL_PREFIX" "$LAPACKPP_COMMIT" "$MARCH" "$BUILD_SHARED_LIBS" "$BUILD_TYPE"

# Install libint2
echo "=== Installing libint2 ==="
LIBINT_TARBALL=$(download_and_verify "$CGMANIFEST" "Libint") || exit 1
if [[ "$MAC_BUILD" == "ON" ]]; then
    tar xzf "$LIBINT_TARBALL"
else
    tar xzf "$LIBINT_TARBALL" --warning=no-unknown-keyword
fi
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
shallow_checkout https://github.com/robashaw/libecpint "$LIBECPINT_COMMIT" ecpint
cd ecpint
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
shallow_checkout https://github.com/wavefunction91/gauxc.git "$GAUXC_COMMIT" gauxc
cd gauxc
mkdir -p build
cd build
gauxc_cmake_args=(
  ..
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DBUILD_TESTING=OFF # GauXC testing requires Catch2 v2
  -DEXCHCXX_ENABLE_LIBXC=OFF
  -DGAUXC_ENABLE_HDF5=OFF
  -DGAUXC_ENABLE_MAGMA=OFF
  -DGAUXC_ENABLE_CUDA=OFF
  -DGAUXC_ENABLE_MPI=OFF
  -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
)

if [[ "$MAC_BUILD" == "ON" ]]; then
  gauxc_cmake_args+=(
    -DGAUXC_ENABLE_CUTLASS=OFF
    -DGAUXC_ENABLE_OPENMP=OFF
  )
else
  gauxc_cmake_args+=(
    -DGAUXC_ENABLE_CUTLASS=ON
  )
fi
cmake "${gauxc_cmake_args[@]}"
make -j"$JOBS"
make install
cd "$BUILD_DIR"
rm -rf gauxc

# Cleanup
if [[ "$KEEP_BUILD_DIR" != "1" ]]; then
  cd /
  rm -rf "$BUILD_DIR"
fi

echo "=== All dependencies installed successfully ==="
