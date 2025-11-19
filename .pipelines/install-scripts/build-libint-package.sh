#!/bin/bash
set -e

# Script to build libint package for Azure Artifacts
# Usage: build-libint-package.sh <source_dir> <build_dir> <version> [build_type] [march]

LIBINT_SOURCE_DIR=${1:-/workspace/libint-source}
LIBINT_BUILD_DIR=${2:-/workspace/libint-build}
LIBINT_VERSION=${3:-2.11.1}
BUILD_TYPE=${4:-Release}
MARCH=${5:-x86-64-v3}

# Libint configuration options with defaults
LIBINT_ENABLE_1BODY=${LIBINT_ENABLE_1BODY:-3}
LIBINT_ENABLE_ERI=${LIBINT_ENABLE_ERI:-2}
LIBINT_ENABLE_ERI3=${LIBINT_ENABLE_ERI3:-2}
LIBINT_ENABLE_ERI2=${LIBINT_ENABLE_ERI2:-2}
LIBINT_MAX_AM=${LIBINT_MAX_AM:-4}
LIBINT_OPT_AM=${LIBINT_OPT_AM:-4}
LIBINT_ERI_MAX_AM=${LIBINT_ERI_MAX_AM:-4}
LIBINT_ERI_OPT_AM=${LIBINT_ERI_OPT_AM:-4}
LIBINT_ENABLE_GENERIC_CODE=${LIBINT_ENABLE_GENERIC_CODE:-true}
LIBINT_ENABLE_CONTRACTED_INTS=${LIBINT_ENABLE_CONTRACTED_INTS:-true}
LIBINT_DISABLE_T1G12_SUPPORT=${LIBINT_DISABLE_T1G12_SUPPORT:-true}

# Build parallelization
if [ -z "${BUILD_JOBS}" ] || [ "${BUILD_JOBS}" = "0" ]; then
    BUILD_JOBS=$(nproc)
fi

echo "=== Libint Package Build Script ==="
echo "Source Directory: ${LIBINT_SOURCE_DIR}"
echo "Build Directory: ${LIBINT_BUILD_DIR}"
echo "Libint Version: ${LIBINT_VERSION}"
echo "Build Type: ${BUILD_TYPE}"
echo "Architecture: ${MARCH}"
echo "Build Jobs: ${BUILD_JOBS}"
echo ""
echo "Libint Configuration:"
echo "  1-body integrals: ${LIBINT_ENABLE_1BODY}"
echo "  ERI integrals: ${LIBINT_ENABLE_ERI}"
echo "  ERI3 integrals: ${LIBINT_ENABLE_ERI3}"
echo "  ERI2 integrals: ${LIBINT_ENABLE_ERI2}"
echo "  Max AM: ${LIBINT_MAX_AM}"
echo "  Opt AM: ${LIBINT_OPT_AM}"
echo "  ERI Max AM: ${LIBINT_ERI_MAX_AM}"
echo "  ERI Opt AM: ${LIBINT_ERI_OPT_AM}"
echo "  Generic code: ${LIBINT_ENABLE_GENERIC_CODE}"
echo "  Contracted ints: ${LIBINT_ENABLE_CONTRACTED_INTS}"
echo "  Disable T1G12: ${LIBINT_DISABLE_T1G12_SUPPORT}"
echo "=================================="

# Validate source directory exists
if [ ! -d "${LIBINT_SOURCE_DIR}" ]; then
    echo "Error: Source directory ${LIBINT_SOURCE_DIR} not found!"
    exit 1
fi

# Create build directory
mkdir -p "${LIBINT_BUILD_DIR}"

# Install missing libint prerequisites
echo "=== Installing Libint Prerequisites ==="

# Check and install missing packages
echo "Checking for required libint prerequisites..."

# Install GMP library with C++ support
if ! dpkg -l | grep -q libgmp-dev; then
    echo "Installing GMP library..."
    apt-get update && apt-get install -y \
        libgmp-dev \
        libgmpxx4ldbl
else
    echo "GMP library already installed"
fi

# Install MPFR library (optional, for high-precision testing)
if ! dpkg -l | grep -q libmpfr-dev; then
    echo "Installing MPFR library..."
    apt-get update && apt-get install -y \
        libmpfr-dev
else
    echo "MPFR library already installed"
fi

echo "Prerequisites installation completed."

# Verify prerequisites
echo "=== Verifying Prerequisites ==="
echo "C++ Compiler: $(g++ --version | head -1)"
echo "Boost: $(dpkg -l | grep -E '^ii +libboost' | head -1 | awk '{print $2, $3}')"
echo "GMP: $(dpkg -l | grep libgmp-dev | head -1 | awk '{print $2, $3}')"
echo "MPFR: $(dpkg -l | grep libmpfr-dev | head -1 | awk '{print $2, $3}')"
echo "Doxygen: $(doxygen --version 2>/dev/null || echo 'Not available')"
echo "Make: $(make --version | head -1)"
echo "Autoconf: $(autoconf --version | head -1)"
echo "Automake: $(automake --version | head -1)"
echo "Libtool: $(libtool --version | head -1 || libtoolize --version | head -1)"
echo "Git: $(git --version)"
echo "Prerequisites verification completed."

# Check available resources
echo "System Information:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem' | awk '{print $2}')"
echo "Disk space: $(df -h /workspace 2>/dev/null || df -h / | tail -1)"

# Configure libint compiler
echo "=== Configuring Libint Compiler ==="
cd "${LIBINT_SOURCE_DIR}"

# Generate the configure script (requires autotools)
echo "Running autogen.sh to generate configure script..."
if [ -x "./autogen.sh" ]; then
    ./autogen.sh
else
    echo "Warning: autogen.sh not found or not executable; proceeding if configure exists."
fi

# Create separate build directory (must be outside source tree as per documentation)
echo "Creating build directory outside source tree..."
cd "${LIBINT_BUILD_DIR}"

# Configure the libint compiler with standard options
echo "Configuring libint compiler..."
# Map MARCH string to compiler -march flag when possible
MARCH_FLAG=""
case "${MARCH}" in
    x86-64-v3) MARCH_FLAG="-march=x86-64-v3" ;;
    x86-64) MARCH_FLAG="-march=x86-64" ;;
    native) MARCH_FLAG="-march=native" ;;
    *) MARCH_FLAG="" ;;
esac

# Build configure arguments based on configuration variables
CONFIGURE_ARGS=(
    "--enable-1body=${LIBINT_ENABLE_1BODY}"
    "--enable-eri=${LIBINT_ENABLE_ERI}"
    "--enable-eri3=${LIBINT_ENABLE_ERI3}"
    "--enable-eri2=${LIBINT_ENABLE_ERI2}"
    "--with-max-am=${LIBINT_MAX_AM}"
    "--with-opt-am=${LIBINT_OPT_AM}"
    "--with-eri-max-am=${LIBINT_ERI_MAX_AM}"
    "--with-eri-opt-am=${LIBINT_ERI_OPT_AM}"
)

# Add conditional flags
if [ "${LIBINT_ENABLE_GENERIC_CODE}" = "true" ]; then
    CONFIGURE_ARGS+=("--enable-generic-code")
fi

if [ "${LIBINT_ENABLE_CONTRACTED_INTS}" = "true" ]; then
    CONFIGURE_ARGS+=("--enable-contracted-ints")
fi

if [ "${LIBINT_DISABLE_T1G12_SUPPORT}" = "true" ]; then
    CONFIGURE_ARGS+=("--disable-t1g12-support")
fi

CXXFLAGS_COMBINED="-O3 -fPIC ${MARCH_FLAG} ${CXXFLAGS:-} ${CMAKE_CXX_FLAGS:-}"
"${LIBINT_SOURCE_DIR}/configure" \
    "${CONFIGURE_ARGS[@]}" \
    CXX=g++ \
    CXXFLAGS="${CXXFLAGS_COMBINED}"

echo "Libint compiler configuration completed successfully."

# Build the libint compiler using configured number of jobs
echo "=== Building Libint Compiler ==="
echo "Building libint compiler using ${BUILD_JOBS} parallel jobs..."
make -j${BUILD_JOBS}

echo "Libint compiler build completed successfully."

# Generate the libint library
echo "=== Generating Libint Library ==="
echo "Running 'make export' to generate libint library tarball..."
make export

echo "Library generation completed successfully."

# Move the generated tarball to workspace for external access
echo "=== Moving Generated Library ==="
GENERATED_TARBALL=$(find . -maxdepth 2 \( -name "libint-*.tgz" -o -name "libint-*.tar.gz" \) -type f | head -1)

if [ -n "${GENERATED_TARBALL}" ]; then
    echo "Found generated tarball: ${GENERATED_TARBALL}"
    cp "${GENERATED_TARBALL}" /workspace/output/libint-${LIBINT_VERSION}.tar.gz
    echo "Copied to: /workspace/output/libint-${LIBINT_VERSION}.tar.gz"
    # Verify the tarball
    echo "Tarball size: $(du -h /workspace/output/libint-${LIBINT_VERSION}.tar.gz)"
else
    echo "Error: No libint tarball found!"
    echo "Available files in build directory:"
    ls -la
    exit 1
fi

echo "=== Libint Compiler Build Completed Successfully ==="
echo "Generated library: /workspace/output/libint-${LIBINT_VERSION}.tar.gz"
