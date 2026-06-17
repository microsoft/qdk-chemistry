#!/bin/bash
set -ex

MARCH=${1:-x86-64-v3}
PYTHON_VERSION=${2:-3.11}
BUILD_TYPE=${3:-Release}
BUILD_TESTING=${4:-ON}
ENABLE_COVERAGE=${5:-OFF}
HDF5_VERSION=${6:-1.13.0}
BLIS_VERSION=${7:-2.0}
LIBFLAME_VERSION=${8:-5.2.0}
MAC_BUILD=${9:-OFF}

export CFLAGS="-fPIC -Os"
# Use sudo for system-level installs when running as a non-root pipeline agent.
SUDO=""
[ "$(id -u)" != "0" ] && SUDO="sudo"

if [ "$MAC_BUILD" == "OFF" ]; then # Build/install Linux dependencies
    # Update and install dependencies (Azure Linux 3 / tdnf)
    echo "Installing tdnf dependencies..."
    $SUDO tdnf update -y
    $SUDO tdnf install -y \
        binutils \
        boost-devel \
        bzip2-devel \
        cmake \
        curl \
        fmt-devel \
        gcc \
        gcc-c++ \
        git \
        glibc-devel \
        gtest-devel \
        libffi-devel \
        libxml2-devel \
        make \
        ncurses-devel \
        ninja-build \
        nlohmann-json-devel \
        openssl-devel \
        patchelf \
        pugixml-devel \
        pybind11-devel \
        python3 \
        python3-devel \
        readline-devel \
        sqlite-devel \
        unzip \
        wget \
        xmlsec1-devel \
        xz \
        xz-devel \
        zlib-devel
    cmake --version

    # Eigen3 is not packaged in Azure Linux 3; download from internal Azure Artifacts feed
    echo "Installing Eigen3 headers..."
    AZURE_DEVOPS_EXT_PAT=$SYSTEM_ACCESSTOKEN az artifacts universal download \
        --organization https://dev.azure.com/ms-azurequantum \
        --project AzureQuantum \
        --scope project \
        --feed quantum-apps-dependencies \
        --name eigen3 \
        --version 3.4.0 \
        --path eigen3
    cmake -S eigen3 -B eigen3/build -DCMAKE_INSTALL_PREFIX=/usr/local
    $SUDO cmake --install eigen3/build
    rm -rf eigen3

    # We use BLIS/libflame as the BLAS/LAPACK vendors to prevent symbol collisions
    # with qiskit's shared OpenBLAS
    echo "Downloading and installing BLIS..."
    $SUDO bash .pipelines/install-scripts/install-blis.sh /usr/local ${MARCH} ${BLIS_VERSION} "${CFLAGS}"

    echo "Downloading and installing libflame..."
    $SUDO bash .pipelines/install-scripts/install-libflame.sh /usr/local ${MARCH} ${LIBFLAME_VERSION} "${CFLAGS}"
elif [ "$MAC_BUILD" == "ON" ]; then
    arch -arm64 brew update
    arch -arm64 brew upgrade
    arch -arm64 brew install \
        boost \
        cmake \
        curl \
        eigen \
        gcc \
        ncurses \
        ninja \
        pybind11 \
        python \
        wget
    export CMAKE_PREFIX_PATH="/opt/homebrew"
    # Make sure Homebrew's python3 is preferred when bootstrapping conda.
    export PATH="/opt/homebrew/bin:$PATH"
fi

echo "Installing HDF5..."
$SUDO bash .pipelines/install-scripts/install-hdf5.sh /usr/local ${BUILD_TYPE} ${PWD} "${CFLAGS}" ${MAC_BUILD} ${HDF5_VERSION}

# Bootstrap Anaconda's `conda` and create the build env. See header of the
# sourced script for full rationale (CFSClean / ms-ensureconda platform gaps /
# Azure Artifacts feed auth).
# shellcheck disable=SC1091
. .pipelines/pip-scripts/bootstrap-conda.sh buildenv
conda activate buildenv

python3 --version

# Update pip and install build tools
python3 -m pip install --upgrade pip

# This is necessary for 1ES Geneva telemetry during the Linux builds.
python3 -m pip install -r .pipelines/requirements.txt

# Snapshot the full env and feed it to a dry-run `pip install --report` so
# Component Governance's PipReportDetector sees every package in buildenv.
# The report is auto-discovered when it sits next to a setup.py or
# requirements.txt in a non-hidden directory (the detector skips dotdirs
# like .pipelines/). See:
#   https://github.com/microsoft/component-detection/blob/main/docs/detectors/pip.md
#   https://github.com/microsoft/component-detection/issues/243
mkdir -p python/build/build-manifest
echo "------------------ Installed Python packages (buildenv) ------------------"
python3 -m pip list --format=freeze | tee python/build/build-manifest/requirements.txt
echo "---------------------------------------------------------------------------"
python3 -m pip install --dry-run --ignore-installed --quiet \
    --report python/build/build-manifest/component-detection-pip-report.json \
    -r python/build/build-manifest/requirements.txt

# Prepare README for PyPI
bash .pipelines/pip-scripts/prepare-readme.sh

# Install Python package
cd python

# Build wheel with all necessary CMake flags
if [ "$MAC_BUILD" == "OFF" ]; then
    # We need to include the -g flag so that we can publish our symbols internally
    if [ "$MARCH" == "x86-64-v3" ]; then
        export CMAKE_C_FLAGS="-march=${MARCH} -fPIC -Os -fvisibility=hidden -g"
        export CMAKE_CXX_FLAGS="-march=${MARCH} -fPIC -Os -fvisibility=hidden -g"
    else
        export CMAKE_C_FLAGS="-march=${MARCH} -fPIC -Os -fvisibility=hidden"
        export CMAKE_CXX_FLAGS="-march=${MARCH} -fPIC -Os -fvisibility=hidden"
    fi
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
    python3 -m build --wheel \
        -C build-dir="build/{wheel_tag}" \
        -C cmake.define.QDK_UARCH=${MARCH} \
        -C cmake.define.BUILD_SHARED_LIBS=OFF \
        -C cmake.define.QDK_CHEMISTRY_ENABLE_MPI=OFF \
        -C cmake.define.QDK_ENABLE_OPENMP=OFF \
        -C cmake.define.QDK_CHEMISTRY_ENABLE_COVERAGE=${ENABLE_COVERAGE} \
        -C cmake.define.BUILD_TESTING=${BUILD_TESTING} \
        -C cmake.define.CMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
        -C cmake.define.CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"

    echo "Checking shared dependencies..."
    ldd build/cp*/_core.*.so

    # Repair wheel
    auditwheel repair dist/qdk_chemistry-*.whl -w repaired_wheelhouse/

    # Fix RPATH
    WHEEL_FILE=$(ls repaired_wheelhouse/qdk_chemistry-*.whl)
    FULL_WHEEL_PATH="$PWD/$WHEEL_FILE"
    TEMP_DIR=$(mktemp -d)
    python3 -m zipfile -e "$WHEEL_FILE" "$TEMP_DIR"

    find "$TEMP_DIR" -name '*.so*' -type f -not -path '*/qdk_chemistry.libs/*' | while read so_file; do
        echo "Fixing RPATH for main package: $so_file"
        patchelf --set-rpath '$ORIGIN/../../qdk_chemistry.libs' "$so_file" || true
    done

    # We need to do this in order to publish our C++ debug symbols internally.
    if [ "$MARCH" == "x86-64-v3" ]; then
        export debugdir="debug_symbols"
        mkdir -p "${debugdir}"
        CORE_SO="$(find "$TEMP_DIR" -type f -name '_core*.so' | head -n 1)"
        if [ -z "$CORE_SO" ]; then
            echo "ERROR: Could not find _core*.so in repaired wheel contents."
            exit 1
        fi
        debugfile="$(basename "$CORE_SO").debug"
        objcopy --only-keep-debug "$CORE_SO" "${debugdir}/${debugfile}"
        strip --strip-debug --strip-unneeded "$CORE_SO"
        objcopy --add-gnu-debuglink="${debugdir}/${debugfile}" "$CORE_SO"
        echo "Extracted debug symbols to ${debugdir}/${debugfile}"
        ls "${debugdir}"
    fi

    find "$TEMP_DIR" -path '*/qdk_chemistry.libs/*' -name '*.so*' -type f | while read so_file; do
        echo "Fixing RPATH for bundled library: $so_file"
        patchelf --set-rpath '$ORIGIN' "$so_file" || true
    done

    rm "$WHEEL_FILE"
    (cd "$TEMP_DIR" && python3 -m zipfile -c "$FULL_WHEEL_PATH" .)
    rm -rf "$TEMP_DIR"

elif [ "$MAC_BUILD" == "ON" ]; then
    export CMAKE_C_FLAGS="-fPIC -Os -fvisibility=hidden -target arm64-apple-darwin"
    export CMAKE_CXX_FLAGS="-fPIC -Os -fvisibility=hidden -target arm64-apple-darwin"
    python3 -m build --wheel \
        -C build-dir="build/{wheel_tag}" \
        -C cmake.define.QDK_UARCH=native \
        -C cmake.define.BUILD_SHARED_LIBS=OFF \
        -C cmake.define.QDK_CHEMISTRY_ENABLE_MPI=OFF \
        -C cmake.define.QDK_ENABLE_OPENMP=OFF \
        -C cmake.define.QDK_CHEMISTRY_ENABLE_COVERAGE=${ENABLE_COVERAGE} \
        -C cmake.define.BUILD_TESTING=${BUILD_TESTING} \
        -C cmake.define.CMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
        -C cmake.define.CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
        -C cmake.define.CMAKE_PREFIX_PATH="/opt/homebrew"
    echo "Repairing wheel for macOS..."
    WHEEL_FILE=$(ls dist/qdk_chemistry-*.whl)
    delocate-wheel -w repaired_wheelhouse/ "$WHEEL_FILE"
    delocate-listdeps --all repaired_wheelhouse/qdk_chemistry*.whl

    echo "Checking shared dependencies..."
    otool -L build/cp*/_core.*.so
fi
