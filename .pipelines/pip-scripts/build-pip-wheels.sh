#!/bin/bash
set -ex

MARCH=${1:-x86-64-v3}
PYTHON_VERSION=${2:-3.11}
BUILD_TYPE=${3:-Release}
BUILD_TESTING=${4:-ON}
ENABLE_COVERAGE=${5:-OFF}
CMAKE_VERSION=${6:-3.28.3}
HDF5_VERSION=${7:-1.13.0}
BLIS_VERSION=${8:-2.0}
LIBFLAME_VERSION=${9:-5.2.0}
MAC_BUILD=${10:-OFF}

export CFLAGS="-fPIC -Os"
if [ "$MAC_BUILD" == "OFF" ]; then # Build/install Linux dependencies
    export DEBIAN_FRONTEND=noninteractive
    # Try to prevent stochastic segfault from libc-bin
    echo "Reinstalling libc-bin..."
    rm /var/lib/dpkg/info/libc-bin.*
    apt-get clean
    apt-get update -q
    apt-get install -y -q libc-bin

    # Update and install dependencies
    echo "Installing apt dependencies..."
    apt-get update -q
    apt-get install -y -q \
        build-essential \
        curl \
        gcc g++ \
        git \
        libboost-all-dev \
        libbz2-dev \
        libeigen3-dev \
        libffi-dev \
        libfmt-dev \
        libgmock-dev \
        libgtest-dev \
        liblzma-dev \
        libncursesw5-dev \
        libpugixml-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        make \
        ninja-build \
        nlohmann-json3-dev \
        patchelf \
        pybind11-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-pybind11 \
        python3-venv \
        tk-dev \
        unzip \
        wget \
        xz-utils \
        zlib1g-dev

    # Upgrade cmake as Ubuntu 22.04 only has up to v3.22 in apt
    echo "Downloading and installing CMake ${CMAKE_VERSION}..."
    export CMAKE_CHECKSUM=72b7570e5c8593de6ac4ab433b73eab18c5fb328880460c86ce32608141ad5c1
    wget -q https://cmake.org/files/v3.28/cmake-${CMAKE_VERSION}.tar.gz -O cmake-${CMAKE_VERSION}.tar.gz
    echo "${CMAKE_CHECKSUM}  cmake-${CMAKE_VERSION}.tar.gz" | shasum -a 256 -c || exit 1
    tar -xzf cmake-${CMAKE_VERSION}.tar.gz
    rm cmake-${CMAKE_VERSION}.tar.gz
    cd cmake-${CMAKE_VERSION}
    ./bootstrap --parallel=$(nproc) --prefix=/usr/local
    make --silent -j$(nproc)
    make install
    cd ..
    rm -r cmake-${CMAKE_VERSION}
    cmake --version

    # We use BLIS/libflame as the BLAS/LAPACK vendors to prevent symbol collisions
    # with qiskit's shared OpenBLAS
    echo "Downloading and installing BLIS..."
    bash .pipelines/install-scripts/install-blis.sh /usr/local ${MARCH} ${BLIS_VERSION} "${CFLAGS}"

    echo "Downloading and installing libflame..."
    bash .pipelines/install-scripts/install-libflame.sh /usr/local ${MARCH} ${LIBFLAME_VERSION} "${CFLAGS}"
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

echo "Downloading HDF5 $HDF5_VERSION..."
# Clean up any leftover state from a previous (possibly failed) attempt on
# this self-hosted agent — the workspace persists across builds and retries.
rm -rf hdf5 hdf5-${HDF5_VERSION} hdf5-${HDF5_VERSION}.tar.bz2
export HDF5_CHECKSUM=1826e198df8dac679f0d3dc703aba02af4c614fd6b7ec936cf4a55e6aa0646ec
wget -q -nc --no-check-certificate https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.bz2
echo "${HDF5_CHECKSUM}  hdf5-${HDF5_VERSION}.tar.bz2" | shasum -a 256 -c || exit 1
tar -xjf hdf5-${HDF5_VERSION}.tar.bz2
rm hdf5-${HDF5_VERSION}.tar.bz2
mv hdf5-${HDF5_VERSION} hdf5
echo "HDF5 $HDF5_VERSION downloaded and extracted successfully"

echo "Installing HDF5..."
bash .pipelines/install-scripts/install-hdf5.sh /usr/local ${BUILD_TYPE} ${PWD} "${CFLAGS}" ${MAC_BUILD}

# =============================================================================
# Bootstrap Anaconda's `conda` for Python ${PYTHON_VERSION}.
# =============================================================================
#
# Anaconda is the officially approved Python distribution for Microsoft CI
# builds (Azure Pipelines / OneBranch); see
#   https://eng.ms/docs/more/languages-at-microsoft/python/articles/anaconda/install
# (section "Setting up Conda in CI builds"). The MS-vetted bootstrapper is
# `ms-ensureconda`.
#
# !!! HEADS UP: ms-ensureconda has incomplete platform coverage.            !!!
# !!! As of writing it only ships `manylinux_2_7_x86_64` and `win_amd64`    !!!
# !!! wheels — there are NO wheels for macOS (any arch) or Linux aarch64.   !!!
# !!! See:                                                                  !!!
# !!!   https://pkgs.dev.azure.com/ms-azurequantum/AzureQuantum/_packaging/quantum-apps-dependencies/pypi/simple/ms-ensureconda/
# !!! On those platforms `pip install ms-ensureconda` aborts with           !!!
# !!! "No matching distribution found".                                     !!!
# !!!                                                                       !!!
# !!! Until ms-ensureconda publishes broader wheel coverage we fall back to !!!
# !!! the upstream public `ensureconda` package (pure-Python `py3-none-any`,!!!
# !!! same module name). It is fetched from the same Azure Artifacts feed   !!!
# !!! via its public PyPI upstream so it still flows through CFS.           !!!
# !!!                                                                       !!!
# !!! IMPORTANT: ms-ensureconda has --envfile (dumps CONDA_BASH_HOOK etc);  !!!
# !!! public ensureconda does NOT. The bootstrap logic branches on package. !!!
# !!!                                                                       !!!
# !!! TODO: revert to ms-ensureconda on every platform once arm64 / macOS   !!!
# !!! wheels are published. File via python@microsoft.com if needed.        !!!
#
# Prereq: the AzureQuantum/quantum-apps-dependencies feed must have
# azure-feed://mseng/Anaconda@Published configured as an upstream so that
# ms-ensureconda resolves on Linux x86_64. PIP_INDEX_URL is set by
# PipAuthenticate@1 at job level; on Linux it's forwarded into the docker
# container via -e PIP_INDEX_URL.
case "$(uname -s):$(uname -m)" in
    Linux:x86_64)
        ENSURECONDA_PKG="ms-ensureconda==2026.2.1"
        ;;
    *)
        # macOS (arm64) and Linux aarch64 — see HEADS UP block above.
        ENSURECONDA_PKG="ensureconda==2025.1.0"
        ;;
esac

echo "Installing ${ENSURECONDA_PKG} (on $(uname -s):$(uname -m)) and bootstrapping conda..."
# Use a throwaway venv so we don't fight PEP 668 (Homebrew Python on macOS
# and apt Python on Ubuntu 24.04 are both externally managed).
# Clean any stale venv first (idempotent on self-hosted agents / retries).
rm -rf /tmp/bootstrap-venv
python3 -m venv /tmp/bootstrap-venv
# shellcheck disable=SC1091
. /tmp/bootstrap-venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install "${ENSURECONDA_PKG}"

if [ "$ENSURECONDA_PKG" = "ms-ensureconda" ]; then
    # ms-ensureconda's --envfile flag dumps CONDA_BASH_HOOK + friends to a
    # dotenv file we then source to wire up conda for this shell.
    python3 -m ensureconda --envfile /tmp/ensureconda.env
    deactivate
    set -a; . /tmp/ensureconda.env; set +a
    # shellcheck disable=SC1090
    . "$CONDA_BASH_HOOK"
else
    # Public ensureconda has no --envfile (it's an ms-ensureconda extension);
    # it just prints the discovered / installed conda binary path on stdout.
    # Force it to install conda-standalone (default would be micromamba, which
    # uses a different CLI: `micromamba shell hook --shell bash` instead of
    # `conda shell.bash hook`, plus mamba-specific `create`/`activate`).
    CONDA_EXE=$(python3 -m ensureconda --no-mamba --no-micromamba --conda --conda-exe)
    deactivate
    export CONDA_EXE
    # shellcheck disable=SC1090
    eval "$("$CONDA_EXE" shell.bash hook)"
fi

echo "Creating conda environment with Python ${PYTHON_VERSION}..."
# Explicitly include `pip` — fresh conda envs created with conda-standalone do
# not include pip by default, which would break `python3 -m pip ...` below.
#
# On Linux x86_64 (ms-ensureconda path) we are running under 1ES network
# isolation (CFSClean): public conda channels (conda.anaconda.org,
# repo.anaconda.com) are blocked. Force conda to install everything from the
# Azure Artifacts feed's Conda channel (proxied through the
# azure-feed://mseng/Anaconda@Published upstream).
#
# The feed exposes its upstream conda channels as named subpaths under
# /Conda/repo/<channel>/ (the feed root /Conda/repo/ itself returns 404 — it
# is not a channel). We use `main` (the Anaconda defaults channel that hosts
# python+pip) and `conda-forge` as a fallback for any package that defaults
# wouldn't carry.
#
# Auth: the conda install shipped by ms-ensureconda has a pre-registered
# azure_artifacts_conda_auth plugin that injects auth on every Azure
# Artifacts HTTPS request by reading $ARTIFACTS_CONDA_TOKEN. We just need to
# set that env var (to the pipeline's System.AccessToken). The plugin then
# handles auth; the channel URL itself must NOT inline the token (the plugin
# crashes with UnboundLocalError if the var is missing, even when basic-auth
# creds are present in the URL).
#
# Remove any existing conda env with the same name first (idempotent on
# self-hosted agents and on retries).
conda env remove -y -n buildenv 2>/dev/null || true
if [ "$ENSURECONDA_PKG" = "ms-ensureconda==2026.2.1" ]; then
    : "${SYSTEM_ACCESSTOKEN:?SYSTEM_ACCESSTOKEN must be set when bootstrapping conda from the Azure Artifacts feed}"
    { set +x; } 2>/dev/null
    export ARTIFACTS_CONDA_TOKEN="${SYSTEM_ACCESSTOKEN}"
    set -x
    CONDA_FEED_ROOT="https://pkgs.dev.azure.com/ms-azurequantum/AzureQuantum/_packaging/quantum-apps-dependencies/Conda/repo"
    conda create --override-channels \
                 --channel "${CONDA_FEED_ROOT}/main" \
                 --channel "${CONDA_FEED_ROOT}/conda-forge" \
                 --yes --quiet --name buildenv "python=${PYTHON_VERSION}" pip
else
    conda create --yes --quiet --name buildenv "python=${PYTHON_VERSION}" pip
fi
conda activate buildenv

python3 --version

# Update pip and install build tools
python3 -m pip install --upgrade pip

# This is necessary for 1ES Geneva telemetry during the Linux builds.
python3 -m pip install -r .pipelines/requirements.txt

# Print installed packages for debugging
echo "------------------ Installed Python packages ------------------"
python3 -m pip freeze
echo "---------------------------------------------------------------"

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
