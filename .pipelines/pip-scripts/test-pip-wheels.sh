#!/bin/bash
set -ex
PYTHON_VERSION=${1:-3.11}
MAC_BUILD=${2:-OFF}
export MAC_BUILD

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -d "/workspace/qdk-chemistry/python" ]; then
    PYTHON_DIR="/workspace/qdk-chemistry/python"
else
    PYTHON_DIR="$REPO_ROOT/python"
fi

# ExaChem/TAMM (built separately, in a preliminary Linux x86_64-only pipeline job -- see
# .pipelines/templates/build-exachem-linux.yml -- and downloaded here as a pipeline artifact) is optional: only
# present for the Linux x86_64 leg. Auto-detected the same way PYTHON_DIR is above, rather than via a script
# argument, so this script needs no changes for legs that never build/download it (aarch64, macOS, Windows) --
# the ExaChem CCSD integration tests (tests/test_exachem_ccsd_integration.py) already self-skip via
# shutil.which("ExaChem") when it is not on PATH.
if [ -d "/workspace/qdk-chemistry/exachem_install" ]; then
    EXACHEM_INSTALL_DIR="/workspace/qdk-chemistry/exachem_install"
else
    EXACHEM_INSTALL_DIR=""
fi

export DEBIAN_FRONTEND=noninteractive

if [ "$MAC_BUILD" == "OFF" ]; then
    # CFSClean3: redirect Ubuntu apt endpoints to the Azure-internal mirror.
    # Note: azure.archive.ubuntu.com only carries amd64/i386. Non-x86 architectures
    # (e.g. arm64) live under ubuntu-ports and must go to azure.ports.ubuntu.com,
    # otherwise apt gets a 404 for binary-<arch>/Packages and exits with code 100.
    _cfs_apt_redirect() {
        sed -i \
            -e 's|https\?://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
            -e 's|https\?://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
            -e 's|https\?://ports.ubuntu.com/ubuntu-ports|http://azure.ports.ubuntu.com/ubuntu-ports|g' \
            "$1"
    }
    [ -f /etc/apt/sources.list.d/ubuntu.sources ] && _cfs_apt_redirect /etc/apt/sources.list.d/ubuntu.sources
    [ -f /etc/apt/sources.list ]                  && _cfs_apt_redirect /etc/apt/sources.list

    # Try to prevent stochastic segfault from libc-bin
    echo "Reinstalling libc-bin..."
    rm /var/lib/dpkg/info/libc-bin.*
    apt-get clean
    apt-get update -q
    apt-get install -y -q libc-bin

    # Update and install dependencies needed for testing
    echo "Installing apt dependencies..."
    apt-get update -q
    apt-get install -y -q \
        build-essential \
        curl \
        git \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        make \
        python3 \
        python3-pip \
        python3-venv \
        tk-dev \
        unzip \
        wget \
        xz-utils \
        zlib1g-dev

    # ExaChem is a separate MPI process (not linked into the qdk_chemistry wheel), so it needs its own runtime
    # deps installed here -- this is a fresh container, not the one it was built in. Matches exactly the package
    # set build-exachem-linux.sh installs on the build side (see that script for why each one is needed); apt
    # pulls in any further transitive runtime libs (e.g. libopenmpi3, libgfortran5) automatically.
    if [ -n "$EXACHEM_INSTALL_DIR" ]; then
        echo "Installing ExaChem runtime apt dependencies..."
        apt-get install -y -q \
            gfortran \
            libhdf5-dev \
            libnuma-dev \
            openmpi-bin \
            libopenmpi-dev
    fi

elif [ "$MAC_BUILD" == "ON" ]; then
    arch -arm64 brew update
    arch -arm64 brew upgrade
    arch -arm64 brew install \
        curl \
        ncurses \
        python \
        unzip \
        wget
    # Make sure Homebrew's python3 is preferred when bootstrapping conda.
    export PATH="/opt/homebrew/bin:$PATH"
fi

# Bootstrap Anaconda's `conda` and create the test env. See header of the
# sourced script for full rationale (CFSClean / ms-ensureconda platform gaps /
# Azure Artifacts feed auth).
# shellcheck disable=SC1091
. "$SCRIPT_DIR/bootstrap-conda.sh" testenv
conda activate testenv

python3 --version

python3 -m pip install --upgrade pip

# Install the wheel in the clean environment
cd "$PYTHON_DIR"

# Install built wheel with test dependencies
WHEEL=(repaired_wheelhouse/qdk_chemistry*.whl)
if [ ${#WHEEL[@]} -ne 1 ] || [ ! -f "${WHEEL[0]}" ]; then
    echo "ERROR: Expected exactly 1 wheel, found ${#WHEEL[@]}: ${WHEEL[*]}"
    exit 1
fi
python3 -m pip install "${WHEEL[0]}[test]"

# Snapshot the full env and feed it to a dry-run `pip install --report` so
# Component Governance's PipReportDetector sees every package in testenv.
# The report is auto-discovered when it sits next to a setup.py or
# requirements.txt in a non-hidden directory (the detector skips dotdirs
# like .pipelines/). The locally-built qdk_chemistry wheel is excluded
# because it is not resolvable from any index. See:
#   https://github.com/microsoft/component-detection/blob/main/docs/detectors/pip.md
#   https://github.com/microsoft/component-detection/issues/243
mkdir -p "$PYTHON_DIR/build/test-manifest"
echo "------------------ Installed Python packages (testenv) ------------------"
python3 -m pip list --format=freeze --exclude qdk_chemistry \
    | tee "$PYTHON_DIR/build/test-manifest/requirements.txt"
echo "-------------------------------------------------------------------------"
python3 -m pip install --dry-run --ignore-installed --quiet \
    --report "$PYTHON_DIR/build/test-manifest/component-detection-pip-report.json" \
    -r "$PYTHON_DIR/build/test-manifest/requirements.txt"

# Disable telemetry during testing
export QSHARP_PYTHON_TELEMETRY=false

# ExaChem's binary/libs are only needed here, for the ExaChem CCSD integration test's shutil.which("ExaChem")
# lookup -- scoped to just this pytest invocation (matching the same scoping decision already made in GHA's
# build-and-test.yaml) rather than exported earlier, so nothing else in this script (pip install, Component
# Governance's pip report, ...) is affected by it.
if [ -n "$EXACHEM_INSTALL_DIR" ]; then
    export PATH="${EXACHEM_INSTALL_DIR}/bin:${PATH}"
    export LD_LIBRARY_PATH="${EXACHEM_INSTALL_DIR}/lib:${EXACHEM_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"
fi

# Run pytest suite
echo '=== Running pytest suite ==='
python3 -m pytest -v ./tests
