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

if [ "$MAC_BUILD" == "OFF" ]; then
    # Update and install dependencies needed for testing (Ubuntu / apt-get)
    SUDO=""
    [ "$(id -u)" != "0" ] && SUDO="sudo"
    export DEBIAN_FRONTEND=noninteractive
    echo "Installing apt dependencies..."
    $SUDO apt-get update -q
    $SUDO apt-get install -y -q \
        curl \
        gcc \
        g++ \
        git \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncurses-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        make \
        python3 \
        python3-dev \
        python3-venv \
        unzip \
        wget \
        xz-utils \
        zlib1g-dev

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

# Run pytest suite
echo '=== Running pytest suite ==='
python3 -m pytest -v ./tests
