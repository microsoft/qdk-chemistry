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

if [ "$MAC_BUILD" == "OFF" ] && [ -d "/workspace" ]; then
    VENV_DIR="/workspace/test_wheel_env"
else
    VENV_DIR="$REPO_ROOT/.test_wheel_env"
fi

export DEBIAN_FRONTEND=noninteractive

if [ "$MAC_BUILD" == "OFF" ]; then
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
        tk-dev \
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
        unzip \
        wget
fi

# On Linux, install Python ${PYTHON_VERSION} inside the container via Anaconda's conda
# package manager, bootstrapped by Microsoft's CFS-compliant ms-ensureconda tool.
# Anaconda is the officially approved Python distribution for Microsoft CI builds
# (Azure Pipelines / OneBranch); see
# https://eng.ms/docs/more/languages-at-microsoft/python/articles/anaconda/install
# (section "Setting up Conda in CI builds").
#
# Prereq: the AzureQuantum/quantum-apps-dependencies Azure Artifacts feed must have
# azure-feed://mseng/Anaconda@Published configured as an upstream so that the
# ms-ensureconda pip package resolves inside this container. PIP_INDEX_URL is
# forwarded into the container by the pipeline template after PipAuthenticate@1.
#
# On macOS, the requested Python is provided by the UsePythonVersion@0 ADO task in
# the pipeline template, which puts it on PATH for this script.
if [ "$MAC_BUILD" == "OFF" ]; then
    echo "Installing ms-ensureconda and bootstrapping conda..."
    python3 -m pip install --upgrade pip
    python3 -m pip install ms-ensureconda
    python3 -m ensureconda --envfile /tmp/ensureconda.env
    set -a; . /tmp/ensureconda.env; set +a
    # shellcheck disable=SC1090
    . "$CONDA_BASH_HOOK"

    echo "Creating fresh conda environment for wheel test with Python ${PYTHON_VERSION}..."
    conda create --yes --quiet --name testenv "python=${PYTHON_VERSION}"
    conda activate testenv
fi

python3 --version

# On macOS we still use a venv for isolation; on Linux the conda env above already
# provides an isolated Python.
if [ "$MAC_BUILD" == "ON" ]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    . "$VENV_DIR/bin/activate"
fi

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

# Print installed packages for debugging
echo "------------------ Installed Python packages ------------------"
python3 -m pip freeze
echo "---------------------------------------------------------------"

# Disable telemetry during testing
export QSHARP_PYTHON_TELEMETRY=false

# Run pytest suite
echo '=== Running pytest suite ==='
python3 -m pytest -v ./tests

if [ "$MAC_BUILD" == "ON" ]; then
    deactivate
fi
