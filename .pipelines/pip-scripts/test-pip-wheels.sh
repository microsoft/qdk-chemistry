#!/bin/bash
set -e
PYTHON_VERSION=${1:-3.11}
MAC_BUILD=${2:-OFF}
PYENV_VERSION=${3:-2.6.15}
export MAC_BUILD

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -d "/workspace/qdk-chemistry/python" ]; then
    PYTHON_DIR="/workspace/qdk-chemistry/python"
else
    PYTHON_DIR="$REPO_ROOT/python"
fi

if [ "$MAC_BUILD" == "OFF" ] && [ -d "/workspace" ]; then
    export PYENV_ROOT="/workspace/.pyenv"
    VENV_DIR="/workspace/test_wheel_env"
else
    export PYENV_ROOT="$REPO_ROOT/.pyenv"
    VENV_DIR="$REPO_ROOT/.test_wheel_env"
fi

export DEBIAN_FRONTEND=noninteractive

if [ "$MAC_BUILD" == "OFF" ]; then
    # Try to prevent stochastic segfault from libc-bin
    echo "Reinstalling libc-bin..."
    rm /var/lib/dpkg/info/libc-bin.*
    apt-get clean
    apt-get update
    apt install -q libc-bin

    # Update and install dependencies needed for testing
    echo "Installing apt dependencies..."
    apt-get update
    apt-get install -q -y \
        python3 python3-pip python3-venv python3-dev \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
        libffi-dev liblzma-dev \
        libopenblas-dev \
        libboost-all-dev \
        wget \
        curl \
        unzip
elif [ "$MAC_BUILD" == "ON" ]; then
    arch -arm64 brew update
    arch -arm64 brew upgrade
    arch -arm64 brew install \
        wget \
        curl \
        unzip
fi

# Install pyenv to use non-system python3 versions
if [ ! -d "$PYENV_ROOT" ]; then
    echo "Installing pyenv ${PYENV_VERSION}..."
    export PYENV_CHECKSUM=95187d6ad9bc8310662b5b805a88506e5cbbe038f88890e5aabe3021711bf3c8
    wget -q https://github.com/pyenv/pyenv/archive/refs/tags/v${PYENV_VERSION}.zip -O pyenv.zip
    echo "${PYENV_CHECKSUM}  pyenv.zip" | shasum -a 256 -c || exit 1
    unzip -q pyenv.zip
    mv pyenv-${PYENV_VERSION} "$PYENV_ROOT"
    rm pyenv.zip
    "$PYENV_ROOT/bin/pyenv" install ${PYTHON_VERSION}
    "$PYENV_ROOT/bin/pyenv" global ${PYTHON_VERSION}
    export PATH="$PYENV_ROOT/versions/${PYTHON_VERSION}/bin:$PATH"
    export PATH="$PYENV_ROOT/shims:$PATH"
fi

# Install and activate the specific Python version
"$PYENV_ROOT/bin/pyenv" install $PYTHON_VERSION --skip-existing
"$PYENV_ROOT/bin/pyenv" global $PYTHON_VERSION
export PATH="$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"
export PATH="$PYENV_ROOT/shims:$PATH"

python3 --version

# Create a clean virtual environment for testing the wheel
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"

# Upgrade pip packages
python3 -m pip install --upgrade pip
python3 -m pip install "fonttools>=4.61.0" "urllib3>=2.6.0"

# Install the wheel in the clean environment
cd "$PYTHON_DIR"
python3 -m pip install pytest pyscf
pip3 install repaired_wheelhouse/qdk_chemistry*.whl

# Install qiskit-extras if supported (not available on Python 3.14+)
# qiskit-aer and qiskit-nature do not yet have Python 3.14 wheels
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MINOR" -lt 14 ]; then
    echo "Installing qiskit-extras..."
    python3 -m pip install qiskit-aer qiskit-nature
else
    echo "Skipping qiskit-extras (not supported on Python 3.14+)"
fi

# Run pytest suite
echo '=== Running pytest suite ==='
python3 -m pytest -v ./tests

deactivate
