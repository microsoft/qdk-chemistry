#!/bin/bash
set -e
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
    brew update
    brew upgrade
    brew install \
        wget \
        curl \
        unzip
fi

# Install pyenv to use non-system python3 versions
if [ ! -d "$PYENV_ROOT" ]; then
    wget -q https://github.com/pyenv/pyenv/archive/refs/heads/master.zip -O pyenv.zip
    unzip -q pyenv.zip
    mv pyenv-master "$PYENV_ROOT"
    rm pyenv.zip
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

# Run pytest suite
echo '=== Running pytest suite ==='
python3 -m pytest -v ./tests

deactivate
