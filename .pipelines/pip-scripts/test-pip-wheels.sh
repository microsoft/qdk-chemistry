#!/bin/bash
set -e
PYTHON_VERSION=${1:-3.11}

export DEBIAN_FRONTEND=noninteractive

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

# Install pyenv to use non-system python3 versions
export PYENV_ROOT="/workspace/.pyenv"
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
python3 -m venv /workspace/test_wheel_env
. /workspace/test_wheel_env/bin/activate

# Upgrade pip packages
python3 -m pip install --upgrade pip
python3 -m pip install "fonttools>=4.61.0" "urllib3>=2.6.0"

# Install the wheel in the clean environment
cd /workspace/qdk-chemistry/python
python3 -m pip install pytest pyscf
pip3 install repaired_wheelhouse/qdk_chemistry*.whl

# Run pytest suite
echo '=== Running pytest suite ==='
python3 -m pytest -v ./tests

deactivate
