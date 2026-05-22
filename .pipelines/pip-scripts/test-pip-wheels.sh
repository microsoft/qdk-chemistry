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
        python3 \
        python3-pip \
        python3-venv \
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
        python \
        unzip \
        wget
    # Make sure Homebrew's python3 is preferred when bootstrapping conda.
    export PATH="/opt/homebrew/bin:$PATH"
fi

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
# !!! identical `python -m ensureconda --envfile` CLI). It is fetched from  !!!
# !!! the same Azure Artifacts feed via its public PyPI upstream so it      !!!
# !!! still flows through CFS.                                              !!!
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
        ENSURECONDA_PKG="ms-ensureconda"
        ;;
    *)
        # macOS (arm64) and Linux aarch64 — see HEADS UP block above.
        ENSURECONDA_PKG="ensureconda"
        ;;
esac

echo "Installing ${ENSURECONDA_PKG} (on $(uname -s):$(uname -m)) and bootstrapping conda..."
# Use a throwaway venv so we don't fight PEP 668 (Homebrew Python on macOS
# and apt Python on Ubuntu 24.04 are both externally managed).
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

echo "Creating fresh conda environment for wheel test with Python ${PYTHON_VERSION}..."
# Explicitly include `pip` — fresh conda envs created with conda-standalone do
# not include pip by default, which would break `python3 -m pip ...` below.
conda create --yes --quiet --name testenv "python=${PYTHON_VERSION}" pip
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

# Print installed packages for debugging
echo "------------------ Installed Python packages ------------------"
python3 -m pip freeze
echo "---------------------------------------------------------------"

# Disable telemetry during testing
export QSHARP_PYTHON_TELEMETRY=false

# Run pytest suite
echo '=== Running pytest suite ==='
python3 -m pytest -v ./tests
