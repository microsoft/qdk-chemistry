# =============================================================================
# Bootstrap Anaconda's `conda` and create a conda environment.
# =============================================================================
#
# This script is meant to be *sourced* (not executed) from the pip-script
# entrypoints. After sourcing, the caller can `conda activate ${CONDA_ENV_NAME}`
# in the same shell.
#
# Required positional arg:
#   $1 = CONDA_ENV_NAME    e.g. "buildenv" or "testenv"
#
# Required env vars:
#   PYTHON_VERSION         e.g. "3.13"
#
# Additional required env on Linux x86_64 (only):
#   SYSTEM_ACCESSTOKEN     PAT used by the azure_artifacts_conda_auth plugin
#                          to authenticate against the Azure Artifacts feed.
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
# ms-ensureconda resolves. PIP_INDEX_URL is set by PipAuthenticate@1 at job
# level; on Linux it's forwarded into the docker container via -e PIP_INDEX_URL.

CONDA_ENV_NAME="${1:?bootstrap-conda.sh: missing CONDA_ENV_NAME (positional arg \$1)}"
: "${PYTHON_VERSION:?bootstrap-conda.sh: PYTHON_VERSION must be set before sourcing}"

# case "$(uname -s):$(uname -m)" in
#     Linux:x86_64)
#         ENSURECONDA_PKG="ms-ensureconda==2026.6.1"
#         USE_MS_ENSURECONDA=1
#         ;;
#     *)
#         # macOS (arm64) and Linux aarch64 — see HEADS UP block above.
#         ENSURECONDA_PKG="ensureconda==1.6.0"
#         USE_MS_ENSURECONDA=0
#         ;;
# esac
# https://dev.azure.com/mseng/Python/_artifacts/feed/Anaconda/PyPI/ms-ensureconda/overview/2026.6.1
# The platforms now include:
# win_amd64
# manylinux_2_35_x86_64 (created/tested on Azure Linux 3)
# manylinux_2_35_aarch64 (also on AL3)
# macos_15_0_x86_64 (using the standard macos-15 image)
# macos_15_0_arm64 (using the macos-15-arm image)
ENSURECONDA_PKG="ms-ensureconda==2026.6.1"
USE_MS_ENSURECONDA=1


# All network-sensitive operations (pip install, conda download, env creation)
# are wrapped in a single retryable function to handle transient failures.
_BOOTSTRAP_MAX_ATTEMPTS=3
_BOOTSTRAP_RETRY_DELAY=30

_bootstrap_conda() {
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

    if [ "$USE_MS_ENSURECONDA" = "1" ]; then
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

    echo "Creating conda environment '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}..."
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
    conda env remove -y -n "${CONDA_ENV_NAME}" 2>/dev/null || true
    if [ "$USE_MS_ENSURECONDA" = "1" ]; then
        : "${SYSTEM_ACCESSTOKEN:?SYSTEM_ACCESSTOKEN must be set when bootstrapping conda from the Azure Artifacts feed}"
        { set +x; } 2>/dev/null
        export ARTIFACTS_CONDA_TOKEN="${SYSTEM_ACCESSTOKEN}"
        set -x
        CONDA_FEED_ROOT="https://pkgs.dev.azure.com/ms-azurequantum/AzureQuantum/_packaging/quantum-apps-dependencies/Conda/repo"
        conda create --override-channels \
                     --channel "${CONDA_FEED_ROOT}/main" \
                     --channel "${CONDA_FEED_ROOT}/conda-forge" \
                     --yes --quiet --name "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" pip
    else
        conda create --yes --quiet --name "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" pip
    fi
}

_bootstrap_attempt=1
while true; do
    if _bootstrap_conda; then
        break
    fi
    if [ "$_bootstrap_attempt" -ge "$_BOOTSTRAP_MAX_ATTEMPTS" ]; then
        echo "ERROR: bootstrap-conda.sh failed after ${_BOOTSTRAP_MAX_ATTEMPTS} attempts." >&2
        return 1 2>/dev/null || exit 1
    fi
    echo "WARNING: bootstrap attempt ${_bootstrap_attempt}/${_BOOTSTRAP_MAX_ATTEMPTS} failed; retrying in ${_BOOTSTRAP_RETRY_DELAY}s..." >&2
    sleep "$_BOOTSTRAP_RETRY_DELAY"
    _bootstrap_attempt=$((_bootstrap_attempt + 1))
done
unset _bootstrap_attempt _BOOTSTRAP_MAX_ATTEMPTS _BOOTSTRAP_RETRY_DELAY
unset -f _bootstrap_conda
