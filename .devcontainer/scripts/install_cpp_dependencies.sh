#!/bin/bash
set -e

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <cpp_cgmanifest_path> <macis_cgmanifest_path>" >&2
    exit 1
fi

INSTALL_SCRIPTS="${INSTALL_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.pipelines/install-scripts" && pwd)}"
PARALLELISM_HELPER="/usr/local/share/qdk/parallelism.sh"

if [[ -f "$PARALLELISM_HELPER" ]]; then
    # shellcheck source=/dev/null
    source "$PARALLELISM_HELPER"
fi

if ! command -v parallel_jobs_for_memory >/dev/null 2>&1; then
    echo "Error: parallel_jobs_for_memory is unavailable; install $PARALLELISM_HELPER first." >&2
    exit 1
fi

export MARCH="${MARCH:-native}"
export BLAS_VENDOR="${BLAS_VENDOR:-openblas}"
export INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
export BUILD_DIR="${BUILD_DIR:-/tmp/qdk_deps_build}"
export BUILD_TYPE="${BUILD_TYPE:-Release}"
export BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-OFF}"
export KEEP_BUILD_DIR="${KEEP_BUILD_DIR:-0}"
export JOBS="${JOBS:-$(parallel_jobs_for_memory 1)}"
export LIBINT_JOBS="${LIBINT_JOBS:-$(parallel_jobs_for_memory 4)}"

exec bash "${INSTALL_SCRIPTS}/install-cpp-deps.sh" "$1" "$2" "$BLAS_VENDOR"
