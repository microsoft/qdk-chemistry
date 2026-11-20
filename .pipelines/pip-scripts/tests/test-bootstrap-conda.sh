#!/bin/bash
# Run with bash; mocks keep this test offline and leave /tmp bootstrap files untouched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SCRIPT="${1:-$SCRIPT_DIR/../bootstrap-conda.sh}"
REAL_PYTHON="$(command -v python3)"

run_case() (
    set -e
    scenario="$1"
    export PYTHON_VERSION=3.13 SYSTEM_ACCESSTOKEN=test-token MAC_BUILD=OFF
    unset PYTHONHOME PYTHONPATH __PYVENV_LAUNCHER__ VIRTUAL_ENV
    attempts=0
    snapshots=0
    hooks=0
    creates=0
    runs=0
    case "$scenario" in mac-*) MAC_BUILD=ON ;; esac

    if [ "$scenario" = inherited-paths ]; then
        export PYTHONHOME=/qdk-test-missing-home PYTHONPATH=/qdk-test-wrong-path
        export __PYVENV_LAUNCHER__=/qdk-test-wrong-python
    fi

    rm() { [ "$*" = '-rf /tmp/bootstrap-venv' ]; }
    sleep() { :; }
    python3() {
        [ "$*" = '-m venv /tmp/bootstrap-venv' ] || return 1
        [ -z "${PYTHONHOME+x}${PYTHONPATH+x}${__PYVENV_LAUNCHER__+x}" ] || return 1
        "$REAL_PYTHON" -c 'import encodings'
    }

    # Bash permits path-named functions, so the real bootstrap interpreter is
    # never created or executed.
    function /tmp/bootstrap-venv/bin/python3() {
        [ -z "${VIRTUAL_ENV+x}" ] || return 1
        case "$*" in
            '-m pip install --upgrade pip') return 0 ;;
            '-m pip install ms-ensureconda=='*) return 0 ;;
            '-m ensureconda --envfile /tmp/ensureconda.env')
                attempts=$((attempts + 1))
                if [ "$scenario" = bootstrap-failure ] ||
                   { [ "$scenario" = retry ] && [ "$attempts" -eq 1 ]; }; then
                    return 1
                fi
                ;;
            *) return 1 ;;
        esac
    }

    function .() {
        case "$1" in
            /tmp/ensureconda.env)
                snapshots=$((snapshots + 1))
                export CONDA_BASH_HOOK=/qdk-test-conda-hook
                export PYTHONHOME=/qdk-test-missing-home PYTHONPATH=/qdk-test-wrong-path
                export __PYVENV_LAUNCHER__=/qdk-test-wrong-python
                [ "$scenario" != snapshot-failure ]
                ;;
            /qdk-test-conda-hook)
                hooks=$((hooks + 1))
                [ -z "${PYTHONHOME+x}${PYTHONPATH+x}${__PYVENV_LAUNCHER__+x}" ] || return 1
                "$REAL_PYTHON" -c 'import encodings' || return 1
                [ "$scenario" != hook-failure ]
                ;;
            *) echo "Unexpected source: $1" >&2; return 1 ;;
        esac
    }

    conda() {
        case "$*" in
            'env remove -y -n testenv') return 1 ;;
            create*)
                creates=$((creates + 1))
                [[ " $* " = *' --name testenv python=3.13 pip '* ]] || return 1
                if [ "$MAC_BUILD" = ON ]; then
                    [[ " $* " = *' --platform osx-arm64 '* ]] || return 1
                else
                    [[ " $* " != *' --platform '* ]] || return 1
                fi
                [ "$scenario" != create-failure ]
                ;;
            'run --no-capture-output --name testenv python -c '*)
                runs=$((runs + 1))
                "$REAL_PYTHON" -c '
import platform
import sys

code = sys.argv.pop(1)
scenario = sys.argv.pop(1)
sys.version_info = (3, 12, 0) if scenario == "wrong-version" else (3, 13, 0)
if scenario.startswith("mac-"):
    sys.platform = "darwin"
    platform.machine = lambda: "x86_64" if scenario == "mac-wrong-arch" else "arm64"
exec(code)
' "$7" "$scenario" "$8" "$9"
                ;;
            *) return 1 ;;
        esac
    }

    if builtin source "$BOOTSTRAP_SCRIPT" testenv; then
        result=0
    else
        result=$?
    fi

    case "$scenario" in
        bootstrap-failure)
            [ "$result" -ne 0 ]
            [ "$attempts" -eq 3 ]
            [ "$snapshots" -eq 0 ]
            [ "$creates" -eq 0 ]
            ;;
        snapshot-failure|hook-failure)
            [ "$result" -ne 0 ]
            [ "$attempts" -eq 3 ]
            [ "$creates" -eq 0 ]
            if [ "$scenario" = snapshot-failure ]; then
                [ "$hooks" -eq 0 ]
            fi
            ;;
        create-failure|wrong-version|mac-wrong-arch)
            [ "$result" -ne 0 ]
            [ "$attempts" -eq 3 ]
            [ "$creates" -eq 3 ]
            if [ "$scenario" = create-failure ]; then
                [ "$runs" -eq 0 ]
            else
                [ "$runs" -eq 3 ]
            fi
            ;;
        *)
            [ "$result" -eq 0 ]
            [ "$snapshots" -eq 1 ]
            [ "$hooks" -eq 1 ]
            [ "$creates" -eq 1 ]
            [ "$runs" -eq 1 ]
            [ -z "${VIRTUAL_ENV+x}${PYTHONHOME+x}${PYTHONPATH+x}${__PYVENV_LAUNCHER__+x}" ]
            "$REAL_PYTHON" -c 'import encodings'
            if [ "$scenario" = retry ]; then
                [ "$attempts" -eq 2 ]
            else
                [ "$attempts" -eq 1 ]
            fi
            ;;
    esac
)

log="$(mktemp)"
trap 'rm -f "$log"' EXIT
for scenario in snapshot-paths inherited-paths retry bootstrap-failure snapshot-failure hook-failure \
                create-failure wrong-version mac-arm64 mac-wrong-arch; do
    set +e
    run_case "$scenario" >"$log" 2>&1
    result=$?
    set -e
    if [ "$result" -ne 0 ]; then
        cat "$log"
        echo "FAIL: $scenario" >&2
        exit 1
    fi
    echo "PASS: $scenario"
done
