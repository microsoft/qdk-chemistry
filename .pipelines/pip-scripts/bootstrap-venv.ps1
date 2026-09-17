<#
.SYNOPSIS
    Create a throwaway virtual environment and emit the path to its python.exe.

.DESCRIPTION
    Conda-free alternative to bootstrap-conda.ps1 for platforms conda does not
    package — notably Windows ARM64, where conda-forge's win-arm64 support is
    cross-compiled and experimental and Miniforge ships no ARM64 installer.

    The base interpreter is whichever `python` is on PATH, so the caller is
    responsible for selecting the right version and architecture beforehand
    (e.g. with the UsePythonVersion pipeline task).

    Writes all diagnostic output to Write-Host; emits exactly one line to
    stdout: the resolved path to the environment's python.exe. Callers capture
    it with:

        $envPython = & "$PSScriptRoot\bootstrap-venv.ps1" -EnvName buildenv -PythonVersion 3.13
#>
param(
    # Name of the environment to create (e.g. "buildenv" or "testenv").
    [Parameter(Mandatory)] [string]$EnvName,
    # Python version the base interpreter is expected to provide (e.g. "3.13").
    [Parameter(Mandatory)] [string]$PythonVersion
)
$ErrorActionPreference = 'Stop'

$basePython = (Get-Command python -ErrorAction Stop).Source
$actual = & $basePython -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))"
if ($LASTEXITCODE -ne 0) { throw "Could not query the base interpreter ($LASTEXITCODE)" }
if ($actual.Trim() -ne $PythonVersion) {
    throw "bootstrap-venv.ps1: 'python' on PATH is $($actual.Trim()), expected $PythonVersion ($basePython)"
}

# Surface the interpreter's architecture: an x64 interpreter running under
# emulation on an ARM64 agent would silently produce win_amd64 wheels.
$machine = & $basePython -c "import platform; print(platform.machine())"
Write-Host "Base interpreter: $basePython (Python $($actual.Trim()), $($machine.Trim()))"

$venvDir = Join-Path $env:TEMP "qdk-venv-$EnvName"
Remove-Item -Recurse -Force $venvDir -ErrorAction SilentlyContinue
& $basePython -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "venv creation failed ($LASTEXITCODE)" }

$venvPython = Join-Path $venvDir 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) { throw "venv python not found at $venvPython" }
Write-Host "Virtual environment '$EnvName' created at $venvDir"

Write-Output $venvPython
