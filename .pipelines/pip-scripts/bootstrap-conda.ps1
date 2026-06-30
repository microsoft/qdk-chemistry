<#
.SYNOPSIS
    Bootstrap ms-ensureconda and create a named conda environment.

.DESCRIPTION
    Installs ms-ensureconda into a throwaway venv, uses it to bootstrap conda,
    then creates the named environment with the requested Python version.
    Writes all diagnostic output to Write-Host; emits exactly one line to stdout:
    the resolved path to conda.exe. Callers capture it with:

        $condaExe = & "$PSScriptRoot\bootstrap-conda.ps1" -EnvName buildenv -PythonVersion 3.11

    Requires SYSTEM_ACCESSTOKEN to be set (mapped from $(System.AccessToken)).
#>
param(
    # Name of the conda environment to create (e.g. "buildenv" or "testenv").
    [Parameter(Mandatory)] [string]$EnvName,
    # Python version for the new environment (e.g. "3.11").
    [Parameter(Mandatory)] [string]$PythonVersion
)
$ErrorActionPreference = 'Stop'

if (-not $env:SYSTEM_ACCESSTOKEN) {
    throw "bootstrap-conda.ps1: SYSTEM_ACCESSTOKEN must be set before calling this script."
}

$ENSURECONDA_PKG    = 'ms-ensureconda==2026.6.1'
$MAX_ATTEMPTS       = 3
$RETRY_DELAY_SEC    = 30
$CONDA_FEED_ROOT    = 'https://pkgs.dev.azure.com/ms-azurequantum/AzureQuantum/_packaging/quantum-apps-dependencies/Conda/repo'

function _Bootstrap {
    param([string]$EnvName, [string]$PythonVersion)

    Write-Host "Installing $ENSURECONDA_PKG and bootstrapping conda..."

    # Throwaway venv avoids PEP 668 issues; cleaned on retries.
    $bootstrapVenv = Join-Path $env:TEMP "qdk-bootstrap-venv-$EnvName"
    Remove-Item -Recurse -Force $bootstrapVenv -ErrorAction SilentlyContinue
    python -m venv $bootstrapVenv
    if ($LASTEXITCODE -ne 0) { throw "Bootstrap venv creation failed ($LASTEXITCODE)" }
    $venvPy = Join-Path $bootstrapVenv 'Scripts\python.exe'

    & $venvPy -m pip install --quiet $ENSURECONDA_PKG 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "ms-ensureconda install failed ($LASTEXITCODE)" }

    # Parse CONDA_EXE from the KEY=VALUE envfile written by ms-ensureconda.
    $condaEnvFile = Join-Path $env:TEMP "qdk-ensureconda-$EnvName.env"
    $env:ARTIFACTS_CONDA_TOKEN = $env:SYSTEM_ACCESSTOKEN
    & $venvPy -m ensureconda --envfile $condaEnvFile 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "ensureconda failed ($LASTEXITCODE)" }

    $condaExe = Get-Content $condaEnvFile |
        Where-Object { $_ -match "^(?:export\s+)?CONDA_EXE=" } |
        Select-Object -First 1 |
        ForEach-Object { ($_ -split '=', 2)[1].Trim().Trim("'`"") }
    if (-not $condaExe -or -not (Test-Path $condaExe)) {
        throw "CONDA_EXE not found or path does not exist: '$condaExe'"
    }
    Write-Host "conda: $condaExe  ($( & $condaExe --version 2>&1 ))"

    # Remove any pre-existing env (idempotent).
    & $condaExe env remove -y -n $EnvName 2>&1 | Out-Null
    $LASTEXITCODE = 0  # env remove exits 1 when env doesn't exist; ignore

    # Route all installs through the Azure Artifacts feed (proxies conda main + conda-forge).
    $env:ARTIFACTS_CONDA_TOKEN = $env:SYSTEM_ACCESSTOKEN
    & $condaExe create --override-channels `
        --channel "$CONDA_FEED_ROOT/main" `
        --channel "$CONDA_FEED_ROOT/conda-forge" `
        --yes --quiet --name $EnvName "python=$PythonVersion" pip 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "conda create '$EnvName' failed ($LASTEXITCODE)" }
    Write-Host "Conda env '$EnvName' created with Python $PythonVersion."

    return $condaExe
}

$attempt = 1
while ($true) {
    try {
        $condaExe = _Bootstrap -EnvName $EnvName -PythonVersion $PythonVersion
        break
    } catch {
        if ($attempt -ge $MAX_ATTEMPTS) {
            Write-Error "bootstrap-conda.ps1: failed after $MAX_ATTEMPTS attempts: $_"
            exit 1
        }
        Write-Warning "Bootstrap attempt $attempt/$MAX_ATTEMPTS failed: $_  Retrying in ${RETRY_DELAY_SEC}s..."
        Start-Sleep -Seconds $RETRY_DELAY_SEC
        $attempt++
    }
}

Write-Output $condaExe
