<#
.SYNOPSIS
    Bootstrap ms-ensureconda and create a named conda environment.

.DESCRIPTION
    PowerShell equivalent of bootstrap-conda.sh for Windows 1ES builds.

    This script is meant to be *called* (not dot-sourced) from other scripts
    or YAML powershell steps. It writes all diagnostic output via Write-Host
    and emits exactly one line to stdout: the resolved path to conda.exe.
    Callers capture it with:

        $condaExe = & "$PSScriptRoot\bootstrap-conda.ps1" -EnvName buildenv -PythonVersion 3.11

    Required env var:
        SYSTEM_ACCESSTOKEN   Pipeline access token, mapped from $(System.AccessToken).
                             Used by the azure_artifacts_conda_auth plugin that is
                             pre-registered in ms-ensureconda's conda distribution.

    Rationale: ms-ensureconda is the Microsoft-approved conda bootstrapper for CI
    builds. See:
        https://eng.ms/docs/more/languages-at-microsoft/python/articles/anaconda/install
    All network access goes through the Azure Artifacts feed (1ES CFSClean blocks
    public conda channels). The azure_artifacts_conda_auth plugin reads
    ARTIFACTS_CONDA_TOKEN; we set it to SYSTEM_ACCESSTOKEN before every conda call.
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

    # Use a throwaway venv so we don't fight PEP 668 (externally-managed Python).
    # Clean any stale venv first (idempotent on retries / self-hosted agents).
    $bootstrapVenv = Join-Path $env:TEMP "qdk-bootstrap-venv-$EnvName"
    Remove-Item -Recurse -Force $bootstrapVenv -ErrorAction SilentlyContinue
    python -m venv $bootstrapVenv
    if ($LASTEXITCODE -ne 0) { throw "Bootstrap venv creation failed ($LASTEXITCODE)" }
    $venvPy = Join-Path $bootstrapVenv 'Scripts\python.exe'

    & $venvPy -m pip install --quiet $ENSURECONDA_PKG
    if ($LASTEXITCODE -ne 0) { throw "ms-ensureconda install failed ($LASTEXITCODE)" }

    # ms-ensureconda --envfile writes a shell-sourceable KEY=VALUE file with
    # CONDA_EXE, CONDA_BASH_HOOK, etc. We parse CONDA_EXE from it.
    $condaEnvFile = Join-Path $env:TEMP "qdk-ensureconda-$EnvName.env"
    $env:ARTIFACTS_CONDA_TOKEN = $env:SYSTEM_ACCESSTOKEN
    & $venvPy -m ensureconda --envfile $condaEnvFile
    if ($LASTEXITCODE -ne 0) { throw "ensureconda failed ($LASTEXITCODE)" }

    $condaExe = Get-Content $condaEnvFile |
        Where-Object { $_ -match "^(?:export\s+)?CONDA_EXE=" } |
        Select-Object -First 1 |
        ForEach-Object { ($_ -split '=', 2)[1].Trim().Trim("'`"") }
    if (-not $condaExe -or -not (Test-Path $condaExe)) {
        throw "CONDA_EXE not found or path does not exist: '$condaExe'"
    }
    Write-Host "conda: $condaExe  ($( & $condaExe --version 2>&1 ))"

    # Remove any pre-existing env (idempotent on self-hosted agents and retries).
    & $condaExe env remove -y -n $EnvName 2>&1 | Out-Null
    $LASTEXITCODE = 0  # env remove exits 1 when env doesn't exist; ignore

    # Public channels (conda.anaconda.org, repo.anaconda.com) are blocked under
    # 1ES network isolation (CFSClean). Force all installs through the Azure
    # Artifacts feed, which proxies `main` and `conda-forge` as named subpaths.
    # The `main` channel carries python + pip; `conda-forge` is a fallback.
    $env:ARTIFACTS_CONDA_TOKEN = $env:SYSTEM_ACCESSTOKEN
    & $condaExe create --override-channels `
        --channel "$CONDA_FEED_ROOT/main" `
        --channel "$CONDA_FEED_ROOT/conda-forge" `
        --yes --quiet --name $EnvName "python=$PythonVersion" pip
    if ($LASTEXITCODE -ne 0) { throw "conda create '$EnvName' failed ($LASTEXITCODE)" }
    Write-Host "Conda env '$EnvName' created with Python $PythonVersion."

    # Return the resolved conda exe path (captured by caller).
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

# Emit only the conda exe path to stdout; callers capture this line.
Write-Output $condaExe
