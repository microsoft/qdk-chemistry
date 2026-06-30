<#
.SYNOPSIS
    Install the built Windows wheel and run pytest.

.DESCRIPTION
    Bootstraps a conda test environment, installs the wheel with its [test]
    extras, generates a Component Governance PipReport (non-fatal), and runs pytest.
#>
param(
    [string]$SrcDir        = (Resolve-Path "$PSScriptRoot\..\.." -ErrorAction Stop),
    [string]$PythonVersion = '3.11',
    [string]$RunSlowTests  = 'true'
)

$ErrorActionPreference = 'Stop'

$pythonDir = "$SrcDir\python"

# ─── 1. Bootstrap conda test environment ─────────────────────────────────────
Write-Host "=== Set up conda test environment (Python $PythonVersion) ==="
$condaExe = & "$PSScriptRoot\bootstrap-conda.ps1" -EnvName testenv -PythonVersion $PythonVersion
if ($LASTEXITCODE -ne 0) { throw "Conda bootstrap failed ($LASTEXITCODE)" }

# ─── 2. Install wheel with test dependencies ──────────────────────────────────
Write-Host "=== Install wheel with test dependencies ==="
$wheels = Get-ChildItem "$pythonDir\repaired_wheelhouse\qdk_chemistry*.whl"
if ($wheels.Count -ne 1) {
    throw "Expected exactly 1 wheel, found $($wheels.Count): $($wheels.Name -join ', ')"
}
$wheel = $wheels[0].FullName
Write-Host "Installing: $wheel"
& $condaExe run -n testenv python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed ($LASTEXITCODE)" }
& $condaExe run -n testenv python -m pip install "$wheel[test]"
if ($LASTEXITCODE -ne 0) { throw "pip install wheel[test] failed ($LASTEXITCODE)" }

# ─── 3. Component Governance PipReport (non-fatal) ───────────────────────────
Write-Host "=== Generate Component Governance PipReport ==="
try {
    $manifestDir = "$pythonDir\build\test-manifest"
    New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
    $reqs = & $condaExe run -n testenv python -m pip list --format=freeze --exclude qdk_chemistry
    if ($LASTEXITCODE -ne 0) { throw "pip list failed ($LASTEXITCODE)" }
    $reqs | Set-Content -Encoding utf8 "$manifestDir\requirements.txt"
    $reqs | ForEach-Object { Write-Host $_ }
    & $condaExe run -n testenv python -m pip install `
        --dry-run --ignore-installed --quiet `
        --report "$manifestDir\component-detection-pip-report.json" `
        -r "$manifestDir\requirements.txt"
} catch {
    Write-Warning "PipReport generation failed (non-fatal): $_"
}

# ─── 4. Run pytest ────────────────────────────────────────────────────────────
Write-Host "=== Running pytest suite ==="
$env:QSHARP_PYTHON_TELEMETRY      = 'false'
$env:QDK_CHEMISTRY_RUN_SLOW_TESTS = $RunSlowTests
$env:OMP_NUM_THREADS              = '2'
Push-Location $pythonDir
& $condaExe run -n testenv --no-capture-output python -m pytest -v tests/
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "pytest failed ($code)" }
