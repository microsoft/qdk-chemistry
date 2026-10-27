<#
.SYNOPSIS
    Install the built Windows wheel and run pytest.

.DESCRIPTION
    Bootstraps a Python test environment using ms-ensureconda, or a plain venv
    on platforms conda does not package. Audits the wheel's native DLL imports,
    installs the base wheel using binary distributions only, verifies a clean
    import, then installs [test] extras and runs pytest.
#>
param(
    [string]$SrcDir        = (Resolve-Path "$PSScriptRoot\..\.." -ErrorAction Stop),
    [string]$PythonVersion = '3.11',
    [string]$RunSlowTests  = 'true',
    [string]$Triplet       = $env:VCPKG_TRIPLET,
    [ValidateSet('conda', 'venv')]
    [string]$PythonEnv     = 'conda'
)

$ErrorActionPreference = 'Stop'

$pythonDir = "$SrcDir\python"

# ─── 1. Bootstrap the Python test environment ────────────────────────────────
# ms-ensureconda everywhere except Windows ARM64, where conda has no usable
# win-arm64 packages; there the wheel is tested in a venv.
if ($PythonEnv -eq 'venv') {
    Write-Host "=== Set up venv test environment (Python $PythonVersion) ==="
    $runExe = & "$PSScriptRoot\bootstrap-venv.ps1" -EnvName testenv -PythonVersion $PythonVersion
    if ($LASTEXITCODE -ne 0) { throw "venv bootstrap failed ($LASTEXITCODE)" }
    $runArgs   = @()
    $runNoCapArgs = @()
} else {
    Write-Host "=== Set up conda test environment (Python $PythonVersion) ==="
    $runExe = & "$PSScriptRoot\bootstrap-conda.ps1" -EnvName testenv -PythonVersion $PythonVersion
    if ($LASTEXITCODE -ne 0) { throw "Conda bootstrap failed ($LASTEXITCODE)" }
    $runArgs   = @('run', '-n', 'testenv', 'python')
    $runNoCapArgs = @('run', '-n', 'testenv', '--no-capture-output', 'python')
}

# ─── 2. Audit wheel DLL imports ───────────────────────────────────────────────
$wheels = Get-ChildItem "$pythonDir\repaired_wheelhouse\qdk_chemistry*.whl"
if ($wheels.Count -ne 1) {
    throw "Expected exactly 1 wheel, found $($wheels.Count): $($wheels.Name -join ', ')"
}
$wheel = $wheels[0].FullName
& $runExe @runArgs -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed ($LASTEXITCODE)" }
& $runExe @runArgs -m pip install pefile==2024.8.26
if ($LASTEXITCODE -ne 0) { throw "pefile install failed ($LASTEXITCODE)" }
& $runExe @runArgs "$PSScriptRoot\check-wheel-dll-imports.py" $wheel
if ($LASTEXITCODE -ne 0) { throw "Windows wheel DLL import validation failed ($LASTEXITCODE)" }

# ─── 3. Validate the normal end-user installation path ───────────────────────
Write-Host "=== Install base wheel using binary distributions only ==="
Write-Host "Installing: $wheel"
& $runExe @runArgs -m pip install --only-binary=:all: $wheel
if ($LASTEXITCODE -ne 0) { throw "minimal pip install failed ($LASTEXITCODE)" }
& $runExe @runArgs -I -c "import qdk_chemistry; print(qdk_chemistry.__version__)"
if ($LASTEXITCODE -ne 0) { throw "minimal qdk_chemistry import failed ($LASTEXITCODE)" }

# ─── 4. Install wheel with test dependencies ──────────────────────────────────
Write-Host "=== Install wheel with test dependencies ==="
& $runExe @runArgs -m pip install "$wheel[test]"
if ($LASTEXITCODE -ne 0) { throw "pip install wheel[test] failed ($LASTEXITCODE)" }
if ($Triplet -ne 'arm64-windows-static-md') {
    & $runExe @runArgs -c "import mcp; from qdk_chemistry.ui._mcp import MCP_AVAILABLE; assert MCP_AVAILABLE"
    if ($LASTEXITCODE -ne 0) { throw "MCP installation smoke test failed ($LASTEXITCODE)" }
}

# ─── 5. Component Governance PipReport (non-fatal) ───────────────────────────
Write-Host "=== Generate Component Governance PipReport ==="
try {
    $manifestDir = "$pythonDir\build\test-manifest"
    New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
    $reqs = & $runExe @runArgs -m pip list --format=freeze --exclude qdk_chemistry
    if ($LASTEXITCODE -ne 0) { throw "pip list failed ($LASTEXITCODE)" }
    $reqs | Set-Content -Encoding utf8 "$manifestDir\requirements.txt"
    $reqs | ForEach-Object { Write-Host $_ }
    & $runExe @runArgs -m pip install `
        --dry-run --ignore-installed --quiet `
        --report "$manifestDir\component-detection-pip-report.json" `
        -r "$manifestDir\requirements.txt"
} catch {
    Write-Warning "PipReport generation failed (non-fatal): $_"
}

# ─── 6. Run pytest ────────────────────────────────────────────────────────────
Write-Host "=== Running pytest suite ==="
$env:QSHARP_PYTHON_TELEMETRY      = 'false'
$env:QDK_CHEMISTRY_RUN_SLOW_TESTS = $RunSlowTests
$env:OMP_NUM_THREADS              = '2'
Push-Location $pythonDir
& $runExe @runNoCapArgs -m pytest -v tests/
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "pytest failed ($code)" }
