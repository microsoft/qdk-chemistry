<#
.SYNOPSIS
    Build a Python wheel for qdk-chemistry on Windows.

.DESCRIPTION
    Bootstraps a conda environment, installs build tooling, then builds the
    wheel with scikit-build-core. C++ deps are pre-installed by the deps job
    and found via CMAKE_PREFIX_PATH. Mirrors the approach of build-pip-wheels.sh.
#>
param(
    [Parameter(Mandatory)] [string]$SrcDir,
    [Parameter(Mandatory)] [string]$ClPath,
    [string]$March          = 'x86-64-v3',
    [string]$BuildType      = 'Release',
    [string]$BuildTesting   = 'OFF',
    [string]$EnableCoverage = 'OFF',
    [string]$PythonVersion  = '3.11',
    [string]$DevTag         = 'None',
    [string]$VcpkgRoot,
    [string]$DepsInstallDir
)
$ErrorActionPreference = 'Stop'

if (-not $VcpkgRoot) {
    $VcpkgRoot = if ($env:VCPKG_INSTALLATION_ROOT) { $env:VCPKG_INSTALLATION_ROOT } else { 'C:\vcpkg' }
}
if (-not $DepsInstallDir) { $DepsInstallDir = "$SrcDir\deps-install-msvc" }

if (-not $env:CMAKE_BUILD_PARALLEL_LEVEL) {
    $cpu   = [int]$env:NUMBER_OF_PROCESSORS
    $ramGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $jobs  = [math]::Min($cpu, [math]::Max(1, [math]::Floor($ramGB / 3.5)))
    Write-Host "CPUs=$cpu  RAM=${ramGB} GB  -> CMAKE_BUILD_PARALLEL_LEVEL=$jobs"
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $jobs
}

# ─── Conda bootstrap ─────────────────────────────────────────────────────────
Write-Host "=== Conda bootstrap ==="
$condaExe = & "$PSScriptRoot\bootstrap-conda.ps1" -EnvName buildenv -PythonVersion $PythonVersion
if ($LASTEXITCODE -ne 0) { throw "Conda bootstrap failed ($LASTEXITCODE)" }

# ─── Install Python build tooling ────────────────────────────────────────────
Write-Host "=== pip install build tooling ==="
& $condaExe run -n buildenv python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $condaExe run -n buildenv python -m pip install -r "$SrcDir\.pipelines\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "pip install requirements failed" }

# ─── Component Governance PipReport (non-fatal) ──────────────────────────────
$manifestDir = "$SrcDir\python\build\build-manifest"
New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
$reqs = & $condaExe run -n buildenv python -m pip list --format=freeze
if ($LASTEXITCODE -ne 0) { throw "pip list failed ($LASTEXITCODE)" }
$reqs | Set-Content -Encoding utf8 "$manifestDir\requirements.txt"
$reqs | ForEach-Object { Write-Host $_ }
& $condaExe run -n buildenv python -m pip install `
    --dry-run --ignore-installed --quiet `
    --report "$manifestDir\component-detection-pip-report.json" `
    -r "$manifestDir\requirements.txt"
# PipReport failures are non-fatal (Component Governance is best-effort).
$LASTEXITCODE = 0

# ─── Prepare README ──────────────────────────────────────────────────────────
Write-Host "=== Prepare README ==="
try {
    & "$PSScriptRoot\prepare-readme.ps1" -SrcDir $SrcDir
} catch {
    Write-Warning "prepare-readme failed: $_  (continuing)"
}

# ─── Build Python wheel ───────────────────────────────────────────────────────
# scikit-build-core builds the full C++ library; pre-installed deps (vcpkg
# packages + libint2/ecpint/gauxc) are found via CMAKE_PREFIX_PATH.
Write-Host "=== python -m build --wheel ==="
# Default to a clean release version (python/_dev_version.py) when run standalone.
if (-not $env:QDK_CHEMISTRY_RELEASE_BUILD) { $env:QDK_CHEMISTRY_RELEASE_BUILD = '1' }
$prefix = "$DepsInstallDir;$SrcDir\vcpkg_installed\x64-windows-static-md"
$buildArgs = @(
    'run', '-n', 'buildenv',
    'python', '-m', 'build', '--wheel',
    "-C=build-dir=build/{wheel_tag}",
    "-C=cmake.args=-GNinja",
    "-C=cmake.define.QDK_UARCH=$March",
    '-C=cmake.define.BUILD_SHARED_LIBS=OFF',
    '-C=cmake.define.QDK_CHEMISTRY_ENABLE_MPI=OFF',
    '-C=cmake.define.QDK_ENABLE_OPENMP=OFF',
    "-C=cmake.define.QDK_CHEMISTRY_ENABLE_COVERAGE=$EnableCoverage",
    "-C=cmake.define.BUILD_TESTING=$BuildTesting",
    "-C=cmake.define.MACIS_ENABLE_TESTS=$BuildTesting",
    '-C=cmake.define.CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST',
    "-C=cmake.define.QDK_ALLOW_DEPENDENCY_FETCH=OFF",
    '-C=cmake.define.VCPKG_APPLOCAL_DEPS=OFF',
    "-C=cmake.define.CMAKE_BUILD_TYPE=$BuildType",
    "-C=cmake.define.CMAKE_C_COMPILER=$ClPath",
    "-C=cmake.define.CMAKE_CXX_COMPILER=$ClPath",
    "-C=cmake.define.CMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake",
    "-C=cmake.define.VCPKG_CHAINLOAD_TOOLCHAIN_FILE=$SrcDir\.pipelines\toolchains\windows.cmake",
    '-C=cmake.define.VCPKG_TARGET_TRIPLET=x64-windows-static-md',
    "-C=cmake.define.VCPKG_INSTALLED_DIR=$SrcDir\vcpkg_installed",
    "-C=cmake.define.CMAKE_PREFIX_PATH=$prefix",
    '-C=cmake.define.FETCHCONTENT_QUIET=OFF'
)
Push-Location "$SrcDir\python"
& $condaExe @buildArgs
$wheelCode = $LASTEXITCODE
Pop-Location
if ($wheelCode -ne 0) { throw "python -m build --wheel failed ($wheelCode)" }

# ─── Copy wheel to repaired_wheelhouse ───────────────────────────────────────
# No wheel repair needed: x64-windows-static-md statically links all vcpkg deps.
$distDir   = "$SrcDir\python\dist"
$outputDir = "$SrcDir\python\repaired_wheelhouse"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$wheels = Get-ChildItem "$distDir\qdk_chemistry*.whl"
if ($wheels.Count -ne 1) {
    throw "Expected exactly 1 wheel in dist/, found $($wheels.Count): $($wheels.Name -join ', ')"
}
Copy-Item $wheels[0].FullName $outputDir
Write-Host "Wheel : $($wheels[0].Name)"
Write-Host "Output: $outputDir"
