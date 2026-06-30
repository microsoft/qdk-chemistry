<#
.SYNOPSIS
    Build the full qdk C++ library, run C++ tests, and produce a Python wheel
    using a conda-isolated build environment.

.DESCRIPTION
    This script is always invoked (unlike the deps script which is skipped on
    cache hit). It:
      1. Configures and builds the full qdk library using pre-installed deps
         from $DepsInstallDir (the cached install prefix populated by the deps
         script). FetchContent is disabled (QDK_ALLOW_DEPENDENCY_FETCH=OFF).
      2. Runs the C++ test suite (ctest); results are written to an XML file
         that the calling YAML template publishes with PublishTestResults@2.
      3. Installs the C++ library under install-msvc/.
      4. Bootstraps a conda environment via ms-ensureconda (the Microsoft-approved
         conda bootstrapper). Public channels are blocked in 1ES CFSClean; all
         packages are fetched from the Azure Artifacts Conda/PyPI feed.
      5. Installs build tooling (pip, build, scikit-build-core, etc.) and
         generates a Component Governance PipReport.
      6. Rewrites relative README links to absolute GitHub URLs (equivalent to
         prepare-readme.sh).
      7. Builds the distribution wheel with scikit-build-core. No wheel repair
         is needed: x64-windows-static-md statically links all vcpkg deps.
      8. Copies the wheel to python/repaired_wheelhouse/.

    Prerequisites (set by the YAML template before this script runs):
      - INCLUDE, LIB, PATH already contain MSVC entries.
      - CMAKE_BUILD_PARALLEL_LEVEL is set (or computed here).
      - SYSTEM_ACCESSTOKEN is in the environment (mapped by the YAML step via
        env: SYSTEM_ACCESSTOKEN: $(System.AccessToken)).
      - PIP_INDEX_URL is set by PipAuthenticate@1 at job level.
      - The C++ deps install prefix ($DepsInstallDir) has been restored from
        cache (or built by build-pip-wheels-windows-deps.ps1).
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

$buildDir  = "$SrcDir\cpp\build-msvc"
$installDir = "$SrcDir\install-msvc"

if (-not $env:CMAKE_BUILD_PARALLEL_LEVEL) {
    $cpu   = [int]$env:NUMBER_OF_PROCESSORS
    $ramGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $jobs  = [math]::Min($cpu, [math]::Max(1, [math]::Floor($ramGB / 3.5)))
    Write-Host "CPUs=$cpu  RAM=${ramGB} GB  -> CMAKE_BUILD_PARALLEL_LEVEL=$jobs"
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $jobs
}

# ─── CMake configure + full build ────────────────────────────────────────────
# Always configure fresh (no cached build dir). Pre-installed deps are found via
# CMAKE_PREFIX_PATH; FetchContent is disabled to enforce use of installed copies.
# This build dir is for qdk only and is NOT cached between pipeline runs.
Write-Host "=== CMake configure (full build) ==="
$vcpkgInstalled = "$SrcDir\vcpkg_installed\x64-windows-static-md"
$cmakeArgs = @(
    '-S', "$SrcDir\cpp",
    '-B', $buildDir,
    '-GNinja',
    "-DQDK_UARCH=$March",
    "-DQDK_CHEMISTRY_ENABLE_COVERAGE=$EnableCoverage",
    '-DQDK_CHEMISTRY_ENABLE_MPI=OFF',
    "-DMACIS_ENABLE_TESTS=$BuildTesting",
    '-DBUILD_SHARED_LIBS=OFF',
    "-DBUILD_TESTING=$BuildTesting",
    '-DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST',
    '-DVCPKG_APPLOCAL_DEPS=OFF',
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_C_COMPILER=$ClPath",
    "-DCMAKE_CXX_COMPILER=$ClPath",
    "-DCMAKE_INSTALL_PREFIX=$installDir",
    "-DCMAKE_PREFIX_PATH=$DepsInstallDir;$vcpkgInstalled",
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake",
    "-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$SrcDir\.pipelines\toolchains\windows.cmake",
    '-DVCPKG_TARGET_TRIPLET=x64-windows-static-md',
    "-DVCPKG_INSTALLED_DIR=$SrcDir\vcpkg_installed",
    '-DQDK_ALLOW_DEPENDENCY_FETCH=OFF',
    '-DQDK_ENABLE_OPENMP=OFF',
    '-DFETCHCONTENT_QUIET=OFF'
)
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

Write-Host "=== CMake build ==="
cmake --build $buildDir
if ($LASTEXITCODE -ne 0) { throw "CMake build failed ($LASTEXITCODE)" }

# ─── C++ tests ───────────────────────────────────────────────────────────────
$ctestCode = 0
if ($BuildTesting -ne 'OFF') {
    # Exclude MACIS_SERIAL_TEST and libint2/unit (compile-at-test-time meta-test)
    # which can exceed the ctest timeout under MSVC.
    # Save the ctest exit code; throw AFTER the wheel has been built so the
    # PublishTestResults@2 task in the YAML always has something to publish.
    Write-Host "=== ctest ==="
    Push-Location $buildDir
    ctest --output-on-failure --verbose --timeout 400 `
          --output-junit ctest_results.xml `
          -E "MACIS_SERIAL_TEST|libint2/unit"
    $ctestCode = $LASTEXITCODE
    Pop-Location
    if ($ctestCode -ne 0) {
        Write-Warning "ctest returned $ctestCode — continuing to build wheel, then will throw."
    }
}

# ─── Install C++ library ─────────────────────────────────────────────────────
Write-Host "=== cmake --install ==="
cmake --install $buildDir
if ($LASTEXITCODE -ne 0) { throw "CMake install failed ($LASTEXITCODE)" }

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

# ─── Component Governance PipReport ──────────────────────────────────────────
# Snapshot the buildenv and feed it to pip install --report so Component
# Governance's PipReportDetector sees every package.
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

# ─── Prepare README (equivalent to prepare-readme.sh) ────────────────────────
Write-Host "=== Prepare README ==="
try {
    & "$PSScriptRoot\prepare-readme.ps1" -SrcDir $SrcDir
} catch {
    Write-Warning "prepare-readme failed: $_  (continuing)"
}

# ─── Build Python wheel ───────────────────────────────────────────────────────
# scikit-build-core picks up the pre-built C++ library via CMAKE_PREFIX_PATH so
# FetchContent is never triggered. QDK_ALLOW_DEPENDENCY_FETCH=OFF enforces this.
Write-Host "=== python -m build --wheel ==="
$prefix   = "$SrcDir\vcpkg_installed\x64-windows-static-md;$installDir;$DepsInstallDir"
$buildArgs = @(
    'run', '-n', 'buildenv',
    'python', '-m', 'build', '--wheel',
    "-C=build-dir=build/{wheel_tag}",
    "-C=cmake.args=-GNinja",
    "-C=cmake.define.QDK_UARCH=$March",
    '-C=cmake.define.BUILD_SHARED_LIBS=OFF',
    '-C=cmake.define.QDK_CHEMISTRY_ENABLE_MPI=OFF',
    '-C=cmake.define.QDK_ENABLE_OPENMP=OFF',
    '-C=cmake.define.QDK_CHEMISTRY_ENABLE_COVERAGE=OFF',
    '-C=cmake.define.BUILD_TESTING=OFF',
    "-C=cmake.define.QDK_ALLOW_DEPENDENCY_FETCH=OFF",
    "-C=cmake.define.CMAKE_C_COMPILER=$ClPath",
    "-C=cmake.define.CMAKE_CXX_COMPILER=$ClPath",
    "-C=cmake.define.CMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake",
    "-C=cmake.define.VCPKG_CHAINLOAD_TOOLCHAIN_FILE=$SrcDir\.pipelines\toolchains\windows.cmake",
    '-C=cmake.define.VCPKG_TARGET_TRIPLET=x64-windows-static-md',
    "-C=cmake.define.VCPKG_INSTALLED_DIR=$SrcDir\vcpkg_installed",
    "-C=cmake.define.CMAKE_PREFIX_PATH=$prefix"
)
Push-Location "$SrcDir\python"
& $condaExe @buildArgs
$wheelCode = $LASTEXITCODE
Pop-Location
if ($wheelCode -ne 0) { throw "python -m build --wheel failed ($wheelCode)" }

# ─── Copy wheel to repaired_wheelhouse ───────────────────────────────────────
# No wheel repair needed: x64-windows-static-md statically links all vcpkg
# deps; the only dynamic runtime deps (MSVCP140.dll, VCRUNTIME140.dll, UCRT)
# are always present on modern Windows.
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

# Deferred ctest failure: publish results step has already run by this point.
if ($ctestCode -ne 0) { throw "ctest failed ($ctestCode)" }
