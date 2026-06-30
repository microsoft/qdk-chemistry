<#
.SYNOPSIS
    Build C++ dependencies (vcpkg + FetchContent targets) for the Windows wheel pipeline.

.DESCRIPTION
    Invoked only on a dependency cache miss. Installs vcpkg packages, runs the CMake
    configure pass, and builds the slow FetchContent targets (libint2, ecpint, gauxc,
    blaspp, lapackpp). The build directory is subsequently cached by the calling
    pipeline job; the full qdk build then starts from a warm build tree.

    Prerequisites (set by the YAML template before this script runs):
      - INCLUDE, LIB, PATH already contain MSVC entries
        (applied via ##vso[task.setvariable] / ##vso[task.prependpath]).
      - CMAKE_BUILD_PARALLEL_LEVEL is set if caller wants a specific level
        (otherwise computed here from CPU count and available RAM).
#>
param(
    [Parameter(Mandatory)] [string]$SrcDir,
    [Parameter(Mandatory)] [string]$ClPath,
    [string]$March          = 'x86-64-v3',
    [string]$BuildType      = 'Release',
    [string]$BuildTesting   = 'OFF',
    [string]$EnableCoverage = 'OFF',
    [string]$VcpkgRoot
)
$ErrorActionPreference = 'Stop'

# Fall back to well-known vcpkg location on MMS images.
if (-not $VcpkgRoot) {
    $VcpkgRoot = if ($env:VCPKG_INSTALLATION_ROOT) { $env:VCPKG_INSTALLATION_ROOT } else { 'C:\vcpkg' }
}
if (-not (Test-Path "$VcpkgRoot\vcpkg.exe")) { throw "vcpkg.exe not found under '$VcpkgRoot'" }

$buildDir = "$SrcDir\cpp\build-msvc"

# Cap Ninja parallelism by available RAM (~4 GB/job for MSVC TUs pulling in
# libint2 headers). On the HB120 runner this typically allows all 120 cores.
if (-not $env:CMAKE_BUILD_PARALLEL_LEVEL) {
    $cpu   = [int]$env:NUMBER_OF_PROCESSORS
    $ramGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $jobs  = [math]::Min($cpu, [math]::Max(1, [math]::Floor($ramGB / 4)))
    Write-Host "CPUs=$cpu  RAM=${ramGB} GB  -> CMAKE_BUILD_PARALLEL_LEVEL=$jobs"
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $jobs
}

# ─── vcpkg install ────────────────────────────────────────────────────────────
# Route all vcpkg source downloads through the Terrapin internal mirror to
# avoid hitting external hosts (gitlab.com etc.) that are blocked by the
# 1ES CFSClean network isolation policy.
# See: https://eng.ms/docs/.../vcpkg  (Step 4: Use Terrapin for Asset Caching)
# Do NOT set x-block-origin: github.com is accessible from the runner, but
# gitlab.com (used by eigen3) is blocked; Terrapin is the preferred source and
# falls back to the authoritative URL only on a miss.
$env:X_VCPKG_ASSET_SOURCES = "x-azurl,https://vcpkg.storage.devpackages.microsoft.io/artifacts/"

Write-Host "=== vcpkg install ==="
& "$VcpkgRoot\vcpkg.exe" install `
    --triplet x64-windows-static-md `
    --x-manifest-root="$SrcDir" `
    --x-install-root="$SrcDir\vcpkg_installed" `
    --overlay-ports="$SrcDir\vcpkg-overlay\ports"
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed ($LASTEXITCODE)" }

# ─── CMake configure (deps pass) ─────────────────────────────────────────────
Write-Host "=== CMake configure ==="
$cmakeArgs = @(
    '-S', "$SrcDir\cpp",
    '-B', $buildDir,
    '-GNinja',
    "-DQDK_UARCH=$March",
    '-DQDK_CHEMISTRY_ENABLE_COVERAGE=OFF',
    '-DQDK_CHEMISTRY_ENABLE_MPI=OFF',
    "-DMACIS_ENABLE_TESTS=$BuildTesting",
    '-DBUILD_SHARED_LIBS=OFF',
    "-DBUILD_TESTING=$BuildTesting",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_C_COMPILER=$ClPath",
    "-DCMAKE_CXX_COMPILER=$ClPath",
    "-DCMAKE_INSTALL_PREFIX=$SrcDir\install-msvc",
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake",
    "-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$SrcDir\.pipelines\toolchains\windows.cmake",
    '-DVCPKG_TARGET_TRIPLET=x64-windows-static-md',
    "-DVCPKG_INSTALLED_DIR=$SrcDir\vcpkg_installed",
    '-DFETCHCONTENT_QUIET=OFF'
)
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

# ─── Build slow FetchContent dependency targets ───────────────────────────────
# Ninja reuses any previously-built artefacts from a restored partial cache, so
# this step is incremental when a stale cache is present.
Write-Host "=== Building C++ dependencies ==="
cmake --build $buildDir --target libint2 ecpint gauxc blaspp lapackpp
if ($LASTEXITCODE -ne 0) { throw "Dependency build failed ($LASTEXITCODE)" }
