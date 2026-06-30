<#
.SYNOPSIS
    Build and install C++ dependencies for the Windows wheel pipeline.

.DESCRIPTION
    Invoked only on a dependency cache miss. Runs vcpkg, configures the standalone
    CMake project (.pipelines/cmake/deps-install), builds all targets, and installs
    to a path-independent prefix. macis is excluded — it is always built from source
    in the main build.
#>
param(
    [Parameter(Mandatory)] [string]$SrcDir,
    [Parameter(Mandatory)] [string]$ClPath,
    [string]$March          = 'x86-64-v3',
    [string]$BuildType      = 'Release',
    [string]$BuildTesting   = 'OFF',
    [string]$EnableCoverage = 'OFF',
    [string]$VcpkgRoot,
    [string]$DepsInstallDir
)
$ErrorActionPreference = 'Stop'

# Fall back to well-known vcpkg location on MMS images.
if (-not $VcpkgRoot) {
    $VcpkgRoot = if ($env:VCPKG_INSTALLATION_ROOT) { $env:VCPKG_INSTALLATION_ROOT } else { 'C:\vcpkg' }
}
if (-not (Test-Path "$VcpkgRoot\vcpkg.exe")) { throw "vcpkg.exe not found under '$VcpkgRoot'" }

if (-not $DepsInstallDir) { $DepsInstallDir = "$SrcDir\deps-install-msvc" }

# Standalone cmake project for external deps only (no macis, no chemistry).
$depsProjectDir = "$SrcDir\.pipelines\cmake\deps-install"
$buildDir = "$SrcDir\deps-build-msvc"

# Cap Ninja parallelism by available RAM (~4 GB/job for MSVC TUs pulling in
# libint2 headers). On the HB120 runner this typically allows all 120 cores.
if (-not $env:CMAKE_BUILD_PARALLEL_LEVEL) {
    $cpu   = [int]$env:NUMBER_OF_PROCESSORS
    $ramGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $jobs  = [math]::Min($cpu, [math]::Max(1, [math]::Floor($ramGB / 3.5)))
    Write-Host "CPUs=$cpu  RAM=${ramGB} GB  -> CMAKE_BUILD_PARALLEL_LEVEL=$jobs"
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $jobs
}

# ─── vcpkg install ────────────────────────────────────────────────────────────
$env:X_VCPKG_ASSET_SOURCES = "x-azurl,https://vcpkg.storage.devpackages.microsoft.io/artifacts/"

Write-Host "=== vcpkg install ==="
& "$VcpkgRoot\vcpkg.exe" install `
    --triplet x64-windows-static-md `
    --x-manifest-root="$SrcDir" `
    --x-install-root="$SrcDir\vcpkg_installed" `
    --overlay-ports="$SrcDir\vcpkg-overlay\ports"
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed ($LASTEXITCODE)" }

# ─── CMake configure (standalone deps project) ───────────────────────────────
Write-Host "=== CMake configure (deps standalone project) ==="
$cmakeArgs = @(
    '-S', $depsProjectDir,
    '-B', $buildDir,
    '-GNinja',
    "-DQDK_UARCH=$March",
    '-DQDK_CHEMISTRY_ENABLE_MPI=OFF',
    '-DBUILD_SHARED_LIBS=OFF',
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_C_COMPILER=$ClPath",
    "-DCMAKE_CXX_COMPILER=$ClPath",
    "-DCMAKE_INSTALL_PREFIX=$DepsInstallDir",
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake",
    "-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$SrcDir\.pipelines\toolchains\windows.cmake",
    '-DVCPKG_TARGET_TRIPLET=x64-windows-static-md',
    "-DVCPKG_INSTALLED_DIR=$SrcDir\vcpkg_installed",
    '-DQDK_ENABLE_OPENMP=OFF',
    '-DFETCHCONTENT_QUIET=OFF'
)
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

# ─── Build all targets in the standalone deps project ────────────────────────
Write-Host "=== Building C++ dependencies ==="
cmake --build $buildDir
if ($LASTEXITCODE -ne 0) { throw "Dependency build failed ($LASTEXITCODE)" }

# ─── Install deps to path-independent prefix ─────────────────────────────────
Write-Host "=== cmake --install (deps prefix: $DepsInstallDir) ==="
cmake --install $buildDir
if ($LASTEXITCODE -ne 0) { throw "CMake install (deps) failed ($LASTEXITCODE)" }
