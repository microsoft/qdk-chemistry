<#
.SYNOPSIS
    Build C++ dependencies (vcpkg + external FetchContent targets) for the Windows wheel pipeline
    and install them to a path-independent prefix for caching.

.DESCRIPTION
    Invoked only on a dependency cache miss. Installs vcpkg packages, runs the CMake
    configure pass, builds the slow external FetchContent targets (libint2, ecpint,
    gauxc), then installs to $DepsInstallDir via cmake --install.

    macis is intentionally excluded: it is internal code and is always built fresh
    from the local external/macis source directory in the main qdk build step.
    Its transitive deps (blaspp, lapackpp, lobpcgxx, etc.) are built as part of
    the main build along with macis.

    Only the installed prefix ($DepsInstallDir) and vcpkg_installed/ are cached, NOT
    the build directory. Installed CMake package config files use ${_IMPORT_PREFIX}
    (computed at find_package time), so they are path-independent and can be restored
    on a runner with a different workspace drive or path without any CMakeCache mismatch.

    The subsequent full qdk build uses:
      -DCMAKE_PREFIX_PATH="<DepsInstallDir>;<vcpkg_installed/triplet>"
      -DQDK_ALLOW_DEPENDENCY_FETCH=OFF
    so that find_package() finds all deps from the install prefix and FetchContent is
    never triggered.

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
# Ephemeral build dir — only the install prefix is cached.
$buildDir = "$SrcDir\deps-build-msvc"

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
$env:X_VCPKG_ASSET_SOURCES = "x-azurl,https://vcpkg.storage.devpackages.microsoft.io/artifacts/"

Write-Host "=== vcpkg install ==="
& "$VcpkgRoot\vcpkg.exe" install `
    --triplet x64-windows-static-md `
    --x-manifest-root="$SrcDir" `
    --x-install-root="$SrcDir\vcpkg_installed" `
    --overlay-ports="$SrcDir\vcpkg-overlay\ports"
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed ($LASTEXITCODE)" }

# ─── CMake configure (standalone deps project) ───────────────────────────────
# Configures .pipelines/cmake/deps-install/CMakeLists.txt which includes
# third_party.cmake directly from the main project — same versions, same flags.
# No changes to cpp/CMakeLists.txt required.
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
    '-DFETCHCONTENT_QUIET=OFF'
)
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

# ─── Build all targets in the standalone deps project ────────────────────────
# Build everything: this produces int2.lib (STATIC library that archives
# libint2_obj), ecpint.lib, gauxc.lib, etc. so that cmake --install succeeds.
# macis is internal code — intentionally excluded from the standalone project.
Write-Host "=== Building C++ dependencies ==="
cmake --build $buildDir
if ($LASTEXITCODE -ne 0) { throw "Dependency build failed ($LASTEXITCODE)" }

# ─── Install deps to path-independent prefix ─────────────────────────────────
# All FetchContent sub-project install() rules are registered in this build tree
# and fire here. Generated cmake config files use ${_IMPORT_PREFIX} (computed
# relative to the config file at find_package time) — fully path-independent.
Write-Host "=== cmake --install (deps prefix: $DepsInstallDir) ==="
cmake --install $buildDir
if ($LASTEXITCODE -ne 0) { throw "CMake install (deps) failed ($LASTEXITCODE)" }
