<#
.SYNOPSIS
    Fetch and install C++ dependencies for Windows CI (GHA and ADO).

.DESCRIPTION
    Reads dependency versions from the cgmanifest JSON files, runs vcpkg for
    vcpkg-managed packages, then builds libint2, ecpint, and gauxc from source.
    gauxc fetches its own transitive deps (ExchCXX, IntegratorXX, gau2grid,
    linalg-cmake-modules) via CMake FetchContent. Mirrors the approach of
    .devcontainer/scripts/install_cpp_dependencies.sh for Linux/macOS.

.PARAMETER SrcDir
    Root of the repository checkout.

.PARAMETER ClPath
    Full path to cl.exe (MSVC) or clang-cl.exe. Used for all cmake builds.

.PARAMETER BuildType
    CMake build type. Default: RelWithDebInfo.

.PARAMETER VcpkgRoot
    Path to the vcpkg installation. Defaults to VCPKG_INSTALLATION_ROOT or C:\vcpkg.

.PARAMETER DepsInstallDir
    Install prefix for all cmake-built dependencies. Defaults to $SrcDir\deps-install-msvc.

.PARAMETER KeepBuildDir
    If set, the temporary build directory is not deleted after installation.
#>
param(
    [Parameter(Mandatory)] [string]$SrcDir,
    [Parameter(Mandatory)] [string]$ClPath,
    [string]$BuildType    = 'RelWithDebInfo',
    [string]$VcpkgRoot,
    [string]$DepsInstallDir,
    [switch]$KeepBuildDir
)
$ErrorActionPreference = 'Stop'

if (-not $VcpkgRoot) {
    $VcpkgRoot = if ($env:VCPKG_INSTALLATION_ROOT) { $env:VCPKG_INSTALLATION_ROOT } else { 'C:\vcpkg' }
}
if (-not (Test-Path "$VcpkgRoot\vcpkg.exe")) { throw "vcpkg.exe not found under '$VcpkgRoot'" }
if (-not $DepsInstallDir) { $DepsInstallDir = "$SrcDir\deps-install-msvc" }

$buildDir = "$SrcDir\deps-build-msvc"
New-Item -ItemType Directory -Force -Path $DepsInstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $buildDir       | Out-Null

# ─── Memory-aware build parallelism ──────────────────────────────────────────
if (-not $env:CMAKE_BUILD_PARALLEL_LEVEL) {
    $cpu   = [int]$env:NUMBER_OF_PROCESSORS
    $ramGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $jobs  = [math]::Min($cpu, [math]::Max(1, [math]::Floor($ramGB / 3.5)))
    Write-Host "CPUs=$cpu  RAM=${ramGB}GB  -> CMAKE_BUILD_PARALLEL_LEVEL=$jobs"
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $jobs
}
# libint2 TUs under MSVC peak at several GB; cap separately to avoid OOM.
$libintJobs = [math]::Min(4, [int]$env:CMAKE_BUILD_PARALLEL_LEVEL)

$isClangCl = $ClPath -imatch 'clang'

# ─── Read dependency versions from cgmanifest ─────────────────────────────────
function Get-ManifestCommit([string]$ManifestPath, [string]$RepoPattern) {
    $data = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    $reg  = $data.registrations |
            Where-Object { $_.component.type -eq 'git' -and
                           $_.component.git.repositoryUrl -match $RepoPattern } |
            Select-Object -First 1
    if (-not $reg) { throw "No '$RepoPattern' entry in $ManifestPath" }
    return $reg.component.git.commitHash.Trim()
}

function Get-ManifestUrl([string]$ManifestPath, [string]$Name) {
    $data = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    $reg  = $data.registrations |
            Where-Object { $_.component.type -eq 'other' -and
                           $_.component.other.name -eq $Name } |
            Select-Object -First 1
    if (-not $reg) { throw "No '$Name' entry in $ManifestPath" }
    return $reg.component.other.downloadUrl
}

$cppManifest  = "$SrcDir\cpp\manifest\qdk-chemistry\cgmanifest.json"
$libintUrl    = Get-ManifestUrl    $cppManifest 'Libint'
$ecpintCommit = Get-ManifestCommit $cppManifest 'robashaw/libecpint'
$gauxcCommit  = Get-ManifestCommit $cppManifest 'wavefunction91/gauxc'

Write-Host "=== Dependency versions ==="
Write-Host "  libint2 : $libintUrl"
Write-Host "  ecpint  : $ecpintCommit"
Write-Host "  gauxc   : $gauxcCommit"

# ─── Common CMake flags (applied to every dep build) ─────────────────────────
$commonArgs = @(
    '-GNinja',
    "-DCMAKE_BUILD_TYPE=$BuildType",
    '-DCMAKE_CXX_STANDARD=20', '-DCMAKE_CXX_STANDARD_REQUIRED=ON',
    '-DBUILD_SHARED_LIBS=OFF',
    '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
    "-DCMAKE_C_COMPILER=$ClPath",
    "-DCMAKE_CXX_COMPILER=$ClPath",
    "-DCMAKE_INSTALL_PREFIX=$DepsInstallDir",
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake",
    "-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$SrcDir\.pipelines\toolchains\windows.cmake",
    '-DVCPKG_TARGET_TRIPLET=x64-windows-static-md',
    "-DVCPKG_INSTALLED_DIR=$SrcDir\vcpkg_installed",
    '-DFETCHCONTENT_QUIET=OFF'
)

function Invoke-CMakeDep([string]$Name, [string]$SrcPath, [string[]]$ExtraArgs, [int]$Jobs = 0) {
    $depBuild = "$buildDir\$Name-build"
    Write-Host "--- cmake configure: $Name ---"
    cmake -S $SrcPath -B $depBuild @commonArgs @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for $Name ($LASTEXITCODE)" }

    $savedJobs = $env:CMAKE_BUILD_PARALLEL_LEVEL
    if ($Jobs -gt 0) { $env:CMAKE_BUILD_PARALLEL_LEVEL = $Jobs }
    Write-Host "--- cmake build: $Name (jobs=$($env:CMAKE_BUILD_PARALLEL_LEVEL)) ---"
    cmake --build $depBuild
    $buildCode = $LASTEXITCODE
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $savedJobs
    if ($buildCode -ne 0) { throw "cmake build failed for $Name ($buildCode)" }

    Write-Host "--- cmake install: $Name ---"
    cmake --install $depBuild
    if ($LASTEXITCODE -ne 0) { throw "cmake install failed for $Name ($LASTEXITCODE)" }

    Remove-Item $depBuild -Recurse -Force -ErrorAction SilentlyContinue
}

# ─── vcpkg install ────────────────────────────────────────────────────────────
# Provides eigen3, openblas, hdf5, boost-headers, spdlog, nlohmann-json, etc.
$env:X_VCPKG_ASSET_SOURCES = 'x-azurl,https://vcpkg.storage.devpackages.microsoft.io/artifacts/'
Write-Host "=== vcpkg install ==="
& "$VcpkgRoot\vcpkg.exe" install `
    --triplet x64-windows-static-md `
    --x-manifest-root="$SrcDir" `
    --x-install-root="$SrcDir\vcpkg_installed" `
    --overlay-ports="$SrcDir\vcpkg-overlay\ports"
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed ($LASTEXITCODE)" }

# ─── libint2 ─────────────────────────────────────────────────────────────────
Write-Host "=== Installing libint2 ==="
$libintExtract = "$buildDir\libint2-extract"
New-Item -ItemType Directory -Force -Path $libintExtract | Out-Null

$tarball = Join-Path $libintExtract (Split-Path $libintUrl -Leaf)
Invoke-WebRequest $libintUrl -OutFile $tarball
tar xzf $tarball -C $libintExtract
Remove-Item $tarball

# Locate the cmake project root (CMakeLists.txt may be at the top of the
# extracted directory or one level deeper, depending on the tarball layout).
$libintTop = Get-ChildItem $libintExtract -Directory |
             Where-Object { $_.Name -match '^libint' } |
             Sort-Object Name | Select-Object -First 1
if (-not $libintTop) { throw "Cannot find libint source directory after extraction" }

if (Test-Path "$($libintTop.FullName)\CMakeLists.txt") {
    $libintSrc  = $libintTop.FullName
    $patchBase  = $libintExtract       # patch uses "libint-x.y.z/..." relative to here
} else {
    # Tarball strips its own root: CMakeLists.txt is inside a libint-* subdirectory
    $libintSub = Get-ChildItem $libintTop.FullName -Directory |
                 Where-Object { $_.Name -match '^libint' } | Select-Object -First 1
    if (-not ($libintSub -and (Test-Path "$($libintSub.FullName)\CMakeLists.txt"))) {
        throw "Cannot find CMakeLists.txt under $($libintTop.FullName)"
    }
    $libintSrc = $libintSub.FullName
    $patchBase = $libintTop.FullName
}

if (-not $isClangCl) {
    Write-Host "Applying libint2 MSVC patches..."
    Push-Location $patchBase
    try   { cmake -P "$SrcDir\cpp\cmake\patches\libint2-msvc-sse-macros.cmake" }
    finally { Pop-Location }
}

Invoke-CMakeDep 'libint2' $libintSrc @('-DBUILD_TESTING=OFF') $libintJobs
Remove-Item $libintExtract -Recurse -Force

# ─── ecpint ──────────────────────────────────────────────────────────────────
Write-Host "=== Installing ecpint ==="
$ecpintSrc = "$buildDir\ecpint-src"
git clone https://github.com/robashaw/libecpint $ecpintSrc --no-local
git -C $ecpintSrc checkout $ecpintCommit

if (-not $isClangCl) {
    Write-Host "Applying ecpint MSVC patches..."
    Push-Location $ecpintSrc
    try   { cmake -P "$SrcDir\cpp\cmake\patches\ecpint-msvc-vla.cmake" }
    finally { Pop-Location }
}

Invoke-CMakeDep 'ecpint' $ecpintSrc @(
    '-DLIBECPINT_BUILD_TESTS=OFF',
    '-DLIBECPINT_USE_PUGIXML=OFF'
)
Remove-Item $ecpintSrc -Recurse -Force

# ─── gauxc ───────────────────────────────────────────────────────────────────
# gauxc fetches its own transitive deps (ExchCXX, IntegratorXX, gau2grid,
# linalg-cmake-modules) via FetchContent — all pinned in gauxc's cmake.
Write-Host "=== Installing gauxc ==="
$gauxcSrc = "$buildDir\gauxc-src"
git clone https://github.com/wavefunction91/gauxc.git $gauxcSrc --no-local
git -C $gauxcSrc checkout $gauxcCommit

Invoke-CMakeDep 'gauxc' $gauxcSrc @(
    '-DBUILD_TESTING=OFF',
    '-DEXCHCXX_ENABLE_LIBXC=OFF',
    '-DGAUXC_ENABLE_HDF5=OFF',
    '-DGAUXC_ENABLE_MAGMA=OFF',
    '-DGAUXC_ENABLE_CUDA=OFF',
    '-DGAUXC_ENABLE_CUTLASS=OFF',
    '-DGAUXC_ENABLE_MPI=OFF',
    '-DGAUXC_ENABLE_OPENMP=OFF'
)
Remove-Item $gauxcSrc -Recurse -Force

# ─── Cleanup ─────────────────────────────────────────────────────────────────
if (-not $KeepBuildDir) {
    Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "=== All C++ dependencies installed to: $DepsInstallDir ==="
