<#
.SYNOPSIS
    Build Python wheels for qdk-chemistry on Windows using vcpkg and clang-cl.

.PARAMETER March
    Target microarchitecture (e.g. x86-64-v3, armv8-a)

.PARAMETER PythonVersion
    Python version to build for (e.g. 3.12)

.PARAMETER BuildType
    CMake build type (default: Release)

.PARAMETER VcpkgTriplet
    vcpkg triplet (e.g. x64-windows, arm64-windows)

.PARAMETER DevTag
    Optional development tag for versioning (e.g. dev1, rc1)
#>
param(
    [string]$March = "x86-64-v3",
    [string]$PythonVersion = "3.12",
    [string]$BuildType = "Release",
    [string]$VcpkgTriplet = "x64-windows",
    [string]$DevTag = "None"
)

$ErrorActionPreference = "Stop"

Write-Host "=== QDK Chemistry Windows Wheel Build ==="
Write-Host "March: $March"
Write-Host "Python: $PythonVersion"
Write-Host "BuildType: $BuildType"
Write-Host "VcpkgTriplet: $VcpkgTriplet"
Write-Host "DevTag: $DevTag"

# ---------------------------------------------------------------------------
# 1. Setup vcpkg
# ---------------------------------------------------------------------------
if ($env:VCPKG_INSTALLATION_ROOT -and (Test-Path "$env:VCPKG_INSTALLATION_ROOT\vcpkg.exe")) {
    $vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
    Write-Host "Using pre-installed vcpkg at $vcpkgRoot"
} elseif (Test-Path "C:\vcpkg\vcpkg.exe") {
    $vcpkgRoot = "C:\vcpkg"
    Write-Host "Using vcpkg at $vcpkgRoot"
} else {
    Write-Host "Cloning and bootstrapping vcpkg..."
    git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
    & C:\vcpkg\bootstrap-vcpkg.bat -disableMetrics
    $vcpkgRoot = "C:\vcpkg"
}

$vcpkgToolchain = "$vcpkgRoot\scripts\buildsystems\vcpkg.cmake"

# ---------------------------------------------------------------------------
# 2. Install vcpkg dependencies
# ---------------------------------------------------------------------------
Write-Host "Installing vcpkg dependencies for triplet $VcpkgTriplet..."
& "$vcpkgRoot\vcpkg.exe" install `
    --triplet $VcpkgTriplet `
    --overlay-ports=vcpkg-overlay/ports `
    --x-manifest-root=. `
    --x-install-root="$vcpkgRoot\installed"

$vcpkgInstalledDir = "$vcpkgRoot\installed\$VcpkgTriplet"
Write-Host "vcpkg installed directory: $vcpkgInstalledDir"

# ---------------------------------------------------------------------------
# 3. Locate clang-cl
# ---------------------------------------------------------------------------
# clang-cl ships with Visual Studio's LLVM/Clang component
$clangClCandidates = @(
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm\x64\bin\clang-cl.exe",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Tools\Llvm\x64\bin\clang-cl.exe",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clang-cl.exe",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin\clang-cl.exe",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm\x64\bin\clang-cl.exe",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Professional\VC\Tools\Llvm\x64\bin\clang-cl.exe",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clang-cl.exe",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin\clang-cl.exe"
)

$clangCl = $null
foreach ($candidate in $clangClCandidates) {
    if (Test-Path $candidate) {
        $clangCl = $candidate
        break
    }
}

# Fallback: try clang-cl from PATH
if (-not $clangCl) {
    $clangCl = (Get-Command clang-cl -ErrorAction SilentlyContinue).Source
}

if (-not $clangCl) {
    Write-Error "clang-cl not found. Install the 'C++ Clang tools for Windows' component in Visual Studio."
    exit 1
}
Write-Host "Using clang-cl: $clangCl"

# ---------------------------------------------------------------------------
# 4. Set environment variables
# ---------------------------------------------------------------------------
$env:CC = $clangCl
$env:CXX = $clangCl
$env:CMAKE_BUILD_PARALLEL_LEVEL = [Environment]::ProcessorCount
$env:CMAKE_TOOLCHAIN_FILE = $vcpkgToolchain
$env:VCPKG_TARGET_TRIPLET = $VcpkgTriplet
$env:CMAKE_PREFIX_PATH = $vcpkgInstalledDir

# ---------------------------------------------------------------------------
# 5. Upgrade pip and install build tools
# ---------------------------------------------------------------------------
Write-Host "Upgrading pip and installing build tools..."
python -m pip install --upgrade pip
python -m pip install build delvewheel

# ---------------------------------------------------------------------------
# 6. Prepare README for PyPI
# ---------------------------------------------------------------------------
Write-Host "Preparing README..."
$GITHUB_BLOB = "https://github.com/microsoft/qdk-chemistry/blob/main"
$GITHUB_TREE = "https://github.com/microsoft/qdk-chemistry/tree/main"
$readme = Get-Content README.md -Raw
$readme = $readme -replace '\]\(\./([^)]+)\)', "](${GITHUB_BLOB}/`$1)"
$readme = $readme -replace '\]\(([A-Z][A-Z_]*\.(md|txt))\)', "](${GITHUB_BLOB}/`$1)"
$readme | Set-Content python/README.md -Encoding utf8

# ---------------------------------------------------------------------------
# 7. Build wheel
# ---------------------------------------------------------------------------
Write-Host "Building wheel..."
Push-Location python

$buildArgs = @(
    "-m", "build", "--wheel",
    "-C", "build-dir=build/{wheel_tag}",
    "-C", "cmake.define.QDK_UARCH=$March",
    "-C", "cmake.define.BUILD_SHARED_LIBS=OFF",
    "-C", "cmake.define.QDK_CHEMISTRY_ENABLE_MPI=OFF",
    "-C", "cmake.define.QDK_ENABLE_OPENMP=OFF",
    "-C", "cmake.define.QDK_CHEMISTRY_ENABLE_COVERAGE=OFF",
    "-C", "cmake.define.BUILD_TESTING=OFF",
    "-C", "cmake.define.CMAKE_C_COMPILER=$clangCl",
    "-C", "cmake.define.CMAKE_CXX_COMPILER=$clangCl",
    "-C", "cmake.define.CMAKE_TOOLCHAIN_FILE=$vcpkgToolchain",
    "-C", "cmake.define.CMAKE_PREFIX_PATH=$vcpkgInstalledDir"
)

& python @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Wheel build failed with exit code $LASTEXITCODE" }

# ---------------------------------------------------------------------------
# 8. Repair wheel
# ---------------------------------------------------------------------------
Write-Host "Repairing wheel with delvewheel..."
New-Item -ItemType Directory -Force -Path repaired_wheelhouse | Out-Null
$wheelFile = Get-ChildItem dist\qdk_chemistry-*.whl | Select-Object -First 1

Write-Host "Wheel file: $($wheelFile.FullName)"
delvewheel show $wheelFile.FullName
delvewheel repair $wheelFile.FullName -w repaired_wheelhouse\
if ($LASTEXITCODE -ne 0) { throw "delvewheel repair failed with exit code $LASTEXITCODE" }

Write-Host "Repaired wheel:"
Get-ChildItem repaired_wheelhouse\

Pop-Location
Write-Host "=== Windows wheel build complete ==="
