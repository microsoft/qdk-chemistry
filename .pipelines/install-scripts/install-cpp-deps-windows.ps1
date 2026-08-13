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

.PARAMETER Triplet
    vcpkg target triplet. Defaults to VCPKG_TRIPLET or x64-windows-static-md.

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
    [string]$Triplet,
    [string]$DepsInstallDir,
    [switch]$KeepBuildDir,
    # Print a hash of the build plan and exit without building anything. The
    # pipeline uses this for the dependency cache key, so that the key is derived
    # from the same description of the build that the build itself consumes.
    [switch]$EmitFingerprint
)
$ErrorActionPreference = 'Stop'

if (-not $VcpkgRoot) {
    $VcpkgRoot = if ($env:VCPKG_INSTALLATION_ROOT) { $env:VCPKG_INSTALLATION_ROOT } else { 'C:\vcpkg' }
}
if (-not (Test-Path "$VcpkgRoot\vcpkg.exe")) { throw "vcpkg.exe not found under '$VcpkgRoot'" }
if (-not $Triplet) {
    $Triplet = if ($env:VCPKG_TRIPLET) { $env:VCPKG_TRIPLET } else { 'x64-windows-static-md' }
}
if (-not $DepsInstallDir) { $DepsInstallDir = "$SrcDir\deps-install-msvc" }
Write-Host "vcpkg triplet: $Triplet"

# ─── Prerequisite check ──────────────────────────────────────────────────────
# Verify the tools the dependency builds need before starting. Several of these
# fail in ways that only surface hours in and are hard to read: without a Python
# interpreter, for example, ecpint's configure silently generates an empty
# source list and dies with "add_custom_command Wrong syntax", and gau2grid
# generates its sources the same way. Checking up front turns a multi-hour build
# into a one-minute failure that names the missing tool.
$required = @(
    @{ Name = 'cmake';  Args = @('--version') },
    @{ Name = 'ninja';  Args = @('--version') },
    @{ Name = 'git';    Args = @('--version') },
    @{ Name = 'python'; Args = @('--version') }
)
$missing = @()
foreach ($tool in $required) {
    $cmd = Get-Command $tool.Name -ErrorAction SilentlyContinue
    if (-not $cmd) { $missing += $tool.Name; continue }
    $ver = (& $cmd.Source @($tool.Args) 2>&1 | Select-Object -First 1)
    Write-Host ("  {0,-8} {1}  ({2})" -f $tool.Name, $ver, $cmd.Source)
}
if ($missing) {
    throw ("Missing required build tools: {0}. The dependency builds need these on PATH; " -f ($missing -join ', ')) +
          "provision them in the pipeline rather than relying on the agent image."
}

$buildDir = "$SrcDir\deps-build-msvc"

# ─── Resumability ────────────────────────────────────────────────────────────
# Each dependency records a stamp naming the exact version that was installed,
# so a restored cache can be inspected and only the missing work redone.
$stampDir = Join-Path $DepsInstallDir '.deps-stamps'

function Set-DepStamp([string]$Name, [string]$Id) {
    New-Item -ItemType Directory -Force -Path $stampDir | Out-Null
    Set-Content -Path (Join-Path $stampDir "$Name.stamp") -Value $Id -NoNewline
}

function Test-DepInstalled([string]$Name, [string]$Id, [string]$ConfigGlob) {
    $stamp = Join-Path $stampDir "$Name.stamp"
    if (Test-Path $stamp) {
        if ((Get-Content $stamp -Raw).Trim() -eq $Id.Trim()) { return $true }
        Write-Host "  $Name stamp does not match the pinned version; rebuilding"
        return $false
    }
    # Caches produced before stamping existed still hold a perfectly good
    # install tree, so fall back to detecting the installed CMake package and
    # adopt it rather than rebuilding for hours over a missing marker file.
    $found = Get-ChildItem (Join-Path $DepsInstallDir 'lib\cmake') -Directory `
                           -Filter $ConfigGlob -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "  $Name found in the restored install tree; adopting it"
        Set-DepStamp $Name $Id
        return $true
    }
    return $false
}

New-Item -ItemType Directory -Force -Path $DepsInstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $buildDir       | Out-Null

# ─── Memory-aware build parallelism ──────────────────────────────────────────
# Cap by RAM (~3.5 GB/job) to avoid OOM on machines with many cores but limited
# memory. On typical CI runners this equals NUMBER_OF_PROCESSORS.
$cpu   = [int]$env:NUMBER_OF_PROCESSORS
$ramGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$jobs  = [math]::Min($cpu, [math]::Max(1, [math]::Floor($ramGB / 3.5)))
Write-Host "CPUs=$cpu  RAM=${ramGB}GB  -> CMAKE_BUILD_PARALLEL_LEVEL=$jobs"
$env:CMAKE_BUILD_PARALLEL_LEVEL = $jobs

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

# ─── Dependency build plan ───────────────────────────────────────────────────
# The single source of truth for everything that determines what gets built.
# The build below consumes it, and -EmitFingerprint hashes it, so the cache key
# and the build can never end up describing different things. Adding a
# dependency or changing a flag here is picked up by the cache key automatically,
# with nothing to remember to bump.
#
# Only portable, semantically meaningful values belong in here. Machine-specific
# paths are kept out by construction (see $machineArgs below) so that the same
# configuration fingerprints identically on every agent.
$plan = [ordered]@{
    BuildType = $BuildType
    Triplet   = $Triplet
    # The compiler's identity matters -- MSVC and clang-cl produce different
    # artifacts, and it selects whether the MSVC patches get applied -- but its
    # install path does not. The toolset version is carried separately in the
    # cache key by the pipeline.
    Compiler   = (Split-Path $ClPath -Leaf)
    CommonArgs = @(
        '-GNinja',
        "-DCMAKE_BUILD_TYPE=$BuildType",
        '-DCMAKE_CXX_STANDARD=20', '-DCMAKE_CXX_STANDARD_REQUIRED=ON',
        '-DBUILD_SHARED_LIBS=OFF',
        "-DVCPKG_TARGET_TRIPLET=$Triplet",
        '-DFETCHCONTENT_QUIET=OFF'
    )
    Deps = @(
        [ordered]@{
            Name       = 'libint2'
            Version    = $libintUrl
            ConfigGlob = 'libint2*'
            CMakeArgs  = @('-DBUILD_TESTING=OFF')
        },
        [ordered]@{
            Name       = 'ecpint'
            Version    = $ecpintCommit
            ConfigGlob = 'ecpint*'
            CMakeArgs  = @(
                '-DLIBECPINT_BUILD_TESTS=OFF',
                '-DLIBECPINT_USE_PUGIXML=OFF'
            )
        },
        [ordered]@{
            Name       = 'gauxc'
            Version    = $gauxcCommit
            ConfigGlob = 'gauxc*'
            CMakeArgs  = @(
                '-DBUILD_TESTING=OFF',
                '-DEXCHCXX_ENABLE_LIBXC=OFF',
                '-DGAUXC_ENABLE_HDF5=OFF',
                '-DGAUXC_ENABLE_MAGMA=OFF',
                '-DGAUXC_ENABLE_CUDA=OFF',
                '-DGAUXC_ENABLE_CUTLASS=OFF',
                '-DGAUXC_ENABLE_MPI=OFF',
                '-DGAUXC_ENABLE_OPENMP=OFF'
            )
        }
    )
}

function Get-DepPlan([string]$Name) {
    $dep = $plan.Deps | Where-Object { $_.Name -eq $Name }
    if (-not $dep) { throw "No plan entry for dependency '$Name'" }
    return $dep
}

if ($EmitFingerprint) {
    # Canonical JSON so that formatting never perturbs the hash. Emit exactly one
    # line on stdout; everything else the script prints goes to Write-Host.
    $json  = $plan | ConvertTo-Json -Depth 10 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $sha   = [System.Security.Cryptography.SHA256]::Create()
    Write-Output ([System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant())
    return
}

# ─── Common CMake flags (applied to every dep build) ─────────────────────────
# Paths differ from agent to agent, so they are deliberately not part of the
# plan: including them would make the fingerprint, and therefore the cache key,
# vary between machines that are building exactly the same thing.
$machineArgs = @(
    "-DCMAKE_C_COMPILER=$ClPath",
    "-DCMAKE_CXX_COMPILER=$ClPath",
    "-DCMAKE_INSTALL_PREFIX=$DepsInstallDir",
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake",
    "-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$SrcDir\.pipelines\toolchains\windows.cmake",
    "-DVCPKG_INSTALLED_DIR=$SrcDir\vcpkg_installed"
)
$commonArgs = $plan.CommonArgs + $machineArgs

function Invoke-CMakeDep([string]$Name, [string]$SrcPath, [string[]]$ExtraArgs) {
    $depBuild = "$buildDir\$Name-build"
    Write-Host "--- cmake configure: $Name ---"
    cmake -S $SrcPath -B $depBuild @commonArgs @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for $Name ($LASTEXITCODE)" }

    Write-Host "--- cmake build: $Name (jobs=$env:CMAKE_BUILD_PARALLEL_LEVEL) ---"
    cmake --build $depBuild
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed for $Name ($LASTEXITCODE)" }

    Write-Host "--- cmake install: $Name ---"
    cmake --install $depBuild
    if ($LASTEXITCODE -ne 0) { throw "cmake install failed for $Name ($LASTEXITCODE)" }

    if (-not $KeepBuildDir) {
        Remove-Item $depBuild -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ─── vcpkg install ────────────────────────────────────────────────────────────
# Provides eigen3, openblas, hdf5, boost-headers, spdlog, nlohmann-json, etc.
$vcpkgInstalled = "$SrcDir\vcpkg_installed"
$env:X_VCPKG_ASSET_SOURCES = 'x-azurl,https://vcpkg.storage.devpackages.microsoft.io/artifacts/'
Write-Host "=== vcpkg install ==="
& "$VcpkgRoot\vcpkg.exe" install `
    --triplet $Triplet `
    --x-manifest-root="$SrcDir" `
    --x-install-root="$vcpkgInstalled" `
    --overlay-ports="$SrcDir\vcpkg-overlay\ports"
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed ($LASTEXITCODE)" }

# ─── libint2 ─────────────────────────────────────────────────────────────────
if (-not (Test-DepInstalled 'libint2' $libintUrl 'libint2*')) {
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

    Invoke-CMakeDep 'libint2' $libintSrc (Get-DepPlan 'libint2').CMakeArgs
    Remove-Item $libintExtract -Recurse -Force
    Set-DepStamp 'libint2' $libintUrl
} else { Write-Host "=== libint2 already installed; skipping ===" }

# ─── ecpint ──────────────────────────────────────────────────────────────────
if (-not (Test-DepInstalled 'ecpint' $ecpintCommit 'ecpint*')) {
    Write-Host "=== Installing ecpint ==="
    $ecpintSrc = "$buildDir\ecpint-src"
    git clone https://github.com/robashaw/libecpint $ecpintSrc
    git -C $ecpintSrc checkout $ecpintCommit

    if (-not $isClangCl) {
        Write-Host "Applying ecpint MSVC patches..."
        Push-Location $ecpintSrc
        try   { cmake -P "$SrcDir\cpp\cmake\patches\ecpint-msvc-vla.cmake" }
        finally { Pop-Location }
    }

    Invoke-CMakeDep 'ecpint' $ecpintSrc (Get-DepPlan 'ecpint').CMakeArgs
    Remove-Item $ecpintSrc -Recurse -Force
    Set-DepStamp 'ecpint' $ecpintCommit
} else { Write-Host "=== ecpint already installed; skipping ===" }

# ─── gauxc ───────────────────────────────────────────────────────────────────
# gauxc fetches its own transitive deps (ExchCXX, IntegratorXX, gau2grid,
# linalg-cmake-modules) via FetchContent — all pinned in gauxc's cmake.
if (-not (Test-DepInstalled 'gauxc' $gauxcCommit 'gauxc*')) {
    Write-Host "=== Installing gauxc ==="
    $gauxcSrc = "$buildDir\gauxc-src"
    git clone https://github.com/wavefunction91/gauxc.git $gauxcSrc
    git -C $gauxcSrc checkout $gauxcCommit

    Invoke-CMakeDep 'gauxc' $gauxcSrc (Get-DepPlan 'gauxc').CMakeArgs
    Remove-Item $gauxcSrc -Recurse -Force
    Set-DepStamp 'gauxc' $gauxcCommit
} else { Write-Host "=== gauxc already installed; skipping ===" }

# ─── Cleanup ─────────────────────────────────────────────────────────────────
if (-not $KeepBuildDir) {
    Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "=== C++ dependencies installed to: $DepsInstallDir ==="
