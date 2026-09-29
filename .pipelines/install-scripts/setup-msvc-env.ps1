<#
.SYNOPSIS
    Set up the MSVC developer environment for Windows CI pipelines (x64 or ARM64).

.DESCRIPTION
    Runs vcvarsall for the requested target architecture, discovers the C++
    compiler, and exports all changed environment variables to the CI system.
    Auto-detects GitHub Actions (GITHUB_ENV set) vs Azure Pipelines (##vso
    commands). Sets $env:MSVC_TOOLSET, $env:CXX_PATH and $env:VCPKG_TRIPLET in
    the current process for the calling step to read.

.PARAMETER Compiler
    'msvc' (default) to use cl.exe, or 'clang-cl' to use clang-cl.exe.

.PARAMETER Arch
    Target architecture: 'x64' (default) or 'arm64'. The vcvarsall argument is
    derived from the host/target pair, so this works both natively and when
    cross-compiling.
#>
param(
    [ValidateSet('msvc', 'clang-cl')]
    [string]$Compiler = 'msvc',
    [ValidateSet('x64', 'arm64')]
    [string]$Arch = 'x64'
)
$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) { throw "No Visual Studio installation found" }
Write-Host "Visual Studio: $vsPath"

$vcvarsall = "$vsPath\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvarsall)) { throw "vcvarsall.bat not found at $vcvarsall" }

# vcvarsall takes "<host>_<target>", or just "<target>" when they are equal.
# PROCESSOR_ARCHITECTURE reports the architecture of the *current process*, so a
# 32-bit or x64-emulated process on an ARM64 machine reports x86/AMD64. Windows
# sets PROCESSOR_ARCHITEW6432 to the real machine architecture in that case, so
# prefer it when present; otherwise a natively-capable agent would silently be
# treated as a cross-compile host.
$hostRaw = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
$hostArch = switch ($hostRaw) {
    'ARM64' { 'arm64' }
    default { 'x64' }
}
$vcvarsArg = if ($hostArch -eq $Arch) { $Arch } else { "${hostArch}_${Arch}" }
Write-Host "Host: $hostArch (PROCESSOR_ARCHITECTURE=$env:PROCESSOR_ARCHITECTURE, PROCESSOR_ARCHITEW6432=$env:PROCESSOR_ARCHITEW6432)"
Write-Host "Target: $Arch  ->  vcvarsall $vcvarsArg"

# Snapshot env, run vcvarsall, diff to capture changes.
$before = @{}
Get-ChildItem env: | ForEach-Object { $before[$_.Name] = $_.Value }
$tmp = [System.IO.Path]::GetTempFileName()
cmd /c "`"$vcvarsall`" $vcvarsArg && set > `"$tmp`""
if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat $vcvarsArg failed ($LASTEXITCODE)" }
$after = @{}
Get-Content $tmp | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { $after[$matches[1]] = $matches[2] }
}
Remove-Item $tmp

# Locate clang-cl if requested (not added by vcvarsall).
# The clang-cl binary must run on the *host*, so pick the host-arch LLVM dir.
$clangDir = $null
if ($Compiler -eq 'clang-cl') {
    if ($hostArch -ne $Arch) {
        throw "clang-cl cross-compilation ($hostArch host -> $Arch target) is not supported by this script."
    }
    $llvmHostDir = if ($hostArch -eq 'arm64') { 'ARM64' } else { 'x64' }
    $candidates = @(
        "$vsPath\VC\Tools\Llvm\$llvmHostDir\bin\clang-cl.exe",
        "$vsPath\VC\Tools\Llvm\bin\clang-cl.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { $clangDir = Split-Path $c; break } }
    if (-not $clangDir) { throw "clang-cl.exe not found under $vsPath\VC\Tools\Llvm" }
}

# Apply vcvarsall changes to the current process.
foreach ($name in $after.Keys) {
    [System.Environment]::SetEnvironmentVariable($name, $after[$name], 'Process')
}
if ($clangDir) { $env:PATH = "$clangDir;$env:PATH" }

# Resolve the compiler path.
$cxx = if ($Compiler -eq 'clang-cl') {
    (Get-Command clang-cl.exe -ErrorAction Stop).Source
} else {
    (Get-Command cl.exe -ErrorAction Stop).Source
}
Write-Host "C++ compiler: $cxx"
if ($Compiler -eq 'clang-cl') { & $cxx --version 2>&1 | Write-Host }

# Stable MSVC toolset version for cache keys.
$toolsetFile = "$vsPath\VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"
$toolset = (Get-Content $toolsetFile -ErrorAction Stop | Select-Object -First 1).Trim()
Write-Host "MSVC toolset: $toolset"

# vcpkg triplet for the requested target architecture.
$triplet = "$Arch-windows-static-md"
Write-Host "vcpkg triplet: $triplet"

# Expose in current process so the calling step can read them immediately.
$env:MSVC_TOOLSET  = $toolset
$env:CXX_PATH      = $cxx
$env:VCPKG_TRIPLET = $triplet

# Propagate vcvarsall env changes and CI-specific outputs.
$beforePath    = @($before['Path'] -split ';')
$newPathEntries = @($after['Path'] -split ';') | Where-Object { $_ -and ($beforePath -notcontains $_) }

if ($env:GITHUB_ENV) {
    # GitHub Actions: write to GITHUB_ENV / GITHUB_PATH.
    foreach ($name in $after.Keys) {
        if ($name -ieq 'Path') { continue }
        if ($before[$name] -ne $after[$name]) { "$name=$($after[$name])" >> $env:GITHUB_ENV }
    }
    if ($clangDir) { $clangDir >> $env:GITHUB_PATH }
    $newPathEntries | ForEach-Object { $_ >> $env:GITHUB_PATH }
    "MSVC_TOOLSET=$toolset"   >> $env:GITHUB_ENV
    "CXX_PATH=$cxx"           >> $env:GITHUB_ENV
    "VCPKG_TRIPLET=$triplet"  >> $env:GITHUB_ENV
} else {
    # Azure Pipelines: use ##vso logging commands.
    foreach ($name in $after.Keys) {
        if ($name -ieq 'Path') { continue }
        if ($before[$name] -ne $after[$name]) {
            Write-Host "##vso[task.setvariable variable=$name]$($after[$name])"
        }
    }
    if ($clangDir) { Write-Host "##vso[task.prependpath]$clangDir" }
    $newPathEntries | ForEach-Object { Write-Host "##vso[task.prependpath]$_" }
    Write-Host "##vso[task.setvariable variable=MSVC_TOOLSET]$toolset"
    Write-Host "##vso[task.setvariable variable=CL_PATH]$cxx"
    Write-Host "##vso[task.setvariable variable=CXX_PATH]$cxx"
    Write-Host "##vso[task.setvariable variable=VCPKG_TRIPLET]$triplet"
}

# Bootstrap vcpkg if not pre-installed.
$vcpkgRoot = if ($env:VCPKG_INSTALLATION_ROOT) { $env:VCPKG_INSTALLATION_ROOT } else { 'C:\vcpkg' }
if (Test-Path "$vcpkgRoot\vcpkg.exe") {
    Write-Host "vcpkg already available at $vcpkgRoot"
} else {
    Write-Host "vcpkg not found — bootstrapping into $vcpkgRoot"
    git clone --depth 1 https://github.com/microsoft/vcpkg.git $vcpkgRoot
    & "$vcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw "vcpkg bootstrap failed ($LASTEXITCODE)" }
    if ($env:GITHUB_ENV) { "VCPKG_INSTALLATION_ROOT=$vcpkgRoot" >> $env:GITHUB_ENV }
    else                  { Write-Host "##vso[task.setvariable variable=VCPKG_INSTALLATION_ROOT]$vcpkgRoot" }
}
# Override VCPKG_ROOT: vcvarsall may set it to VS's bundled copy.
if ($env:GITHUB_ENV) { "VCPKG_ROOT=$vcpkgRoot" >> $env:GITHUB_ENV }
else                  { Write-Host "##vso[task.setvariable variable=VCPKG_ROOT]$vcpkgRoot" }
