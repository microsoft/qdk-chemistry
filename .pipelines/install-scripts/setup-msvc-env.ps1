<#
.SYNOPSIS
    Set up the MSVC x64 developer environment for Windows CI pipelines.

.DESCRIPTION
    Runs vcvarsall x64, discovers the C++ compiler, and exports all changed
    environment variables to the CI system. Auto-detects GitHub Actions
    (GITHUB_ENV set) vs Azure Pipelines (##vso commands). Sets $env:MSVC_TOOLSET
    and $env:CXX_PATH in the current process for the calling step to read.

.PARAMETER Compiler
    'msvc' (default) to use cl.exe, or 'clang-cl' to use clang-cl.exe.
#>
param(
    [string]$Compiler = 'msvc'
)
$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) { throw "No Visual Studio installation found" }
Write-Host "Visual Studio: $vsPath"

$vcvarsall = "$vsPath\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvarsall)) { throw "vcvarsall.bat not found at $vcvarsall" }

# Snapshot env, run vcvarsall x64, diff to capture changes.
$before = @{}
Get-ChildItem env: | ForEach-Object { $before[$_.Name] = $_.Value }
$tmp = [System.IO.Path]::GetTempFileName()
cmd /c "`"$vcvarsall`" x64 && set > `"$tmp`""
if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat failed ($LASTEXITCODE)" }
$after = @{}
Get-Content $tmp | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { $after[$matches[1]] = $matches[2] }
}
Remove-Item $tmp

# Locate clang-cl if requested (not added by vcvarsall).
$clangDir = $null
if ($Compiler -eq 'clang-cl') {
    $candidates = @(
        "$vsPath\VC\Tools\Llvm\x64\bin\clang-cl.exe",
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

# Expose in current process so the calling step can read them immediately.
$env:MSVC_TOOLSET = $toolset
$env:CXX_PATH     = $cxx

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
    "MSVC_TOOLSET=$toolset" >> $env:GITHUB_ENV
    "CXX_PATH=$cxx"         >> $env:GITHUB_ENV
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
