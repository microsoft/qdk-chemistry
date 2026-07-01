<#
.SYNOPSIS
    Build and install C++ dependencies for the Windows wheel pipeline.

.DESCRIPTION
    Thin wrapper around install-cpp-deps-windows.ps1. Invoked only on a
    dependency cache miss.
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

& "$PSScriptRoot\install-cpp-deps-windows.ps1" `
    -SrcDir        $SrcDir `
    -ClPath        $ClPath `
    -BuildType     $BuildType `
    -VcpkgRoot     $VcpkgRoot `
    -DepsInstallDir $DepsInstallDir
if ($LASTEXITCODE -ne 0) { throw "install-cpp-deps-windows.ps1 failed ($LASTEXITCODE)" }
