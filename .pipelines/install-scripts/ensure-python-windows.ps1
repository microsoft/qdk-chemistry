<#
.SYNOPSIS
    Ensure a Python interpreter is on PATH for the Windows C++ dependency build.

.DESCRIPTION
    Two of the C++ dependencies generate their sources with Python at CMake
    configure time: ecpint runs makelist.py to emit its integral kernels, and
    gau2grid runs a generator to emit its collocation code. Neither needs pip or
    any third-party package -- a bare interpreter is enough.

    UsePythonVersion@0 cannot be used on the Windows ARM64 image. That image
    ships an empty tool cache, so the task falls through to downloading Python,
    and its post-install step then tries to bootstrap pip from pypi.org. The 1ES
    network isolation policy set applied to this pipeline is
    "Permissive,CFSClean,CFSClean2,CFSClean3", and CFSClean exists to "block
    access to public package feeds and enforce CFS (Centralized Feed Service)
    for package management". So the pip bootstrap is blocked by design, the task
    treats that as fatal, and it unregisters the Python it just extracted.

    This script provisions the interpreter without ever contacting a package
    feed, which keeps it within the isolation policy. Where the wheel jobs do
    need packages, they get them from the internal feed via PipAuthenticate@1.

    Resolution order, cheapest first:
      1. A working python already on PATH.
      2. A python already in the agent tool cache. This also recovers the
         extracted-but-unregistered interpreter that a failed UsePythonVersion@0
         leaves behind.
      3. A download from the actions/python-versions GitHub release feed, which
         the isolation policy does allow. The archive's own setup.ps1 is
         deliberately not run, because that is the part that reaches for pip.

.PARAMETER Arch
    Target architecture, x64 or arm64. Defaults to the host architecture.

.PARAMETER Version
    Python feature version to provision, e.g. '3.13'. Only used when a download
    is actually required.

.PARAMETER Force
    Skip the PATH and tool cache probes and go straight to the download. Used to
    exercise the download path when testing this script.

.PARAMETER IncludePip
    Install pip as well. The dependency build does not need it, but the wheel
    jobs do, because `python -m venv` provisions pip into the new environment.
    This stays within the isolation policy: the python.org installer bundles pip
    and installs it offline via ensurepip. It is only the actions/python-versions
    setup.ps1 wrapper, which afterwards runs `pip install --upgrade pip` against
    pypi.org, that needs a package feed -- and that wrapper is never used here.
#>
[CmdletBinding()]
param(
    [ValidateSet('x64', 'arm64')]
    [string]$Arch,
    [string]$Version = '3.13',
    [switch]$Force,
    [switch]$IncludePip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $Arch) {
    # PROCESSOR_ARCHITEW6432 reports the real machine architecture when the
    # current process is itself emulated, which PROCESSOR_ARCHITECTURE does not.
    $hostArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    $Arch = if ($hostArch -imatch 'arm64') { 'arm64' } else { 'x64' }
}
Write-Host "Ensuring Python is available (arch=$Arch, requested version=$Version)"

function Test-PythonExe {
    param([string]$Exe)
    if (-not $Exe -or -not (Test-Path $Exe)) { return $null }
    try {
        # Collect all output before inspecting $LASTEXITCODE. Piping a native
        # command straight into `Select-Object -First 1` stops the pipeline
        # early, which can leave $LASTEXITCODE unassigned -- and under
        # Set-StrictMode, reading a never-assigned $LASTEXITCODE throws, so a
        # perfectly good interpreter gets reported as broken.
        $out  = & $Exe --version 2>&1
        $code = if (Test-Path variable:LASTEXITCODE) { $LASTEXITCODE } else { 0 }
        if ($code -ne 0 -or ($out | Out-String) -notmatch 'Python\s+(\d+\.\d+\.\d+)') { return $null }
        $found = $Matches[1]
        # A pip-less interpreter is no use to a caller that asked for pip, so
        # treat it as a miss and keep looking rather than failing later inside
        # venv creation.
        if ($IncludePip) {
            & $Exe -m pip --version 2>&1 | Out-Null
            $pipCode = if (Test-Path variable:LASTEXITCODE) { $LASTEXITCODE } else { 0 }
            if ($pipCode -ne 0) { return $null }
        }
        return $found
    } catch {
        # An interpreter that cannot run is the same as no interpreter. This
        # legitimately happens in the tool cache when an install was interrupted
        # partway through, so fall through to the next candidate.
    }
    return $null
}

function Publish-Python {
    param([string]$Dir, [string]$Ver, [string]$How)
    Write-Host "##vso[task.prependpath]$Dir"
    $scripts = Join-Path $Dir 'Scripts'
    if (Test-Path $scripts) { Write-Host "##vso[task.prependpath]$scripts" }
    $env:PATH = "$Dir;$env:PATH"
    Write-Host "Python $Ver ready via $How"
    Write-Host "  $Dir"
}

# ─── 1. Already on PATH ───────────────────────────────────────────────────────
if (-not $Force) {
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) {
        $ver = Test-PythonExe $onPath.Source
        if ($ver) {
            Write-Host "Python $ver already on PATH"
            Write-Host "  $($onPath.Source)"
            return
        }
        Write-Host "python on PATH at '$($onPath.Source)' is not runnable; continuing to search"
    }
}

# ─── 2. Agent tool cache ──────────────────────────────────────────────────────
if (-not $Force) {
    $roots = @($env:AGENT_TOOLSDIRECTORY, 'C:\ToolCache', 'C:\hostedtoolcache\windows') |
        Where-Object { $_ -and (Test-Path (Join-Path $_ 'Python')) } |
        ForEach-Object { Join-Path $_ 'Python' }

    foreach ($root in $roots) {
        # Newest version first so we do not pin to a stale cache entry.
        $candidates = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object { try { [version]$_.Name } catch { [version]'0.0.0' } } -Descending
        foreach ($verDir in $candidates) {
            $exe = Join-Path $verDir.FullName "$Arch\python.exe"
            $ver = Test-PythonExe $exe
            if ($ver) { Publish-Python (Split-Path $exe -Parent) $ver "tool cache ($root)"; return }
        }
    }
    Write-Host "No usable interpreter in the agent tool cache"
}

# ─── 3. Download from the actions/python-versions release feed ────────────────
# This is the same source UsePythonVersion@0 uses, minus the pip bootstrap.
$manifestUrl = 'https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json'
Write-Host "Fetching Python version manifest"
Write-Host "  $manifestUrl"
$manifest = Invoke-RestMethod -Uri $manifestUrl -UseBasicParsing

$asset = $manifest |
    Where-Object { $_.stable -and $_.version -like "$Version.*" } |
    Sort-Object { [version]$_.version } -Descending |
    ForEach-Object {
        $v = $_.version
        $_.files |
            Where-Object { $_.platform -eq 'win32' -and $_.arch -eq $Arch } |
            ForEach-Object { [pscustomobject]@{ Version = $v; Url = $_.download_url } }
    } |
    Select-Object -First 1

if (-not $asset) {
    throw "No stable Python $Version build for win32/$Arch in the actions/python-versions manifest."
}

$work    = Join-Path ([System.IO.Path]::GetTempPath()) "python-$($asset.Version)-$Arch"
$zipPath = "$work.zip"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
Write-Host "Downloading Python $($asset.Version) for $Arch"
Write-Host "  $($asset.Url)"

$progressPreferenceOriginal = $ProgressPreference
$ProgressPreference = 'SilentlyContinue'   # Progress rendering is very slow over the ADO log pipe.
try {
    Invoke-WebRequest -Uri $asset.Url -OutFile $zipPath -UseBasicParsing
} finally {
    $ProgressPreference = $progressPreferenceOriginal
}

Write-Host "Extracting to $work"
Expand-Archive -Path $zipPath -DestinationPath $work -Force
Remove-Item $zipPath -Force

# The Windows assets in this feed are not portable trees: each one contains the
# official python.org installer plus a setup.ps1 wrapper. Run the installer
# directly and skip that wrapper, because the wrapper's final step is the
# `pip install --upgrade pip` that the isolation policy blocks.
$installer = Get-ChildItem $work -Filter '*.exe' -File | Select-Object -First 1
if (-not $installer) { throw "No installer found under '$work' after extracting $($asset.Url)" }

$targetDir = Join-Path $env:SystemDrive "PythonCI\$($asset.Version)\$Arch"
Write-Host "Running $($installer.Name) -> $targetDir"
# The dependency build only needs the standard library, so pip is left out by
# default and this step then touches no package feed at all. When the caller
# does ask for pip, the installer still provisions it offline from its bundled
# copy, so the CFSClean policy is satisfied either way.
$installArgs = @(
    '/quiet'
    "TargetDir=$targetDir"
    'InstallAllUsers=0'
    "Include_pip=$(if ($IncludePip) { 1 } else { 0 })"
    'Include_test=0'
    'Include_doc=0'
    'Include_tcltk=0'
    'Include_launcher=0'
    'InstallLauncherAllUsers=0'
    'AssociateFiles=0'
    'Shortcuts=0'
    'PrependPath=0'
)
$proc = Start-Process -FilePath $installer.FullName -ArgumentList $installArgs -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    throw "Python installer '$($installer.Name)' failed with exit code $($proc.ExitCode)."
}

$exe = Join-Path $targetDir 'python.exe'
$ver = Test-PythonExe $exe
if (-not $ver) { throw "Installed interpreter at '$exe' failed to run." }
Publish-Python $targetDir $ver 'actions/python-versions installer'
