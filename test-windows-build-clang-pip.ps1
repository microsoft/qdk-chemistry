# test-windows-build-clang-pip.ps1
# Local script to build and test the QDK Chemistry Python package on Windows using Clang and pip.
# pip (via scikit-build-core) handles the full C++ and Python build in one step.
# Run from the repo root in an elevated PowerShell (admin) if VS Build Tools need installing.
#
# Usage:
#   .\test-windows-build-clang-pip.ps1                  # Full build (install deps + build + test)
#   .\test-windows-build-clang-pip.ps1 -SkipPrereqs     # Skip prerequisite installation
#   .\test-windows-build-clang-pip.ps1 -SkipBuild       # Skip build, only run tests
#   .\test-windows-build-clang-pip.ps1 -SkipTests       # Skip test runs

param(
    [switch]$SkipPrereqs,
    [switch]$SkipBuild,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$VcpkgInstalledDir = "$RepoRoot\vcpkg_installed"
$QDK_UARCH = "x86-64-v3"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  QDK Chemistry - Windows Build (pip)       " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot"
Write-Host ""

# --------------------------------------------------------------------------
# Helper: ensure a command exists
# --------------------------------------------------------------------------
function Assert-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "$Name not found in PATH. Please install it first."
        exit 1
    }
}

# ==========================================================================
# STEP 0 - Prerequisites
# ==========================================================================
if (-not $SkipPrereqs) {
    Write-Host ""
    Write-Host "=== Step 0: Checking / installing prerequisites ===" -ForegroundColor Yellow

    # --- 0a. VS Build Tools with clang-cl ---
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $clangCl = $null

    # Search existing VS installations for clang-cl (-products * includes BuildTools)
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -property installationPath 2>$null
        if ($vsPath) {
            $candidates = @(
                "$vsPath\VC\Tools\Llvm\x64\bin\clang-cl.exe",
                "$vsPath\VC\Tools\Llvm\bin\clang-cl.exe"
            )
            foreach ($c in $candidates) {
                if (Test-Path $c) { $clangCl = $c; break }
            }
        }
    }

    if (-not $clangCl) {
        Write-Host "clang-cl not found. Installing VS Build Tools with C++ and Clang components..." -ForegroundColor Magenta
        Write-Host "This requires an elevated (admin) PowerShell and will take several minutes."
        Write-Host ""

        $installerUrl = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
        $installerPath = "$env:TEMP\vs_BuildTools.exe"

        if (-not (Test-Path $installerPath)) {
            Write-Host "Downloading VS Build Tools installer..."
            Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
        }

        Write-Host "Running VS Build Tools installer (this may take 10-20 minutes)..."
        $installArgs = @(
            "--quiet", "--wait", "--norestart",
            "--add", "Microsoft.VisualStudio.Workload.VCTools",
            "--add", "Microsoft.VisualStudio.Component.VC.Llvm.Clang",
            "--add", "Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset",
            "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "--add", "Microsoft.VisualStudio.Component.Windows11SDK.26100",
            "--includeRecommended"
        )
        $proc = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru
        if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
            Write-Error "VS Build Tools installation failed with exit code $($proc.ExitCode)"
            exit 1
        }
        Write-Host "VS Build Tools installed successfully." -ForegroundColor Green

        # Re-search for clang-cl
        $vsPath = & $vswhere -latest -products * -property installationPath 2>$null
        if ($vsPath) {
            $candidates = @(
                "$vsPath\VC\Tools\Llvm\x64\bin\clang-cl.exe",
                "$vsPath\VC\Tools\Llvm\bin\clang-cl.exe"
            )
            foreach ($c in $candidates) {
                if (Test-Path $c) { $clangCl = $c; break }
            }
        }

        if (-not $clangCl) {
            Write-Error "clang-cl still not found after installing VS Build Tools."
            exit 1
        }
    }

    $clangDir = Split-Path $clangCl
    Write-Host "Using clang-cl: $clangCl" -ForegroundColor Green
    & $clangCl --version

    # Add clang-cl to PATH for this session
    $env:PATH = "$clangDir;$env:PATH"

    # --- 0b. Set up MSVC environment (vcvarsall) ---
    Write-Host ""
    Write-Host "Setting up MSVC developer environment..."
    $vsPath = & $vswhere -latest -products * -property installationPath
    $vcvarsall = "$vsPath\VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path $vcvarsall)) {
        Write-Error "vcvarsall.bat not found at $vcvarsall"
        exit 1
    }

    # Capture environment from vcvarsall
    $envBefore = @{}
    Get-ChildItem env: | ForEach-Object { $envBefore[$_.Name] = $_.Value }

    $tempFile = [System.IO.Path]::GetTempFileName()
    cmd /c "`"$vcvarsall`" x64 && set > `"$tempFile`""
    Get-Content $tempFile | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            $name = $matches[1]
            $value = $matches[2]
            if ($envBefore[$name] -ne $value) {
                [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
    }
    Remove-Item $tempFile
    Write-Host "MSVC developer environment configured." -ForegroundColor Green

    # Re-add clang-cl to PATH (vcvarsall may have reset it)
    $env:PATH = "$clangDir;$env:PATH"

    # --- 0c. vcpkg ---
    # Check VS-bundled vcpkg first, then VCPKG_INSTALLATION_ROOT, then bootstrap
    $vcpkgRoot = $null
    $vsVcpkg = "$vsPath\VC\vcpkg\vcpkg.exe"
    if (Test-Path $vsVcpkg) {
        $vcpkgRoot = Split-Path $vsVcpkg
    } elseif ($env:VCPKG_INSTALLATION_ROOT -and (Test-Path "$($env:VCPKG_INSTALLATION_ROOT)\vcpkg.exe")) {
        $vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
    } else {
        $vcpkgRoot = "$RepoRoot\vcpkg-tool"
        if (-not (Test-Path "$vcpkgRoot\vcpkg.exe")) {
            Write-Host ""
            Write-Host "Bootstrapping vcpkg..."
            git clone https://github.com/microsoft/vcpkg.git "$vcpkgRoot"
            & "$vcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
        }
    }
    Write-Host "Using vcpkg: $vcpkgRoot\vcpkg.exe" -ForegroundColor Green

    # --- 0d. Install vcpkg packages ---
    Write-Host ""
    Write-Host "Installing vcpkg dependencies (this may take a while on first run)..."
    & "$vcpkgRoot\vcpkg.exe" install `
        --triplet x64-windows `
        --x-manifest-root="$RepoRoot" `
        --x-install-root="$VcpkgInstalledDir" `
        --overlay-ports="$RepoRoot\vcpkg-overlay\ports"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "vcpkg install failed"
        exit 1
    }
    Write-Host "vcpkg dependencies installed." -ForegroundColor Green

    # --- 0e. Set CMake/vcpkg environment variables ---
    $toolchainFile = "$vcpkgRoot\scripts\buildsystems\vcpkg.cmake"
    $env:CMAKE_TOOLCHAIN_FILE = $toolchainFile
    $env:VCPKG_TARGET_TRIPLET = "x64-windows"
    $env:VCPKG_INSTALLED_DIR = $VcpkgInstalledDir
    $env:CMAKE_PREFIX_PATH = "$VcpkgInstalledDir\x64-windows"
    $env:PATH = "$VcpkgInstalledDir\x64-windows\bin;$VcpkgInstalledDir\x64-windows\debug\bin;$env:PATH"

    # --- 0f. uv (Python package manager) ---
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "uv not found. Installing..." -ForegroundColor Magenta
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Error "uv installation failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        }
    }
    Write-Host "Using uv: $(Get-Command uv | Select-Object -ExpandProperty Source)  ($(uv --version))" -ForegroundColor Green

} else {
    Write-Host "=== Skipping prerequisites (reusing previous environment) ===" -ForegroundColor DarkGray

    # Even when skipping, we need the environment set up
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = & $vswhere -latest -products * -property installationPath 2>$null
    $candidates = @(
        "$vsPath\VC\Tools\Llvm\x64\bin\clang-cl.exe",
        "$vsPath\VC\Tools\Llvm\bin\clang-cl.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $clangDir = Split-Path $c
            $env:PATH = "$clangDir;$env:PATH"
            break
        }
    }
    Assert-Command "clang-cl"

    # Set up MSVC env
    $vcvarsall = "$vsPath\VC\Auxiliary\Build\vcvarsall.bat"
    if (Test-Path $vcvarsall) {
        $tempFile = [System.IO.Path]::GetTempFileName()
        cmd /c "`"$vcvarsall`" x64 && set > `"$tempFile`""
        Get-Content $tempFile | ForEach-Object {
            if ($_ -match "^([^=]+)=(.*)$") {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
        Remove-Item $tempFile
        # Re-add clang-cl to PATH
        $env:PATH = "$clangDir;$env:PATH"
    }

    # vcpkg env - check VS-bundled vcpkg first
    $vcpkgRoot = $null
    $vsVcpkg = "$vsPath\VC\vcpkg\vcpkg.exe"
    if (Test-Path $vsVcpkg) {
        $vcpkgRoot = Split-Path $vsVcpkg
    } elseif ($env:VCPKG_INSTALLATION_ROOT -and (Test-Path "$($env:VCPKG_INSTALLATION_ROOT)\vcpkg.exe")) {
        $vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
    } else {
        $vcpkgRoot = "$RepoRoot\vcpkg-tool"
    }
    $env:CMAKE_TOOLCHAIN_FILE = "$vcpkgRoot\scripts\buildsystems\vcpkg.cmake"
    $env:VCPKG_TARGET_TRIPLET = "x64-windows"
    $env:VCPKG_INSTALLED_DIR = $VcpkgInstalledDir
    $env:CMAKE_PREFIX_PATH = "$VcpkgInstalledDir\x64-windows"
    $env:PATH = "$VcpkgInstalledDir\x64-windows\bin;$VcpkgInstalledDir\x64-windows\debug\bin;$env:PATH"
}

# Verify tools
Write-Host ""
Write-Host "=== Environment summary ===" -ForegroundColor Yellow
Write-Host "  clang-cl:        $(Get-Command clang-cl | Select-Object -ExpandProperty Source)"
Write-Host "  cmake:           $(Get-Command cmake | Select-Object -ExpandProperty Source)"
Write-Host "  ninja:           $(Get-Command ninja | Select-Object -ExpandProperty Source)"
Write-Host "  python:          $(Get-Command python | Select-Object -ExpandProperty Source)  ($(python --version 2>&1))"
Write-Host "  uv:              $(Get-Command uv | Select-Object -ExpandProperty Source)  ($(uv --version 2>&1))"
Write-Host "  TOOLCHAIN_FILE:  $env:CMAKE_TOOLCHAIN_FILE"
Write-Host "  VCPKG_TRIPLET:   $env:VCPKG_TARGET_TRIPLET"
Write-Host "  VCPKG_INSTALLED: $env:VCPKG_INSTALLED_DIR"
Write-Host ""

# ==========================================================================
# STEP 1 - Build and install Python package (C++ is built by scikit-build-core)
# ==========================================================================
if (-not $SkipBuild) {
    Write-Host "=== Step 1: Build and install Python package ===" -ForegroundColor Yellow
    Push-Location "$RepoRoot\python"

    # Set QDK_DLL_DIR so the qdk_chemistry package can find vcpkg DLL
    # dependencies (openblas.dll, hdf5.dll, etc.) at import time.
    $env:QDK_DLL_DIR = "$VcpkgInstalledDir\x64-windows\bin"
    Write-Host "QDK_DLL_DIR: $env:QDK_DLL_DIR" -ForegroundColor Blue

    if (-not (Test-Path .\venv)) {
        uv venv .\venv
    }
    .\venv\Scripts\activate

    $env:CMAKE_BUILD_PARALLEL_LEVEL = "6"
    $env:CMAKE_C_FLAGS="/Os"
    $env:CMAKE_CXX_FLAGS="/Os"

    # pip drives the full build: scikit-build-core invokes CMake to compile the
    # C++ library and pybind11 bindings, then packages everything into a wheel.
    # Do not install:
    # - plugins: pyscf does not build on Windows
    # - jupyter: requires plugins
    uv pip install -v .[coverage,dev,docs,qiskit-extras,openfermion-extras] `
        -C cmake.args=-GNinja `
        -C cmake.define.QDK_UARCH="$QDK_UARCH" `
        -C cmake.define.QDK_CHEMISTRY_ENABLE_MPI=OFF `
        -C cmake.define.QDK_ENABLE_OPENMP=OFF `
        -C cmake.define.QDK_CHEMISTRY_ENABLE_COVERAGE=OFF `
        -C cmake.define.BUILD_SHARED_LIBS=OFF `
        -C cmake.define.BUILD_TESTING=OFF `
        -C cmake.define.CMAKE_BUILD_TYPE=Release `
        -C cmake.define.CMAKE_C_COMPILER=clang-cl `
        -C cmake.define.CMAKE_CXX_COMPILER=clang-cl `
        -C cmake.define.CMAKE_TOOLCHAIN_FILE="$env:CMAKE_TOOLCHAIN_FILE" `
        -C cmake.define.VCPKG_TARGET_TRIPLET="$env:VCPKG_TARGET_TRIPLET" `
        -C cmake.define.VCPKG_INSTALLED_DIR="$env:VCPKG_INSTALLED_DIR"
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Error "Python package install failed"; exit 1 }

    python -c "import qdk_chemistry; print('qdk_chemistry version:', qdk_chemistry.__version__)"
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Error "Python import check failed"; exit 1 }
    Write-Host "Python package installed successfully." -ForegroundColor Green

    Pop-Location
} else {
    Write-Host "=== Skipping build ===" -ForegroundColor DarkGray
}

# ==========================================================================
# STEP 2 - Run Python tests
# ==========================================================================
if (-not $SkipTests) {
    Write-Host ""
    Write-Host "=== Step 2: Run Python tests ===" -ForegroundColor Yellow
    Push-Location "$RepoRoot\python"

    # Ensure venv is activated
    if (Test-Path .\venv\Scripts\activate.ps1) {
        .\venv\Scripts\activate
    }

    pytest -v --tb=short
    $pytestExit = $LASTEXITCODE
    Pop-Location
    if ($pytestExit -ne 0) {
        Write-Warning "Some Python tests failed (exit code: $pytestExit)"
    } else {
        Write-Host "All Python tests passed." -ForegroundColor Green
    }
} else {
    Write-Host "=== Skipping tests ===" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Build script finished!                    " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
