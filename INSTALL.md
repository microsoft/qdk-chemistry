# Installation Instructions for QDK/Chemistry

## Choose Your Installation Path

QDK/Chemistry can be installed in three ways:

| Goal | Method | Time |
|------|--------|------|
| **Use QDK/Chemistry in your own project** | [Install from PyPI](#install-from-pypi) | ~2 minutes |
| **Develop or contribute to QDK/Chemistry** | [VS Code Dev Container](#using-the-vscode-dev-container) | ~30-120 min (one-time build) |
| **Build everything from source** | [Build from Source](#building-from-source) | ~30-60 min |

Most users should start with the PyPI install.

---

## Install from PyPI

### Prerequisites

- Python 3.10 or newer
- pip (on Ubuntu/Debian you may need `sudo apt install python3-pip python3-venv`)

Prebuilt wheels are published for the following platforms:

| Platform | Architecture | Notes |
|----------|--------------|-------|
| Linux | x86_64, arm64 | |
| macOS | arm64 (Apple Silicon) | |
| Windows | x86_64 | See [Windows notes](#notes-for-windows-users) |
| Windows | arm64 | No wheel; use WSL or [build from source](#building-from-source) |

On Windows you can either install natively or work inside the
[Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/windows/wsl/install). Both are
supported: WSL uses the Linux wheels and the Linux instructions throughout this document, and is the
simplest option on arm64 or when you need PySCF.

> **NOTE:** Commands below are given for Linux/macOS (`bash`) and Windows (PowerShell). Where the
> Linux/macOS commands use `python3`, use `python` on Windows.

### Step 1: Create a virtual environment

Use a virtual environment to avoid conflicts with other packages:

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **NOTE:** If PowerShell refuses to run the activation script, activate from `cmd.exe` with
> `venv\Scripts\activate.bat` instead. See the Microsoft
> [execution policy documentation](https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_execution_policies)
> before changing your execution policy.

### Step 2: Install the package

For most users, **`[all]` is the recommended install target**. It pulls in all optional dependencies so that examples and tests work without chasing missing packages:

**Linux / macOS:**

```bash
python3 -m pip install 'qdk-chemistry[all]'
```

**Windows (PowerShell):**

```powershell
python -m pip install "qdk-chemistry[all]"
```

> **Tip:** `[all]` is the path of least resistance if you're just getting started.
> You can always switch to a slimmer install later.

If you prefer a minimal install (core library only, no optional backends):

```bash
python3 -m pip install qdk-chemistry
```

> **NOTE:** The `all` and `qiskit-extras` extras are not supported on Python 3.14 because Qiskit does not yet publish Python 3.14 wheels. See the [Optional Extras](#optional-extras) table below for details and alternative install targets.
>
> **NOTE:** PySCF is the only package in the `plugins` extra and publishes no Windows wheels, so it is skipped on native Windows. See [Notes for Windows users](#notes-for-windows-users).

### Step 3: Verify the installation

```bash
python3 -c "import qdk_chemistry; print(qdk_chemistry.__version__)"
```

### Step 4: Clone the repository (for examples and tests)

The examples and test suite live in the source repository. Clone it and check out the branch that matches your installed version:

```bash
pip show qdk-chemistry          # check your installed version
git clone https://github.com/microsoft/qdk-chemistry.git
cd qdk-chemistry
git checkout stable/1.1          # match major.minor to your version
```

> **NOTE:** The `main` branch is the active development branch and may be incompatible with the released pip package. Always use the `stable/major.minor` branch for examples.

Some examples require additional packages not included in any extra. See the [examples README](examples/README.md) for per-example requirements.

### Optional Extras

If you chose the minimal `pip install qdk-chemistry` above, you can add specific extras as needed:

| Extra | Description | Included Packages |
|-------|-------------|-------------------|
| `jupyter` | Jupyter notebook support | ipykernel, pandas |
| `plugins` | Third-party quantum chemistry backends | PySCF (no Windows wheels) |
| `qiskit-extras` | Qiskit ecosystem packages | qiskit, qiskit-aer, qiskit-nature |
| `openfermion-extras` | OpenFermion ecosystem packages | openfermion |
| `networkx-extras` | NetworkX ecosystem packages | networkx |
| `dev` | Development and testing tools | pytest, ruff, mypy, and related tooling |
| `test` | Testing tools and optional dependencies | pytest, ipykernel, networkx, openfermion, pennylane, pyscf, qiskit, rdkit |
| `all` | **All of the above** | All optional dependencies |

Install one or more extras with:

**Linux / macOS:**

```bash
python3 -m pip install 'qdk-chemistry[plugins,dev]'
```

**Windows (PowerShell):**

```powershell
python -m pip install "qdk-chemistry[plugins,dev]"
```

> **NOTE:** In PowerShell, use double quotes around targets containing square brackets.
> Single quotes also work in PowerShell but not in `cmd.exe`, where no quoting is required at all.

Installing with the `dev` extra lets you run the test suite (you need to clone the repository first; see [Step 4](#step-4-clone-the-repository-for-examples-and-tests)):

```bash
pytest python/tests
```

### Notes for Windows Users

Windows x86_64 is a supported platform: wheels are published for Python 3.10-3.14 and the project is
built and tested on Windows in CI. The following caveats apply to native Windows installs; none of
them apply under [WSL](https://learn.microsoft.com/windows/wsl/install).

| Topic | Detail |
|-------|--------|
| Architecture | Wheels are published for x86_64 only. On arm64, use WSL or [build from source](#building-from-source). |
| PySCF plugin | PySCF publishes no Windows wheels, so the `plugins` extra installs no PySCF and the PySCF plugin is unavailable. The native implementations are unaffected. |
| OpenMP | Shared-memory threading via OpenMP is disabled on Windows. |

---

## Using the VSCode Dev Container

The VS Code Dev Container gives you a ready-made development environment. It builds a Docker container with all C++ and Python dependencies pre-installed.

### Step 1: Clone the repository

```bash
git clone https://github.com/microsoft/qdk-chemistry.git
```

### Step 2: Open in VS Code and reopen in the container

1. Open the `qdk-chemistry` folder in VS Code
2. When prompted, click **"Reopen in Container"** (or use the Command Palette: `Ctrl+Shift+P` / `Cmd+Shift+P` → "Dev Containers: Reopen in Container")
3. VS Code will build and start the development container

Alternatively, click the green button in the bottom-left corner of VS Code and select "Reopen in Container".

### Step 3: Restart VS Code

After the initial build, restart VS Code and reopen in the container to ensure the Python virtual environment is properly loaded.

**NOTE:**

- The first build can take up to two hours on slower systems.
- Docker must be available on your system (may require elevated permissions).
- Subsequent launches reuse the built container and are fast.

---

## Dependencies (for Source Builds)

> **NOTE:** If you are installing from PyPI, skip this section. pip handles all dependencies automatically.

**Disclaimer**: The list of dependencies listed here denotes the *direct* software dependencies of QDK/Chemistry. Each may have dependencies of their own. The Component Governance Manifests for the [C++](cpp/manifest/qdk-chemistry/cgmanifest.json) and [Python](python/manifest/cgmanifest.json) libraries track the full dependency graph. Please refer to linked dependency documentation for their respective dependency trees.

### System Dependencies

These must be installed before starting a from-source build. See [Managed Dependencies](#managed-dependencies) for dependencies that the build system handles automatically.

QDK/Chemistry requires both a C and a C++ compiler supporting the ISO C++20 standard. See [this reference](https://en.cppreference.com/w/cpp/compiler_support/20) to check your compiler's C++20 support.

| Compiler Family | Tested Versions | Platform |
|-----------------|-----------------|----------|
| GNU  | 13+ | Linux |
| AppleClang | 17+ | macOS |
| MSVC | Visual Studio 2022 Build Tools | Windows |
| clang-cl | 17+ | Windows |

**NOTE**: Before installing dependencies on Ubuntu/Debian, update package indices with:

```bash
sudo apt update
```

For Fedora/RHEL systems, update package metadata with:

```bash
sudo dnf makecache
```

On Windows, dependencies come from two places: the compiler, CMake, and Ninja ship with the
Visual Studio 2022 Build Tools, and the C++ libraries come from
[vcpkg](https://learn.microsoft.com/vcpkg/). The Windows column below gives the vcpkg port name for
each library; they are all declared in [`vcpkg.json`](vcpkg.json), so a single
`vcpkg install --overlay-ports=vcpkg-overlay/ports` resolves the whole set.

| Dependency | Description | Requirements | Source Location | Ubuntu / Debian | Redhat | Windows |
|------------|-------------|--------------------|-----------------|-----------------|---------|-----------------|
| Python 3 | Python interpreter and package tools | Version 3.10+ | [source](https://www.python.org/) | `apt install python3 python3-pip python3-venv` | `dnf install python3 python3-pip` | `winget install --id Python.Python.3.12 -e -s winget` |
| CMake | Build system manager | Version > 3.15 | [source](https://github.com/Kitware/CMake) | `apt install cmake` | `dnf install cmake` | VS Build Tools |
| Ninja | Build tool used for the Windows builds | N/A | [source](https://github.com/ninja-build/ninja) | `apt install ninja-build` | `dnf install ninja-build` | VS Build Tools |
| Eigen | C++ linear algebra templates | Version > 3.4.0 | [source](https://libeigen.gitlab.io/) | `apt install libeigen3-dev` | `dnf install eigen3-devel` | `eigen3` |
| LAPACK | C library for linear algebra. See [this note](#note-on-lapack-usage) for further information | N/A | e.g. [source](https://github.com/OpenMathLib/OpenBLAS) | e.g. `apt install libopenblas-dev` | e.g. `dnf install openblas-devel`| `openblas` |
| HDF5 | A portable data file library | Version > 1.12 + C++ bindings | [source](https://www.hdfgroup.org/download-hdf5/) | `apt install libhdf5-serial-dev` | `dnf install hdf5-devel`| `hdf5[cpp]` |
| Boost | A collection of useful C++ libraries | Version > 1.80 | [source](https://github.com/boostorg/wiki/wiki/Getting-Started%3A-Overview) | `apt install libboost-all-dev` | `dnf install boost-devel` | `boost-headers`, `boost-container-hash`, `boost-dynamic-bitset`, `boost-sort` |

See [Python dependencies](#python-dependencies) for a list of dependencies installed by `pip`.

#### Quick install (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv cmake ninja-build libeigen3-dev \
    libopenblas-dev libhdf5-serial-dev libboost-all-dev
```

#### Quick install (Windows)

Install the Visual Studio 2022 Build Tools with the C++ workload, which also provides CMake and
Ninja:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e -s winget `
    --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

`--includeRecommended` is required: the MSVC toolset, CMake, and Ninja are recommended (not
required) components of the `VCTools` workload. This is the
[Microsoft-documented](https://learn.microsoft.com/cpp/overview/acquire-msvc#install-msvc-by-using-winget)
way to script a Build Tools install; the installer itself prompts for elevation.

The remaining C++ dependencies are installed by vcpkg during the build. The
[from-source build scripts](#step-4-build-the-python-package) do this for you.

### Managed Dependencies

These dependencies are automatically downloaded and built by the CMake build system if not found. Pre-installing them is optional but **strongly encouraged** for faster rebuilds. See the [C++ configuration section](#configuring-the-c-library) for how to point the build system at pre-installed locations.

| Dependency | Description | Tested Versions | Source Location | Ubuntu / Debian | Redhat | Windows |
|------------|-------------|--------------------|-----------------|-----------------|---------|-----------------|
| nlohmann/json | A C++ library for JSON manipulation | v3.12.0 | [source](https://github.com/nlohmann/json) | `apt install nlohmann-json3-dev` | `dnf install json-devel` | `nlohmann-json` |
| Libint2 | A C++ library for molecular integral evaluation | v2.9.0 | [source](https://github.com/evaleev/libint) | N/A | N/A | N/A |
| Libecpint | A C++ library for molecular integrals involving [effective core potentials](https://en.wikipedia.org/wiki/Pseudopotential) | v1.0.7 | [source](https://github.com/robashaw/libecpint) | `apt install libecpint-dev` | N/A | N/A |
| GauXC | A C++ library for molecular integrals on numerical grids | v1.0 | [source](https://github.com/wavefunction91/gauxc) | N/A | N/A | N/A |
| MACIS | A C++ library for configuration interaction methods | N/A | [source](https://github.com/wavefunction91/macis) | N/A | N/A | N/A |

**NOTE**: As Libint and GauXC exhibit very long build times, it is **strongly encouraged** that these dependencies are separately installed to avoid excessive build costs. See the [Libint2](https://github.com/evaleev/libint) and [GauXC](https://github.com/wavefunction91/gauxc) project documentation for build instructions.

**NOTE**: The source code of MACIS is included in the `external` directory of QDK/Chemistry. `MACIS` carries its own set of dependencies which are automatically managed by the `MACIS` build system. While building `MACIS` and its dependencies can be time consuming, it is strongly encouraged to allow the QDK/Chemistry build system handle this dependency to ensure proper interaction of up- and down-stream components.

### Note on LAPACK Usage

BLAS (Basic Linear Algebra Subroutines) and LAPACK (Linear Algebra Package) are API standards for libraries implementing linear algebra operations such as matrix multiplication and matrix decomposition. These operations are compute intensive and require careful optimization on modern architectures to achieve optimal performance. As such, we require users have a LAPACK (and transitively BLAS) installation in their environment rather than providing stock implementations. Below are commonly used LAPACK libraries that are regularly tested with QDK/Chemistry.

| Library | Description | Installation Instructions |
|---------|-------------|-------------------|
| Intel MKL | A highly optimized BLAS/LAPACK library targeting Intel CPUs | [Intel Documentation](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html) |
| AMD AOCL | A highly optimized BLAS/LAPACK library targeting AMD CPUs | [AMD Documentation](https://github.com/amd/aocl) |
| OpenBLAS | A high performance, cross-platform, open-source BLAS/LAPACK library | [OpenBLAS Documentation](https://github.com/OpenMathLib/OpenBLAS) |
| BLIS / FLAME | A set of high performance, cross-platform, open source BLAS (BLIS) and LAPACK (FLAME) libraries. BLIS may also be combined with NETLIB-LAPACK to provide LAPACK functionality | [BLIS](https://github.com/flame/blis) and [FLAME](https://github.com/flame/libflame) Documentation |
| NETLIB | Reference implementation of the BLAS / LAPACK standards. Generic but sub-optimal | [NETLIB Documentation](https://netlib.org/) |

---

## Building from Source

Build from source if you need to modify the C++ core, work with unreleased features on `main`, or target a non-standard platform.

### Step 1: Check your platform

**Linux**: A Debian-based distribution is recommended for the broadest package availability. Other distributions may require building some dependencies (e.g. Eigen3, nlohmann-json) from source.

**Windows**: Native builds use MSVC or clang-cl with dependencies supplied by
[vcpkg](https://learn.microsoft.com/vcpkg/). Turnkey build scripts are available under
[`.pipelines/windows/`](.pipelines/windows/) — see [Building on Windows](#building-on-windows).
Source builds work on x86_64 and arm64; [WSL](https://learn.microsoft.com/windows/wsl/install) is
also supported and follows the Linux instructions.

**macOS**: The latest version of [Xcode](https://apps.apple.com/us/app/xcode/id497799835?mt=12) must be installed.

> **NOTE:** QDK/Chemistry requires a **64-bit** platform.

### Step 2: Install system dependencies

All from-source builds require the dependencies listed in the [Dependencies](#dependencies-for-source-builds) section above. Install those before proceeding.

### Step 3: Clone the repository

```bash
git clone https://github.com/microsoft/qdk-chemistry.git
cd qdk-chemistry
```

### Step 4: Build the Python package

The simplest way to build from source is via `pip`. If the C++ library hasn't been separately installed, pip builds it automatically.

> **Tip:** **`[all]` is the recommended install target here too.** It pulls in all optional dependencies so examples and tests work without extra steps. See the [Optional Extras](#optional-extras) table for details and platform-specific exclusions.

**Linux / macOS:**

```bash
cd python
python3 -m pip install '.[all]'
pytest tests/
cd ..
```

**Windows (PowerShell):**

Run from a **Developer PowerShell for VS 2022** so that the compiler is on `PATH`, with the vcpkg
dependencies already installed. The [build scripts](#building-on-windows) do both for you.

```powershell
cd python
python -m pip install ".[all]"
pytest tests/
cd ..
```

**NOTE:** Building this Python package may require significant memory, since the C++ library build uses all available threads by default and some compilations can consume around 3 GB of RAM. To avoid running out of memory, set `CMAKE_BUILD_PARALLEL_LEVEL` to a reasonably small value. For example, `1` performs a single-threaded C++ library build:

```bash
# Linux / macOS
CMAKE_BUILD_PARALLEL_LEVEL=1 python3 -m pip install '.[all]'
```

```powershell
# Windows (PowerShell) - no inline environment variable syntax
$env:CMAKE_BUILD_PARALLEL_LEVEL = 1
python -m pip install ".[all]"
```

> **For active developers:** The pip source build includes a full C++ compilation, which is slow. For faster iteration, [build and install the C++ library separately](#building-the-c-library) first, then [link the Python build to it](#linking-to-an-existing-c-installation). That way `pip install` only rebuilds the pybind11 bindings.

#### Building on Windows

Two scripts under [`.pipelines/windows/`](.pipelines/windows/) perform a complete build — prerequisite
installation, vcpkg dependency resolution, C++ configure/build/test/install, and the Python package
build and test:

| Script | Compiler |
|--------|----------|
| [`windows-build-msvc-cmake.ps1`](.pipelines/windows/windows-build-msvc-cmake.ps1) | MSVC `cl.exe` |
| [`windows-build-clang-cl-cmake.ps1`](.pipelines/windows/windows-build-clang-cl-cmake.ps1) | clang-cl |

Run from the repository root in a normal, non-elevated PowerShell, after installing the
[prerequisites](#quick-install-windows):

```powershell
.\.pipelines\windows\windows-build-msvc-cmake.ps1 -SkipPrereqs
```

Without `-SkipPrereqs` the scripts install any missing prerequisites themselves: they download and
run the Visual Studio Build Tools bootstrapper (which requires administrator rights), clone
[vcpkg](https://github.com/microsoft/vcpkg), and install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). Installing the prerequisites
yourself keeps the build off an elevated shell.

Useful switches:

| Switch | Effect |
|--------|--------|
| `-SkipPrereqs` | Reuse the existing toolchain and vcpkg installation |
| `-SkipConfigure` | Incremental build; skip the CMake configure step |
| `-SkipCpp` / `-SkipPython` | Build only one half of the project |
| `-SkipTests` | Skip the `ctest` and `pytest` runs |
| `-BuildType` | `Release` (default), `RelWithDebInfo`, or `Debug` |
| `-DynamicDeps` | Link dependencies dynamically (`x64-windows`) instead of statically (`x64-windows-static-md`) |

The default `x64-windows-static-md` triplet links the vcpkg dependencies statically while keeping
the dynamic CRT, so no dependency DLLs need to sit alongside the Python extension. `-DynamicDeps`
switches to `x64-windows`, whose DLLs must be discoverable at import time.

Windows builds are pinned to `QDK_UARCH=x86-64-v3`. The
[`.pipelines/toolchains/windows.cmake`](.pipelines/toolchains/windows.cmake) toolchain file is
chainloaded after the vcpkg toolchain to force the dynamic CRT (`/MD`) across subprojects.

#### Accelerating Rebuilds with Build Caching

By default, each `pip install` uses a fresh temporary build directory to ensure reproducible builds and avoid issues with stale CMake cache state. However, for development workflows where you're making frequent changes, you can enable persistent build caching for significantly faster rebuilds:

```bash
python3 -m pip install . -C build-dir="build/{wheel_tag}"
```

**Warning:** When using a persistent build directory, CMake caches configuration decisions (such as whether the C++ library was found pre-installed or built from source). If your environment changes (e.g., you add or remove a pre-installed C++ library, or C++ dependencies change), the cached state may cause subtle build failures. In this case, remove the build directory and try again:

```bash
# Linux / macOS
rm -rf build/
python3 -m pip install .
```

```powershell
# Windows (PowerShell)
Remove-Item -Recurse -Force build
python -m pip install .
```

#### Environment Variables for the Python Build

To control the settings of the internal C++ build in the python package installation, the following environment variables can be set.

|Variable | Description | Possible Values |
|---------|-------------|-----------------|
|`QDK_UARCH` | ISA specification | See [this note](#note-on-qdk_uarch-specification)|
|`CMAKE_BUILD_PARALLEL_LEVEL` | Number of parallel compile jobs | See the [official CMake documentation](https://cmake.org/cmake/help/latest/envvar/CMAKE_BUILD_PARALLEL_LEVEL.html) |
|`CMAKE_BUILD_TYPE`| Build the Release or Debug version of the C++ bindings | See this [table](#configuring-the-c-library) |

#### Python Dependencies

For the most up-to-date list of python dependencies, see [`pyproject.toml`](python/pyproject.toml).

#### Linking to an Existing C++ Installation

If you have already [built and installed](#building-the-c-library) the C++ QDK/Chemistry library, you may link the python package build to your existing installation to avoid rebuilding the C++ library.

The official way to notify the python package build of an existing QDK/Chemistry C++ installation is to append the `CMAKE_PREFIX_PATH` environment variable with the installation prefix. As an environment variable it uses the platform's `PATH` separator — `:` on Linux and macOS, `;` on Windows — unlike the [CMake cache variable](#note-on-cmake-lists-from-the-command-line) of the same name, which is always semicolon-separated. See the [CMake documentation](https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html) for further details.

```bash
# Linux / macOS
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/full/qdk/chemistry/prefix"
```

```powershell
# Windows (PowerShell)
$env:CMAKE_PREFIX_PATH = "$env:CMAKE_PREFIX_PATH;C:\full\qdk\chemistry\prefix"
```

#### Note on `QDK_UARCH` specification

Specification of the instruction set architecture (ISA) is highly compiler specific and requires careful examination of the compiler documentation to ensure appropriate usage. The accepted values differ by compiler family, and may or may not work on your machine depending on your processor's ISA.

**GNU / Clang / AppleClang** — `QDK_UARCH` is passed through as `-march=<value>`:

| `QDK_UARCH` | Description |
|--------------|-----|
|`native` | Generates code for your native ISA. This will likely result in binaries which are not portable to other systems. Use with caution |
|`x86-64-v3` | AMD64: x86_64 + AVX2 + FMA. Applicable to most modern x86_64 processors |
|`armv8-a` | AARCH64: 64-bit ARM. Applicable to Apple Silicon and Microsoft Surface ARM architectures |

**MSVC (`cl.exe`)** — `QDK_UARCH` is passed through as `/arch:<value>`, so the GNU-style values above are *not* valid. MSVC has no generic baseline `/arch:` flag; when `QDK_UARCH` is left unset, the compiler default ISA is used. See the [MSVC `/arch` documentation](https://learn.microsoft.com/cpp/build/reference/arch-x64) for the full list:

| `QDK_UARCH` | Description |
|--------------|-----|
| *(unset)* | Compiler default ISA. Portable, but forgoes newer vector instructions |
|`AVX2` | x86_64 + AVX2 + FMA. Roughly equivalent to `x86-64-v3` |
|`AVX512` | x86_64 + AVX-512. Only for processors that support it |

clang-cl accepts MSVC-style flags, so the MSVC values apply there as well. Values that the compiler rejects are detected during configure and dropped with a warning.

> **NOTE:** The Windows [build scripts](#building-on-windows) and release wheels set
> `QDK_UARCH=x86-64-v3`. Recent clang-cl versions accept this; native `cl.exe` does not, so use
> `AVX2` there.

### Building the C++ Library

With all system dependencies installed, the C++ QDK/Chemistry
library may be built and installed via

**Linux / macOS:**

```bash
cd [/full/path/to/qdk-chemistry]
cmake -S cpp -B cpp/build -DQDK_UARCH="x86-64-v3" [CMake Options]  # Adjust -DQDK_UARCH based on your target architecture if necessary
cmake --build cpp/build
[cmake --build cpp/build --target test] # Optional but encouraged, tests the C++ library if testing is enabled
[cmake --install cpp/build] # Optional, installs to CMAKE_INSTALL_PREFIX
```

**Windows (PowerShell):**

From a Developer PowerShell for VS 2022, with the vcpkg dependencies already installed:

```powershell
cd C:\full\path\to\qdk-chemistry
cmake -S cpp -B cpp/build -GNinja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl `
    -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
    -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=".pipelines\toolchains\windows.cmake" `
    -DVCPKG_TARGET_TRIPLET=x64-windows-static-md
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure   # Optional, tests the C++ library if testing is enabled
cmake --install cpp/build                        # Optional, installs to CMAKE_INSTALL_PREFIX
```

Alternatively, the [Windows build scripts](#building-on-windows) wrap this entire sequence.

#### Configuring the C++ Library

The following table contains information pertaining to influential CMake configuration variables for QDK/Chemistry. They may be appended by replacing `[CMake Options]` in the above CMake invocation using the syntax

```bash
cmake [...] -D<VARIABLE>=<VALUE>
```

Where possible, the official CMake documentation is linked for further information.

| Variable | Description | Type | Default | Other Values |
|----------|-------------|------|---------|--------------|
|[`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html) | The optimization level for the C++ build. | String | `Release` | `Debug`, `RelWithDebInfo`|
|[`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)| The desired installation prefix | String | `/usr/local` (Linux/macOS), `C:\Program Files\<project>` (Windows) | User defined |
|[`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)| Location of installed dependencies | [List](#note-on-cmake-lists-from-the-command-line) | N/A | User defined |
|[`CMAKE_CXX_FLAGS`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html#variable:CMAKE_%3CLANG%3E_FLAGS) | Space-delimited set of C++ compilation flags to append to the C++ compilation | String | N/A | User defined |
|[`CMAKE_TOOLCHAIN_FILE`](https://cmake.org/cmake/help/latest/variable/CMAKE_TOOLCHAIN_FILE.html) | Toolchain file. On Windows, set to the vcpkg toolchain | String | N/A | User defined |
|[`VCPKG_TARGET_TRIPLET`](https://learn.microsoft.com/vcpkg/users/triplets) | Windows only. vcpkg triplet selecting the dependency link mode | String | N/A | `x64-windows-static-md`, `x64-windows` |
|[`BUILD_TESTING`](https://cmake.org/cmake/help/latest/variable/BUILD_TESTING.html) | Whether to build unit and integration tests | Bool | `True` | `False` |
|`QDK_UARCH`| The instruction set architecture (ISA) to compile for. This is not a mandatory setting, but it is strongly encouraged for good performance | String | N/A | [See below](#note-on-qdk_uarch-specification) |
|`QDK_CHEMISTRY_ENABLE_COVERAGE` | Enable coverage reports. Requires a `Debug` or `RelWithDebInfo` build and a GNU/Clang compiler; ignored under MSVC | Bool | `False` | `True` |
|`QDK_CHEMISTRY_ENABLE_LONG_TESTS` | Enable long running tests (useful on HPC architectures) | Bool | `False` | `True` |
|`QDK_ENABLE_OPENMP` | Enable OpenMP support. Off by default on Windows and on macOS with AppleClang | Bool | `True` (`False` on Windows / AppleClang) | `False` |

#### Note on CMake Lists from the Command Line

Lists in CMake are stored as semicolon delimited strings. On the command line, List variables must be contained in double quotes to ensure proper execution. This is most commonly encountered when specifying `CMAKE_PREFIX_PATH` when dependencies are installed to multiple prefixes. For example, if one has OpenBLAS installed in `/opt/openblas` and HDF5 installed in `/opt/hdf5`, the proper specification for `CMAKE_PREFIX_PATH` would be:

```bash
cmake [...] -DCMAKE_PREFIX_PATH="/opt/openblas;/opt/hdf5"
```

This applies on Windows too: the CMake cache variable is always semicolon-delimited, regardless of platform.
