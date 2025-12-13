# Installation Instructions for QDK/Chemistry

This file contains instructions for how to configure, build and install QDK/Chemistry via
several common methods.

## Pip Wheel Installation

We are currently in the process of preparing pre-built wheels for QDK/Chemistry.
Once these are available, the preferred installation method will be via `pip`.
For now, please build the package from source using the instructions below.

## Using the VSCode Dev Container

The easiest way to get started with QDK/Chemistry development is to use the provided VS Code Dev Container. The container definition and VSCode files provided in the Git repository will build a development environment (Docker container with Python virtual environment) from scratch.

To get started:

1. Open the repository folder in VS Code
2. When prompted, click "Reopen in Container" (or use the Command Palette: `Ctrl+Shift+P` / `Cmd+Shift+P` and select "Dev Containers: Reopen in Container")
3. VS Code will build and start the development container automatically

Alternatively, you can click the green button in the bottom-left corner of VS Code and select "Reopen in Container" from the menu.

**NOTES:**

- Building the development environment can take some time, on slower systems this can be up to two hours.
- Using a VS Code Dev Container may require elevated permissions on the current system to allow for Docker usage.
- You will likely need to restart VS Code and the Dev Container (reopening in the container) before running examples.  The restart will load the Python virtual environment into the continer.

## Building from Source

### Operating System Specific Notes

**Linux**: We would recommend a Debian distribution due to the broad number of scientific computing packages available through apt. You may need to build certain dependencies from source (e.g. Eigen3, nlohmann-json) if using a distribution with more limited package availability.

**Windows**: We do not currently provide native Windows support for building qdk-chemistry. If building the package on Windows, you will need to use the Windows Subsystem for Linux (WSL). Microsoft's [official installation instructions for WSL](https://learn.microsoft.com/en-us/windows/wsl/install) can guide you through the process of setting up WSL if you do not have it installed.

**macOS**: To build qdk-chemistry on macOS, you will need the latest version of Xcode installed. If you do not already have it installed, it can be downloaded from the [App Store](https://apps.apple.com/us/app/xcode/id497799835?mt=12).

### Building the Python Package

The easiest way to build and install the QDK/Chemistry python package from source is via `pip`.

**NOTE:** If the C++ library has not been installed, these instructions will automatically build the C++ library as a part of the python package build. For developers, it is strongly encouraged that you build and install the C++ library first to avoid long build times. See [these instructions](#linking-to-an-existing-c-installation) for details on how to link the python package to an existing C++ installation.

1. Ensure all system dependencies are installed. See [System Dependencies](#system-dependencies) for details.
2. (Optional) Set any desired environment variables to control the C++ build. See [Environment Variables for the Python Build](#environment-variables-for-the-python-build) for details.
3. From the root of the QDK/Chemistry repository, run the following commands:

```bash
cd qdk-chemistry/python
pip install .
pytest tests/
```

**NOTE:** Building this Python package may require significant memory, since the C++ library build uses all available threads by default and some compilations can consume around 3 GB of RAM. To avoid running out of memory, set `CMAKE_BUILD_PARALLEL_LEVEL` to a reasonably small value. For example, use: `CMAKE_BUILD_PARALLEL_LEVEL=1 pip install .` to perform a single-threaded C++ library build.

#### Accelerating Rebuilds with Build Caching

By default, each `pip install` uses a fresh temporary build directory to ensure reproducible builds and avoid issues with stale CMake cache state. However, for development workflows where you're making frequent changes, you can enable persistent build caching for significantly faster rebuilds:

```bash
pip install . -C build-dir="build/{wheel_tag}"
```

**Warning:** When using a persistent build directory, CMake caches configuration decisions (such as whether the C++ library was found pre-installed or built from source). If your environment changes (e.g., you add or remove a pre-installed C++ library, or C++ dependencies change), the cached state may cause subtle build failures. In this case, remove the build directory and try again:

```bash
rm -rf build/
pip install .
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

The official way to notify the python package build of an existing QDK/Chemistry C++ installation is to append the `CMAKE_PREFIX_PATH` environment variable with the installation prefix: e.g. `CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/full/qdk/chemistry/prefix"`. See the [CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html) for a discussing surrounding the use of environment variables for prefix paths.

#### Note on `QDK_UARCH` specification

Specification of the instruction set architecture (ISA) is highly compiler specific and requires careful examination of the compiler documentation to ensure appropriate usage. Here we provide a number of common possibilities for the GNU family of compilers. These may or may not work on your machine depending on your processor's ISA and the compiler you are using.

| `QDK_UARCH` | Description |
|--------------|-----|
|`native` | Generates code for your native ISA. This will likely result in binaries which are not portable to other systems. Use with caution |
|`x86-64-v3` | AMD64: x86_64 + AVX2 + FMA. Applicable to most modern x86_64 processors |
|`armv8-a` | AARCH64: 64-bit ARM. Applicable to Apple Silicon and Microsoft Surface ARM architectures |

### Building the C++ Library

With all system dependencies installed, the C++ QDK/Chemistry
library may be build and installed via

```bash
cd [/full/path/to/qdk-chemistry]
cmake -S cpp -B cpp/build -DQDK_UARCH="x86-64-v3" [CMake Options]  # Adjust -DQDK_UARCH based on your target architecture if necessary
cmake --build cpp/build
[cmake --build cpp/build --target test] # Optional but encouraged, tests the C++ library if testing is enabled
[cmake --install cpp/build] # Optional, installs to CMAKE_INSTALL_PREFIX
```

#### Configuring the C++ Library

The following table contains information pertaining to influential CMake configuration variables for QDK/Chemistry. They may be appended by replacing `[CMake Options]` in the above CMake invocation using the syntax

```bash
cmake [...] -D<VARIABLE>=<VALUE>
```

Where possible, the official CMake documentation is linked for further information.

| Variable | Description | Type | Default | Other Values |
|----------|-------------|------|---------|--------------|
|[`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html) | The optimization level for the C++ build. | String | `Release` | `Debug`, `RelWithDebInfo`|
|[`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)| The desired installation prefix | String | `/usr/local` | User defined |
|[`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)| Location of installed dependencies | [List](#note-on-cmake-lists-from-the-command-line) | N/A | User defined |
|[`CMAKE_CXX_FLAGS`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html#variable:CMAKE_%3CLANG%3E_FLAGS) | Space-delimited set of C++ compilation flags to append to the C++ compilation | String | N/A | User defined |
|[`BUILD_TESTING`](https://cmake.org/cmake/help/latest/variable/BUILD_TESTING.html) | Whether to build unit and integration tests | Bool | `True` | `False` |
|`QDK_UARCH`| The instruction set architecture (ISA) to compile for. This is not a mandatory setting, but it is strongly encouraged for good performance | String | N/A | [See below](#note-on-qdk_uarch-specification) |
|`QDK_CHEMISTRY_ENABLE_COVERAGE` | Enable coverage reports | Bool | `True` | `False` |
|`QDK_CHEMISTRY_ENABLE_LONG_TESTS` | Enable long running tests (useful on HPC architectures) | Bool | `False` | `True` |

#### Note on CMake Lists from the Command Line

Lists in CMake are stored as semicolon delimited strings. On the command line, List variables must be contained in double quotes to ensure proper execution. This is most commonly encountered when specifying `CMAKE_PREFIX_PATH` when dependencies are installed to multiple prefixes. For example, if one has OpenBLAS installed in `/opt/openblas` and HDF5 installed in `/opt/hdf5`, the proper specification for `CMAKE_PREFIX_PATH` would be:

```bash
cmake [...] -DCMAKE_PREFIX_PATH="/opt/openblas;/opt/hdf5"
```

## Dependencies

**Disclaimer**: The list of dependencies listed in this file denotes the *direct* software dependencies of QDK/Chemistry. Each of these dependencies may come with dependencies of their own. The Component Governance Manifests for the [C++](cpp/manifest/qdk-chemistry/cgmanifest.json) and [Python](python/manifest/cgmanifest.json) libraries keep track of our current understanding of the complete dependency graph of QDK/Chemistry, but are subject to inaccuracies given changes in upstream dependencies. Please refer to the documentation of linked dependencies for more information of their respective dependency trees.

```txt
TODO (DBWY): Flesh out DNF Installation instructions
Work Item: 41432
```

### System Dependencies

This section details the QDK/Chemistry dependencies which must be installed prior to
starting from-source builds, as they are not managed by the build system. See [Managed Dependencies](#managed-dependencies) for a discussion on the dependencies which are managed by the C++ build system. See the [C++ configuration section](#configuring-the-c-library) for
instructions on how to notify the build system where dependencies have been installed.

QDK/Chemistry requires both a C and a C++ compiler to be installed. Additionally, the C++ compiler must support the ISO C++20 standard. See [this website](https://en.cppreference.com/w/cpp/compiler_support/20) to determine if your compiler admits appropriate C++20 support. Below is a table of the compilers and versions tested for the full-stack QDK/Chemistry build.

| Compiler Family | Versions |
|-----------------|----------|
| GNU  | 11.0 + |
| Clang | `TODO (DBWY)` |
| MCVC | `TODO (DBWY)` |

```txt
TODO (DBWY): Add compiler support documentation
Work Item: 41432
```

Additionally, QDK/Chemistry requires the following software dependencies:

| Dependency | Description | Requirements | Source Location | Ubuntu / Debian | Redhat |
|------------|-------------|--------------------|-----------------|-----------------|---------|
| CMake | Build system manager | Version > 3.15 | [source](https://github.com/Kitware/CMake) | `apt install cmake` | `dnf install cmake` |
| Eigen | C++ linear algebra templates | Version > 3.4.0 | [source](https://eigen.tuxfamily.org/index.php?title=Main_Pag) | `apt install libeigen3-dev` | `dnf install eigen3-devel` |
| LAPACK | C library for linear algebra. See [this note](#note-on-lapack-usage) for further information | N/A | e.g. [source](https://github.com/OpenMathLib/OpenBLAS) | e.g. `apt install libopenblas-dev` | e.g. `dnf install TODO(DBWY)`|
| HDF5 | A portable data file library | Version > 1.12 + C++ bindings | [source](https://www.hdfgroup.org/download-hdf5/) | `apt install libhdf5-serial-dev` | `dnf install TODO(DBWY)`|
| Boost | A collection of useful C++ libraries | Version > 1.80 | [source](https://github.com/boostorg/wiki/wiki/Getting-Started%3A-Overview) | `apt install libboost-all-dev` | `dnf install TODO (DBWY)` |

See [Python dependencies](#python-dependencies) for a list of dependencies installed by `pip`.

### Managed Dependencies

Other than the system dependencies outlined [above](#system-dependencies), QDK/Chemistry has a number of other dependencies which, if left unspecified, will be automatically handled by the build system
when compiling the top-level C++ library. For active developers, it is strongly suggested that one has these dependencies pre-installed to lower build times. As with the system
dependencies, see the [C++ build instructions](#configuring-the-c-library) for guidance on how to notify the build system of install locations for locally built dependencies.

| Dependency | Description | Tested Versions | Source Location | Ubuntu / Debian | Redhat |
|------------|-------------|--------------------|-----------------|-----------------|---------|
| nlohmann/json | A C++ library for JSON manipulation | v3.12.0 | [source](https://github.com/nlohmann/json) | `apt install nlohmann-json3-dev` | `dnf install TODO(DBWY)` |
| Libint2 | A C++ library for molecular integral evaluation | v2.9.0 | [source](https://github.com/evaleev/libint) | N/A | N/A |
| Libecpint | A C++ library for molecular integrals involving [effective core potentials](https://en.wikipedia.org/wiki/Pseudopotential) | v1.0.7 | [source](https://github.com/robashaw/libecpint) | `apt install libecpint-dev` | `dnf install TODO (DBWY)`|
| GauXC | A C++ library for molecular integrals on numerical grids | v1.0 | [source](https://github.com/wavefunction91/gauxc) | N/A | N/A |
| MACIS | A C++ library for configuration interaction methods | N/A | [source](https://github.com/wavefunction91/macis) | N/A | N/A |

**NOTE**: As Libint and GauXC exhibit very long build times, it is **strongly encouraged** that these dependencies are separately installed to avoid excessive build costs. Example CMake invocations for these libraries may be found in [install-libint2.sh](.pipelines/install-scripts/install-libint2.sh) and [install-gauxc.sh](.pipelines/install-scripts/install-gauxc.sh), respectively.

**NOTE**: The source code of MACIS is included in the `external` directory of QDK/Chemistry. `MACIS` carries it's own set of dependencies which are automatically managed by the `MACIS` build system. While building `MACIS` and its dependencies can be time consuming, it is strongly encouraged to allow the QDK/Chemistry build system handle this dependency to ensure proper interaction of up- and down-stream components.

### Note on LAPACK Usage

BLAS (Basic Linear Algebra Subroutines) and LAPACK (Linear Algebra Package) are API standards for libraries implementing linear algebra operations such as matrix multiplication and matrix decomposition. These operations are compute intensive and require careful optimization on modern architectures to achieve optimal performance. As such, we require users have a LAPACK (and transitively BLAS) installation in their environment rather than providing stock implementations. Below are a list of commonly used LAPACK libraries that are regularly tested with QDK/Chemistry.

| Library | Description | Installation Instructions |
|---------|-------------|-------------------|
| Intel MKL | A highly optimized BLAS/LAPACK library targeting Intel CPUs | [Intel Documentation](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html) |
| AMD AOCL | A highly optimized BLAS/LAPACK library targeting AMD CPUs | [AMD Documentation](https://github.com/amd/aocl) |
| OpenBLAS | A high performance, cross-platform, open-source BLAS/LAPACK library | [OpenBLAS Documentation](https://github.com/OpenMathLib/OpenBLAS) |
| BLIS / FLAME | A set of high performance, cross-platform, open source BLAS (BLIS) and LAPACK (FLAME) libraries. BLIS may also be combined with NETLIB-LAPACK to provide LAPACK functionality | [BLIS](https://github.com/flame/blis) and [FLAME](https://github.com/flame/libflame) Documentation |
| NETLIB | Reference implementation of the BLAS / LAPACK standards. Generic but sub-optimal | [NETLIB Documentation](https://netlib.org/) |
