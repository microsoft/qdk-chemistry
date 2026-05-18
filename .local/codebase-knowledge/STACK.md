# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary languages | C++20 (core) + Python 3.10+ (API/bindings) | `cpp/CMakeLists.txt`, `python/pyproject.toml` |
| Current version | 2.0.0 | `VERSION` |
| Python package manager | pip (with scikit-build-core backend) | `python/pyproject.toml` |
| C++ build system | CMake ≥3.15 | `cpp/CMakeLists.txt`, `python/pyproject.toml` |
| Python build backend | scikit-build-core ≥0.10.0 | `python/pyproject.toml` |
| Tested compilers | GNU 13+, AppleClang 17+ | `INSTALL.md` |

### 2) Production Frameworks and Dependencies

**Python runtime dependencies:**

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| numpy | ≥2.0.0, <2.4.0 | Numerical arrays, integral data | `python/pyproject.toml` |
| scipy | ≥1.7.0 | Scientific computing utilities | `python/pyproject.toml` |
| matplotlib | ≥3.10.0 | Visualization | `python/pyproject.toml` |
| ruamel-yaml | ≥0.18.10 | YAML config handling | `python/pyproject.toml` |
| qdk[jupyter] | ≥1.27.0 | Azure Quantum SDK integration | `python/pyproject.toml` |
| h5py | ≥3.0.0 | HDF5 file I/O | `python/pyproject.toml` |
| pybind11 | ≥2.13.6 | C++/Python bindings (build+runtime stubs) | `python/pyproject.toml` |
| pybind11-stubgen | ≥2.5.1 | Auto-generate `.pyi` type stubs | `python/pyproject.toml` |

**C++ system dependencies:**

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| Eigen3 | ≥3.4.0 | Linear algebra | `cpp/cmake/third_party.cmake`, `INSTALL.md` |
| HDF5 (C++) | ≥1.12 | Serialization | `cpp/cmake/third_party.cmake`, `INSTALL.md` |
| Boost | ≥1.80 | Misc utilities | `INSTALL.md` |
| LAPACK/BLAS | — | Dense linear algebra backend | `INSTALL.md` |
| OpenMP | — | Shared-memory parallelism (enabled by default) | `cpp/CMakeLists.txt` |

**C++ fetched dependencies (managed via CMake FetchContent):**

| Dependency | Version/Tag | Role in system | Evidence |
|------------|-------------|----------------|----------|
| nlohmann/json | v3.12.0 | JSON serialization | `cpp/cmake/third_party.cmake` |
| Libint2 | v2.9.0 | Electron repulsion integrals | `cpp/cmake/third_party.cmake` |
| Libecpint | v1.0.7 | Effective core potential integrals | `cpp/cmake/third_party.cmake` |
| GauXC | commit 62fea07 | Exchange-correlation grid integration | `cpp/cmake/third_party.cmake` |
| MACIS | vendored | Many-body CI (FCI/CAS/sCI) | `external/macis/` |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| Ruff | Python lint + format (line-length=120, double quotes) | `python/pyproject.toml`, `.pre-commit-config.yaml` |
| mypy | Python type checking | `python/pyproject.toml`, `.pre-commit-config.yaml` |
| clang-format | C++ formatting (Google style, 80-col) | `.clang-format`, `.pre-commit-config.yaml` |
| interrogate | Docstring coverage (≥80%) | `.pre-commit-config.yaml` |
| sphinx-lint | RST documentation linting | `.pre-commit-config.yaml` |
| markdownlint | Markdown linting | `.pre-commit-config.yaml` |
| gitleaks | Secret detection | `.pre-commit-config.yaml` |
| pytest | Python test runner | `python/pyproject.toml` |
| GoogleTest | C++ test framework | `cpp/tests/CMakeLists.txt` |
| gcovr | C++ coverage reporting | `.github/workflows/build-and-test.yaml` |
| pytest-cov | Python coverage reporting | `python/pyproject.toml` |
| Sphinx + Breathe + Doxygen | Documentation generation | `docs/source/conf.py`, `docs/Doxyfile` |

### 4) Key Commands

```bash
# Install (from PyPI)
python3 -m pip install 'qdk-chemistry[all]'

# Install (from source)
cd python && python3 -m pip install '.[all]'

# C++ build
cmake -B build -S cpp -DBUILD_TESTING=ON && cmake --build build

# C++ tests
cd build && ctest

# Python tests
pytest python/tests/ -ra -v

# Lint (pre-commit)
pre-commit run --all-files
```

### 5) Container and Dev Environment

- Dev Container base image: `mcr.microsoft.com/mirror/docker/library/ubuntu:24.04` (`.devcontainer/Dockerfile:1`)
- No production Docker image — this is a library distributed via PyPI, not a deployable service
- Dev Container configures VS Code extensions (Python, Pylance, Ruff, C/C++, CMake Tools, Jupyter)

### 6) Environment and Config

- Config sources: `python/pyproject.toml`, `cpp/CMakeLists.txt`, `.pre-commit-config.yaml`, `.clang-format`
- Required env vars: `QSHARP_PYTHON_TELEMETRY` (optional — disables telemetry when set to `false`/`disabled`/`none`/`0`)
- Dev container sets: `CPATH` for HDF5/Eigen include paths
- Deployment: Pure library — no server, no runtime config files required. Distributed via PyPI as `qdk-chemistry`.

### 7) Evidence

- `python/pyproject.toml`
- `cpp/CMakeLists.txt`
- `cpp/cmake/third_party.cmake`
- `.pre-commit-config.yaml`
- `.clang-format`
- `INSTALL.md`
- `VERSION`
- `.devcontainer/devcontainer.json`
