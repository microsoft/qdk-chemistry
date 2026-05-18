# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `cpp/` | C++ core library — algorithms, data structures, utilities | `cpp/CMakeLists.txt` |
| `cpp/include/qdk/chemistry/` | Public C++ headers (algorithms, data, utils) | Directory listing |
| `cpp/src/qdk/chemistry/` | C++ implementations | Directory listing |
| `cpp/tests/` | C++ unit tests (GoogleTest) | `cpp/tests/CMakeLists.txt` |
| `python/` | Python package — bindings, pure-Python algorithms, plugins | `python/pyproject.toml` |
| `python/src/qdk_chemistry/` | Main Python package source | `python/src/qdk_chemistry/__init__.py` |
| `python/src/pybind11/` | pybind11 C++→Python binding wrappers | `python/src/pybind11/module.cpp` |
| `python/tests/` | Python test suite (pytest) | `python/pyproject.toml` |
| `docs/` | Sphinx documentation (RST, Doxygen, Breathe) | `docs/Makefile`, `docs/source/conf.py` |
| `examples/` | Example notebooks, scripts, interop demos | `examples/README.md` |
| `external/macis/` | Vendored MACIS library (many-body CI) | `external/macis/CMakeLists.txt` |
| `.github/` | GitHub workflows, issue templates, skills, copilot instructions | `.github/workflows/` |
| `.pipelines/` | Azure DevOps pipeline configs for wheel building | `.pipelines/python-wheels.yaml` |
| `.devcontainer/` | Dev Container setup (Dockerfile, scripts) | `.devcontainer/devcontainer.json` |

### 2) Entry Points

- Main runtime entry: `python/src/qdk_chemistry/__init__.py` — initializes resources, loads plugins, registers algorithms
- C++ library entry: `python/src/pybind11/module.cpp` — pybind11 module definition (`_core`)
- CLI entry: None — this is a library, not an application
- Example entry points: Jupyter notebooks in `examples/` and Python scripts in `examples/language/`
- C++ test entry: `cpp/tests/qdk_test_main.cpp`

### 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| `cpp/include/qdk/chemistry/data/` | Immutable data classes (Structure, Hamiltonian, Wavefunction, Orbitals, Settings) | Algorithm logic, I/O operations |
| `cpp/include/qdk/chemistry/algorithms/` | Algorithm interfaces (SCF, MCSCF, ActiveSpace, etc.) | Data class definitions |
| `cpp/src/qdk/chemistry/algorithms/microsoft/` | Microsoft-specific algorithm implementations | Public API headers |
| `python/src/qdk_chemistry/algorithms/` | Python algorithm base classes, registry, pure-Python algorithms | Data class definitions |
| `python/src/qdk_chemistry/data/` | Python data wrappers, enums, validation | Algorithm logic |
| `python/src/qdk_chemistry/plugins/` | Third-party library bridges (PySCF, Qiskit, OpenFermion) | Core algorithm logic, data definitions |
| `python/src/qdk_chemistry/utils/` | Shared utilities (telemetry, model Hamiltonians, Pauli helpers) | Business logic, data classes |
| `python/src/pybind11/` | C++ ↔ Python binding code only | Pure-Python logic, algorithm implementations |

### 4) Naming and Organization Rules

- **C++ headers**: `cpp/include/qdk/chemistry/{subsystem}/{name}.hpp` — mirrors namespace `qdk::chemistry::{subsystem}`
- **C++ sources**: `cpp/src/qdk/chemistry/{subsystem}/{name}.cpp` — mirrors header layout
- **C++ tests**: `cpp/tests/test_{feature}.cpp`
- **Python modules**: `python/src/qdk_chemistry/{subsystem}/{name}.py` — snake_case
- **Python tests**: `python/tests/test_{feature}.py`
- **Pybind11 bindings**: `python/src/pybind11/{subsystem}/{name}.cpp` — one binding file per C++ class
- **Directory organization**: Layer-based (data / algorithms / plugins / utils) rather than feature-based

### 5) Evidence

- `cpp/CMakeLists.txt`
- `python/pyproject.toml`
- `python/src/qdk_chemistry/__init__.py`
- `python/src/pybind11/module.cpp`
- Directory listings from scan output
