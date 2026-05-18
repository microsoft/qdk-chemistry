# Architecture

## Core Sections (Required)

### 1) Architectural Style

- Primary style: **Layered** — strict separation of data, algorithms, plugins, and utilities
- Why this classification: The codebase enforces immutable data classes, stateless algorithms configured via Settings, and plugin bridges that register into a central algorithm registry. Each layer has clear ownership boundaries.
- Primary constraints:
  1. Data classes are immutable; all configuration lives in `Settings` objects
  2. Algorithms are stateless — `run()` locks settings before dispatching to `_run_impl()`
  3. Plugins bridge external libraries but must register via the algorithm registry

### 2) System Flow

```text
Structure (molecular geometry)
  → SCF Algorithm (e.g., PyscfScfSolver)
    → Wavefunction + Orbitals
      → Hamiltonian Constructor (integral transformation)
        → Hamiltonian (molecular integrals)
          → Qubit Mapper (fermion→qubit encoding)
            → QubitHamiltonian (Pauli operators)
              → Quantum Algorithm (QPE, state prep, energy estimator)
                → Results (energies, measurements)
```

1. **Molecular setup**: User creates a `Structure` with atomic coordinates and charge/multiplicity (`data/structure.hpp`)
2. **Electronic structure**: An SCF algorithm (native C++ or PySCF plugin) computes a `Wavefunction` containing orbitals and energies
3. **Hamiltonian generation**: A `HamiltonianConstructor` algorithm transforms integrals from the wavefunction into a `Hamiltonian` object
4. **Qubit mapping**: A `QubitMapper` algorithm encodes the fermionic Hamiltonian into a `QubitHamiltonian` (Pauli operators)
5. **Circuit construction**: State preparation and time evolution algorithms build quantum circuits
6. **Execution**: A circuit executor runs the circuit on a simulator backend, returning measurement results

### 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| `data/` (C++ and Python) | Immutable data containers: Structure, Hamiltonian, Wavefunction, Orbitals, Settings, QubitHamiltonian | Algorithm logic, external library calls | `cpp/include/qdk/chemistry/data/`, `python/src/qdk_chemistry/data/` |
| `algorithms/` (C++ and Python) | Algorithm interfaces, base classes, registry, pure-Python algorithm implementations | Data structure definitions, plugin-specific code | `python/src/qdk_chemistry/algorithms/base.py`, `algorithms/registry.py` |
| `plugins/` (Python only) | Bridges to PySCF, Qiskit, OpenFermion — converting data and registering algorithm implementations | Core data/algorithm definitions | `python/src/qdk_chemistry/plugins/` |
| `utils/` | Shared helpers: telemetry, model Hamiltonians, Pauli math, logging | Business logic, data definitions | `python/src/qdk_chemistry/utils/` |
| `pybind11/` | C++ ↔ Python binding layer | Pure-Python logic, algorithm implementations | `python/src/pybind11/` |

### 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| **Algorithm Registry (Factory)** | `python/src/qdk_chemistry/algorithms/registry.py` | Central `create()` / `register()` / `available()` API — decouples algorithm selection from implementation. Plugins register factories at import time. |
| **Settings (Configuration Object)** | `python/src/qdk_chemistry/data/__init__.py`, `cpp/include/qdk/chemistry/data/settings.hpp` | All algorithm configuration lives in a `Settings` dict-like object, declared via `set_default()`. Settings are locked before `run()` to prevent mutation during execution. |
| **Plugin auto-loading** | `python/src/qdk_chemistry/__init__.py:114-129` | On package import, available plugins (PySCF, Qiskit, OpenFermion) are detected and their algorithm factories registered automatically. |
| **AlgorithmRef (Nested Algorithms)** | `python/src/qdk_chemistry/algorithms/base.py:193-217` | Algorithms can reference other algorithms by type+name via `AlgorithmRef` in Settings. `create_from_ref()` resolves references through the registry at runtime. |
| **Template Method (`_run_impl`)** | `python/src/qdk_chemistry/algorithms/base.py:95-110` | `Algorithm.run()` handles settings locking and dispatches to `_run_impl()` — subclasses override only the implementation method. |
| **Immutable Data Classes** | `cpp/include/qdk/chemistry/data/data_class.hpp` | Data objects are constructed once and not mutated, ensuring thread safety and reproducibility. |

### 5) Background Workers and Async Components

- **Background workers/queues**: None. This is a synchronous computation library — no job queues, event buses, or worker processes.
- **Telemetry batching**: The telemetry module uses a background thread to batch and send anonymous usage metrics (`python/src/qdk_chemistry/utils/telemetry.py`), but this is not part of the application's computational architecture.

### 6) Known Architectural Risks

- **Plugin import fragility**: Plugin loading at import time means missing optional dependencies can cause import warnings or subtle registration failures. The system uses `importlib.util.find_spec()` for detection, but edge cases exist.
- **Large C++ files**: Several implementation files exceed 2000 lines (`wavefunction.cpp`, `basis_set.cpp`, `orbitals.cpp`, `element_data.cpp`) — these accumulate complexity and are harder to maintain.
- **Cartesian basis set support**: Multiple TODO markers in both C++ and Python indicate incomplete Cartesian AO support (`pyscf/conversion.py:160,271`, `utils.cpp:189`).

### 7) Evidence

- `python/src/qdk_chemistry/__init__.py`
- `python/src/qdk_chemistry/algorithms/base.py`
- `python/src/qdk_chemistry/algorithms/registry.py`
- `python/src/qdk_chemistry/plugins/pyscf/scf_solver.py`
- `python/src/qdk_chemistry/plugins/qiskit/qubit_mapper.py`
- `cpp/include/qdk/chemistry.hpp`
- `cpp/include/qdk/chemistry/data/structure.hpp`
