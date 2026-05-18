# External Integrations

## Core Sections (Required)

### 1) Integration Inventory

| System | Type | Purpose | Auth model | Criticality | Evidence |
|--------|------|---------|------------|-------------|----------|
| PySCF | Library plugin | Classical electronic structure (SCF, MCSCF, CC, localization, AVAS) | None (local library) | High — primary classical backend | `python/src/qdk_chemistry/plugins/pyscf/` |
| Qiskit / Qiskit Aer / Qiskit Nature | Library plugin | Qubit mapping, circuit simulation, QIR interop | None (local library) | Medium — quantum simulation backend | `python/src/qdk_chemistry/plugins/qiskit/` |
| OpenFermion | Library plugin | Hamiltonian ↔ fermion/qubit operator conversion | None (local library) | Low — alternative qubit mapping | `python/src/qdk_chemistry/plugins/openfermion/` |
| Azure Monitor (Application Insights) | HTTP API | Anonymous telemetry collection | Instrumentation key | Low — opt-out telemetry | `python/src/qdk_chemistry/utils/telemetry.py` |
| PyQIR | Library | QIR (Quantum Intermediate Representation) parsing | None | Medium — circuit format interop | `python/src/qdk_chemistry/plugins/qiskit/_interop/qir.py` |
| qdk (Q#) | Library | Azure Quantum SDK, resource estimation | None | Medium — quantum resource estimation | `python/pyproject.toml` |

### 2) Data Stores

**Databases connected**: None — no SQL, NoSQL, or networked database connections. All persistence is file-based.

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| HDF5 files | Persistent serialization of Hamiltonians, Wavefunctions | C++ via HDF5 C++ API; Python via h5py | Large file sizes for big molecules | `cpp/src/qdk/chemistry/data/hdf5_serialization.cpp`, `python/pyproject.toml` |
| JSON files | Lightweight serialization, configuration | C++ via nlohmann/json | None significant | `cpp/src/qdk/chemistry/data/json_serialization.cpp` |
| XYZ files | Molecular geometry input | Parsed by Structure class | Malformed input | `examples/data/*.xyz` |

### 3) Secrets and Credentials Handling

- Credential sources: Telemetry uses a hardcoded Azure Application Insights instrumentation key and endpoint in `telemetry.py:54-61`
- Hardcoding checks: Default Application Insights key and endpoint are embedded but can be overridden via `QSHARP_PYTHON_AI_KEY` and `QSHARP_PYTHON_AI_URL` env vars (`telemetry.py:55-59`)
- No other secrets, API keys, or credentials are used — this is a local computation library
- Telemetry is opt-out via `QSHARP_PYTHON_TELEMETRY` environment variable

### 4) Reliability and Failure Behavior

- Retry/backoff: Telemetry HTTP calls use basic error handling but no retry logic (`telemetry.py:236-263`)
- Timeout policy: Telemetry has a batch interval for buffering events
- Circuit-breaker: None — telemetry failures are silently ignored to avoid impacting computation
- Plugin loading: Uses `importlib.util.find_spec()` for graceful detection of optional dependencies; missing plugins produce warnings, not errors

### 5) Observability for Integrations

- Telemetry: Anonymous usage metrics sent to Azure Monitor (import events, algorithm run events)
- Logging: Custom `Logger` class with trace/debug levels used in plugins
- Missing visibility: No structured metrics for computation performance (SCF convergence, circuit depth, etc.) beyond what algorithms return as data

### 6) Evidence

- `python/src/qdk_chemistry/plugins/pyscf/` (PySCF integration)
- `python/src/qdk_chemistry/plugins/qiskit/` (Qiskit integration)
- `python/src/qdk_chemistry/plugins/openfermion/` (OpenFermion integration)
- `python/src/qdk_chemistry/utils/telemetry.py` (telemetry)
- `python/src/qdk_chemistry/utils/telemetry_events.py` (telemetry events)
- `python/pyproject.toml` (optional dependency groups)
- `cpp/src/qdk/chemistry/data/hdf5_serialization.cpp` (HDF5 I/O)
