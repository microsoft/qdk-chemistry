# Testing Patterns

## Core Sections (Required)

### 1) Test Stack and Commands

- Primary Python test framework: pytest
- Primary C++ test framework: GoogleTest (fetched via CMake FetchContent)
- Assertion tools: pytest assertions (Python), GoogleTest `EXPECT_*`/`ASSERT_*` (C++)
- Coverage: pytest-cov (Python), gcovr (C++)
- Commands:

```bash
# Python tests
pytest python/tests/ -ra -v

# Python tests with coverage
pytest python/tests/ --cov=qdk_chemistry --cov-report=term

# C++ build with tests
cmake -B build -S cpp -DBUILD_TESTING=ON -DQDK_CHEMISTRY_ENABLE_COVERAGE=ON
cmake --build build

# C++ tests
cd build && ctest

# C++ coverage (after running tests)
gcovr --root . --filter cpp/src/
```

### 2) Test Layout

- **Python test placement**: Separate `python/tests/` directory (not co-located)
- **Python naming convention**: `test_{feature}.py` (e.g., `test_scf.py`, `test_hamiltonian.py`, `test_interop_qiskit_qubit_mapper.py`)
- **Python fixtures**: `python/tests/conftest.py` — provides orbital fixtures, temp directories, Hamiltonian/Wavefunction fixtures, error profiles
- **Python test data**: `python/tests/test_data/` — JSON, XYZ, QASM files and generator scripts
- **C++ test placement**: `cpp/tests/test_{feature}.cpp`
- **C++ test main**: `cpp/tests/qdk_test_main.cpp`
- **C++ test helpers**: `cpp/tests/testing_utilities.hpp`, `cpp/tests/ut_common.hpp`
- **C++ test discovery**: `gtest_discover_tests` in `cpp/tests/CMakeLists.txt`

### 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | Yes | Data classes, algorithms, utilities, serialization | Majority of test suite — both C++ and Python |
| Integration | Yes | Plugin conversions (PySCF, Qiskit, OpenFermion), end-to-end workflows | `test_interop_*` files, `test_pyscf_plugin.py` |
| E2E | Yes | Full pipeline tests (structure → SCF → Hamiltonian → qubit mapping → simulation) | Example notebooks tested via `test_docs_examples.py`, `test_readme_snippets.py` |
| Doc examples | Yes | README code snippets, documentation examples | `test_readme_snippets.py`, `test_docs_examples.py` |

### 4) Mocking and Isolation Strategy

- **Python mocking**: Standard `unittest.mock` for isolating external dependencies
- **Fixture-based setup**: `conftest.py` provides pre-built molecular structures, Hamiltonians, and wavefunctions to avoid repeated expensive computations
- **Test markers**: `@pytest.mark.slow` for long-running tests
- **Warning filters**: pytest configured to suppress known deprecation warnings from Qiskit/Qiskit Aer
- **Telemetry disabled**: `QSHARP_PYTHON_TELEMETRY=disabled` set in pytest config to prevent telemetry during tests
- **C++ isolation**: Each test file is self-contained; `ut_common.hpp` provides shared test utilities

### 5) Coverage and Quality Signals

- **Python coverage tool**: pytest-cov — config in `python/pyproject.toml:104-141`
- **C++ coverage tool**: gcovr — invoked in CI workflow
- **Coverage threshold**: [TODO] — no explicit minimum threshold found in configuration
- **CI enforcement**: `.github/workflows/build-and-test.yaml` runs both C++ and Python tests with coverage enabled
- **Known gaps**: Plugin integration tests depend on optional dependencies (PySCF, Qiskit, OpenFermion) being installed

### 6) Evidence

- `python/pyproject.toml:154-169` (pytest config)
- `python/tests/conftest.py` (fixtures)
- `cpp/tests/CMakeLists.txt` (GoogleTest setup)
- `.github/workflows/build-and-test.yaml:248-430` (CI test commands)
- `python/tests/test_data/` (test fixtures)
