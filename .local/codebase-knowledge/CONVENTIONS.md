# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

**Python:**

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files | snake_case | `scf_solver.py`, `qubit_mapper.py` | `python/src/qdk_chemistry/plugins/` |
| Classes | PascalCase | `PyscfScfSolver`, `QiskitQubitMapper`, `EnergyEstimator` | `plugins/pyscf/scf_solver.py`, `algorithms/energy_estimator/` |
| Functions/methods | snake_case | `_run_impl`, `inspect_settings`, `create_from_ref` | `algorithms/base.py`, `algorithms/registry.py` |
| Private members | `_` prefix | `_settings`, `_run_impl` | `algorithms/base.py` |
| Constants | UPPER_SNAKE_CASE | Module-level constants | `constants.py` |

**C++:**

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files | snake_case | `basis_set.hpp`, `active_space.cpp` | `cpp/include/qdk/chemistry/data/` |
| Classes | PascalCase | `Structure`, `BasisSet`, `Wavefunction` | `cpp/include/qdk/chemistry/data/structure.hpp` |
| Functions/methods | snake_case | `num_atoms()`, `to_json()` | `cpp/include/qdk/chemistry/data/structure.hpp` |
| Private members | `_` prefix | `_coordinates`, `_elements`, `_settings` | `cpp/include/qdk/chemistry/data/structure.hpp` |
| Namespaces | `qdk::chemistry::...` | `qdk::chemistry::data`, `qdk::chemistry::algorithms` | `cpp/include/qdk/chemistry/` |

### 2) Formatting and Linting

**Python:**

- Formatter: Ruff (`ruff format`) — config in `python/pyproject.toml`
- Linter: Ruff (`ruff check --fix`) — config in `python/pyproject.toml`
- Type checker: mypy — config in `python/pyproject.toml`
- Line length: 120 characters
- Quote style: double quotes
- Docstring style: Google (Parameters, Returns, Raises, Examples)
- Docstring coverage: ≥80% enforced by interrogate
- Single-line parameter descriptions required (multi-line causes Sphinx errors)

**C++:**

- Formatter: clang-format — config in `.clang-format`
- Style: Google-based, 80-column limit
- Indent: 2 spaces
- Pointer alignment: Left (`int* ptr`)
- Braces: Attached (Attach style)
- Includes: sorted

**Pre-commit hooks** (`.pre-commit-config.yaml`):
`ruff-format` → `ruff` → `mypy` → `clang-format` → `interrogate` → `sphinx-lint` → `markdownlint` → `gitleaks` → local checks (license headers, version alignment, `.pyi` stubs)

### 3) Import and Module Conventions

- **Import grouping**: Standard library → third-party → local imports (enforced by Ruff)
- **TYPE_CHECKING guard**: Heavy type-only imports are placed inside `if TYPE_CHECKING:` blocks to avoid circular dependencies
- **Lazy imports**: Plugin modules use delayed imports inside functions to avoid loading optional dependencies at import time
- **Barrel exports**: `__init__.py` files re-export public API; `__all__` is used selectively
- **C++ includes**: Sorted automatically by clang-format; project headers use `#include "qdk/chemistry/..."` path style

### 4) Error and Logging Conventions

**Error handling:**

- Python uses explicit `raise ValueError(...)`, `KeyError(...)`, `RuntimeError(...)` with descriptive messages
- Registry raises `KeyError` for missing algorithm types/names
- Settings layer raises `SettingNotFoundError`, `SettingTypeMismatchError`, `SettingsAreLockedError`
- `NotImplementedError` used for abstract `_run_impl()` methods

**Logging:**

- Custom `Logger` class in `python/src/qdk_chemistry/utils/` wrapping Python `logging`
- Provides `Logger.trace_entering()`, `Logger.debug(...)` for structured tracing
- Log format: [TODO] — exact format string not determined; uses Python `logging` defaults
- Used in plugins (`plugins/pyscf/scf_solver.py`, `plugins/qiskit/__init__.py`) and algorithm modules

### 5) Testing Conventions

- Python test files: `python/tests/test_{feature}.py`
- C++ test files: `cpp/tests/test_{feature}.cpp`
- Python mocking: Standard `unittest.mock`, fixtures in `conftest.py`
- C++ mocking: GoogleTest assertions, utility headers (`testing_utilities.hpp`, `ut_common.hpp`)
- Coverage: pytest-cov for Python, gcovr for C++

### 6) Evidence

- `python/pyproject.toml` (Ruff, mypy, pytest config)
- `.clang-format` (C++ formatting)
- `.pre-commit-config.yaml` (all hooks)
- `python/src/qdk_chemistry/algorithms/base.py` (naming, patterns)
- `python/src/qdk_chemistry/algorithms/registry.py` (error handling)
- `cpp/include/qdk/chemistry/data/structure.hpp` (C++ naming)
