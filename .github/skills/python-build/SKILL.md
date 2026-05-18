---
name: python-build
description: "Use when: building/installing the Python package, running Python tests, diagnosing pip install or pytest errors, reinstalling after code changes. Handles venv activation, pip install, and pytest for the qdk-chemistry project."
---

# Python Build & Install

## Build Variant

Default variant is `release`. All paths use `.local/$VARIANT/` (e.g., `.local/release/`).

| Path | Purpose |
|------|---------|
| `.local/$VARIANT/venv/` | Python virtual environment |
| `.local/$VARIANT/install/` | C++ install prefix (must exist before Python install) |

## Logging

All commands **must** tee their full output to a log file so the user can monitor progress. Use `2>&1 | tee <logfile>` on every install/test command. The log filename is flexible — just tell the user where it is so they can `tail -f` it. These log files are also your primary tool for diagnosing errors — read the log file rather than re-running commands with full output in the terminal or piping output to `/dev/null`.

## Procedure

### 1. Ensure venv exists

```bash
test -d .local/$VARIANT/venv || python3 -m venv .local/$VARIANT/venv
.local/$VARIANT/venv/bin/pip install --upgrade pip
```

### 2. Activate venv

```bash
source .local/$VARIANT/venv/bin/activate
```

### 3. Ensure C++ is built and installed (MANDATORY — do this BEFORE pip install)

The Python package wraps C++ via pybind11. It **cannot** be installed without the C++ library already built and installed.

**You must rebuild C++ yourself** unless you already did so earlier in this session. You cannot trust a pre-existing install — it may be stale from a previous session, a different branch, or a git checkout that changed timestamps. The only way to be sure is to run the full C++ build & install yourself (see the `cpp-build` skill) before continuing with the Python install.

### 4. Install Python package

```bash
cd python && CMAKE_PREFIX_PATH=$(pwd)/../.local/$VARIANT/install \
  CMAKE_BUILD_PARALLEL_LEVEL=4 \
  OMP_NUM_THREADS=1 \
  pip install .[all] -v 2>&1 | tee ../pip-install.log
```

- Always use `CMAKE_BUILD_PARALLEL_LEVEL=4` (4 cores) unless the user specifies otherwise.
- Always set `OMP_NUM_THREADS=1` unless the user specifies otherwise.
- **Never** use `pip install -e` (editable installs). Always do a full install with `pip install .`.
- `.[all]` pulls in all dependencies (main + every extra group defined in `pyproject.toml`). **Do not** `pip install` individual packages separately — let `qdk-chemistry`'s declared dependencies handle it. Only install a package directly if it is genuinely not covered by the main or extra dependency groups.

Then return to repo root:

```bash
cd ..
```

### 5. Run tests

```bash
OMP_NUM_THREADS=1 pytest python/tests/ -v 2>&1 | tee pytest.log
```

For specific tests:

```bash
pytest python/tests/test_specific.py -v 2>&1 | tee pytest.log
pytest python/tests/test_specific.py::TestClass::test_method -v 2>&1 | tee pytest.log
```

Skip slow tests:

```bash
pytest python/tests/ -v -m "not slow" 2>&1 | tee pytest.log
```

### 6. Linting

```bash
pre-commit run --all-files 2>&1 | tee pre-commit.log
```

Or specific hooks:

```bash
pre-commit run ruff --all-files 2>&1 | tee pre-commit.log
pre-commit run mypy --all-files 2>&1 | tee pre-commit.log
```

## Key Paths

| Path | Purpose |
|------|---------|
| `python/pyproject.toml` | Package config, dependencies, tool settings |
| `python/CMakeLists.txt` | pybind11 extension build config |
| `python/src/qdk_chemistry/` | Package source |
| `python/src/pybind11/` | C++ binding source files |
| `python/tests/` | pytest test suite |
| `.pre-commit-config.yaml` | Linting tool configuration |

## Build Backend

The package uses **scikit-build-core**. It handles CMake integration for the pybind11 extension module (`qdk_chemistry._core`).

## Surgical Testing

Tests are long — always run the **minimum relevant subset**, not the full suite. CI/CD handles full verification.

**How to pick which tests to run:**

1. **Changed a specific Python module?** Run the matching test file:
   ```bash
   # Test files mirror source: algorithms/foo.py → tests/test_foo.py
   pytest python/tests/test_foo.py -v 2>&1 | tee pytest.log
   ```

2. **Run a specific test function or class:**
   ```bash
   pytest python/tests/test_foo.py::TestClass::test_method -v 2>&1 | tee pytest.log
   ```

3. **Skip slow tests** when iterating quickly:
   ```bash
   pytest python/tests/test_foo.py -v -m "not slow" 2>&1 | tee pytest.log
   ```

4. **Changed a plugin?** Run only that plugin's tests:
   ```bash
   pytest python/tests/test_pyscf_*.py -v 2>&1 | tee pytest.log
   pytest python/tests/test_qiskit_*.py -v 2>&1 | tee pytest.log
   ```

5. **Only run the full suite if the change is cross-cutting** (e.g., modifying a core data class or the algorithm registry).

## Common Issues

When something fails, **read the log file first** (e.g., `pip-install.log`, `pytest.log`) rather than re-running with verbose flags or keeping full output in context. The logs contain the complete output needed for diagnosis.

- **"Could not find qdk" during pip install**: The C++ library isn't installed. Run the C++ build and install first (see `cpp-build` skill).
- **Import errors after C++ changes**: Rebuild C++ → reinstall C++ → reinstall Python (all three steps required).
- **After editing Python docstrings**: The package must be reinstalled before rebuilding docs, since Sphinx reads from the installed copy.
