---
name: cpp-build
description: "Use when: building C++ code, running C++ tests, diagnosing CMake or compilation errors, rebuilding after C++ source changes. Handles cmake configure, build, install, and ctest for the qdk-chemistry project."
---

# C++ Build & Install

## Build Variant

Default variant is `release`. All paths use `.local/$VARIANT/` (e.g., `.local/release/`).

| Path | Purpose |
|------|---------|
| `.local/$VARIANT/build/` | CMake build output |
| `.local/$VARIANT/install/` | CMake install prefix |

## CMake Presets

The project uses `cpp/CMakePresets.json` with predefined configurations:

| Preset | Build Type | Coverage |
|--------|-----------|----------|
| `release` | Release | Off |
| `debug` | Debug | Off |
| `coverage` | RelWithDebInfo | On |

## Logging

All commands **must** redirect their full output to a log file so the user can `tail -f` it. Use `> <logfile> 2>&1` to keep output out of agent context. Always tell the user which log file you're writing to. These log files are your primary tool for diagnosing errors — read the log file rather than re-running commands or keeping full output in context.

## Procedure

### 1. Configure (only if build dir doesn't exist or CMakeLists.txt changed)

Using presets:

```bash
cmake --preset $VARIANT -S cpp > cmake-configure.log 2>&1
```

Or with explicit flags:

```bash
cmake -S cpp -B .local/$VARIANT/build \
  -DCMAKE_INSTALL_PREFIX=.local/$VARIANT/install \
  -DCMAKE_BUILD_TYPE=Release > cmake-configure.log 2>&1
```

### 2. Build

Using presets:

```bash
cmake --build --preset $VARIANT > cmake-build.log 2>&1
```

Or explicitly:

```bash
cmake --build .local/$VARIANT/build -j 4 > cmake-build.log 2>&1
```

- Presets default to `-j 4`. Override with `CMAKE_BUILD_PARALLEL_LEVEL` env var if needed.

### 3. Install (required before Python package can use C++ library)

```bash
cmake --install .local/$VARIANT/build > cmake-install.log 2>&1
```

### 4. Test

Using presets:

```bash
OMP_NUM_THREADS=1 ctest --preset $VARIANT > ctest.log 2>&1
```

Or explicitly:

```bash
OMP_NUM_THREADS=1 ctest --test-dir .local/$VARIANT/build --output-on-failure > ctest.log 2>&1
```

Run a specific test executable:

```bash
OMP_NUM_THREADS=1 ./.local/$VARIANT/build/tests/test_name > ctest.log 2>&1
```

List available tests:

```bash
ls .local/$VARIANT/build/tests/
```

## Key CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `QDK_CHEMISTRY_ENABLE_COVERAGE` | OFF | Code coverage |
| `QDK_CHEMISTRY_ENABLE_GPU` | OFF | GPU/CUDA support |
| `QDK_ENABLE_OPENMP` | ON (Linux) | OpenMP parallelism |
| `BUILD_TESTING` | ON | Build test executables |

## Key Paths

| Path | Purpose |
|------|---------|
| `cpp/CMakeLists.txt` | Top-level CMake config |
| `cpp/CMakePresets.json` | Preset definitions for configure/build/test |
| `cpp/cmake/third_party.cmake` | External dependencies (Eigen3, HDF5, libint2, etc.) |
| `cpp/cmake/qdk-uarch.cmake` | Microarchitecture detection |
| `cpp/include/qdk/chemistry/` | Public headers |
| `cpp/src/qdk/chemistry/` | Implementations |
| `cpp/tests/` | Google Test sources |

## Surgical Testing

Tests are long — always run the **minimum relevant subset**, not the full suite. CI/CD handles full verification.

**How to pick which tests to run:**

1. **Changed a specific C++ source file?** Find the test that covers it:
   ```bash
   # List all test executables
   ls .local/$VARIANT/build/tests/
   # Test names generally match: scf changes → test_scf, hamiltonian changes → test_hamiltonian
   ```

2. **Run just that test:**
   ```bash
   OMP_NUM_THREADS=1 ./.local/$VARIANT/build/tests/test_name > ctest.log 2>&1
   ```

3. **Run a specific test case within an executable:**
   ```bash
   OMP_NUM_THREADS=1 ./.local/$VARIANT/build/tests/test_name --gtest_filter="TestSuite.TestCase" > ctest.log 2>&1
   ```

4. **Only run the full suite if the change is cross-cutting** (e.g., modifying a core data structure used everywhere).

## Error Diagnosis

When something fails, **read the log file first** (e.g., `cmake-build.log`, `ctest.log`) rather than re-running with verbose flags or keeping full output in context. The logs contain the complete output needed for diagnosis.

- **Missing dependency**: Check `cpp/cmake/third_party.cmake` and installed system packages.
- **Test failure**: Re-run with `--gtest_filter=TestName` and read the test source in `cpp/tests/`.
- **After C++ changes that affect Python bindings**: The Python package must be reinstalled (see the `python-build` skill).
