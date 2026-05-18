# Design Rationale

## Settings System — Type-Safe Configuration

Settings are the configuration mechanism for all algorithms. They enforce schema at construction time.

### Lifecycle

1. **Define**: Algorithm constructor creates `Settings` and calls `set_default(key, type, value, description, constraints)`
2. **Modify**: User modifies via `create("algo", key=value)` or `algo.settings()["key"] = value`
3. **Lock**: `run()` calls `settings().lock()` — no further changes allowed
4. **Read**: `_run_impl()` reads settings via `settings().get("key")`

### Types Supported

| C++ Type | Python Type | Description |
|----------|-------------|-------------|
| `bool` | `bool` | Boolean flag |
| `int64_t` | `int` | 64-bit integer |
| `double` | `float` | Double-precision float |
| `string` | `str` | String value |
| `vector<int64_t>` | `list[int]` | Integer list |
| `vector<double>` | `list[float]` | Float list |
| `vector<string>` | `list[str]` | String list |

### Constraints

- **`BoundConstraint<T>`**: Enforces a min/max range for numeric values
- **`ListConstraint<T>`**: Enforces an enum of allowed values (e.g., `["jordan-wigner", "bravyi-kitaev", "parity"]`)

### Common Settings Per Algorithm Type

| Algorithm | Key Settings |
|-----------|-------------|
| SCF | `method` (hf/dft/rohf/uks), `convergence_threshold` (1e-7), `max_iterations` (50), `scf_type` (auto/restricted/unrestricted) |
| Qubit Mapper | `encoding` (jordan-wigner/bravyi-kitaev/parity) |
| Active Space | `num_active_electrons`, `num_active_orbitals` |
| Phase Estimation | `num_bits`, `num_iterations` |

## The pybind11 Binding Layer

**Location**: `python/src/pybind11/` — one `.cpp` file per C++ class.

**Module structure**: `_core` module with submodules: `data`, `_algorithms`, `utils`

### Key Binding Patterns

- **`py::smart_holder`**: Automatic memory management across the C++/Python language boundary
- **Trampoline classes** (`PyDataClass`, `PySettings`): Enable Python subclassing of C++ abstract bases
- **Wrapper lambdas**: Type conversion between C++ and Python (e.g., `nlohmann::json` ↔ Python `dict`, `pathlib.Path` ↔ `std::string`)
- **GIL release** (`py::gil_scoped_release`): Released during I/O and heavy compute for thread safety
- **Factory template** `bind_algorithm_factory<FactoryType, AlgorithmType, BaseType>()`: Generic binding for all factory classes
- **Property mapping**: C++ getters become Python properties via lambda wrappers

### Binding Order

**Binding order matters**: `module.cpp` comments state dependencies must be bound first (e.g., `Element` enums before `Structure`). When adding new bindings, ensure all types referenced by a class are already bound.

## Build System

### C++ Build

- **Build tool**: CMake 3.15+, C++20
- **Main target**: `chemistry` library (alias `qdk::chemistry`, output: `libqdk_chemistry`)
- **Dependencies**: Eigen3, HDF5, nlohmann_json (fetched), libint2 (fetched), ecpint (fetched), gauxc (fetched), MACIS (submodule at `external/macis/`)
- **Version pinning**: `cpp/manifest/cgmanifest.json` files specify exact versions for FetchContent dependencies
- **Architecture**: `QDK_UARCH` env var or auto-detect (x86-64-v3, armv8-a, native)
- **Key options**: `QDK_CHEMISTRY_ENABLE_GPU` (CUDA ≥ 7.0), `QDK_ENABLE_OPENMP`, `QDK_CHEMISTRY_ENABLE_COVERAGE`

### Python Build

- **Build backend**: scikit-build-core (in `python/pyproject.toml`)
- **What it builds**: pybind11 extension module `_core` from `python/src/pybind11/*.cpp`
- **Dependency on C++**: If C++ library is installed at `CMAKE_PREFIX_PATH`, links against it. Otherwise builds C++ from source.
- **Key**: C++ must be installed first for iterative development; `pip install .` rebuilds the extension

### Build Variants

All artifacts organized by variant (`release`, `debug`):

| Path | Purpose |
|------|---------|
| `.local/$VARIANT/build/` | CMake build output |
| `.local/$VARIANT/venv/` | Python virtual environment |
| `.local/$VARIANT/install/` | C++ library install prefix (headers + lib + cmake config) |

### Dependency Chain for Rebuild

```
C++ source change
  → cmake --build .local/release/build
  → cmake --install .local/release/build
  → cd python && pip install .[all]    ← REQUIRED because pybind11 links C++
  → pytest python/tests/
```

Python-only changes (no C++ binding changes) only need `pip install` + `pytest`.

## Documentation Pipeline

Five-stage pipeline run via `make clean all` from `docs/`:

1. **Doxygen** — C++ headers → XML (`source/api/doxygen/xml/`)
2. **Breathe** — Doxygen XML → RST stubs (`source/api/breathe_api_autogen/`)
3. **sphinx-apidoc** — Python source → RST stubs (`source/api/api_autogen/`)
4. **Sphinx autosummary** — first pass, generates summary tables
5. **Sphinx build** — final HTML output (`build/html/`)

**Build fails on any warnings** — Makefile enforces zero-warning policy.

### Critical Constraints

- `sphinx_autodoc_typehints` loads **before** `sphinx.ext.napoleon` in `conf.py`. This is intentional but means multi-line continuation in `Args:`/`Returns:`/`Raises:` docstring sections triggers `(ERROR/3) Unexpected indentation`. **Every parameter description must be a single line.**
- Files in `api_autogen/` and `breathe_api_autogen/` are **auto-generated** — never edit manually.
- Sphinx reads docstrings from the **installed** package. After editing docstrings, reinstall Python before rebuilding docs.
- `conf.py` has extensive `suppress_warnings` and `nitpick_ignore_regex` for known external types. These are stable and rarely need changes.
- Debug output saved to `docs/*.txt` files for diagnosis.

### Documentation Config (`docs/source/conf.py`)

Key features:

- **Module aliasing**: `_core` internal paths rewritten to public paths in docs (e.g., `qdk_chemistry._core.data.Structure` → `qdk_chemistry.data.Structure`)
- **Autodoc skip**: Filters out re-exported standard library and third-party modules
- **Mock imports**: C++ extension modules and optional deps mocked for docs builds
- **Citation handling**: Custom `transform_doctree_citations()` converts `:cite:` markers from Doxygen to sphinxcontrib-bibtex references

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| C++ core + Python wrapper | Compute-heavy linear algebra (Eigen3) and integral evaluation in C++; user-facing workflow in Python |
| Factory/registry pattern | Swappable backends (QDK, PySCF, Qiskit, OpenFermion) without user code changes |
| Immutable data classes | Prevents accidental mutation of molecular data mid-pipeline |
| Type-safe Settings | Catches misconfiguration at set time, not deep in algorithm execution |
| Container pattern (Hamiltonian/Wavefunction) | Different storage formats (dense, Cholesky, sparse) behind single interface |
| Lazy plugin loading | Optional dependencies don't cause import errors; plugins activate only when deps are available |
| Blocked spin-orbital ordering (QDK default) | Industry standard for most quantum chemistry codes; interleaved available via conversion |
| Little-endian Pauli strings | Qubit 0 is rightmost in string representation for consistency with circuit conventions |
