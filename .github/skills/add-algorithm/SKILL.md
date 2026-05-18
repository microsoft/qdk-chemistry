---
name: add-algorithm
description: "Use when: adding a new algorithm variant, registering a new algorithm factory, implementing the Algorithm interface, creating plugin algorithms. Step-by-step guide for the qdk-chemistry algorithm registry."
---

# Add Algorithm Variant

## Overview

This skill walks through registering a new algorithm variant in the qdk-chemistry factory/registry system. The registry lets users create algorithm instances by type and variant name, with optional keyword settings:

```python
solver = create("scf_solver")                                      # default (QDK C++)
solver = create("scf_solver", "pyscf", max_iterations=100)         # PySCF variant + settings
mapper = create("qubit_mapper", "openfermion", encoding="jordan-wigner")
```

## How the Registry Works

1. `create(type_name, variant_name, **kwargs)` finds the `AlgorithmFactory` whose `algorithm_type_name() == type_name`
2. Factory calls `factory.create(variant_name)` → returns a fresh `Algorithm` instance
3. If `kwargs` provided, calls `instance.settings().update(kwargs)` to configure it
4. Returns the configured instance

### Registration chain (happens at import time)

- C++ algorithm factories registered via `_register_cpp_factories()` — pybind11-bound factory classes
- Python algorithm factories registered via `_register_python_factories()`
- Plugin algorithms (PySCF, Qiskit, OpenFermion) registered lazily — each plugin's `load()` function checks if the dependency is installed, then calls `register(lambda: PluginClass())`

### Algorithm interface contract

Every algorithm must implement:

| Method | Purpose |
|--------|---------|
| `type_name() → str` | Registry key for the algorithm type (e.g., `"scf_solver"`) |
| `name() → str` | Registry key for this variant (e.g., `"pyscf"`) |
| `_run_impl(*args, **kwargs)` | Core logic (called by `run()` after locking settings) |
| `settings() → Settings` | Type-safe configuration object |

### Settings system

Settings use `variant<bool, int64_t, double, string, vector<...>>`. Schema defined at construction, locked before `run()`.

- `set_default(key, type, value, description, constraints)` — define a setting
- `BoundConstraint<T>` (min/max range) or `ListConstraint<T>` (enum of allowed values)

## Procedure

### Adding a Pure Python Algorithm Variant

#### Step 1: Create the algorithm class

Create a new file in the appropriate algorithm directory, e.g., `python/src/qdk_chemistry/algorithms/<type>/my_variant.py`:

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qdk_chemistry.algorithms.base import Algorithm


class MyVariantAlgorithm(Algorithm):
    """One-line description of what this algorithm variant does."""

    def __init__(self) -> None:
        super().__init__()
        # Define settings schema
        self.settings().set_default("my_param", 1.0, "Description of my_param")

    @staticmethod
    def type_name() -> str:
        return "algorithm_type"  # e.g., "scf_solver", "qubit_mapper"

    @staticmethod
    def name() -> str:
        return "my_variant"  # unique name for this variant

    def _run_impl(self, *args, **kwargs):
        # Read settings
        my_param = self.settings()["my_param"]
        # Implement the algorithm
        ...
```

#### Step 2: Register the algorithm

**Option A — For core algorithms** (always available):

Add registration to `python/src/qdk_chemistry/algorithms/__init__.py` in the `_register_python_factories()` function:

```python
from qdk_chemistry.algorithms import register
from qdk_chemistry.algorithms.<type>.my_variant import MyVariantAlgorithm
register(lambda: MyVariantAlgorithm())
```

**Option B — For plugin algorithms** (depends on external library):

Add registration to the plugin's `__init__.py`, e.g., `python/src/qdk_chemistry/plugins/<plugin_name>/__init__.py`:

```python
_loaded = False
def load():
    global _loaded
    if _loaded: return
    _loaded = True
    if importlib.util.find_spec("external_lib") is None: return  # graceful skip
    from qdk_chemistry.algorithms import register
    register(lambda: MyVariantAlgorithm())
```

#### Step 3: Add tests

Create `python/tests/test_my_variant.py`:

```python
from qdk_chemistry.algorithms import create

def test_my_variant_creates():
    algo = create("algorithm_type", "my_variant")
    assert algo.name() == "my_variant"
    assert algo.type_name() == "algorithm_type"

def test_my_variant_settings():
    algo = create("algorithm_type", "my_variant", my_param=2.0)
    assert algo.settings()["my_param"] == 2.0

def test_my_variant_runs():
    algo = create("algorithm_type", "my_variant")
    result = algo.run(...)  # provide appropriate inputs
    assert ...
```

#### Step 4: Add docstrings and type hints

Google-style docstrings with Parameters, Returns, Raises, Examples sections. Type hints required (PEP 484). Line length 120 characters. Keep each parameter description on a single line.

#### Step 5: Build and test

```bash
source .local/release/venv/bin/activate
cd python && CMAKE_PREFIX_PATH=$(pwd)/../.local/release/install pip install .[all] -v
cd ..
pytest python/tests/test_my_variant.py -v
```

### Adding a C++ Algorithm with Python Bindings

Follow the same pattern but:

1. Implement the C++ algorithm class in `cpp/include/qdk/chemistry/algorithms/` and `cpp/src/qdk/chemistry/algorithms/`
2. Create the factory class deriving from the appropriate base factory
3. Add pybind11 bindings (see the `add-binding` skill)
4. Register via `_register_cpp_factories()` in `python/src/qdk_chemistry/algorithms/__init__.py`

## Key Paths

| Path | Purpose |
|------|---------|
| `python/src/qdk_chemistry/algorithms/` | Algorithm implementations |
| `python/src/qdk_chemistry/algorithms/__init__.py` | Registry, `create()`, `register()` |
| `python/src/qdk_chemistry/algorithms/base.py` | Base `Algorithm` class |
| `python/src/qdk_chemistry/plugins/` | Plugin algorithm implementations |
| `cpp/include/qdk/chemistry/algorithms/` | C++ algorithm headers |
| `cpp/src/qdk/chemistry/algorithms/` | C++ algorithm implementations |
| `CODEBASE_MAP.md` | Full architecture reference |

## Algorithm Types Reference

| Type | Default | Variants | Run Signature |
|------|---------|----------|---------------|
| `scf_solver` | `qdk` | `pyscf` | `(Structure, charge, spin_mult, basis) → (energy, Wavefunction)` |
| `hamiltonian_constructor` | `qdk` | — | `(Orbitals) → Hamiltonian` |
| `active_space_selector` | `qdk_valence` | `qdk_occupation`, `qdk_entropy`, `qdk_autocas`, `pyscf_avas` | `(Wavefunction, **params) → Wavefunction` |
| `qubit_mapper` | `qdk` | `qiskit`, `openfermion` | `(Hamiltonian, Symmetries?) → QubitHamiltonian` |
| `orbital_localizer` | `qdk_pipek_mezey` | `qdk_vvhv`, `qdk_mp2_natural`, `pyscf` | `(Orbitals) → Orbitals` |
| `stability_checker` | `qdk` | `pyscf` | `(Wavefunction) → StabilityResult` |
| `multi_configuration_calculator` | `qdk` | — | `(Wavefunction, Hamiltonian) → Wavefunction` |
| `circuit_executor` | `qdk` | `qiskit` | `(Circuit) → result` |
| `phase_estimation` | — | `iterative`, `qiskit_standard` | `(QubitHamiltonian, Circuit) → QpeResult` |
| `energy_estimator` | — | `qsharp`, `qiskit` | `(QubitHamiltonian, Circuit) → EnergyExpectationResult` |
| `state_preparation` | — | `sparse_isometry`, `qiskit_regular` | `(Wavefunction) → Circuit` |
| `time_evolution_builder` | — | `trotter`, `qdrift`, `partially_randomized` | `(QubitHamiltonian) → TimeEvolutionUnitary` |

## Common Pitfalls

- **Duplicate `name()` or `type_name()`**: The registry will raise an error if you register a variant name that already exists for a given type.
- **Settings not defined in `__init__`**: All settings must be declared via `set_default()` before `run()` is called. Settings are locked when `run()` starts.
- **Wrong `_run_impl` signature**: The arguments must match the expected signature for the algorithm type (see table above).
- **Plugin not loading**: Ensure the `load()` function is called at import time and handles missing dependencies gracefully with `importlib.util.find_spec()`.
- **Forgetting to register**: The algorithm won't be discoverable via `create()` unless `register(lambda: YourAlgorithm())` is called.
