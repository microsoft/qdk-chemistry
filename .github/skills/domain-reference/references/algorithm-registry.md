# The Algorithm Registry — How `create()` Works

## Overview

All algorithms use a **factory + registry** pattern. Users never instantiate directly:

```python
solver = create("scf_solver")                                    # default (QDK C++)
solver = create("scf_solver", "pyscf", max_iterations=100)       # PySCF variant + settings
mapper = create("qubit_mapper", "openfermion", encoding="jordan-wigner")
```

## How It Works Internally

1. `create(type_name, variant_name, **kwargs)` finds the `AlgorithmFactory` whose `algorithm_type_name() == type_name`
2. Factory calls `factory.create(variant_name)` → returns a fresh `Algorithm` instance
3. If `kwargs` provided, calls `instance.settings().update(kwargs)` to configure it
4. Returns the configured instance

**Important**: `create()` kwargs go to `Settings.update()` — they are configuration, not runtime kwargs. The `_run_impl()` signature is fixed per algorithm type.

## Registration Chain

Registration happens at import time in three phases:

1. **C++ algorithm factories** registered via `_register_cpp_factories()` — these are pybind11-bound factory classes
2. **Python algorithm factories** registered via `_register_python_factories()`
3. **Plugin algorithms** (PySCF, Qiskit, OpenFermion) registered lazily — each plugin's `load()` function checks if the dependency is installed, then calls `register(lambda: PluginClass())`

## Algorithm Interface Contract

Every algorithm must implement:

| Method | Purpose |
|--------|---------|
| `type_name() → str` | Registry key for the algorithm type (e.g., `"scf_solver"`) |
| `name() → str` | Registry key for this variant (e.g., `"pyscf"`) |
| `_run_impl(*args, **kwargs)` | Core logic (called by `run()` after locking settings) |
| `settings() → Settings` | Type-safe configuration object |

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
