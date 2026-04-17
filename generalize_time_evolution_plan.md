# Plan: Generalize Time-Evolution to Support Block Encoding (PR1)

## TL;DR

Refactor the `time_evolution` algorithms and data classes so the base abstractions are model-agnostic (not Trotter/time-evolution-specific), enabling future block encoding support (PR2). PR1 renames base classes, moves `evolution_time` out of the QPE base, puts phase→energy conversion on the builder, and renames factory strings — all with backward-compatible aliases + deprecation warnings.

---

## Phase 1: Generalize Data Layer

**Goal:** Make the unitary data classes model-agnostic so they can hold both time-evolution and block-encoding representations.

### Step 1.1 — Rename `TimeEvolutionUnitaryContainer` → `UnitaryContainer`
- File: `python/src/qdk_chemistry/data/time_evolution/containers/base.py`
- Rename class `TimeEvolutionUnitaryContainer` → `UnitaryContainer`
- Add alias `TimeEvolutionUnitaryContainer = UnitaryContainer` with deprecation warning
- `PauliProductFormulaContainer` already inherits from this — update its base class reference

### Step 1.2 — Rename `TimeEvolutionUnitary` → `UnitaryRepresentation`
- File: `python/src/qdk_chemistry/data/time_evolution/base.py`
- Rename class `TimeEvolutionUnitary` → `UnitaryRepresentation`
- Add `TimeEvolutionUnitary = UnitaryRepresentation` alias with deprecation
- Update `__init__` to take `UnitaryContainer` (new name)
- Consider renaming directory `data/time_evolution/` → `data/unitary/` (or keep and re-export from new location)

### Step 1.3 — Rename `ControlledTimeEvolutionUnitary` → `ControlledUnitary`
- File: `python/src/qdk_chemistry/data/time_evolution/controlled_time_evolution.py`
- Rename class → `ControlledUnitary`
- Add `ControlledTimeEvolutionUnitary = ControlledUnitary` alias with deprecation
- Update internal references from `TimeEvolutionUnitary` → `UnitaryRepresentation`

### Step 1.4 — Update `data/__init__.py` exports
- File: `python/src/qdk_chemistry/data/__init__.py`
- Export new names (`UnitaryRepresentation`, `UnitaryContainer`, `ControlledUnitary`)
- Keep old names in `__all__` for backward compat, add deprecation

---

## Phase 2: Generalize Algorithm Layer

**Goal:** Create a model-agnostic `UnitaryBuilder` base and generalized circuit mapper.

### Step 2.1 — Introduce `UnitaryBuilder` base class
- File: `python/src/qdk_chemistry/algorithms/time_evolution/builder/base.py`
- Create abstract class `UnitaryBuilder(Algorithm)`:
  - Abstract `_run_impl(self, qubit_hamiltonian: QubitHamiltonian, *args, **kwargs) → UnitaryRepresentation`
  - Abstract `phase_to_energy(self, phase_fraction: float) → float` — model-dependent conversion
  - Abstract `energy_alias_candidates(self, raw_energy: float, shift_range=...) → list[float]` — model-dependent aliasing
  - Optional `resolve_energy(self, raw_energy: float, reference_energy: float) → float` — default: pick closest alias
  - Move `_pauli_label_to_map()` static method here (shared utility)
- `TimeEvolutionBuilder(UnitaryBuilder)`:
  - Keeps `_run_impl(self, qubit_hamiltonian, time)` signature
  - Implements `phase_to_energy`: delegates to existing `energy_from_phase(φ, t)` using `self._get_evolution_time()`
  - Implements `energy_alias_candidates`: delegates to existing `energy_alias_candidates(E, t)` 
  - Needs to store/access `evolution_time` — either passed at run time or from settings
  - `TimeEvolutionBuilder = TimeEvolutionBuilder` (already correct name, no rename needed)
- Add `TimeEvolutionBuilder` alias to old name if any references exist
- Rename factory: `TimeEvolutionBuilderFactory` → `UnitaryBuilderFactory`, keep alias

### Step 2.2 — Rename `ControlledEvolutionCircuitMapper` → `ControlledCircuitMapper`
- File: `python/src/qdk_chemistry/algorithms/time_evolution/controlled_circuit_mapper/base.py`
- Rename class → `ControlledCircuitMapper`
- Add alias `ControlledEvolutionCircuitMapper = ControlledCircuitMapper` with deprecation
- Update `_run_impl` parameter type: `ControlledTimeEvolutionUnitary` → `ControlledUnitary`
- Rename factory: `ControlledEvolutionCircuitMapperFactory` → `ControlledCircuitMapperFactory`, keep alias

### Step 2.3 — Update `__init__.py` exports in builder and mapper packages
- `builder/__init__.py`: Export `UnitaryBuilder`, `UnitaryBuilderFactory` + old aliases
- `controlled_circuit_mapper/__init__.py`: Export `ControlledCircuitMapper`, `ControlledCircuitMapperFactory` + old aliases

### Step 2.4 — Update factory registration strings
- File: `python/src/qdk_chemistry/algorithms/registry.py`
- Primary type name: `"unitary_builder"` (new), with backward fallback to `"time_evolution_builder"`
- Primary type name: `"controlled_circuit_mapper"` (new), with backward fallback to `"controlled_evolution_circuit_mapper"`
- Implement backward compat in registry lookup: if no factory matches, try known aliases
- `UnitaryBuilderFactory.algorithm_type_name()` returns `"unitary_builder"`
- `ControlledCircuitMapperFactory.algorithm_type_name()` returns `"controlled_circuit_mapper"`

### Step 2.5 — Consider directory rename
- Current: `algorithms/time_evolution/` contains both builder/ and controlled_circuit_mapper/
- Option A: Keep directory name (less churn, importable from old paths via __init__ re-exports)
- Option B: Rename to `algorithms/unitary/` with old paths as re-export shims
- **Recommendation: Keep `algorithms/time_evolution/` directory for PR1 to minimize churn, add `algorithms/unitary/` as re-export package pointing to same modules**

---

## Phase 3: Generalize QPE

**Goal:** Decouple phase estimation from time-evolution-specific assumptions.

### Step 3.1 — Remove `evolution_time` from `PhaseEstimationSettings`
- File: `python/src/qdk_chemistry/algorithms/phase_estimation/base.py`
- Remove `evolution_time` from `PhaseEstimationSettings.__init__`
- Remove `evolution_time` parameter from `PhaseEstimation.__init__`
- Keep `num_bits` in the base (applicable to all QPE variants)

### Step 3.2 — Move `evolution_time` to concrete QPE subclasses
- File: `iterative_phase_estimation.py`
  - Add `evolution_time` to `IterativePhaseEstimationSettings`
  - Add `evolution_time` parameter to `IterativePhaseEstimation.__init__`
- File: `standard_phase_estimation.py`
  - Add `evolution_time` to `QiskitStandardPhaseEstimationSettings`
  - Add `evolution_time` parameter to `QiskitStandardPhaseEstimation.__init__`
- **Note:** In PR2, block-encoding QPE subclass will NOT have `evolution_time` but may have `lambda_norm`

### Step 3.3 — Generalize `_run_impl` signature in PhaseEstimation base
- Replace `evolution_builder: TimeEvolutionBuilder` → `unitary_builder: UnitaryBuilder`
- Replace `circuit_mapper: ControlledEvolutionCircuitMapper` → `circuit_mapper: ControlledCircuitMapper`
- Rename helper `_create_time_evolution` → `_create_unitary` (generalized)
  - Signature: `_create_unitary(qubit_hamiltonian, unitary_builder, **model_kwargs) → UnitaryRepresentation`
  - For time evolution, `model_kwargs` includes `time=evolution_time`
  - The concrete subclass passes the right kwargs
- Rename `_create_ctrl_time_evol_circuit` → `_create_controlled_circuit`

### Step 3.4 — Use builder's phase→energy methods in QpeResult construction
- In `IterativePhaseEstimation._run_impl`: after measuring phase, call `unitary_builder.phase_to_energy(phase_fraction)` to get `raw_energy`
- Pass the builder (or its phase model methods) into `QpeResult` construction
- The builder knows `evolution_time` (for time evolution) or `lambda_norm` (for block encoding)

### Step 3.5 — Update `QpeResult.from_phase_fraction()`
- File: `python/src/qdk_chemistry/data/qpe_result.py`
- Option A: Accept a callable `phase_to_energy_fn` + `energy_alias_fn` instead of `evolution_time`
- Option B: Accept a `UnitaryBuilder` reference
- **Recommendation: Option A** — keep `QpeResult` decoupled from algorithm classes
  - New signature: `from_phase_fraction(method, phase_fraction, *, phase_to_energy, energy_alias_candidates, resolve_energy=None, ...)`
  - Add backward-compat overload: if `evolution_time` is passed (old API), construct lambdas from existing `phase.py` functions
  - Deprecation warning when `evolution_time` is used directly

### Step 3.6 — Keep `phase.py` utilities intact
- The functions in `utils/phase.py` (`energy_from_phase`, `energy_alias_candidates`, `resolve_energy_aliases`) remain as-is
- They become the implementation used by `TimeEvolutionBuilder.phase_to_energy()` etc.
- No changes needed to these utilities

---

## Phase 4: Update PauliSequenceMapper

### Step 4.1 — Update type references
- File: `pauli_sequence_mapper.py`
- Update import: `ControlledTimeEvolutionUnitary` → `ControlledUnitary`
- Update `_run_impl` parameter type accordingly
- The mapper still validates for `PauliProductFormulaContainer` internally — this is fine (it's a Pauli-specific mapper, not a generic one)

---

## Phase 5: Update Tests

### Step 5.1 — Test files using time_evolution/QPE types (update imports, verify aliases)
Files to update:
- `tests/test_time_evolution_circuit_mapper.py` — imports `TimeEvolutionUnitary`, `ControlledTimeEvolutionUnitary`
- `tests/test_time_evolution_trotter.py` — imports `TimeEvolutionUnitary`
- `tests/test_time_evolution_qdrift.py` — imports `TimeEvolutionUnitary`
- `tests/test_time_evolution_partially_randomized.py` — imports `TimeEvolutionUnitary`
- `tests/test_time_evolution_container.py` — imports container types
- `tests/test_controlled_time_evolution.py` — imports `ControlledTimeEvolutionUnitary`, `TimeEvolutionUnitary`, `TimeEvolutionUnitaryContainer`
- `tests/test_trotter_error.py` — likely no rename needed (Trotter-specific)
- `tests/test_phase_estimation_iterative.py` — uses `evolution_time`, `TimeEvolutionBuilder` indirectly via `create()`
- `tests/test_interop_qiskit_phase_estimation_standard.py` — same pattern
- `tests/test_qpe_result.py` — uses `evolution_time` in `QpeResult.from_phase_fraction()`
- `tests/test_algorithms_registry.py` — tests factory type strings

### Step 5.2 — Strategy: use aliases initially, update to new names in a follow-up commit
- Since aliases with deprecation warnings exist, tests can initially keep old names
- Update test imports to new names for cleanliness
- Verify all tests pass with new names

---

## Phase 6: Update Examples and Documentation

### Step 6.1 — Example notebooks/scripts to update
- `examples/qpe_stretched_n2.ipynb` — uses `create("time_evolution_builder", "trotter")` and `create("controlled_evolution_circuit_mapper", "pauli_sequence")`
- `examples/extended_hubbard.ipynb` — same pattern
- `examples/interoperability/qiskit/iqpe_trotter.py` — same + `evolution_time` param
- `examples/interoperability/qiskit/iqpe_model_hamiltonian.py` — same pattern
- `examples/utils/qpe_utils.py` — `compute_evolution_time` function (utility, no direct API change needed)

### Step 6.2 — Documentation examples
- `docs/source/_static/examples/python/phase_estimation.py` — uses `create()` with old strings
- `docs/source/_static/examples/python/time_evolution_builder.py` — builder demo
- `docs/source/_static/examples/python/circuit_mapper.py` — imports `ControlledTimeEvolutionUnitary`

### Step 6.3 — Update strategy: use new factory strings, verify backward compat
- Examples should be updated to new strings (`"unitary_builder"`, `"controlled_circuit_mapper"`)
- Old strings still work via backward compat in registry

---

## Relevant Files

### Core source (to modify)
- `python/src/qdk_chemistry/data/time_evolution/containers/base.py` — rename `TimeEvolutionUnitaryContainer` → `UnitaryContainer`
- `python/src/qdk_chemistry/data/time_evolution/base.py` — rename `TimeEvolutionUnitary` → `UnitaryRepresentation`
- `python/src/qdk_chemistry/data/time_evolution/controlled_time_evolution.py` — rename `ControlledTimeEvolutionUnitary` → `ControlledUnitary`
- `python/src/qdk_chemistry/data/__init__.py` — update exports
- `python/src/qdk_chemistry/algorithms/time_evolution/builder/base.py` — introduce `UnitaryBuilder` base, `TimeEvolutionBuilder` inherits
- `python/src/qdk_chemistry/algorithms/time_evolution/builder/__init__.py` — update exports
- `python/src/qdk_chemistry/algorithms/time_evolution/builder/trotter.py` — update base class import
- `python/src/qdk_chemistry/algorithms/time_evolution/builder/qdrift.py` — update base class import
- `python/src/qdk_chemistry/algorithms/time_evolution/builder/partially_randomized.py` — update base class import
- `python/src/qdk_chemistry/algorithms/time_evolution/controlled_circuit_mapper/base.py` — rename `ControlledEvolutionCircuitMapper` → `ControlledCircuitMapper`
- `python/src/qdk_chemistry/algorithms/time_evolution/controlled_circuit_mapper/__init__.py` — update exports
- `python/src/qdk_chemistry/algorithms/time_evolution/controlled_circuit_mapper/pauli_sequence_mapper.py` — update type refs
- `python/src/qdk_chemistry/algorithms/phase_estimation/base.py` — remove `evolution_time`, generalize types
- `python/src/qdk_chemistry/algorithms/phase_estimation/iterative_phase_estimation.py` — own `evolution_time`, use builder for phase→energy
- `python/src/qdk_chemistry/algorithms/phase_estimation/__init__.py` — update if needed
- `python/src/qdk_chemistry/algorithms/__init__.py` — update re-exports
- `python/src/qdk_chemistry/algorithms/registry.py` — new factory type strings + backward compat
- `python/src/qdk_chemistry/data/qpe_result.py` — generalize `from_phase_fraction` to accept callables
- `python/src/qdk_chemistry/plugins/qiskit/standard_phase_estimation.py` — own `evolution_time`, update types

### Tests (to update)
- `python/tests/test_time_evolution_circuit_mapper.py`
- `python/tests/test_time_evolution_trotter.py`
- `python/tests/test_time_evolution_qdrift.py`
- `python/tests/test_time_evolution_partially_randomized.py`
- `python/tests/test_time_evolution_container.py`
- `python/tests/test_controlled_time_evolution.py`
- `python/tests/test_phase_estimation_iterative.py`
- `python/tests/test_interop_qiskit_phase_estimation_standard.py`
- `python/tests/test_qpe_result.py`
- `python/tests/test_algorithms_registry.py`

### Examples/docs (to update)
- `examples/qpe_stretched_n2.ipynb`
- `examples/extended_hubbard.ipynb`
- `examples/interoperability/qiskit/iqpe_trotter.py`
- `examples/interoperability/qiskit/iqpe_model_hamiltonian.py`
- `docs/source/_static/examples/python/phase_estimation.py`
- `docs/source/_static/examples/python/time_evolution_builder.py`
- `docs/source/_static/examples/python/circuit_mapper.py`

### Unchanged
- `python/src/qdk_chemistry/utils/phase.py` — utilities remain, used by `TimeEvolutionBuilder` internally
- `python/src/qdk_chemistry/algorithms/time_evolution/builder/trotter_error.py` — Trotter-specific, no rename needed
- `examples/utils/qpe_utils.py` — `compute_evolution_time` is Trotter-specific utility, no API change

---

## Verification

1. **Unit tests pass:** Run full test suite `pytest python/tests/` — all existing tests must pass with both old and new names
2. **Deprecation warnings:** Verify importing old class names (`TimeEvolutionUnitary`, `ControlledTimeEvolutionUnitary`, etc.) triggers `DeprecationWarning`
3. **Factory backward compat:** Verify `create("time_evolution_builder", "trotter")` still works and `create("unitary_builder", "trotter")` also works
4. **QpeResult backward compat:** Verify `QpeResult.from_phase_fraction(..., evolution_time=t)` still works with deprecation warning
5. **Notebook smoke test:** Run `examples/qpe_stretched_n2.ipynb` end-to-end
6. **Type checking:** Run `mypy` or `pyright` if configured, verify no type regressions

---

## Decisions

- **Naming:** `UnitaryBuilder` / `UnitaryRepresentation` / `ControlledUnitary` / `UnitaryContainer`
- **Phase→energy:** Method on `UnitaryBuilder` (not a separate ABC) — builder knows its own model
- **Backward compat:** Old names remain as aliases with `DeprecationWarning` on import
- **Factory strings:** Rename primary to `"unitary_builder"` / `"controlled_circuit_mapper"`, old strings fall back with deprecation
- **Directory structure:** Keep `algorithms/time_evolution/` and `data/time_evolution/` for PR1 (minimize churn)
- **Excluded from PR1:** Block encoding builder, block encoding container, block encoding phase model — all deferred to PR2

---

## Open Design Questions

1. **How should `TimeEvolutionBuilder` store `evolution_time` for phase→energy?** The builder's `_run_impl` receives `time` as an argument, but `phase_to_energy()` needs it later. Options: (A) builder caches last-used `time` as state, (B) `phase_to_energy` takes `time` as argument, (C) `time` is in builder settings. Recommendation: (B) — `phase_to_energy(phase_fraction, **model_params)` where time_evolution passes `time=t`.

2. **Should `UnitaryBuilder._run_impl` have a fixed signature or use `**kwargs`?** If `**kwargs`, subclass signatures are implicit. If fixed to `(qubit_hamiltonian)`, time_evolution needs an override. Recommendation: base takes `(qubit_hamiltonian, **kwargs)`, subclasses override with specific signatures.
