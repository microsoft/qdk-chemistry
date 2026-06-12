# Wavefunction/Orbitals Refactor — Resume Notes (SBID + Sz-removal)

Branch: `feature/dbwy/flat_wfn`. All changes below are **uncommitted** in the
working tree. Builds use the cpp-build skill paths (`.local/release/...`);
**run ctest serially with `OMP_NUM_THREADS=1` (never `-j`)**.

This file is the pickup point for the in-flight refactor. It captures what is
done, what was last verified, and the remaining deferred Python work.

---

## 1. Goal (developer's stated intent)

- **No base/implicit Sz assumption anywhere.** Single-particle symmetry is
  always *explicit* data; the default is *no symmetry* (trivial). Restricted-ness
  is *inferred from symmetries*, never passed as a flag.
- **Active/inactive spaces are expressed as `SymmetryBlockedIndexSet` (SBID)**,
  not alpha/beta index tuples. The `Orbitals::RestrictedCASIndices` /
  `UnrestrictedCASIndices` tuple typedefs are to be **removed** ("garbage").
- Model systems (e.g. `ModelOrbitals` from `SparseHamiltonianContainer`) default
  to **spinless**; spin is requested by the top-level caller (chosen option:
  electronic model systems default spinless, v1 spin accessors throw until spin
  is attached).
- Sectors: production code stays explicit; constructors carry a default
  `sector = DEFAULT_SECTOR` (`"electrons"`) so tests can omit it.

---

## 2. Last known-GREEN state

After the **sectors test migration** (before the ModelOrbitals SBID redesign),
the full C++ suite was green: `OMP_NUM_THREADS=1 ctest` → **875/875 passed**.
That commit-worthy checkpoint = Sz-removal + sectors-default migration. If you
want to bail out of the SBID redesign, revert the ModelOrbitals-ctor changes in
`orbitals.{hpp,cpp}`, `python/src/pybind11/data/orbitals.cpp`, and the
`test_hamiltonian.cpp`/`test_model_orbitals.cpp` edits to get back here.

---

## 3. What is DONE (uncommitted)

### Sz removal (verified green earlier)
- `Orbitals::symmetries()` / `mo_extents()` no longer fabricate a spin axis;
  default = trivial. `_build_space_index_sets()` is symmetry-driven
  (`orbitals.cpp`).
- `_is_restricted_closed_shell()` gated on a declared spin axis
  (`wavefunction.cpp`); single-determinant lazy RDMs + `has_*_spin_dependent()`
  gated on spin axis (`state_vector.cpp`).

### Sectors default migration (verified green earlier)
- `DEFAULT_SECTOR = "electrons"` constant in `configuration_set.hpp`.
- `sector` defaulted + relocated *after the RDM/amplitude payload* in
  `StateVectorContainer` / `AmplitudeContainer` / `ConfigurationSet` ctors
  (C++ + pybind), so production passes it explicitly and tests omit it.
- Legacy `"sd"` deserialization version fixed (accepts `0.1.0`).
- `SectorLayout` was tried then **reverted to `std::pair`** per review.

### ModelOrbitals SBID redesign
- `orbitals.hpp`: replaced the 3 old ctors with **two**:
  - `explicit ModelOrbitals(size_t basis_size, shared_ptr<const SymmetryProduct> symmetries = nullptr)` — full active space; restricted inferred.
  - `explicit ModelOrbitals(shared_ptr<const SymmetryBlockedIndexSet> active, shared_ptr<const SymmetryBlockedIndexSet> inactive = nullptr)`.
  Sz-implying doc comments removed.
- `orbitals.cpp`: new ctor impls + file-local static helpers
  `model_restricted_from_symmetries`, `model_make_index_set`, and shared
  v1/SBID projection helpers. `from_json`/`from_hdf5` reconstruct via the new
  ctors.
- `python/src/pybind11/data/orbitals.cpp`: ModelOrbitals pybind updated to the
  two new ctors (`SymmetryBlockedIndexSet` is already bound in Python).
- `sparse.cpp::_make_orbitals` → `ModelOrbitals(n)` (trivial, spinless).
- `test_hamiltonian.cpp`: all `ModelOrbitals(n, true/false)` migrated. Added
  static helpers `model_spin_symmetry(bool)` and `trivial_index_set(n, idx)`.
  Cholesky tests opt into spin (`model_spin_symmetry(true/false)`); other
  container tests use trivial `ModelOrbitals(n)`.

### CASIndices removal / base `Orbitals` SBID migration
- Removed `Orbitals::RestrictedCASIndices` and
  `Orbitals::UnrestrictedCASIndices` from the public C++ header.
- Converted all dense and SBT-native `Orbitals` ctors to take
  `shared_ptr<const SymmetryBlockedIndexSet> active = nullptr,
  inactive = nullptr`.
- Migrated C++ source/tests away from tuple active/inactive constructor args.
  Scoped search is clean:
  `rg "RestrictedCASIndices|UnrestrictedCASIndices" cpp/include cpp/src cpp/tests python/src/pybind11`
  returns no matches.
- Localization and active-space code now carry SBIDs through constructor calls.
- Pybind `Orbitals` ctor signatures were source-migrated to SBID parameters so
  `_core` should compile, but Python plugins/tests are still deferred.

---

## 4. Current verification state

- `cmake --build .local/release/build -j 4` passed after the CASIndices/SBID
  cleanup.
- Full serial C++ ctest was started with `OMP_NUM_THREADS=1 ctest` and then
  stopped at the developer's request so CI/CD can run it. It did not complete in
  this workspace.
- Python package builds and installs successfully (`pip install .[all]`).
- Python non-slow test suite: **1769 passed** (5 previously-failing tests fixed,
  all now pass). Full suite excluding pyscf integration and qiskit tests ran
  successfully.

---

## 5. Remaining Python work (DEFERRED — needs full `pip install .[all]` ≈30 min)

> Editable installs are banned (full rebuild). Use `pip install .[all]`.
> Quick iteration during dev: rebuild `_core` in `.local/release/build_py` and
> copy the `.so` into the venv site-packages (see how this was done earlier),
> but the package wrappers there are stale — only good for targeted probes.

### 5a. PySCF plugin construction sites
- `plugins/pyscf/mcscf.py:132` — `ModelOrbitals(norb, True)` → `ModelOrbitals(norb)`
  (or attach spin via `SymmetryProduct([axes.spin(1, True)])` if the MCSCF path
  needs spin-resolved quantities downstream — check what consumes it).
- Base `Orbitals` ctors now take SBIDs, so these `Orbitals(...)` calls must
  build/pass `SymmetryBlockedIndexSet` objects (or pass nothing for full-active):
  `mcscf.py:416`, `scf_solver.py:303, 314, 324`,
  `active_space_avas.py:181, 192`, `localization.py:216, 238`.

### 5b. Python tests
- ModelOrbitals/Orbitals construction: `python/tests/test_orbitals.py`,
  `test_pyscf_plugin.py`, `test_hamiltonian.py`.
- **Sectors Python test migration was never run/verified.** The pybind `sector`
  defaults *should* make most of the (numerous) sector call sites work without
  edits, but this is unverified. Run the Python suite after a full rebuild and
  fix fallout. Suspect spots: `test_wavefunction.py`, `test_algorithms.py`,
  `test_rdms`/`test_configuration_set.py`, `test_qiskit_conversion.py`,
  `examples/extended_hubbard.ipynb` (QPE trial-state cell builds
  `StateVectorContainer(coeffs, det_keys, orbitals)` — fine via default sector).

### 5c. Verification
- `pip install .[all]` then `pytest` (slow tests gated by
  `QDK_CHEMISTRY_RUN_SLOW_TESTS`). Fix runtime fallout from spinless-by-default
  model orbitals (v1 spin accessors / spin-dependent RDMs now throw for trivial
  bases — callers wanting spin must opt in).

---

## 6. Suggested order to resume

1. Let CI/CD run the full serial C++ suite.
2. Do Python plugin/test migration for ModelOrbitals and Orbitals SBID ctor
   changes.
3. Run full `pip install .[all]` + pytest.

## 7. .pyi stubs
No `.pyi` stub updates were needed for ModelOrbitals (none found). Re-check if
the Python package migration adds generated/static stubs for the new SBID ctor
surface.
