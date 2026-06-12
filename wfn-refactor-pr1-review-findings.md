# PR #497 Review Findings — Feature-Branch Rubric

Scope: `feature/dbwy/symmetry_blocked_tensor`, 50 files, ~8000 LOC. Reviewed
against the feature-branch rubric, the planning doc
(`wfn-refactor-pr1-symmetry-blocked-tensor.md`), and the `domain-reference`
skill (`data-structures`, `design-rationale`, `notation-conventions`,
`plugin-conversions`).

Items are continuously numbered so you can reference them as "#N" in
line comments. Each item is self-contained — title, description, citation,
optional verification.

---

## Blockers

### 1. `CholeskyHamiltonianContainer.three_center()` throws at runtime in Python
- **What:** Pybind def is registered, but the return type `SymmetryBlockedTensor<3, double>` is not bound to Python.
- **Where:**
  - Bound: `python/src/pybind11/data/hamiltonian.cpp:500-511`
  - Missing rank-3 binding: `python/src/pybind11/data/symmetry/symmetry_blocked_tensor.cpp:124-134` (only ranks 1, 2, 4)
  - Missing Python export: `python/src/qdk_chemistry/data/symmetry.py`
- **Verified:** Called from Python against installed wheel — `TypeError: Unable to convert function return value to a Python type! The signature was (...) -> SymmetryBlockedTensor<3ul, double>`.
- **Also wrong:** Docstring at line 504 says "rank-2 symmetry-blocked tensor" and "Returns: SymmetryBlockedTensorRank2Real" — neither matches reality.
- **Fix:** Bind `SymmetryBlockedTensor<3, double>` (+ complex), export `SymmetryBlockedTensorRank3{,Complex}` from `data/symmetry.py`, and correct the docstring. Or remove the Python def and pick a different replacement story for the deprecated `get_three_center_integrals`.

### 2. `SparseHamiltonianContainer.two_body_integrals_sparse()` missing in Python
- **What:** `_v1_deprecation.py` advertises this as the replacement for two deprecated methods, but no Python binding exists. `SymmetryBlockedSparseMap` is also entirely unbound.
- **Where:**
  - Advertised: `python/src/qdk_chemistry/data/_v1_deprecation.py:101-102`
  - Missing pybind: nothing in `python/src/pybind11/data/hamiltonian.cpp` after `sparse_container.def(...)` block
- **Verified:** `hasattr(SparseHamiltonianContainer, 'two_body_integrals_sparse')` → `False`
- **Fix:** Bind `two_body_integrals_sparse()` and `SymmetryBlockedSparseMap`, or change the deprecation messages.

### 3. Five new anonymous `namespace { … }` blocks
- **What:** Direct violation of the stored repo rule "I want *zero* `namespace {`". The most recent commit on this branch (`4248378a2`) explicitly converted other test files away from this pattern.
- **Where:**
  - `cpp/src/qdk/chemistry/data/orbitals.cpp:26` (`to_u32_indices`)
  - `cpp/src/qdk/chemistry/data/wavefunction.cpp:427` (`_make_active_one_rdm_sbt_block`, `_make_active_two_rdm_sbt_block`)
  - `cpp/tests/data/test_orbitals_sbt.cpp:19`
  - `cpp/tests/data/test_basis_set_ao_symmetries.cpp:19`
  - `cpp/tests/symmetry/test_symmetry_blocked_index_set.cpp:15`
- **Note:** Pybind11 source files use anonymous namespaces by established convention — those are NOT flagged.
- **Fix:** Convert to `static` free functions (or named file-detail namespace for types).

### 4. Plugin migration absent; deprecation install fires in unmigrated code paths
- **What:** Planning doc §10/§15 require PySCF (and audit Qiskit/OpenFermion) plugin migration to SBT-native ctors in this PR, with exit criterion §15.4: "PySCF plugin emits zero warnings under `pytest -W error::DeprecationWarning`". Zero plugin files changed on this branch, but `_v1_deprecation.py` is installed unconditionally at package import.
- **Unmigrated call sites** still using v1 accessors:
  - `python/src/qdk_chemistry/plugins/pyscf/conversion.py:642,644`
  - `python/src/qdk_chemistry/plugins/pyscf/mcscf.py:56,143` (constructs the deprecated v1 ctor)
  - `python/src/qdk_chemistry/plugins/qiskit/qubit_mapper.py:136-137`
  - `python/src/qdk_chemistry/plugins/openfermion/conversion.py:141-142`
- **Verified:** `python -W error::DeprecationWarning -c "...get_one_body_integrals()"` raises `DeprecationWarning: Hamiltonian.get_one_body_integrals is deprecated. Use get_container().one_body_integrals() instead.`
- **Fix:** Either complete the migration (matching planning doc §10), or explicitly defer §10 in the PR description AND gate `_v1_deprecation._install_deprecation_warnings()` so it doesn't trip the unmigrated plugin paths.

### 5. Python deprecation message for spin-traced RDMs silently changes semantics
- **What:** The C++ deprecation tells users to "trace over the spin variant". The Python wrapper drops that half. `active_one_rdm()` / `active_two_rdm()` return spin-**dependent** SBTs, not spin-traced ones — distinct quantities per `data-structures.md`.
- **Where:** `python/src/qdk_chemistry/data/_v1_deprecation.py:116-117`
- **Compare:** C++ header `wavefunction.hpp` says `"Use @ref active_one_rdm() and trace over the spin variant instead."`
- **Fix:** Extend the Python wrapper messages to include the trace step, OR implement an SBT-aware spin-trace helper and point to that.

### 6. Orphaned Doxygen block on the deprecated Cholesky restricted ctor
- **What:** Two `/** … */` blocks stacked above one declaration. The original detailed param-list block was left in place when a new `@brief … @deprecated` block was added.
- **Where:** `cpp/include/qdk/chemistry/data/hamiltonian_containers/cholesky.hpp:35-64` (lines 35-52 are orphaned; 53-56 are the intended new block)
- **Why it's a blocker, not a nit:** Per `design-rationale.md`, the docs build is zero-warning-strict ("Build fails on any warnings — Makefile enforces zero-warning policy"). Orphaned blocks typically produce Doxygen warnings.
- **Asymmetry:** The deprecated unrestricted ctor (lines 66-80) lost its docblock entirely. Pick one shape for both.
- **Fix:** Remove the orphan; ensure both deprecated ctors carry the same minimal-but-complete docblock.

---

## Should-fix

### 7. `axes::spin(int two_s, bool equivalent)` — `two_s` is unused
- **What:** Speculative parameter; impl is `int /*two_s*/`; comment admits "currently ignored — the returned axis always carries the two spin-½ labels."
- **Where:**
  - Decl: `cpp/include/qdk/chemistry/data/symmetry/symmetry.hpp:688`
  - Impl: `cpp/src/qdk/chemistry/data/symmetry/symmetry.cpp:578`
- **Inconsistent call sites:** `basis_set.cpp:924` passes `1`; `orbitals.cpp:650,704`, `wavefunction.cpp:442,481`, and every test pass `0`. Both produce identical axes.
- **Fix:** Drop the parameter (rubric §1/§5: speculative configuration).

### 8. Cholesky "non-owning views" claim is wrong
- **What:** Header says "Non-owning views into _three_center_sbt blocks (for v1 dense access)"; implementation makes a full `MatrixXd` copy via `Eigen::Map`.
- **Where:**
  - Lie: `cpp/include/qdk/chemistry/data/hamiltonian_containers/cholesky.hpp:233-235`
  - Implementation: `cpp/src/qdk/chemistry/data/hamiltonian_containers/cholesky.cpp:340-353` (lambda `make_view`)
- **Contrast:** `canonical_four_center.cpp _init_h2_views` IS genuinely non-owning (uses `block_ptr` directly). Comment was likely copy-pasted across containers.
- **Fix:** Either change the doc to "owning copies" or store actual `Eigen::Map` views.

### 9. Stray TODO without an issue link
- **Where:** `cpp/src/qdk/chemistry/data/hamiltonian_containers/cholesky.cpp` `_init_three_center_views`: `// TODO: eliminate this copy by storing rank-3 blocks as MatrixXd directly.`
- **Rubric §4:** "TODO/FIXME/NOTE blocks that ship without an issue link and without changing behavior — they age into lies."
- **Fix:** File an issue and link it, or delete.

### 10. Two redundant `const_cast`s
- **What:** `block` is `std::shared_ptr<SparseMapBlock<4>>` (non-const) — `block.get()` already returns a non-const pointer; the cast is a no-op and suggests confusion about the type.
- **Where:**
  - `cpp/src/qdk/chemistry/data/hamiltonian_containers/sparse.cpp:653`
  - `cpp/include/qdk/chemistry/data/symmetry/symmetry_blocked_sparse_map.hpp:289`
- **Fix:** Drop both `const_cast`s.

### 11. Quartic-root inferred from array size
- **Where:** `cpp/src/qdk/chemistry/data/wavefunction.cpp` `_make_active_two_rdm_sbt_block`:
  ```cpp
  std::size_t n4 = static_cast<std::size_t>(aaaa.size());
  std::size_t n = 1;
  while (n * n * n * n < n4) ++n;
  ```
- **Why:** The caller `_build_active_two_rdm_sbt` already has `get_orbitals()` and could pass the active-space size directly.
- **Fix:** Pass `n_active` in as a parameter.

### 12. "Eagerly" comment vs lazy implementation
- **What:** Header doc says SBT RDM members are "built eagerly"; implementation is short-circuit lazy (`if (_active_one_rdm_sbt) return;`) called from the accessor.
- **Where:** `cpp/include/qdk/chemistry/data/wavefunction.hpp:640-644` vs `wavefunction.cpp _build_active_*_rdm_sbt`
- **Fix:** Align doc to actual behavior.

### 13. `SymmetryBlockedSparseMap` skips the codebase-wide serialization-version convention
- **What:** Every other DataClass in the diff carries `SERIALIZATION_VERSION = "0.1.0"` and validates via `validate_serialization_version`. This class has no version, no Python binding, and HDF5 paths that always `throw`.
- **Where:** `cpp/include/qdk/chemistry/data/symmetry/symmetry_blocked_sparse_map.hpp`
- **Fix:** Either complete the surface (add version + JSON validation + HDF5), or drop the half-implementation and document as runtime-only (the type's only consumer is `SparseHamiltonianContainer`, no Python exposure, no test coverage).

### 14. `mutable _ao_symmetries` / `_ao_extents` as defaulted config, not derived cache
- **What:** Existing `mutable` fields in `BasisSet` (`_basis_to_atom_map`, `_cached_num_atomic_orbitals`) are caches of derived quantities. The new fields are *defaulted configuration*, which crosses a different line from `design-rationale.md`: "Immutable data classes | Prevents accidental mutation of molecular data mid-pipeline."
- **Where:** `cpp/include/qdk/chemistry/data/basis_set.hpp:878-881`
- **Fix:** Extract `_init_ao_symmetries(...)` from the new SBT-aware ctor, call it from every ctor, drop the `mutable`.

### 15. Awkward `throw` inside ternary in a member-init list
- **Where:** `cpp/src/qdk/chemistry/data/basis_set.cpp:352-360`:
  ```cpp
  : BasisSet(name, shells,
             structure ? *structure
                       : throw std::invalid_argument("..."),
             nullptr, {}, atomic_orbital_type) {
  ```
- **Why:** Replaces a plain `if (!structure) throw …;`. Non-idiomatic and harder to read.
- **Fix:** Restore the explicit guard at the top of the ctor body, or delegate via a `shared_ptr<Structure>` ctor that validates.

---

## Nits

### 16. Dead `return "unknown";` branch in `to_string(AxisName)`
- **Where:** `cpp/src/qdk/chemistry/data/symmetry/symmetry.cpp:20-26`
- **Why:** Only `AxisName::Spin` exists today; the fallback is unreachable.

### 17. `active_*_rdm_block` returns by value — every call copies
- **Where:** `cpp/src/qdk/chemistry/data/wavefunction.cpp` `active_one_rdm_block`, `active_two_rdm_block`
- **Why:** Probably intentional (variant-of-references is awkward), but worth a one-line contract note on the docstring.

### 18. Untracked test artifacts in repo root
- **What:** `test_cas_rdm_serialization.h5`, `test_unrestricted.hamiltonian.h5` — tests aren't cleaning up.
- **Note:** Not in this branch's diff, but symptomatic; worth filing.

---

## Confirmation (no action needed)

- **C++ `Symmetries` → `SymmetryProduct` rename was correct.** Pre-existing pure-Python `Symmetries` data class (`data-structures.md`) made the original name a hard collision. The rename in `015e904e7` is the right resolution.
- **Hamiltonian `aaaa/aabb/bbbb` spin-channel block keys** match documented shape.
- **2-RDM chemist-convention flat layout** (`idx = p·n³ + q·n² + r·n + s`) preserved by the new rank-4 SBT.
- **pybind11 binding order** in `module.cpp` follows spec (`bind_symmetry` → SBT/SBIS → `single_particle_basis` → `orbitals`).
- **No `@copydoc`** in the new code (matches stated preference).
- **All new DataClass subclasses (except `SymmetryBlockedSparseMap`)** consistently follow the `SERIALIZATION_VERSION` + `QDK_LOG_TRACE_ENTERING` + `DATACLASS_TO_SNAKE_CASE` + JSON-error-wrapping pattern.

---

## Decision-pending (for you)

Several Blockers (#1, #2, #4, #6) depend on whether the planning doc's §10/§15 scope is in this PR or has been deferred. The PR is `draft` and the wfn-refactor is a multi-PR plan. If you've decided to ship PR-1 without the plugin migration:

- **(a)** Say so explicitly in the PR description, citing §10/§15.
- **(b)** Gate `_v1_deprecation._install_deprecation_warnings()` so it doesn't fire from inside unmigrated plugins (e.g., env var, or skip until plugins migrate).
- **(c)** Decide whether the partial Python SBT-3 surface (#1) ships at all in this PR, or gets removed until its consumer follow-on PR is ready.
- **(d)** Decide the same about `two_body_integrals_sparse()` (#2) and `SymmetryBlockedSparseMap` (#13).
