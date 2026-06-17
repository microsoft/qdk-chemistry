# Design Doc — CT-F12 (Canonical Transcorrelated F12) Hamiltonian Construction

**Audience:** the coding agent implementing this feature. **Status:** approved design, not yet
implemented. **Branch:** `feature/ct-f12`.

**Primary sources (in repo root):**
- `084107_1_online.pdf` — Yanai & Shiozaki, *J. Chem. Phys.* **136**, 084107 (2012) ("the paper").
- `2211.09685v2.pdf` — Masteran *et al.*, *Comment* + Supplemental Material ("the Comment"). **This is
  the corrected reference of truth for the equations and ships reference numbers.**

**Supporting analysis:** `.local/codebase-knowledge/{STACK,STRUCTURE,ARCHITECTURE,CONVENTIONS,INTEGRATIONS,TESTING,CONCERNS}.md`.

---

## 0. How to use this doc

Implement in the milestone order of §9, gating each on the verification in that milestone. Do **not**
write new integral recurrences — wire libint2's existing geminal engines (§5). Validate against the
exact Ne numbers in §8 before considering the math correct. Honor the conventions in §10. Where a
design choice remains open (§11), the **recommended default** is given so you are never blocked; flag
the choice in your PR description for the maintainer (@wavefunction91).

---

## 1. What CT-F12 is (and the single place RDMs enter)

CT-F12 produces an *a priori*, Hermitian, **two-body effective Hamiltonian** H̄_F12 by an approximate
canonical (unitary) similarity transformation of the molecular Hamiltonian H with a fixed-amplitude
projected Slater-type geminal generator A:

> H̄_F12 = H + [H, A]_{1,2} + ½ [[F, A]_{1,2}, A]_{1,2}   (paper Eq. 9; Comment SM Eq. S2)

The output is exactly a dressed `(h̄, ḡ, core energy, Fock)` — the **same shape and quartic complexity
as the bare Hamiltonian** (paper §IV) — so it plugs into every existing downstream solver and qubit
mapper unchanged.

**Why RDMs appear at all (read this before the math).** The exact e^{−A}He^{A} is reference-free but
**3-body**. CT-F12 forces a 2-body result via generalized normal ordering + cumulant decomposition
**relative to a reference |Ψ₀⟩**: each BCH-generated 3-body operator string is rewritten as
Σ (contraction) × (≤2-body operator), where the contraction coefficients **are the reference 1-RDM
`D^p_q` and 2-RDM `D^{pq}_{rs}`** (paper Eqs. 5-6); the discarded piece is the connected 3-body
cumulant λ₃ (this discard *is* the `[...]_{1,2}` approximation). That is the **only** place a density
enters. It is why `D` and the pair-cumulant `D̄^{pr}_{qs} = 2(D^p_q D^r_s − ½ D^p_s D^r_q) − D^{pr}_{qs}`
(paper Eq. 28) show up in the final tensor elements. The Comment SM makes this literal: Eq. S1 still
carries 3-body (6-index) Ê operators; Eqs. S8/S10 rewrite one as [D×Ê] + [D×D] + [D₂×Ê].

---

## 2. Architectural decision (settled)

**CT-F12 is a NEW algorithm type with signature `(Wavefunction) → Hamiltonian`** — distinct from the
existing `hamiltonian_constructor`, which is `(Orbitals) → Hamiltonian` and must not change (the
codebase enforces one fixed `_run_impl` signature per algorithm type; configuration lives in
`Settings`, never runtime kwargs).

The input is a reference **`Wavefunction`** because it carries *both* things the construction needs:
1. its `Orbitals` via `Wavefunction::get_orbitals()` (`cpp/include/qdk/chemistry/data/wavefunction.hpp:909`)
   → AO basis, full MO coefficients, active/inactive/virtual partition;
2. its **active 1-/2-RDMs** via `get_active_one_rdm_spin_traced()` / `get_active_two_rdm_spin_traced()`
   (and spin-dependent / SBT-block variants), guarded by `has_active_one_rdm()` / `has_active_two_rdm()`
   (`wavefunction.hpp:1203,1209,1246,1285`).

**The decisive consequence — there is NO SR-vs-MR code branch.** The construction contracts whatever
RDMs the reference supplies. The container flattening (#523) makes this even cleaner: every
determinant-expansion reference is now a single **`StateVectorContainer`**
(`cpp/include/qdk/chemistry/data/wavefunction_containers/state_vector.hpp`), which subsumes the former
`SlaterDeterminant`/`Cas`/`Sci` containers (structurally identical apart from a type tag). The flavor
is set by the *expansion*, not the container type:

| Reference (provenance) | RDMs it provides | Flavor |
|---|---|---|
| HF — single-determinant `StateVectorContainer` | occupation RDMs, **lazily generated from the determinant occupations** when none were supplied (`state_vector.hpp:34-39,293-307,435`) → `D^i_j=2δ`, `D^{ik}_{jl}=4δδ−2δδ` | **SR** closed-shell |
| CASSCF/CASCI — multi-determinant `StateVectorContainer` | genuine active RDMs (built by `macis::CASRDMFunctor`, `algorithms/microsoft/macis_cas.cpp:99`; container at `:105`) | **MR** |
| selected CI/ASCI — multi-determinant `StateVectorContainer` | RDMs from the selected-CI expansion (`macis_asci.cpp:99`; container at `:138`) | **MR** from SCI |

Implement the **general RDM-contracted expressions once** (Comment SM S3/S4). SR is simply the
single-determinant limit (lazy occupation RDMs). Feeding a multi-determinant (CAS/SCI) wavefunction
"just works" and yields the multireference operator.

**Methods rationale (for context, not action):** the λ₃ truncation error is set by the reference. An
HF reference on a strongly correlated system makes λ₂/λ₃ large (poor closure, wrong integer
occupations); a CAS/SCI reference absorbs static correlation into D₁/D₂ so residual cumulants are
dynamic-only and small, and the strong-orthogonality projector Q̂₁₂ stays consistent (no
static/dynamic double counting). Net: balanced, state-specific near-CBS corrections for the
strong-correlation regime this toolkit targets.

---

## 3. Public API & registration surface

Mirror the existing `HamiltonianConstructor` stack 1:1, swapping the input type and adding a new
factory. (A new *type* — unlike a new *variant* — requires a new abstract base, factory, and pybind
binding.)

### C++ base + factory (new)
`cpp/include/qdk/chemistry/algorithms/effective_hamiltonian.hpp` (model on `algorithms/hamiltonian.hpp`):

```cpp
namespace qdk::chemistry::algorithms {

class EffectiveHamiltonianConstructor
    : public Algorithm<EffectiveHamiltonianConstructor,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Wavefunction>> {
 public:
  using Algorithm::run;
  virtual std::string name() const = 0;
  std::string type_name() const final { return "effective_hamiltonian_constructor"; }
 protected:
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference) const = 0;
};

struct EffectiveHamiltonianConstructorFactory
    : public AlgorithmFactory<EffectiveHamiltonianConstructor,
                              EffectiveHamiltonianConstructorFactory> {
  static std::string algorithm_type_name() { return "effective_hamiltonian_constructor"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_ct_f12"; }
};
}
```

### C++ registration
`cpp/src/qdk/chemistry/algorithms/effective_hamiltonian.cpp` (model on
`algorithms/hamiltonian.cpp:30-36`):

```cpp
std::unique_ptr<EffectiveHamiltonianConstructor> make_microsoft_ctf12_hamiltonian() {
  return std::make_unique<microsoft::CtF12HamiltonianConstructor>();
}
void EffectiveHamiltonianConstructorFactory::register_default_instances() {
  EffectiveHamiltonianConstructorFactory::register_instance(&make_microsoft_ctf12_hamiltonian);
}
```

### Python binding + registry (new type ⇒ binding required; use the `add-binding` skill)
- `python/src/pybind11/algorithms/effective_hamiltonian.cpp` — bind base + factory via the generic
  `bind_algorithm_factory<Factory, Base, …>()` template (model on the hamiltonian-constructor
  binding); wire into `module.cpp` **after** `Wavefunction` and `Hamiltonian` are bound (binding order
  matters).
- `python/src/qdk_chemistry/algorithms/registry.py` — `register_factory(EffectiveHamiltonianConstructorFactory)`
  (model on `registry.py:643-663`).
- `python/src/qdk_chemistry/algorithms/effective_hamiltonian_constructor.py` — re-export module; add
  `EffectiveHamiltonianConstructor` and `QdkCtF12HamiltonianConstructor` to `algorithms/__init__.py`
  `__all__`.

### Resulting user API
```python
from qdk_chemistry.algorithms import create
ctf12 = create("effective_hamiltonian_constructor", "qdk_ct_f12", gamma=1.5, cabs_basis="aug-cc-pvtz-optri")
h_bar = ctf12.run(reference_wavefunction)   # SD wfn → SR; CAS/SCI wfn → MR
```

---

## 4. Concrete implementation (the `qdk_ct_f12` backend)

`cpp/src/qdk/chemistry/algorithms/microsoft/ctf12_hamiltonian.{hpp,cpp}`:
`class CtF12HamiltonianConstructor : public algorithms::EffectiveHamiltonianConstructor`,
`name() == "qdk_ct_f12"`, owning a `CtF12HamiltonianSettings` (§7).

`_run_impl(reference)` orchestration:
1. `auto orbitals = reference->get_orbitals();` — reuse the AO core-Hamiltonian build and active-space
   folding already in `microsoft/hamiltonian.cpp` (T+V(+ECP) via `OneBodyIntegral`; inactive folding
   into core energy + inactive Fock). **Fold using the F12-dressed Fock**, not the bare Coulomb Fock,
   so the reference is the F12-HF reference (paper §III.B).
2. Read reference RDMs (`get_active_one/two_rdm_spin_traced()`, guarded by `has_active_*`); these feed
   the cumulant contractions.
3. Build the internal SCF `BasisSet` (OBS) via `microsoft/utils.cpp` `convert_basis_set_from_qdk`
   (`utils.cpp:264-288`) and the **CABS** (§5).
4. Compute Coulomb `g` and the **F12-family MO integrals** over the index spaces occ {i,j}, OBS-virt
   {a,b}, CABS {x,y} (§5).
5. Build the geminal coefficients `G^{αβ}_{ij}` (paper Eqs. 10-13), the F12-dressed Fock `f`, and the
   intermediates `V, X, B, U, S` (§6) — with the **Eq. 27 correction**.
6. Assemble `h̄` and `ḡ` per **Comment SM Eqs. S3/S4**, contracting the reference RDMs.
7. Emit `std::make_shared<data::Hamiltonian>(std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
   H_active, two_body_vector, orbitals, core_energy, inactive_fock, HamiltonianType::Hermitian))`
   (restricted ctor; see `hamiltonian_containers/canonical_four_center.hpp`). The container stores the
   **full dense n⁴** 2-body tensor — required because ḡ is only 4-fold symmetric (§6).

New integral/CABS helpers live under the SCF subsystem (mirror existing Coulomb/DF style):
`cpp/src/qdk/chemistry/algorithms/microsoft/scf/.../{geminal_eri,cabs}.{h,cpp}` — add to the relevant
`CMakeLists.txt`.

---

## 5. Integrals & CABS strategy — lean on libint2, **no new integral recursions**

**Decisive fact:** the pinned integral provider is the **MPQC4 libint export**
(`libint-2.9.0-mpqc4.tgz`, `cpp/cmake/third_party.cmake:39-51`; `cgmanifest.json:39-46`) — the same
libint family used by the Comment's authors (MPQC). It is generated with geminal (F12) integral
classes, so the AO kernels are expected to be available. **We only wire them**, mirroring the existing
Coulomb path (`scf/.../util/libint2_util.cpp` `opt_eri`/`debug_eri`; `scf/.../core/moeri.h`
`MOERI::compute(nao,nt,Ci,Cj,Ck,Cl,out)`).

| Need | libint2 `Operator` | Used in |
|------|--------------------|---------|
| Coulomb (pq\|rs) | `coulomb` (already wired) | g, V, U, Fock |
| Slater factor F12 = −γ⁻¹·(1−e^{−γr}) ... (geminal) | `stg` (genuine) or `cgtg` (Gaussian fit) | G, X |
| F12·Coulomb | `stg_x_coulomb` / `cgtg_x_coulomb` | V, U |
| \|∇F12\|² / [t, F12] kernel | `delcgtg2` | B |

Two new wiring layers:
1. **Geminal AO-ERI helper** — given a libint2 geminal `Operator` (+ γ) and up to four (possibly
   different) basis sets, return AO integrals. Parameterize the existing `opt_eri` pattern by operator;
   reuse libint2 `Engine`.
2. **Geminal MO transform** — reuse the mixed-coefficient `MOERI::compute(nao,nt,Ci,Cj,Ck,Cl,out)`
   overload, feeding CABS MO coefficients as one or more C-blocks to reach occ × {OBS-virt ⊕ CABS}.

**CABS construction (new — none exists today).** Adopt Valeev's **CABS+** (paper refs 12, 14): load an
RI/OptRI auxiliary basis; form OBS∪RI; build the cross-overlap S(OBS, RI) with a libint2 1-body
overlap `Engine` over two basis sets; canonical-orthogonalize (SVD) the RI block against OBS to get
orthonormal CABS {x,y}. Reuse the existing two-basis plumbing (`eri_df(obs, abs, …)`, `metric_df(abs)`
in `libint2_util.h`) and the name→`BasisSet` loader (`BasisSet::from_database_json`,
`scf/.../core/basis_set.cpp:233-315`), which builds a basis for any *named* set independent of the
molecule's MOs.

**Basis data dependency:** OptRI/CABS bases (`aug-cc-pVXZ/OptRI`, optionally `cc-pVXZ-F12`) are **not
bundled** — only `aug-cc-pV{D,T,Q,5,6}Z` OBS are present under
`scf/.../resources/compressed/`. You must vendor OptRI tarballs there (and register them in
`basis_summary.json`), or source them from `qdk-chemistry-data` (§11.3).

---

## 6. The math to implement (corrected)

Use **Comment SM Eqs. S2-S4** as the source of truth for the final `h̄`/`ḡ` matrix elements (they are
the automated-SeQuant-verified, corrected forms). Key generator/projector definitions (paper):
- A_F12 = ½ G^{αβ}_{ij}(Ê^{αβ}_{ij} − Ê^{ij}_{αβ})   (Eq. 10)
- G^{αβ}_{ij} = (3/8)⟨αβ|Q̂₁₂ F12|ij⟩ + (1/8)⟨αβ|Q̂₁₂ F12|ji⟩   (Eq. 11; SP/"fixed" amplitudes)
- F12 = −γ⁻¹ exp(−γ r₁₂)   (Eq. 12)
- Q̂₁₂ = (1−Ô₁)(1−Ô₂) − V̂₁V̂₂   (Eq. 13); α,β span complete-virtual = OBS-virt {a,b} ⊕ CABS {x,y}; i,j
  are valence-occupied.

Intermediates (paper Eqs. 23-28; corrected):
- V^{pq}_{ij} = g^{pq}_{αβ} G^{αβ}_{ij}   (Eq. 23; non-Hermitian — also need its adjoint Ṽ, SM S6-S7)
- X^{kl}_{ij} = G^{αβ}_{ij} G^{αβ}_{kl}   (Eq. 24)
- B^{kl}_{ij} = G^{αβ}_{ij} f^α_γ G^{γβ}_{kl} + G^{αβ}_{ij} f^β_γ G^{αγ}_{kl}   (Eq. 25; from [[F,A],A])
- U^{prs}_{ijb} = g^{pr}_{xs} G^{xb}_{ij}   (Eq. 26)
- **S^{klb}_{ija} = G^{xa}_{ij} G^{yb}_{kl}[f^x_y + δ^x_y(f^a_a − f^i_i − f^j_j)]   (Eq. 27 — DROP the
  leading ½ shown in the paper; the Comment's bullet 1 identifies it as a spurious factor. Mandatory
  to reproduce the reference numbers.)**
- RDM factors `D^p_q`, `D^{pr}_{qs}`, and pair-cumulant `D̄^{pr}_{qs}` (Eq. 28) are **read from the
  reference wavefunction**, not hard-coded. Determinant limit → occupation identities (SR).

**Permutational symmetry (critical for downstream):** ḡ keeps **4-fold** symmetry
`ḡ^{pr}_{qs}=ḡ^{rp}_{sq}=ḡ^{qs}_{pr}=ḡ^{sq}_{rp}` (Hermiticity) but **NOT** `ḡ^{pr}_{qs}=ḡ^{rp}_{qs}`
(paper p.4). Therefore:
- Produce and store the **full dense n⁴ tensor** (the container does not assume 8-fold compression).
- Native QDK MP2 is safe: `microsoft/mp2.cpp:316-336` reads `(ia|jb)` and `(ib|ja)` independently from
  the full tensor (no 8-fold assumption).
- For solvers that *do* assume 8-fold (CAS via MACIS, PySCF-CC), expose an optional
  `symmetrize_two_body` that returns `(ḡ^{pr}_{qs}+ḡ^{rp}_{qs})/2` (Yanai's TCE path — small, known
  error). **Probe these solvers before relying on them (M0/G2).**

**Frozen core:** implement formulation **(a)** (core excluded from the geminal-generating occupied set
only). The SM S3/S4 expressions are written in this formulation with primed (core-including) OBS
indices. Avoid (a+b2), which introduces core-orbital RDM elements.

---

## 7. Settings schema (`CtF12HamiltonianSettings : data::Settings`)

Declare all via `set_default(...)` in the constructor (locked at `run()`):

| Key | Type | Default | Constraint | Meaning |
|-----|------|---------|-----------|---------|
| `gamma` | double | `1.0` | `BoundConstraint{0, 100}` | Slater exponent γ (use **1.5** for the Ne benchmark) |
| `cabs_basis` | string | `""` | — | Named OptRI/CABS auxiliary basis; empty ⇒ derive from OBS |
| `frozen_core` | int | `0` | `BoundConstraint{0, …}` | # frozen core orbitals (formulation (a)) |
| `eri_method` | string | `"direct"` | `ListConstraint{"direct","incore"}` | mirror existing `HamiltonianSettings` |
| `slater_factor` | string | `"stg"` | `ListConstraint{"stg","cgtg"}` | genuine STG vs Gaussian fit |
| `symmetrize_two_body` | bool | `false` | — | symmetrize ḡ for 8-fold-only solvers |

---

## 8. Validation plan (oracle = Comment Table I, Ne atom)

Conditions (Comment p.2 / Table I): Ne; OBS = aug-cc-pV{D,T,Q}Z; CABS = aug-cc-pVXZ/OptRI; **γ = 1.5
a.u.**; genuine Slater factor (no Gaussian fit); **no density fitting**; frozen-core (a); no 8-fold
symmetrization. Validate **SR-first with an SD (HF) reference wavefunction**, then MR.

Ladder (each a regression test):
1. **G1 build gate** — assert libint2 geminal engines exist; compute one F12 AO block, compare to a
   hand/MPQC value.
2. **CABS sanity** — `S(OBS,CABS) ≈ 0`, `S(CABS,CABS) = I`.
3. **SD-RDM sanity** — with an HF determinant reference, the contracted RDMs equal `2δ` / `4δδ−2δδ`.
4. **Intermediate cross-check** — V, X, B match standard MP2-F12 intermediates (Ne/aDZ).
5. **F12-HF** (SD reference; no correlated solver):
   `E = E_core + Σ_i 2 h̄_ii + Σ_{ij}(2 ḡ_{iijj} − ḡ_{ijji})` over occupied; assert `E − E_HF` equals
   Table I "F12-HF (this work)": **aDZ −0.111555079**, aTZ −0.042845640, aQZ −0.019939990 Eh.
   *This is the decisive end-to-end test of h̄/ḡ.*
6. **F12-MP2** (SD reference) — native QDK MP2 on H̄ (re-canonicalize w.r.t. the F12-dressed Fock,
   paper §III.B): assert correlation energy equals Table I **aDZ −0.301361902**, aTZ −0.308391143,
   aQZ −0.313067546 Eh.
7. **MR check (CAS/SCI reference)** — feed a multi-determinant CASSCF/ASCI `StateVectorContainer`; run
   CASCI (ultimately
   CTSD) on H̄; target the paper's CH₂ ¹A₁–³B₁ gap (Table I, ~9 kcal/mol), CAS(6e,6o), VXZ-F12.
8. **F12-CCSD/CCSDT** (follow-on) — only via the PySCF plugin, which likely assumes ERI symmetry; use
   `symmetrize_two_body` or defer (gated on M0/G2).

Test files (auto-discovered / mirror-source):
- `cpp/tests/test_ctf12_hamiltonian.cpp` (gtest; globbed by `file(GLOB_RECURSE test_*.cpp)` — no
  CMake edit needed; run serially with `OMP_NUM_THREADS=1`).
- `python/tests/test_ct_f12_hamiltonian.py` (create→run on an SD Ne wfn→assert F12-HF/F12-MP2; later a
  CAS-wfn MR case).

---

## 9. Milestones & verification gates

- **M0 — Feasibility gates (before bulk math).** G1: confirm mpqc4 libint geminal engines at build
  time. G2: probe whether MACIS CAS/ASCI and PySCF-CC assume 8-fold ERI symmetry (native MP2 already
  shown safe). Outcome decides CC/CAS scope.
- **M1 — New-type scaffold + integrals + CABS.** Stand up the `effective_hamiltonian_constructor`
  base/factory/binding with an empty body; verify `create(...).run(wfn)` reaches the backend. Add
  geminal AO/MO helpers + CABS builder. Gate: CABS sanity + single-block F12 integral check.
- **M2 — General RDM-contracted assembly.** V/X/B/U/S (Eq. 27 corrected) → h̄/ḡ per SM S3/S4, RDMs
  from the reference. Validate SR with an SD reference. Gate: **SD-RDM sanity + F12-HF Ne/aDZ matches
  Table I**.
- **M3 — F12-MP2 validation** (SD reference) across aDZ/aTZ/aQZ; lint, docstrings, docs page, tests.
- **M4 — MR validation (CAS/SCI reference).** CH₂ S-T gap; **no new API** (same constructor, correlated
  input). (Semi-internal geminal excitations omitted, as in the paper.)
- **M5 — (optional) `symmetrize_two_body` + F12-CCSD**, or defer.

---

## 10. Conventions & gotchas (must follow)

Build / test / lint (see the `cpp-build`, `python-build`, `lint` skills):
- C++: `cmake --preset release -S cpp && cmake --build --preset release && cmake --install .local/release/build`.
- **ctest is serial-only: `OMP_NUM_THREADS=1 ctest --preset release` — NEVER pass `-j`** (cardinal rule).
- Python: C++ must be installed **first**; then
  `cd python && CMAKE_PREFIX_PATH=$(pwd)/../.local/release/install CMAKE_BUILD_PARALLEL_LEVEL=4 OMP_NUM_THREADS=1 pip install .[all] -v`.
  **Never** use `pip install -e` (editable) — it forces a ~30-min full C++ rebuild.
- Run only the relevant test subset, not the full suite (CI does full verification).
- `pre-commit run --all-files` (clang-format, Ruff @120 cols/double quotes, mypy, license-header check,
  `.pyi` stub check).

Code style:
- C++ namespace `qdk::chemistry`, backend under `…::algorithms::microsoft`; private members `_prefix`.
- **Comments minimal** — only clarify non-obvious math; no narrative comments.
- Python: Google-style docstrings, type hints required, **single-line** `Args:`/`Returns:`/`Raises:`
  params (Sphinx breaks on multi-line continuations).
- MIT license header at the top of every new file.
- New algorithm classes re-exported via `algorithms/__init__.py` `__all__`.

Other:
- **Spherical AO only** initially (Cartesian AO is an open `TODO` in `microsoft/utils.cpp:189`).
- In-core `O(N⁴)` integrals/intermediates (mirrors the paper's pilot) — document basis-size limits;
  defer DF/out-of-core.
- Do **not** modify the existing Coulomb integral paths — add geminal helpers alongside them.
- Do **not** commit unless the maintainer asks.

---

## 11. Open decisions (recommended defaults — confirm in PR)

1. **Type name:** `effective_hamiltonian_constructor` (generic — *recommended*) vs
   `transcorrelated_hamiltonian_constructor`.
2. **Slater factor:** genuine libint2 **STG** (*recommended* — digit-matches the reference table) vs
   Gaussian `cgtg` fit (~µEh deviation). Exposed via the `slater_factor` setting either way.
3. **CABS basis data:** vendor aug-cc-pVXZ/OptRI tarballs into `scf/.../resources/compressed/`
   (*recommended*), or source from `qdk-chemistry-data`.
4. **Frozen core:** formulation **(a)** as default (*recommended*; matches the reference table).
5. **Validation tolerance:** ≤ 1e-6 Eh vs the printed Table I digits for F12-HF/F12-MP2 (*recommended*).
6. **`symmetrize_two_body`:** provide the setting, default **off** (*recommended*).

---

## 12. Evidence index (file:line anchors)

- Base/factory pattern: `cpp/include/qdk/chemistry/algorithms/hamiltonian.hpp:31-137`;
  `cpp/src/qdk/chemistry/algorithms/hamiltonian.cpp:30-36`.
- Existing qdk constructor (reuse core/folding): `cpp/src/qdk/chemistry/algorithms/microsoft/hamiltonian.cpp:85-495`;
  settings `microsoft/hamiltonian.hpp:13-23`; second variant pattern `microsoft/cholesky_hamiltonian.hpp`.
- Wavefunction RDM + orbitals accessors: `cpp/include/qdk/chemistry/data/wavefunction.hpp:909,1203,1209,1246,1285`;
  single-determinant lazy RDMs in the unified `StateVectorContainer` (subsumes SD/CAS/SCI)
  `wavefunction_containers/state_vector.hpp:34-39,293-307,435`; CAS/SCI RDM build + container
  `microsoft/macis_cas.cpp:99,105`, `microsoft/macis_asci.cpp:99,138`.
- Hamiltonian container (output): `cpp/include/qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp`
  (restricted/unrestricted ctors; flat index `i·n³+j·n²+k·n+l`; full dense storage).
- Orbitals accessors: `cpp/include/qdk/chemistry/data/orbitals.hpp` (`get_basis_set`, `coefficients`/`get_coefficients`,
  `energies`/`get_energies`, `get_active_space_indices`, `get_inactive_space_indices`, `get_virtual_space_indices`,
  `get_num_molecular_orbitals`, `get_mo_overlap_*`).
- Integral engine: `scf/.../eri/eri_multiplexer.h` (`ERIMultiplexer::create`, `build_JK`);
  `scf/.../core/moeri.h` (`MOERI::compute`); `scf/.../util/int1e.h` (`OneBodyIntegral`);
  `scf/.../util/libint2_util.{h,cpp}` (Coulomb-only today; `eri_df`/`metric_df` two-basis helpers);
  basis lookup `scf/.../core/basis_set.cpp:233-315`; basis conversion `microsoft/utils.cpp:264-288`.
- libint2 (mpqc4, F12-enabled): `cpp/cmake/third_party.cmake:39-51`; `cpp/manifest/qdk-chemistry/cgmanifest.json:39-46`.
- MP2 reduced-symmetry safety: `microsoft/mp2.cpp:316-336`.
- Registry (Python): `python/src/qdk_chemistry/algorithms/registry.py:643-663`; `algorithms/__init__.py:29-33`.
- Docs wiring: `docs/source/user/comprehensive/algorithms/index.rst`.
- Corrected equations + reference numbers: `2211.09685v2.pdf` (Comment bullet 1 = Eq. 27 fix; SM S2-S4;
  Table I); original derivation `084107_1_online.pdf` (Eqs. 9-28, §III).
