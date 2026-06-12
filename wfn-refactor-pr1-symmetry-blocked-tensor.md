# Wavefunction Refactor — PR-1: Symmetry-Blocked Storage Primitive

**This document is self-contained.** It specifies the scope, design, file
plan, and exit criteria for the first PR of the multi-PR wavefunction
refactor. An implementing agent should be able to execute this PR using only
this document plus the existing codebase. No external spec documents need
to be consulted.

---

## 1. Goal

Land the single-particle symmetry vocabulary and the `SymmetryBlockedTensor`
storage primitive; migrate the **internal storage** of `BasisSet`, `Orbitals`,
`HamiltonianContainer` (base + `CanonicalFourCenter` subclass), and
`WavefunctionContainer`'s RDM fields onto it; **add new symmetry-keyed
SBT-native accessors** alongside the v1 surface; and **mark the v1 dense /
spin-channel-named accessors `[[deprecated]]`** (Python: emit
`DeprecationWarning`). The Sz-symmetric convention from v1 (closed-shell +
open-shell with α/β labels) is the only supported configuration in this PR;
non-Sz / GHF / SOC / spinor pathways are out of scope.

This is a **refactor of underlying storage plus a formal deprecation pass**.
After this PR merges:

- Every container that today holds dense Eigen integrals or RDMs stores them
  inside `SymmetryBlockedTensor` blocks instead, behind unchanged v1 accessors
  that now emit `DeprecationWarning` on call.
- A parallel set of new SBT-native accessors (`h1()`, `h2()`, `one_rdm()`,
  `two_rdm()`, `*_block(SymmetryLabel, ...)`) is the recommended path for new
  and migrated consumers.
- The single-particle symmetry vocabulary (`Symmetries` / `SymmetryAxis` /
  `SymmetryLabel` / `axes::*`) exists and is the only construction surface for
  AO and MO label vocabularies.
- The many-body symmetry layer (`HilbertSpace`, `ManyBodyBasis`,
  `ManyBodySymmetryCondition`, `SymmetrySector`, `RdmRequest`, `RdmFacade`,
  `Ansatz` reshape, encoded-wavefunction work, serialization v2) is **not** in
  this PR. Follow-on PRs will layer those facade types on top.

The slogan: *land the storage primitive everywhere it should live, expose it
via formal symmetry-vocabulary accessors, and start the v1-surface
deprecation clock.*

---

## 2. Repository layout (reference)

| Layer | Path |
|---|---|
| C++ headers / impl / tests | `cpp/include/qdk/chemistry/` · `cpp/src/qdk/chemistry/` · `cpp/tests/` |
| pybind11 bindings | `python/src/pybind11/` (mirrors C++ tree) |
| Python package / tests | `python/src/qdk_chemistry/` · `python/tests/` (slow tests gated by `QDK_CHEMISTRY_RUN_SLOW_TESTS`) |
| Plugins | `python/src/qdk_chemistry/plugins/{pyscf,qiskit,openfermion,networkx}/` (auto-loaded on import) |
| Examples | `examples/` |
| Docs | `docs/` (Sphinx + Doxygen + Breathe) |

**Skills.** Use `cpp-build`, `python-build`, `lint`, `docs-build` skills. No `pip install -e` (triggers ~30 min full rebuild; use `pip install .[all]`).

**Conventions.** Data classes are immutable. Algorithms are stateless; configuration via `Settings` only. Google-style docstrings with single-line params. 120-char lines, double quotes (Ruff). Type hints required.

---

## 3. Design: single-particle symmetry vocabulary

All types live in `qdk::chemistry::symmetry`. Python bindings under `qdk_chemistry.symmetries`.

### 3.1 Core types

```cpp
namespace qdk::chemistry::symmetry {

// Closed enum. This PR populates only Spin; others declared for forward compat.
enum class AxisName { Spin, PointGroup, Kramers, Mode, Site, LatticeMomentum };

// Abstract base; held by shared_ptr<const> for interning.
class SymmetryAxisValue {
 public:
  virtual ~SymmetryAxisValue() = default;
  virtual AxisName    axis()                           const = 0;
  virtual std::string kind_name()                      const = 0; // serialization tag
  virtual bool        equals(const SymmetryAxisValue&) const = 0;
  virtual size_t      hash()                           const = 0;
};

// Concrete: spin-½ label. Value is 2·M_s (+1 for α, -1 for β).
class SpinValue : public SymmetryAxisValue {
  int _two_ms;
 public:
  explicit SpinValue(int two_ms);
  int value() const;
  AxisName    axis()      const override { return AxisName::Spin; }
  std::string kind_name() const override { return "spin"; }
  bool        equals(const SymmetryAxisValue& other) const override;
  size_t      hash() const override;
};

// One named symmetry partition the basis is blocked under.
class SymmetryAxis {
  AxisName                                              _name;
  std::vector<std::shared_ptr<const SymmetryAxisValue>> _labels;
  bool                                                  _equivalent;
 public:
  SymmetryAxis(AxisName name,
               std::vector<std::shared_ptr<const SymmetryAxisValue>> labels,
               bool equivalent);
  AxisName name() const;
  const std::vector<std::shared_ptr<const SymmetryAxisValue>>& labels() const;
  bool equivalent() const;
  bool operator==(const SymmetryAxis&) const;
  size_t hash() const;
};

// Vocabulary: names + label sets + equivalent flags.
class Symmetries {
  std::vector<SymmetryAxis> _axes;
 public:
  explicit Symmetries(std::vector<SymmetryAxis> axes);
  const std::vector<SymmetryAxis>& axes() const;
  bool operator==(const Symmetries&) const;
  size_t hash() const;
};

// Composite addressing key. One SymmetryAxisValue per SymmetryAxis.
class SymmetryLabel {
  std::map<AxisName, std::shared_ptr<const SymmetryAxisValue>> _values;
  size_t _hash;
 public:
  SymmetryLabel(std::initializer_list<std::shared_ptr<const SymmetryAxisValue>>);
  std::shared_ptr<const SymmetryAxisValue> get(AxisName) const;
  bool has(AxisName) const;
  bool operator==(const SymmetryLabel&) const;
  size_t hash() const { return _hash; }
};
} // namespace

namespace std {
template <> struct hash<qdk::chemistry::symmetry::SymmetryLabel> {
  size_t operator()(const qdk::chemistry::symmetry::SymmetryLabel& l) const noexcept {
    return l.hash();
  }
};
}
```

### 3.2 Axis factories

```cpp
namespace qdk::chemistry::symmetry::axes {
SymmetryAxis spin(int two_s, bool equivalent = true);

inline const std::shared_ptr<const SpinValue>& alpha() {
  static const auto v = std::make_shared<const SpinValue>(+1);
  return v;
}
inline const std::shared_ptr<const SpinValue>& beta() {
  static const auto v = std::make_shared<const SpinValue>(-1);
  return v;
}
std::shared_ptr<const SpinValue> spin_value(int two_ms);
}
```

Typical call sites — no raw strings:

```cpp
auto rhf_mo_sym = make_shared<Symmetries>({axes::spin(1, /*equivalent=*/true)});
auto uhf_mo_sym = make_shared<Symmetries>({axes::spin(1, /*equivalent=*/false)});
SymmetryLabel L_alpha = {axes::alpha()};
```

### 3.3 Serialisation registry

Per-subclass JSON/HDF5 via a construct-on-first-use registry (each `SymmetryAxisValue` subclass registers `(kind_name, from_json, from_hdf5)`; dispatch by `kind_name`).

### 3.4 NOT in this PR (many-body symmetry vocab)

Deferred: `ManyBodySymmetryCondition` abstract + concrete subclasses (`ParticleNumber`, `SpinProjection`, `TotalSpin`, `PointGroupIrrep`), `SymmetrySector`, many-body shorthand functions.

---

## 4. Design: SymmetryBlockedTensor + SymmetryBlockedIndexSet

### 4.1 SymmetryBlockedTensor

Immutable `DataClass`. Constructed all-at-once with the full block map; no mutation API. Block storage via `shared_ptr<const Tensor>` so orbit-equivalent blocks alias the same data.

```cpp
template <std::size_t Rank, class Scalar> struct TensorType;
template <class S> struct TensorType<1, S> { using T = Eigen::Matrix<S, Eigen::Dynamic, 1>; };
template <class S> struct TensorType<2, S> { using T = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>; };
template <class S> struct TensorType<4, S> { using T = Eigen::Matrix<S, Eigen::Dynamic, 1>; };
template <std::size_t Rank, class Scalar = double>
using Tensor = typename TensorType<Rank, Scalar>::T;

template <std::size_t Rank, class Scalar = double>
class SymmetryBlockedTensor : public DataClass {
 public:
  using Labels   = std::array<SymmetryLabel, Rank>;
  using BlockPtr = std::shared_ptr<const Tensor<Rank, Scalar>>;
  using BlockMap = std::unordered_map<Labels, BlockPtr, LabelsHash<Rank>>;

  SymmetryBlockedTensor(
      std::array<std::shared_ptr<const Symmetries>, Rank> symmetries,
      std::array<std::unordered_map<SymmetryLabel, size_t>, Rank> extents,
      BlockMap blocks);
  // Ctor validates:
  //   (1) Every extent key valid under matching axis's Symmetries.
  //   (2) Every block's shape matches per-axis extents.
  //   (3) Orbit completeness within the supplied set. Orbit equivalence
  //       only on the spin axis with α↔β swap:
  //       - Same Symmetries* across slots (operator-like: RDMs, h1/h2):
  //         α↔β swap simultaneously. Orbit = {(α,...,α), (β,...,β)}.
  //       - Distinct Symmetries* (intertwiner: BasisCoefficients):
  //         NO orbit auto-aliasing. Producer-driven sharing only.
  //   (4) No out-of-set block keys.
  // Errors: BlockExtentMismatchError, BlockAliasMismatchError, BlockLabelInvalidError.

  const Tensor<Rank, Scalar>& block(const Labels&) const;
  bool              has_block(const Labels&) const;
  BlockPtr          block_ptr(const Labels&) const;
  std::vector<Labels> canonical_block_labels() const;
  auto canonical_blocks() const;
  std::array<std::shared_ptr<const Symmetries>, Rank> symmetries() const;
  const std::array<std::unordered_map<SymmetryLabel, size_t>, Rank>& extents() const;
};

template <std::size_t Rank> struct LabelsHash {
  std::size_t operator()(const std::array<SymmetryLabel, Rank>&) const noexcept;
};
```

**Aliasing example (RHF 1-RDM).** MO Symmetries has `equivalent=true`. Producer supplies the αα block; ctor auto-aliases ββ. `block({{β, β}})` and `block({{α, α}})` return references to the same `MatrixXd`.

`is_restricted()` semantics: `true` iff spin-axis `equivalent=true` aliasing is present (pointer-equality on block storage, matching the existing `_coefficients.first == _coefficients.second` check on main).

Ship `Scalar = double` and `Scalar = std::complex<double>` instantiations for all three ranks. v1 already supports complex coefficients / RDMs via `MatrixVariant = std::variant<MatrixXd, MatrixXcd>` and `VectorVariant = std::variant<VectorXd, VectorXcd>` throughout `WavefunctionContainer`, with pybind11 bindings, `is_complex()`, and arithmetic helpers all on main. The SBT must support both scalar types to be a drop-in replacement. Per-instantiation serialisation handler (JSON/HDF5) reused by all downstream containers.

### 4.2 SymmetryBlockedIndexSet

```cpp
class SymmetryBlockedIndexSet : public DataClass {
  std::shared_ptr<const Symmetries> _symmetries;
  std::unordered_map<SymmetryLabel, size_t> _extents;
  std::unordered_map<SymmetryLabel, std::vector<uint32_t>> _indices;
 public:
  SymmetryBlockedIndexSet(std::shared_ptr<const Symmetries>,
                          std::unordered_map<SymmetryLabel, size_t> extents,
                          std::unordered_map<SymmetryLabel, std::vector<uint32_t>> indices);
  // Validates: label admissibility, extent coverage, index < extent, sorted-unique.
  std::span<const uint32_t> indices(const SymmetryLabel&) const;
  std::vector<SymmetryLabel> labels() const;
  std::shared_ptr<const Symmetries> symmetries() const;
};
```

Python binding: `indices(label) → tuple[int, ...]` (immutable copy).

### 4.3 pybind11

Standard pybind11 bindings. SBT blocks are Eigen types so existing pybind11 Eigen support handles the Python ↔ C++ conversion. Expose block accessors that return numpy arrays via the usual Eigen-to-numpy path.

---

## 5. Design: BasisSet AO symmetries

Modify existing `BasisSet`:

- Add `shared_ptr<const Symmetries> _ao_symmetries` and `unordered_map<SymmetryLabel, size_t> _ao_extents`.
- Default: `axes::spin(1, true)`, each spin label extent = `num_basis_functions()`.
- Accessors: `ao_symmetries()`, `ao_extents()`.
- v1 ctor unchanged. Add overloaded ctor for non-default `ao_symmetries`.
- Validates: extent keys admissible; orbit-equal extents (`BasisSetSpinExtentMismatchError`).
- `num_basis_functions()` unchanged (spatial count from `_shells`). `sum(ao_extents)` ≠ `num_basis_functions()` when `equivalent=true`.
- Free helper: `ao_symmetries(shared_ptr<const SingleParticleBasis>)` → `nullptr` for `ModelOrbitals`, real for `Orbitals`.

---

## 6. Design: SingleParticleBasis + Orbitals refactor

### 6.1 Abstract base

```cpp
class SingleParticleBasis : public DataClass {
  virtual std::shared_ptr<const Symmetries> symmetries() const = 0;
  virtual std::unordered_map<SymmetryLabel, size_t> mo_extents() const = 0;
  virtual size_t num_modes() const = 0;
};
```

### 6.2 Orbitals

Reparented under `SingleParticleBasis`. Storage rewired to:

- `shared_ptr<const BasisCoefficients>` — `SymmetryBlockedTensor<2>` with axes `[ao_symmetries, mo_symmetries]`. Intertwiner case: distinct AO/MO Symmetries, no auto-aliasing. RHF/ROHF: producer supplies both αα/ββ with same `shared_ptr<const Tensor>`. UHF: distinct.
- `shared_ptr<const OrbitalEnergies>` — `SymmetryBlockedTensor<1>` over MO Symmetries.
- `shared_ptr<const OrbitalSpacePartitioning>` — 5 `SymmetryBlockedIndexSet` (Frozen=0, Inactive=1, Active=2, Virtual=3, External=4). Default-empty = all-active. Static `all_active(...)`.
- `optional<MatrixXd> _ao_overlap` — stays on `Orbitals` (relocation deferred).

**BasisCoefficients block shapes:**

| Case | AO Symmetries | MO Symmetries | Blocks |
|---|---|---|---|
| RHF | spin(eq=true) | spin(eq=true) | (α,α); (β,β) aliases via same shared_ptr |
| UHF | spin(eq=true) | spin(eq=false) | (α,α) and (β,β) separately |
| GHF | spin(eq=true) | no spin axis | (α,*) and (β,*) rectangular |
| ROHF | spin(eq=true) | spin(eq=true) | same as RHF; ROHF in OrbitalSpacePartitioning |

### 6.3 ModelOrbitals

Reparented under `Orbitals` (v1 substitutability). Members requiring `BasisSet` raise `ModelOrbitalsNoBasisSetError`.

### 6.4 New accessors

- `basis_coefficients() → shared_ptr<const BasisCoefficients>`
- `orbital_energies() → shared_ptr<const OrbitalEnergies>`
- `orbital_space_partitioning() → shared_ptr<const OrbitalSpacePartitioning>`

### 6.5 v1 deprecation

| Pattern | Behaviour | Deprecated? |
|---|---|---|
| `unchanged` | No change (e.g. `basis_set()`) | No |
| `deprecated-forwarder` | One-line into new container. E.g. `get_coefficients_alpha()` → `basis_coefficients()->block({{α-AO, α-MO}})` | **Yes** |
| `computational-method` | Body preserved, internals re-rooted. E.g. `get_mo_overlap_*`, `calculate_ao_density_matrix(...)` | **Yes** |
| `typed-error-on-non-Sz` | Raises `SymmetryConditionError` / `ModelOrbitalsNoBasisSetError` | Yes (unreachable in Sz-only) |
| `storage-migration-only` | HDF5/JSON deserialisers | No |
| `renamed` | `OrbitalsCannotInBasisError` → `ModelOrbitalsNoBasisSetError` (ship `using` alias + `[[deprecated]]`) | Yes (alias) |

Python: emit `DeprecationWarning` on deprecated methods.

### 6.6 OrbitalSpacePartitioning PySCF mapping

| PySCF | Category | Indices |
|---|---|---|
| `ncore` frozen-core | `Frozen` | `[0, ncore)` |
| (none) | `Inactive` | `[]` |
| `ncas` active | `Active` | `[ncore, ncore+ncas)` |
| `nmo-ncore-ncas` virtual | `Virtual` | `[ncore+ncas, nmo)` |
| (none) | `External` | `[]` |

RHF: same indices both spins. UHF: per-spin. No active space: `all_active(...)`.

---

## 7. Design: Hamiltonian containers storage refactor

### 7.1 Storage scope

**`HamiltonianContainer` base:** one-body integrals → `SymmetryBlockedTensor<2>` keyed by MO Symmetries. Inactive Fock → `SymmetryBlockedTensor<2>` keyed by **AO Symmetries** (§7.4).

**`CanonicalFourCenterHamiltonianContainer`:** two-body → `SymmetryBlockedTensor<4>` keyed by MO Symmetries. Flat VectorXd packing per block preserved.

**`Sparse`/`CholeskyHamiltonianContainer`:** inherit base refactor (one-body + inactive-Fock). Their two-body factor representations (sparse integrals and Cholesky decomposition vectors respectively) are structurally distinct from the canonical 8-fold-packed convention and will require a different SBT mapping strategy. **The implementing agent must also migrate these two-body factors onto SBT-compatible storage in this PR**, with new SBT-native accessors and v1 deprecation following the same pattern as `CanonicalFourCenter`. The exact SBT mapping for each (e.g. whether to use a sparse-block SBT variant, a flat per-channel SBT, or a side-by-side factorised primitive) is an implementation decision — pick the approach that preserves v1 numerical parity and raises to the RFC owner if the mapping is ambiguous.

Spin-channel mapping for unrestricted ctors:

| v1 field | SBT block key |
|---|---|
| `one_body_integrals_alpha` | `({α}, {α})` |
| `one_body_integrals_beta` | `({β}, {β})` |
| `two_body_integrals_aaaa` | `({α}, {α}, {α}, {α})` |
| `two_body_integrals_aabb` | `({α}, {α}, {β}, {β})` |
| `two_body_integrals_bbbb` | `({β}, {β}, {β}, {β})` |

Restricted: `equivalent=true`; producer supplies α block; orbit aliases β.

### 7.2 New SBT-native accessors

On `HamiltonianContainer`:
- `const SymmetryBlockedTensor<2>& h1() const;`
- `const MatrixXd& h1_block(const SymmetryLabel& row, const SymmetryLabel& col) const;`
- `const SymmetryBlockedTensor<2>& inactive_fock() const;` (AO-basis)
- `const MatrixXd& inactive_fock_block(const SymmetryLabel& row, const SymmetryLabel& col) const;`

On `CanonicalFourCenterHamiltonianContainer`:
- `const SymmetryBlockedTensor<4>& h2() const;`
- `const VectorXd& h2_block(const SymmetryLabel& p, const SymmetryLabel& q, const SymmetryLabel& r, const SymmetryLabel& s) const;`

New SBT-native ctors:
- Base: `(SymmetryBlockedTensor<2> one_body, shared_ptr<const Orbitals>, double core_energy, SymmetryBlockedTensor<2> inactive_fock)`. Validates Symmetries.
- CanonicalFourCenter: above + `SymmetryBlockedTensor<4> two_body`.

### 7.3 v1 deprecation

Mark all v1 dense/spin-channel ctors + getters `[[deprecated]]`. `SpinChannel` enum stays **un-deprecated** (removal depends on `Ansatz` dispatch, out of scope).

C++ `[[deprecated]]` suppressable inside implementation TUs. Python: `DeprecationWarning`.

### 7.4 Inactive Fock: AO × AO Symmetries

SBT axes both keyed by AO Symmetries (`BasisSet::ao_symmetries()`). Storage is AO-basis. v1 MO-active getters become `computational-method` deprecated forwarders that perform AO→MO-active transform on access. New `inactive_fock()` returns AO-basis directly.

### 7.5 NOT in this PR

`FermionicHamiltonianContainer`, `SumHamiltonianContainer`, `Hamiltonian` ↔ `HilbertSpace`, `make_hamiltonian`, `Ansatz` reshape, `SpinChannel` deprecation.

---

## 8. Design: WavefunctionContainer RDM storage refactor

### 8.1 Storage migration

- 4 spin-dependent 2-RDM channels → single `SymmetryBlockedTensor<4>` (αααα, ααββ, ββββ blocks).
- 2 spin-dependent 1-RDM channels → single `SymmetryBlockedTensor<2>` (αα, ββ blocks).
- Spin-traced caches stay as lazily-computed `MatrixVariant`/`VectorVariant`.

### 8.2 New SBT-native accessors

- `const SymmetryBlockedTensor<2>& one_rdm() const;`
- `const SymmetryBlockedTensor<4>& two_rdm() const;`
- `const MatrixXd& one_rdm_block(const SymmetryLabel& row, const SymmetryLabel& col) const;`
- `const VectorXd& two_rdm_block(const SymmetryLabel& p, const SymmetryLabel& q, const SymmetryLabel& r, const SymmetryLabel& s) const;`
- `bool has_one_rdm() const;` / `bool has_two_rdm() const;`

New SBT-native ctor: optional `SymmetryBlockedTensor<2>` one_rdm, optional `SymmetryBlockedTensor<4>` two_rdm, plus existing base args.

### 8.3 v1 deprecation

Mark all v1 RDM getters, has-checks, and ctors `[[deprecated]]`:
- `get_active_*_rdm_*` → point at `one_rdm()` / `two_rdm()`.
- `has_*_rdm_*` → point at `has_one_rdm()` / `has_two_rdm()`.
- v1 ctors → point at SBT-native ctor.
- HDF5/JSON deserialiser helpers NOT deprecated (`storage-migration-only`).

### 8.4 Complex scalar — in scope

v1 already supports complex coefficients and RDMs via `MatrixVariant =
std::variant<MatrixXd, MatrixXcd>` and `VectorVariant = std::variant<VectorXd,
VectorXcd>`, with `is_complex()`, arithmetic helpers, and full pybind11
bindings on main. The SBT ships `Scalar = double` **and**
`Scalar = std::complex<double>` instantiations for all three ranks so it is a
complete drop-in replacement for the existing variant-based storage. The new
SBT-native accessors return `const SymmetryBlockedTensor<Rank, double>&` or
`const SymmetryBlockedTensor<Rank, std::complex<double>>&` as appropriate
(dispatched via the same variant pattern or templated accessors — implementation
choice).

---

## 9. Error hierarchy

```
QdkError
├── BasisSetError
│   └── BasisSetSpinExtentMismatchError
├── SingleParticleBasisError
│   ├── ModelOrbitalsNoBasisSetError
│   ├── ModelOrbitalsProjectionError
│   ├── OrbitalSpacePartitioningDisjointnessError
│   ├── IndexSetOutOfRangeError
│   └── IndexSetNotSortedUniqueError
├── SymmetryError
│   ├── SymmetryIncompatibleError
│   ├── SymmetryConditionError
│   └── SymmetryProjectionError
└── SymmetryBlockedTensorError
    ├── BlockExtentMismatchError
    ├── BlockAliasMismatchError
    └── BlockLabelInvalidError
```

Capability errors at access site; compatibility errors carry `std::source_location`; `std::expected<T, E>` for recoverable mismatches; exceptions for invariant violations. Pybind11 translators in `python/src/qdk_chemistry/data/exceptions.py`.

---

## 10. Plugin migration

PySCF migration covers **three surfaces** — all must use SBT-native ctors so the plugin emits **zero** `DeprecationWarning`s:

**Orbitals/BasisSet:** `conversion.py` uses new ctors with default `axes::spin(1, true)` Symmetries. BasisCoefficients per §6.2. OrbitalSpacePartitioning per §6.6. `scf_solver.py`, `active_space_avas.py`, `localization.py`, `mcscf.py` touched as needed.

**Hamiltonian:** whichever module constructs `CanonicalFourCenterHamiltonianContainer` migrates to SBT-native ctor. Inactive Fock in AO basis (PySCF provides AO-basis natively; skip v1 AO→MO transform).

**RDM extraction:** whichever module extracts RDMs constructs `SymmetryBlockedTensor<2>`/`<4>` directly.

**Audit:** grep `plugins/{qiskit,openfermion,networkx}/` for `Orbitals`, `BasisSet`, `ModelOrbitals`, `get_orbitals`, `get_one_body`, `get_two_body`, `get_inactive_fock`, `get_active_one_rdm`, `get_active_two_rdm`, `HamiltonianContainer`, `CanonicalFourCenterHamiltonianContainer`. If any non-PySCF plugin touches these: migrate. If none: document in PR description.

---

## 11. Pybind11 binding order

1. Symmetry vocabulary before SBT/SBIS.
2. SBT/SBIS before `BasisCoefficients`, `OrbitalEnergies`, `OrbitalSpacePartitioning`.
3. `bind_single_particle_basis(...)` **before** `bind_orbitals(...)` (reverse silently drops inheritance).

Ship `test_pybind_init_order.py` asserting `isinstance(Orbitals(...), SingleParticleBasis)` and `isinstance(ModelOrbitals(...), SingleParticleBasis)`.

---

## 12. Explicitly NOT in this PR

| Item | When |
|---|---|
| `ManyBodySymmetryCondition` + conditions + `SymmetrySector` | Follow-on |
| `HilbertSpace` + `ProductHilbertSpace` | Follow-on |
| `ManyBodyBasis` + `FockBasis` + `ConfigurationSet` amendment | Follow-on |
| `WavefunctionContainer` collapse to `StateVectorContainer` | Follow-on |
| `Wavefunction` facade reshape (shared_ptr ownership, getter removal) | Follow-on |
| `AmplitudeContainer` + CC/MP2 | Follow-on |
| `ReducedDensityMatrix` abstract + typed containers + free helpers | Follow-on |
| `RdmRequest` + `RdmFacade` + cache + `with_rdm` | Follow-on |
| Producer `RdmSettings` + focused calculators | Follow-on |
| `FermionicHamiltonianContainer` (new HilbertSpace-coupled sibling) | Follow-on |
| `SumHamiltonianContainer` | Follow-on |
| `make_hamiltonian` shorthand | Follow-on |
| `Ansatz` reshape + `calculate_energy()` dispatch | Follow-on |
| `SpinChannel` enum deprecation | Follow-on |
| `QubitHamiltonian` C++ type | Follow-on |
| TensorNetwork/MPS scaffold | Follow-on |
| Encoding chain (JW/BK/parity) | Post-Majorana |
| Serialization v2 + final sweep | Follow-on |
| Non-Sz / GHF / SOC pathways | Future |
| ~~Complex-scalar SBT~~ | **In scope for this PR** (see §8.4) |
| ~~Sparse/Cholesky two-body factor SBT migration~~ | **In scope for this PR** (see §7.1) |
| AO overlap relocation to BasisSet | Deferred |

**Guardrails.** If you reach for any deferred item, stop and escalate:

- "I need `factor_name`" → no; SBT-native ctors take `shared_ptr<const Orbitals>`.
- "I need `RdmRequest`" → no; SBT keyed by spin labels directly.
- "I need `HilbertSpace`" → validate against `Orbitals::symmetries()`.
- "I need `ConfigurationSet(basis, configs)`" → use v1 ctor.
- "I need to flip Wavefunction's container to shared_ptr" → no.
- "Sparse/Cholesky two-body SBT mapping is unclear" → pick best-fit mapping; escalate to RFC owner if ambiguous.
- "I should add `OneRdmContainer`/`TwoRdmContainer`" → no; this PR adds methods on existing `WavefunctionContainer`.

---

## 13. File-level work breakdown

Estimated: ~8,500–10,500 LOC (±30%).

### 13.1 Create

**C++ symmetry vocabulary:**
- `cpp/include/qdk/chemistry/symmetry/{axis_name,symmetry_axis_value,spin_value,symmetry_axis,symmetries,symmetry_label,axes_factory,labels_hash}.hpp` + mirror `.cpp` (except `labels_hash` header-only)

**C++ blocked tensor:**
- `cpp/include/qdk/chemistry/symmetry/{symmetry_blocked_tensor,symmetry_blocked_index_set}.hpp` + `.cpp`

**C++ single-particle basis:**
- `cpp/include/qdk/chemistry/data/single_particle_basis.hpp` + `.cpp`
- `cpp/include/qdk/chemistry/data/orbital_containers/{basis_coefficients,orbital_energies,orbital_space_partitioning}.hpp` + `.cpp`

**C++ errors:**
- `cpp/include/qdk/chemistry/errors/{qdk_error,basis_set_error,single_particle_basis_error,symmetry_error,symmetry_blocked_tensor_error}.hpp`

**pybind11:**
- `python/src/pybind11/symmetry/{axes_factory,symmetries,symmetry_label,symmetry_axis,spin_value,symmetry_blocked_tensor,symmetry_blocked_index_set}.cpp`
- `python/src/pybind11/data/{single_particle_basis,basis_coefficients,orbital_energies,orbital_space_partitioning}.cpp`

**Python:**
- `python/src/qdk_chemistry/symmetries/__init__.py`
- `python/src/qdk_chemistry/data/exceptions.py` (extend if exists)

**Tests:**
- `cpp/tests/symmetry/test_symmetry_vocab.cpp`
- `cpp/tests/symmetry/test_symmetry_blocked_tensor.cpp`
- `cpp/tests/symmetry/test_symmetry_blocked_index_set.cpp`
- `cpp/tests/data/test_single_particle_basis.cpp`
- `cpp/tests/data/test_orbital_containers.cpp`
- `cpp/tests/data/test_orbitals_shim.cpp`
- `cpp/tests/data/test_basis_set_ao_symmetries.cpp`
- `cpp/tests/data/test_hamiltonian_container_sbt_storage.cpp`
- `cpp/tests/data/test_wavefunction_rdm_sbt_storage.cpp`
- `cpp/tests/data/test_v1_deprecation_compile.cpp`
- `python/tests/test_symmetry_vocab.py`
- `python/tests/test_symmetry_blocked_tensor.py`
- `python/tests/test_orbitals_pybind.py`
- `python/tests/test_pybind_init_order.py`
- `python/tests/test_v1_deprecation_warnings.py`
- `python/tests/plugins/test_pyscf_orbitals_migration.py`

**Fixtures** (committed reference data + regeneration scripts):
- `python/tests/test_data/fixture_h2_sto3g_rhf/` (H₂ STO-3G RHF, 2 spatial orbitals)
- `python/tests/test_data/fixture_h2o_631g_rhf/` (H₂O 6-31G RHF, 13 spatial, ncore=1/ncas=7/virtual=5)
- `python/tests/test_data/fixture_h2o_631g_uhf_triplet/` (H₂O 6-31G UHF triplet Mₛ=1)

Tolerances: total energies 1e-6 Hartree; matrix elements 1e-8; integer exact.

### 13.2 Modify

**C++:**
- `basis_set.hpp` + `.cpp` — add `ao_symmetries()` + `ao_extents()` + ctor
- `orbitals.hpp` + `.cpp` — reparent, rewire storage, deprecate v1 methods
- `hamiltonian.hpp` + `.cpp` — storage migration, SBT-native ctors/accessors, deprecate v1
- `hamiltonian_containers/canonical_four_center.hpp` + `.cpp` — same
- `hamiltonian_containers/sparse.hpp` + `.cpp` — inherits base refactor + two-body factor migration to SBT + new accessors + v1 deprecation
- `hamiltonian_containers/cholesky.hpp` + `.cpp` — inherits base refactor + two-body factor migration to SBT + new accessors + v1 deprecation
- `wavefunction.hpp` + `.cpp` — RDM storage migration, SBT-native ctors/accessors, deprecate v1
- Consumers: `shared_ptr<const Orbitals>` for abstract type → `shared_ptr<const SingleParticleBasis>`

**pybind11:**
- `orbitals.cpp` — reparent `Orbitals` under `SingleParticleBasis` in binding
- `basis_set.cpp` — expose new accessors
- `wavefunction.cpp`, `hamiltonian.cpp`, `hamiltonian_containers/*.cpp` — verify bindings
- `module.cpp` — adjust bind order

**Python:**
- `qdk_chemistry/data/__init__.py` — export new types
- `qdk_chemistry/__init__.py` — wire `symmetries` subpackage
- `.pyi` stubs — regenerate

**PySCF plugin:**
- `conversion.py`, `scf_solver.py`, `active_space_avas.py`, `localization.py`, `mcscf.py`
- Hamiltonian construction module, RDM extraction module

### 13.3 Do NOT modify

- `configuration.hpp` / `configuration_set.hpp`
- `wavefunction_containers/*` (CAS/SCI/SD/CC/MP2)
- `ansatz.hpp`
- `algorithms/**`
- `plugins/{qiskit,openfermion,networkx}/**` (unless audit finds breakage)
- `examples/**`, `docs/**` (except new docstrings)

---

## 14. Deprecation policy summary

| Class | Deprecated | NOT deprecated |
|---|---|---|
| `BasisSet` | Nothing (additive) | Everything |
| `Orbitals` / `ModelOrbitals` | `deprecated-forwarder` + `computational-method` methods | `unchanged` methods, serialisation |
| `HamiltonianContainer` base | v1 ctors + dense getters + inactive_fock getters | `SpinChannel` enum |
| `CanonicalFourCenter` | v1 ctors + flat-VectorXd accessors | — |
| `Sparse`/`Cholesky` | Inherited base deprecations + their own two-body factor ctors/accessors | — |
| `WavefunctionContainer` | v1 RDM ctors + getters + has-checks | HDF5/JSON deserialiser helpers |
| `Wavefunction` facade | Nothing | Everything (reshape is follow-on) |
| Error names | `OrbitalsCannotInBasisError` alias | All others |

C++: `[[deprecated("message")]]`, suppressable in impl TUs. Python: `DeprecationWarning`. Final removal: no earlier than follow-on PRs.

---

## 15. Exit criteria

1. **Build green** on CI matrix (Linux + Windows + macOS; debug + release).
2. **Tests green** — full suites, `QDK_CHEMISTRY_RUN_SLOW_TESTS=1`.
3. **Numerical parity with main** on all fixtures (tolerances per §13.1).
4. **Deprecation verified** — every `[[deprecated]]` triggers compiler warning; every Python binding emits `DeprecationWarning`; PySCF plugin emits **zero** warnings under `pytest -W error::DeprecationWarning`.
5. **PySCF smoke** — SCF, active-space, CASSCF, RDM, Hamiltonian all via SBT-native ctors.
6. **Plugin audit** documented in PR description.
7. **Lint green** — `pre-commit run --all-files`.
8. **Docs build green** — Google-style docstrings, single-line params.
9. **`isinstance(Orbitals(...), SingleParticleBasis)`** returns True.
10. **Two reviewers** including ≥1 PySCF-plugin maintainer.
11. **RFC-owner sign-off.**

---

## 16. Resolved decisions

All resolved as of 2026-05-29:

1. ✅ **Complex-scalar SBT** — **in scope for this PR** (v1 already has full complex support via `MatrixVariant`/`VectorVariant`; SBT ships both `double` and `complex<double>` instantiations).
2. ✅ **Sparse/Cholesky two-body migration** — **in scope for this PR** (moved from "next agent" to in-scope per user clarification 2026-05-29). The implementing agent handles the SBT mapping for both.
3. ✅ **AO overlap** — stays on `Orbitals`.
4. ✅ **Inactive Fock Symmetries** — AO × AO by convention.
5. ✅ **`OrbitalsCannotInBasisError` rename** — proceed with `using` alias + `[[deprecated]]`.

---

**End of scope.** Implement within §§3–11 and §13.2. If a change falls in §12, stop and escalate.