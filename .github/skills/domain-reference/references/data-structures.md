# Data Structures — What Each Type Holds

## C++ Core Types

These are bound via pybind11 and live in `qdk_chemistry._core.data`.

| Type | What It Stores | Key Access Patterns |
|------|---------------|-------------------|
| **`Structure`** | N×3 coordinates (Bohr), Element enums, masses, nuclear charges | `get_coordinates()`, `get_elements()`, `calculate_nuclear_repulsion_energy()`, XYZ I/O |
| **`Orbitals`** | Alpha/beta MO coefficient matrices [AO×MO], energy vectors, AO overlap matrix, basis set ref, active/inactive indices | `get_alpha_coefficients()`, `get_mo_overlap_*()`, restricted vs unrestricted auto-detected |
| **`Hamiltonian`** | Wraps `HamiltonianContainer` — core energy + 1e integrals (alpha/beta) + 2e integrals (spin channels: aaaa, aabb, bbbb) | `get_one_body_integrals()`, `get_two_body_integrals()`, `get_core_energy()`, FCIDUMP I/O |
| **`Wavefunction`** | Wraps `WavefunctionContainer` — determinants, coefficients (real or complex), RDMs (spin-traced and spin-dependent), entropies | `truncate(n)`, `overlap(other)`, `norm()`, `get_top_determinants(n)` |
| **`Settings`** | Type-safe key-value store using `variant<bool, int64_t, double, string, vector<...>>`. Schema defined at construction, locked before `run()`. | Dict-like: `settings["key"]`, `settings.set("key", val)`, constraints (bounds or allowed-value lists) |
| **`BasisSet`** | Gaussian basis set specification (shells, exponents, contraction coefficients) | Named lookup: `"sto-3g"`, `"def2-svp"`, `"cc-pvdz"` etc. |
| **`PauliOperator`** | Sparse representation: `vector<pair<qubit_idx, op_type>>` where op=0(I),1(X),2(Y),3(Z). Expression tree: Sum/Product/Single. | `simplify()`, `distribute()`, `to_canonical_terms()`, arithmetic operators |
| **`Configuration`** / **`ConfigurationSet`** | Electronic configurations (occupation vectors) | Determinant enumeration |
| **`Ansatz`** | Combines Hamiltonian + Wavefunction for quantum simulation | Bridge between classical and quantum parts |

## Wavefunction Container Variants

| Container | What It Represents | When Used |
|-----------|-------------------|-----------|
| `SlaterDeterminantContainer` | Single determinant (HF reference) | After SCF |
| `CasWavefunctionContainer` | Complete Active Space CI | After CASCI/CASSCF |
| `SciWavefunctionContainer` | Selected CI | After ASCI |
| `CoupledClusterContainer` | CC amplitudes (T1, T2) | After CCSD |
| `MP2Container` | MP2 correlation | After MP2 |

## Hamiltonian Container Variants

| Container | Storage Format | When Used |
|-----------|---------------|-----------|
| `CanonicalFourCenterHamiltonianContainer` | Full 4-index tensor | Default, small systems |
| `CholeskyHamiltonianContainer` | Cholesky-decomposed 2e integrals | Large systems, memory-efficient |
| `SparseHamiltonianContainer` | Sparse storage | Lattice/model Hamiltonians |

## Pure Python Data Classes

These inherit from `DataClass` and live in pure Python.

| Type | What It Stores | Key Features |
|------|---------------|-------------|
| **`QubitHamiltonian`** | `list[str]` pauli_strings + `ndarray` coefficients + encoding label + fermion_mode_order | `equiv()`, `is_hermitian()`, `group_commuting()`, `to_interleaved()`, `schatten_norm` |
| **`Circuit`** | QASM / QIR / Q# representations (lazy-evaluated) | `get_qasm()`, `get_qiskit_circuit()`, format conversion |
| **`EnergyExpectationResult`** | Energy + variance | From estimators |
| **`QpeResult`** | QPE workflow results | From phase estimation |
| **`Symmetries`** | Physical symmetry info | Passed to qubit mappers |
| **`DataClass` (base)** | Provides immutability (frozen after `__init__`), serialization (JSON/HDF5), filename validation (`name.type.ext` pattern) | `to_file()`, `from_file()`, `to_json()`, `from_json()` |
