# Data Flow — The Core Pipeline

## What This Project Is

QDK/Chemistry is a quantum chemistry package: it takes a molecule → computes its electronic structure → maps the problem to qubits → runs quantum algorithms. The compute-heavy work is C++20; the user-facing API is Python via pybind11. Third-party tools (PySCF, Qiskit, OpenFermion) are integrated as swappable plugins.

**Package identity**: `qdk-chemistry` (pip install), `qdk_chemistry` (import), `qdk::chemistry` (C++ namespace).

## The Core Pipeline

Every workflow follows this chain. Each arrow is a concrete algorithm/constructor call:

```
Structure (atoms + Bohr coordinates)
  │
  │  ScfSolver.run(structure, charge, spin_multiplicity, basis)
  ▼
Wavefunction (MO coefficients, energies, occupations)
  │         └── wraps a WavefunctionContainer (SlaterDeterminant, CAS, SCI, CC, MP2)
  │
  │  ActiveSpaceSelector.run(wavefunction) → Orbitals (active subset)
  ▼
Orbitals (alpha/beta coefficient matrices [AO×MO], energies, active/inactive indices)
  │
  │  HamiltonianConstructor.run(orbitals)
  ▼
Hamiltonian (core energy + 1e integrals [norb×norb] + 2e integrals via container)
  │         └── wraps a HamiltonianContainer (CanonicalFourCenter, Cholesky, Sparse)
  │
  │  QubitMapper.run(hamiltonian) — Jordan-Wigner, Bravyi-Kitaev, etc.
  ▼
QubitHamiltonian (list[str] pauli_strings + ndarray coefficients + encoding label)
  │
  │  [optional] StatePreparation → Circuit → CircuitExecutor → EnergyEstimator
  ▼
Energy estimate (ground state or excited state)
```

## What Each Step Does

| Step | Algorithm Type | Input → Output | Description |
|------|---------------|----------------|-------------|
| SCF | `scf_solver` | `Structure` → `(energy, Wavefunction)` | Self-consistent field calculation (Hartree-Fock or DFT). Produces molecular orbital coefficients, energies, and occupations wrapped in a `Wavefunction` with a `StateVectorContainer` (single determinant). |
| Active Space Selection | `active_space_selector` | `Wavefunction` → `Wavefunction` | Selects a subset of orbitals for the active space. Produces `Orbitals` with active/inactive indices marked. |
| Hamiltonian Construction | `hamiltonian_constructor` | `Orbitals` → `Hamiltonian` | Computes one-electron and two-electron integrals in the molecular orbital basis. Wraps results in a `HamiltonianContainer`. |
| Qubit Mapping | `qubit_mapper` | `Hamiltonian` → `QubitHamiltonian` | Maps fermionic Hamiltonian to qubit operators using Jordan-Wigner, Bravyi-Kitaev, or parity encoding. |
| State Preparation | `state_preparation` | `Wavefunction` → `Circuit` | Prepares an initial quantum state (e.g., via sparse isometry). |
| Time Evolution | `time_evolution_builder` | `QubitHamiltonian` → `TimeEvolutionUnitary` | Builds a time evolution operator for Hamiltonian simulation. |
| Circuit Execution | `circuit_executor` | `Circuit` → result | Executes a quantum circuit on a simulator or hardware. |
| Phase Estimation | `phase_estimation` | `(QubitHamiltonian, Circuit)` → `QpeResult` | Runs quantum phase estimation to extract eigenvalues. |
| Energy Estimation | `energy_estimator` | `(QubitHamiltonian, Circuit)` → `EnergyExpectationResult` | Estimates energy expectation values. |

## Unit Conventions

- **Coordinates**: All internal coordinates are in **Bohr**. Angstrom conversion happens only at I/O boundaries (XYZ files).
- **Energies**: All energies are in **Hartree**.
