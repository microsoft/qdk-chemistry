# Ground-state molecular energies with QPE: completed lab notebook

This completed example records the reference results for the documented QDK/Chemistry 2.0.0 workflow.

## Setup and provenance

- QDK/Chemistry version: 2.0.0
- Built-in implementation verification result: Passed for all required implementations.

## Calculation goal and reference plan

- Molecular system: Neutral singlet N2 at a fixed 1.85 angstrom bond length.
- Target quantity: Fixed-geometry selected-space total ground-state energy.
- Teaching target: Absolute IQPE error no greater than 1 milliHartree.
- Basis-set comparison: Hartree-Fock cc-pVDZ versus cc-pVTZ energy sensitivity.
- Active-space comparison: Initial valence CASCI versus entropy-selected CASCI.
- Quantum-algorithm reference: Exact CASCI energy of the selected active-space Hamiltonian.
- Error sources outside the final algorithmic comparison: Geometry, Born-Oppenheimer approximation, finite basis, and active-space truncation.

## Molecular input and mean-field calculation

- Geometry source: Two nitrogen atoms on the z axis.
- Nitrogen-nitrogen distance: 1.85 angstrom.
- Molecular charge: 0.
- Spin multiplicity: 1 (singlet).
- First basis set: cc-pVDZ.
- First Hartree-Fock energy: -108.418633697214 Hartree.
- Second basis set: cc-pVTZ.
- Second Hartree-Fock energy: -108.445215657498 Hartree.
- Basis-set sensitivity: 0.026581960284 Hartree = 26.581960 milliHartree = 69.790927 kJ/mol.
- Interpretation: This is the observed change between two finite-basis Hartree-Fock calculations, not the exact basis-set error.

## Active-space model

- Orbital representation used for selection: Natural orbitals from the initial valence-space CASCI one-particle RDM.
- Initial active space: CAS(10e,8o), alpha/beta counts (5,5), orbitals 2-9.
- Initial correlated method: Exact CASCI within the initial valence space.
- Initial determinant count: 3,136.
- Initial CASCI energy: -108.778369520882 Hartree.
- Natural-orbital CASCI energy: -108.778369520881 Hartree; change 1.99e-13 Hartree.
- Single-orbital entropies for orbitals 2-9: 0.021695655, 0.029962803, 0.547855061, 0.963884097, 0.963884097, 0.966011090, 0.966011090, 0.554008809.
- Entropy-gap evidence: Orbitals 4-9 form the high-entropy group; orbitals 2-3 are near 0.03 or below.
- Selected active space: CAS(6e,6o), alpha/beta counts (3,3), orbitals 4-9.
- Selected partition: Four inactive, six active, and 18 virtual spatial orbitals.
- Selected determinant count: 400.
- Selected active-space energy: -108.771051792900 Hartree.
- Energy increase from refinement: 0.007317727982 Hartree.
- Interpretation: The compact space retains the strongest entropy-based correlation evidence while documenting excluded correlation as model error.
- Algorithmic reference energy: -108.771051792900 Hartree.

## Qubit representation

- Fermion-to-qubit encoding: Jordan-Wigner with blocked alpha-then-beta mode ordering.
- Active spatial orbitals: 6.
- Active spin orbitals: 12.
- Compute-register qubits: 12.
- Pauli terms: 247 using default mapper thresholds.
- Fixed-electron-number subspace: 3 alpha and 3 beta electrons; 400 basis states.
- Core energy: -99.117775949333 Hartree.
- Mapped active-space ground-state energy: -9.653275843566 Hartree.
- Mapped selected-space total energy: -108.771051792900 Hartree.
- Difference from algorithmic reference: approximately -1.42e-14 Hartree.
- Interpretation: The mapping and fixed-sector construction reproduce the selected-space Hamiltonian within numerical precision.
- Excluded from compute-qubit count: Readout/workspace ancillas, error-correction overhead, and physical qubits.

## Trial state

- Source wavefunction: Selected-space CASCI ground state.
- Leading determinants (occupation, amplitude, weight, cumulative weight):
  - 222000, +0.694657450061, 0.482548972925, 0.482548972925
  - 202200, +0.333212127561, 0.111030321954, 0.593579294879
  - 220020, +0.333212127561, 0.111030321954, 0.704609616833
  - 200220, -0.200148173020, 0.040059291163, 0.731442617514
  - 2uddu0, +0.176754072510, 0.031242002149, 0.762684619662
  - 2duud0, +0.176754072510, 0.031242002149, 0.793926621811
  - 022002, +0.146520843429, 0.021468357559, 0.815394979370
  - 2udud0, +0.117548778913, 0.013817715424, 0.829212694794
- Selection rule: Retain the largest one, two, or four CASCI coefficients, then re-optimize amplitudes with PMC.
- One-determinant trial: Fidelity 0.482548972925; 12 compute qubits; 6 preparation logical gates; {'X': 6}.
- Two-determinant trial: Fidelity 0.586414643728; 12 compute qubits; 14 preparation logical gates; {'CNOT': 6, 'H': 2, 'Rz': 2, 'S': 2, 'X': 2}.
- Four-determinant trial: Fidelity 0.732385015551; 12 compute qubits; 30 preparation logical gates; {'CNOT': 16, 'H': 4, 'Rz': 4, 'S': 4, 'X': 2}.
- State-preparation method: QDK/Chemistry sparse-isometry implementation.
- Interpretation: More determinants improve fidelity but change preparation cost; register size is fixed by the selected spin-orbital space.

## Phase-estimation calculation

- Evolution time: 0.162738441405 inverse Hartree.
- Hamiltonian simulation: First-order Trotter product formula, one division, repeated approximate base unitary for controlled powers.
- Phase bits: 6.
- Shots per bit: 3.
- Complete IQPE runs: 20.
- Readout ancillas: 1 per iteration circuit.
- Simulator seeds: 42-61.
- Complete-run bitstring counts: {'110000': 19, '110001': 1}.
- Most frequent bitstring: 110000.
- Measured active-space energy: -9.652275843566 Hartree.
- Core energy added after phase estimation: -99.117775949333 Hartree.
- Total molecular energy estimate: -108.770051792900 Hartree.
- Algorithmic reference: -108.771051792900 Hartree.
- Signed difference: +0.001000000000 Hartree.
- Teaching-target result: Meets the 1 milliHartree target at its boundary.
- Observed repeated-run phase: approximately six minutes in the measured development run; runtime varies by computer.

## Conclusion

- Main result: The configured IQPE workflow reproduces the selected-space CASCI reference within 1 milliHartree.
- Supporting evidence: Exact mapped/CASCI agreement, 19 of 20 complete runs at bitstring 110000, and the reconstructed total-energy comparison.
- Basis-set limitation: cc-pVDZ is finite; the cc-pVDZ/cc-pVTZ difference is sensitivity evidence, not exact error.
- Active-space limitation: Refinement excludes correlation and raises the CASCI energy by about 7.32 milliHartree relative to the initial valence space.
- Quantum-algorithm limitations: Reference-guided evolution-time tuning is circular; first-order Trotter evolution, six phase bits, and finite sampling remain approximations.
- One strengthening change: Select evolution time without the known target energy, then increase phase precision and Trotter accuracy while measuring the resulting cost.
